//! Model + tokenizer + chat template, and the token generation loop shared by the CLI and server.

use crate::chat::{self, ChatDefaults, ChatTemplate, Message, RenderOpts, Rendered};
use crate::model::{Model, Session};
use crate::sampler::{Sampler, SamplerConfig};
use crate::stop::StopMatcher;
use crate::tokenizer::Tokenizer;
use rli_gguf::{Error, Gguf, Result};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub struct Engine {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub template: ChatTemplate,
    pub chat_defaults: ChatDefaults,
    /// Set when `template` is a guess, for binaries to show if they use chat formatting.
    pub template_warning: Option<String>,
    pub model_name: String,
    stop: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    /// Generation ends before the first of these appears in the output.
    pub stop: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// End-of-turn token generated, or a stop string reached.
    Stop,
    /// `max_tokens` or the context limit was reached.
    Length,
    /// `is_cancelled` or the token callback asked to stop.
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct GenOutput {
    pub text: String,
    pub prompt_tokens: usize,
    /// Prompt tokens actually evaluated (the rest were reused from the session's KV cache).
    pub prompt_evaluated: usize,
    pub completion_tokens: usize,
    pub finish: Finish,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

/// End-of-turn markers used by common chat formats, in addition to the declared EOS/EOT.
const STOP_TOKENS: [&str; 6] = [
    "</s>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "<|end|>",
];

impl Engine {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let gguf = Gguf::open(path)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let jinja = gguf.get("tokenizer.chat_template").and_then(|v| v.as_str());
        let detected = jinja.and_then(ChatTemplate::detect);
        let template = detected.unwrap_or(ChatTemplate::ChatMl);
        let chat_defaults = jinja.map_or_else(ChatDefaults::default, |j| {
            ChatDefaults::from_jinja(template, j)
        });
        let template_warning = detected.is_none().then(|| {
            let problem = if jinja.is_some() {
                "has an unrecognized chat template"
            } else {
                "has no chat template"
            };
            format!("model {problem}; assuming ChatML, so chat output may be wrong")
        });
        let model_name = gguf
            .get("general.name")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
        let mut stop: Vec<u32> = tokenizer.eos.into_iter().collect();
        if let Some(eot) = gguf
            .get("tokenizer.ggml.eot_token_id")
            .and_then(|v| v.as_u64())
        {
            stop.push(eot as u32);
        }
        stop.extend(
            STOP_TOKENS
                .iter()
                .filter_map(|s| tokenizer.token_id(s))
                .filter(|&id| tokenizer.is_control(id)),
        );
        stop.sort_unstable();
        stop.dedup();
        let model = Model::load(gguf)?;
        if model.n_vocab() != tokenizer.vocab_size() {
            return Err(Error::Format(format!(
                "{} tokens but {} embedding rows",
                tokenizer.vocab_size(),
                model.n_vocab()
            )));
        }
        Ok(Self {
            model,
            tokenizer,
            template,
            chat_defaults,
            template_warning,
            model_name,
            stop,
        })
    }

    pub fn new_session(&self, max_ctx: usize) -> Session {
        self.model
            .new_session(max_ctx.min(self.model.cfg.n_ctx_train))
    }

    /// The chat prompt for `messages`, with today's date (UTC) where the template wants one.
    pub fn render_chat(&self, messages: &[Message], opts: &RenderOpts) -> Rendered {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        self.template
            .render_with(messages, &self.chat_defaults, &chat::format_date(now), opts)
    }

    pub fn chat_tokens(&self, messages: &[Message]) -> Vec<u32> {
        self.chat_tokens_with(messages, &RenderOpts::default())
    }

    pub fn chat_tokens_with(&self, messages: &[Message], opts: &RenderOpts) -> Vec<u32> {
        let p = self.render_chat(messages, opts);
        self.tokenizer
            .encode_rendered(&p.text, self.tokenizer.add_bos, &p.untrusted, &p.opaque)
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop.binary_search(&id).is_ok()
    }

    /// Generates a continuation of `prompt` (the full token sequence, BOS included).
    ///
    /// If the session already holds a prefix of `prompt`, that prefix's KV cache is reused.
    /// `on_text` receives decoded text as it becomes valid UTF-8 and returns `false` to cancel.
    /// `is_cancelled` is polled before any work, between prompt slices and before each token, so
    /// a caller that has lost interest doesn't need text to arrive to stop. Cancelling leaves the
    /// session consistent: a retry reuses what was evaluated.
    pub fn generate(
        &self,
        sess: &mut Session,
        prompt: &[u32],
        params: &GenParams,
        is_cancelled: impl Fn() -> bool,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> GenOutput {
        // Before truncating: a request cancelled while queued, or one that can't fit, must not
        // evict the cached prefix.
        let fits = !prompt.is_empty() && prompt.len() <= sess.max_ctx;
        let cancelled = is_cancelled();
        if cancelled || !fits {
            return GenOutput {
                text: String::new(),
                prompt_tokens: prompt.len(),
                prompt_evaluated: 0,
                completion_tokens: 0,
                finish: if cancelled {
                    Finish::Cancelled
                } else {
                    Finish::Length
                },
                prefill_secs: 0.0,
                decode_secs: 0.0,
            };
        }
        let kv_dim = self.model.cfg.kv_dim();
        let common = sess
            .tokens
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        // Always re-evaluate at least the last prompt token: its logits are not cached.
        let keep = common.min(prompt.len().saturating_sub(1));
        sess.truncate(keep, kv_dim);

        let mut out = GenOutput {
            text: String::new(),
            prompt_tokens: prompt.len(),
            prompt_evaluated: prompt.len() - keep,
            completion_tokens: 0,
            finish: Finish::Length,
            prefill_secs: 0.0,
            decode_secs: 0.0,
        };
        let t0 = Instant::now();
        let to_eval = &prompt[keep..];
        let mut evaluated = 0;
        let mut last = None;
        for slice in to_eval.chunks(self.model.prompt_slice()) {
            if evaluated > 0 && is_cancelled() {
                out.prompt_evaluated = evaluated;
                out.finish = Finish::Cancelled;
                out.prefill_secs = t0.elapsed().as_secs_f64();
                return out;
            }
            evaluated += slice.len();
            last = self
                .model
                .forward_batch(sess, slice, evaluated == to_eval.len());
        }
        let mut logits = last.expect("logits requested");
        out.prefill_secs = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut sampler = Sampler::new(params.sampler.clone());
        let mut pending: Vec<u8> = Vec::new();
        let mut stops = StopMatcher::new(&params.stop);
        let mut stopped = false;
        loop {
            if out.completion_tokens >= params.max_tokens || sess.len() >= sess.max_ctx {
                out.finish = Finish::Length;
                break;
            }
            if is_cancelled() {
                out.finish = Finish::Cancelled;
                break;
            }
            let next = sampler.sample(&mut logits, &sess.tokens);
            if self.is_stop(next) {
                out.finish = Finish::Stop;
                break;
            }
            out.completion_tokens += 1;
            self.tokenizer.decode_bytes(next, &mut pending);
            let valid = complete_utf8(&pending);
            if valid > 0 {
                let chunk = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                let (ready, hit) = stops.push(&chunk);
                out.text.push_str(&ready);
                let wanted = ready.is_empty() || on_text(&ready);
                if hit {
                    out.finish = Finish::Stop;
                    stopped = true;
                    break;
                }
                if !wanted {
                    out.finish = Finish::Cancelled;
                    // Keep the KV cache consistent with the tokens actually produced.
                    self.model.forward(sess, next, false);
                    break;
                }
            }
            logits = self
                .model
                .forward(sess, next, true)
                .expect("logits requested");
        }
        if !stopped {
            let rest = String::from_utf8_lossy(&pending).into_owned();
            let (ready, hit) = stops.push(&rest);
            let mut tail = ready.into_owned();
            if hit {
                out.finish = Finish::Stop;
            } else {
                tail.push_str(&stops.finish());
            }
            if !tail.is_empty() {
                on_text(&tail);
                out.text.push_str(&tail);
            }
        }
        out.decode_secs = t1.elapsed().as_secs_f64();
        out
    }
}

/// How many leading bytes of `pending` can be emitted: everything but an incomplete trailing
/// sequence, which waits for the next token. Invalid bytes are emitted (lossily) so a bad token
/// can't stall output, but only up to the end of the last one: a character that starts after it
/// and is still incomplete keeps waiting.
fn complete_utf8(pending: &[u8]) -> usize {
    let mut n = 0;
    loop {
        match std::str::from_utf8(&pending[n..]) {
            Ok(s) => return n + s.len(),
            Err(e) => match e.error_len() {
                None => return n + e.valid_up_to(),
                Some(bad) => n += e.valid_up_to() + bad,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::complete_utf8;

    #[test]
    fn complete_utf8_holds_back_an_unfinished_character_after_an_invalid_byte() {
        // 0x80 is invalid; E6 97 starts "日" (E6 97 A5) and must wait for its last byte.
        assert_eq!(complete_utf8(&[b'a', 0x80, 0xE6, 0x97]), 2);
        assert_eq!(complete_utf8("日".as_bytes()), 3);
        assert_eq!(complete_utf8(&[0xE6, 0x97]), 0);
        assert_eq!(complete_utf8(&[0x80, 0x80]), 2);
    }
}
