//! Model + tokenizer + chat template, and the token generation loop shared by the CLI and server.

use crate::chat::{ChatTemplate, Message};
use crate::model::{Model, PROMPT_SLICE, Session};
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use rli_gguf::{Gguf, Result};
use std::path::Path;
use std::time::Instant;

pub struct Engine {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub template: ChatTemplate,
    pub model_name: String,
    stop: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// End-of-turn token generated.
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
        let template =
            ChatTemplate::detect(gguf.get("tokenizer.chat_template").and_then(|v| v.as_str()));
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
        Ok(Self {
            model,
            tokenizer,
            template,
            model_name,
            stop,
        })
    }

    pub fn new_session(&self, max_ctx: usize) -> Session {
        self.model
            .new_session(max_ctx.min(self.model.cfg.n_ctx_train))
    }

    pub fn chat_tokens(&self, messages: &[Message]) -> Vec<u32> {
        let prompt = self.template.render(messages);
        self.tokenizer
            .encode_untrusted(&prompt.text, self.tokenizer.add_bos, &prompt.untrusted)
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
        // Before truncating: a request cancelled while queued must not evict the cached prefix.
        if is_cancelled() {
            return GenOutput {
                text: String::new(),
                prompt_tokens: prompt.len(),
                prompt_evaluated: 0,
                completion_tokens: 0,
                finish: Finish::Cancelled,
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
        if prompt.len() > sess.max_ctx || prompt.is_empty() {
            return out;
        }

        let t0 = Instant::now();
        let to_eval = &prompt[keep..];
        let mut evaluated = 0;
        let mut last = None;
        for slice in to_eval.chunks(PROMPT_SLICE) {
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
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                // An incomplete trailing sequence waits for the next token; invalid bytes are
                // emitted lossily so a bad token can't stall output forever.
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => pending.len(),
            };
            if valid > 0 {
                let chunk = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                out.text.push_str(&chunk);
                if !on_text(&chunk) {
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
        if !pending.is_empty() {
            let rest = String::from_utf8_lossy(&pending).into_owned();
            on_text(&rest);
            out.text.push_str(&rest);
        }
        out.decode_secs = t1.elapsed().as_secs_f64();
        out
    }
}
