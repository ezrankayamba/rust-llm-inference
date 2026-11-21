//! Model + tokenizer + chat template, and the token generation loop shared by the CLI and server.

use crate::chat::{self, ChatDefaults, ChatTemplate, Message, RenderOpts, Rendered};
use crate::model::{Model, Session};
use crate::sampler::{Sampler, SamplerConfig};
use crate::stop::StopMatcher;
use crate::tokenizer::Tokenizer;
use crate::tools::{Constraint, Grammar, TokenTable};
use rli_gguf::{Error, Gguf, Result};
use std::path::Path;
use std::sync::{Arc, OnceLock};
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
    /// The vocabulary as bytes, for constrained tool calls; built on first use.
    token_table: OnceLock<TokenTable>,
    draft: Option<Draft>,
}

/// A smaller model with the same vocabulary, whose guesses at the next tokens
/// [`Engine::generate`] checks in one pass of the main model, at temperature 0.
struct Draft {
    model: Model,
    /// The most tokens it guesses a pass.
    tokens: usize,
}

#[derive(Clone, Debug, Default)]
pub struct GenParams {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    /// Generation ends before the first of these appears in the output.
    pub stop: Vec<String>,
    /// Restricts what `<tool_call>` blocks in the reply can contain to calls of these tools.
    pub tools: Option<Arc<Grammar>>,
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

/// The share of a draft's recent guesses that must stand for [`Engine::generate`] to keep
/// drafting: about where a batch pass of 1.6-2x a token's time pays for itself.
const MIN_DRAFT_RATE: f32 = 0.6;
/// Most passes it goes without the draft while the draft keeps missing.
const MAX_DRAFT_PAUSE: usize = 16;

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
            token_table: OnceLock::new(),
            draft: None,
        })
    }

    /// Speculates with `draft` (loaded, its backend attached) when sampling at temperature 0:
    /// it guesses `tokens` tokens a pass, and the replies stay what this model alone writes,
    /// up to float rounding (a batch is evaluated in a different order than one token).
    pub fn set_draft(&mut self, draft: Engine, tokens: usize) -> std::result::Result<(), String> {
        let (a, b) = (&self.tokenizer, &draft.tokenizer);
        let same = a.vocab_size() == b.vocab_size()
            && draft.model.n_vocab() == self.model.n_vocab()
            && (0..a.vocab_size() as u32)
                .all(|i| a.token_str(i) == b.token_str(i) && a.is_special(i) == b.is_special(i));
        if !same {
            return Err(format!(
                "{} doesn't have {}'s vocabulary",
                draft.model_name, self.model_name
            ));
        }
        if self.model.logit_rows() < 2 {
            return Err("this backend can't check drafted tokens in one pass".into());
        }
        if tokens == 0 {
            return Err("a draft must guess at least one token".into());
        }
        self.draft = Some(Draft {
            model: draft.model,
            tokens,
        });
        Ok(())
    }

    pub fn new_session(&self, max_ctx: usize) -> Session {
        let max_ctx = max_ctx.min(self.model.cfg.n_ctx_train);
        let mut s = self.model.new_session(max_ctx);
        s.draft = self
            .draft
            .as_ref()
            .map(|d| Box::new(d.model.new_session(max_ctx)));
        s
    }

    /// Up to `room` tokens the draft expects after `tokens` and `batch` (which its session is
    /// brought up to first): `d.tokens` of its own choosing (`true`), each followed by the text
    /// a tool call's grammar then forces (`false`), as `generate` would add it.
    fn propose(
        &self,
        d: &Draft,
        ds: &mut Session,
        mut c: Option<&mut Constraint>,
        tokens: &[u32],
        batch: &[u32],
        room: usize,
    ) -> Vec<(u32, bool)> {
        let all = || tokens.iter().chain(batch).copied();
        let total = tokens.len() + batch.len();
        let common = ds.tokens.iter().zip(all()).take_while(|(a, b)| **a == *b);
        let keep = common.count().min(total - 1);
        ds.truncate(keep, d.model.cfg.kv_dim());
        let rest: Vec<u32> = all().skip(keep).collect();
        let mut logits = d
            .model
            .forward_batch(ds, &rest, true)
            .expect("logits requested");
        let mut out = Vec::new();
        for i in 0..d.tokens {
            if let Some(c) = c.as_deref_mut() {
                c.mask(&mut logits);
            }
            let t = argmax(&logits);
            if out.len() >= room || self.is_stop(t) {
                break;
            }
            out.push((t, true));
            let mut fresh = vec![t];
            if let Some(c) = c.as_deref_mut() {
                c.advance(t);
                let forced = c.jump(|s| self.tokenizer.encode(s, false, false), room - out.len());
                out.extend(forced.iter().map(|&f| (f, false)));
                fresh.extend(forced);
            }
            // The last guess needs no logits; the next pass brings the session up to date.
            if i + 1 < d.tokens && out.len() < room {
                logits = d
                    .model
                    .forward_batch(ds, &fresh, true)
                    .expect("logits requested");
            }
        }
        out
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

    pub fn token_table(&self) -> &TokenTable {
        self.token_table
            .get_or_init(|| TokenTable::new(&self.tokenizer))
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
        let mut constraint = params
            .tools
            .as_deref()
            .map(|g| Constraint::new(g, self.token_table()));
        let mut pending: Vec<u8> = Vec::new();
        let mut stops = StopMatcher::new(&params.stop);
        let mut stopped = false;
        let draft = self
            .draft
            .as_ref()
            .filter(|_| params.sampler.temperature <= 0.0);
        let mut draft_sess = draft.and(sess.draft.take());
        let mut draft_constraint = params
            .tools
            .as_deref()
            .map(|g| Constraint::new(g, self.token_table()));
        let vocab = self.model.n_vocab();
        // Checking guesses costs a batch pass, nearly the same for 2 rows as for 6 (Qwen3-14B
        // on Metal: 1.75x a one-token pass, 2.1x for 5), so a draft guesses all it may, and
        // only while enough of its recent guesses stand to pay for that: below that it waits
        // `backoff` passes, doubled each time it is still below, before trying again. Qwen3-0.6B
        // guesses ~86% of Qwen3-14B's tool calls and ~35% of its Swahili prose.
        let mut rate = 1.0f32;
        let (mut pause, mut backoff) = (0usize, 1);
        'tokens: loop {
            if out.completion_tokens >= params.max_tokens || sess.len() >= sess.max_ctx {
                out.finish = Finish::Length;
                break;
            }
            if is_cancelled() {
                out.finish = Finish::Cancelled;
                break;
            }
            if let Some(c) = &mut constraint {
                c.mask(&mut logits);
            }
            let next = sampler.sample(&mut logits, &sess.tokens);
            if let Some(c) = &mut constraint {
                c.advance(next);
            }
            if self.is_stop(next) {
                out.finish = Finish::Stop;
                break;
            }
            // Inside a tool call, the text the schema leaves no choice over follows the token
            // sampled, all evaluated in one pass.
            let room =
                (params.max_tokens - out.completion_tokens).min(sess.max_ctx - sess.len()) - 1;
            let mut batch = vec![next];
            if let Some(c) = &mut constraint {
                batch.extend(c.jump(|s| self.tokenizer.encode(s, false, false), room));
            }
            // And the draft's guesses at what comes next, checked in the same pass.
            let room = room + 1 - batch.len();
            let drafting = pause == 0;
            pause = pause.saturating_sub(1);
            let drafts = match (draft, draft_sess.as_deref_mut()) {
                (Some(d), Some(ds)) if room > 0 && drafting => {
                    if let (Some(dc), Some(c)) = (&mut draft_constraint, &constraint) {
                        dc.follow(c);
                    }
                    let dc = draft_constraint.as_mut();
                    self.propose(d, ds, dc, &sess.tokens, &batch, room)
                }
                _ => Vec::new(),
            };
            let base = sess.len();
            let mut seq = batch.clone();
            seq.extend(drafts.iter().map(|d| d.0));
            let rows = self
                .model
                .forward_rows(sess, &seq, 1 + drafts.len())
                .expect("logits requested");
            // A guess stands if the model would have chosen it there; the text forced after it
            // stands with it. Row j is the logits after `seq[batch.len() - 1 + j]`.
            let mut kept = batch.len();
            let (mut missed, mut hits) = (false, 0);
            for &(tok, guessed) in &drafts {
                if guessed {
                    let row = kept - batch.len();
                    let mut l = rows[row * vocab..(row + 1) * vocab].to_vec();
                    if let Some(c) = &mut constraint {
                        c.mask(&mut l);
                    }
                    if sampler.sample(&mut l, &sess.tokens[..base + kept]) != tok {
                        missed = true;
                        break;
                    }
                    hits += 1;
                }
                if let Some(c) = &mut constraint {
                    c.advance(tok);
                }
                kept += 1;
            }
            if !drafts.is_empty() {
                // Of the guesses checked: those that stood, and the one that didn't.
                rate = (rate + hits as f32 / (hits + usize::from(missed)) as f32) / 2.0;
                if rate < MIN_DRAFT_RATE {
                    pause = backoff;
                    backoff = (2 * backoff).min(MAX_DRAFT_PAUSE);
                } else {
                    backoff = 1;
                }
            }
            sess.truncate(base + kept, kv_dim);
            logits = if drafts.is_empty() {
                rows
            } else {
                let row = kept - batch.len();
                rows[row * vocab..(row + 1) * vocab].to_vec()
            };
            for (i, &id) in seq[..kept].iter().enumerate() {
                out.completion_tokens += 1;
                self.tokenizer.decode_bytes(id, &mut pending);
                let valid = complete_utf8(&pending);
                if valid == 0 {
                    continue;
                }
                let chunk = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                let (ready, hit) = stops.push(&chunk);
                out.text.push_str(&ready);
                let wanted = ready.is_empty() || on_text(&ready);
                if hit {
                    out.finish = Finish::Stop;
                    stopped = true;
                    // The token that completed the stop is counted but not kept.
                    sess.truncate(base + i, kv_dim);
                    break 'tokens;
                }
                if !wanted {
                    out.finish = Finish::Cancelled;
                    // Keep the KV cache consistent with the tokens actually produced.
                    sess.truncate(base + i + 1, kv_dim);
                    break 'tokens;
                }
            }
        }
        if draft_sess.is_some() {
            sess.draft = draft_sess;
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

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
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
