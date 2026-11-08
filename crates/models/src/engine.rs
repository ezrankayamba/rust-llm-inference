//! Model + tokenizer + chat template, and the token generation loop shared by the CLI and server.

use crate::chat::{ChatTemplate, Message};
use crate::model::{MAX_VERIFY, Model, Session};
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
    /// Prompt-lookup speculative decoding: up to this many tokens guessed per step from an
    /// earlier repeat of the latest n-gram, checked in one batched pass (0 = off). Output
    /// follows the same distribution; with quantized activations batched logits differ from
    /// one-token ones by float rounding, so a near-tie can resolve differently.
    pub lookahead: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// End-of-turn token generated.
    Stop,
    /// `max_tokens` or the context limit was reached.
    Length,
    /// The token callback asked to stop.
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

/// Longest and shortest n-gram (ending at the newest token) that prompt lookup matches.
const LOOKUP_NGRAM: (usize, usize) = (3, 2);

/// Up to `max` tokens that followed the most recent earlier occurrence of the last n-gram of
/// `hist`, trying the longest n-gram first; empty when nothing repeats.
fn lookup_drafts(hist: &[u32], max: usize) -> &[u32] {
    let (longest, shortest) = LOOKUP_NGRAM;
    for n in (shortest..=longest.min(hist.len())).rev() {
        let tail = &hist[hist.len() - n..];
        // Latest start first; the match must end before the tail itself does.
        if let Some(start) = (0..hist.len() - n).rev().find(|&i| &hist[i..i + n] == tail) {
            let from = start + n;
            return &hist[from..(from + max).min(hist.len())];
        }
    }
    &[]
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
        self.tokenizer.encode(&prompt, self.tokenizer.add_bos, true)
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop.binary_search(&id).is_ok()
    }

    /// Counts `token` as generated and passes any text it completes to `on_text`; `false` if
    /// that asked to cancel.
    fn emit(
        &self,
        token: u32,
        pending: &mut Vec<u8>,
        out: &mut GenOutput,
        on_text: &mut impl FnMut(&str) -> bool,
    ) -> bool {
        out.completion_tokens += 1;
        self.tokenizer.decode_bytes(token, pending);
        let valid = match std::str::from_utf8(pending) {
            Ok(s) => s.len(),
            // An incomplete trailing sequence waits for the next token; invalid bytes are
            // emitted lossily so a bad token can't stall output forever.
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => pending.len(),
        };
        if valid == 0 {
            return true;
        }
        let chunk = String::from_utf8_lossy(&pending[..valid]).into_owned();
        pending.drain(..valid);
        out.text.push_str(&chunk);
        on_text(&chunk)
    }

    /// Generates a continuation of `prompt` (the full token sequence, BOS included).
    ///
    /// If the session already holds a prefix of `prompt`, that prefix's KV cache is reused.
    /// `on_text` receives decoded text as it becomes valid UTF-8 and returns `false` to cancel.
    pub fn generate(
        &self,
        sess: &mut Session,
        prompt: &[u32],
        params: &GenParams,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> GenOutput {
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
        let mut logits = self
            .model
            .forward_batch(sess, &prompt[keep..], true)
            .expect("logits requested");
        out.prefill_secs = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut sampler = Sampler::new(params.sampler.clone());
        let mut pending: Vec<u8> = Vec::new();
        let vocab = self.model.vocab_size();
        let lookahead = params.lookahead.min(MAX_VERIFY - 1);
        // A token sampled while checking guesses, still to be emitted and run.
        let mut carried: Option<u32> = None;
        'decode: loop {
            if out.completion_tokens >= params.max_tokens || sess.len() >= sess.max_ctx {
                out.finish = Finish::Length;
                break;
            }
            let next = match carried.take() {
                Some(t) => t,
                None => sampler.sample(&mut logits, &sess.tokens),
            };
            if self.is_stop(next) {
                out.finish = Finish::Stop;
                break;
            }
            if !self.emit(next, &mut pending, &mut out, &mut on_text) {
                out.finish = Finish::Cancelled;
                // Keep the KV cache consistent with the tokens actually produced.
                self.model.forward(sess, next, false);
                break;
            }
            let room = lookahead
                .min(params.max_tokens - out.completion_tokens)
                .min(sess.max_ctx - sess.len() - 1);
            let mut batch = sess.tokens.clone();
            batch.push(next);
            let drafts = lookup_drafts(&batch, room).to_vec();
            if drafts.is_empty() {
                logits = self
                    .model
                    .forward(sess, next, true)
                    .expect("logits requested");
                continue;
            }
            // Run `next` and the guesses together; guess i stands if the token sampled after
            // everything before it equals it (exact for a guess with all its mass on one token).
            let base = sess.len();
            batch.drain(..base);
            batch.extend_from_slice(&drafts);
            let all = self.model.forward_batch_all(sess, &batch);
            let kv_dim = self.model.cfg.kv_dim();
            for (i, &guess) in drafts.iter().enumerate() {
                let mut l = all[i * vocab..(i + 1) * vocab].to_vec();
                let t = sampler.sample(&mut l, &sess.tokens[..base + 1 + i]);
                if t != guess {
                    sess.truncate(base + 1 + i, kv_dim);
                    carried = Some(t);
                    continue 'decode;
                }
                if self.is_stop(t) {
                    sess.truncate(base + 1 + i, kv_dim);
                    out.finish = Finish::Stop;
                    break 'decode;
                }
                if !self.emit(t, &mut pending, &mut out, &mut on_text) {
                    sess.truncate(base + 2 + i, kv_dim);
                    out.finish = Finish::Cancelled;
                    break 'decode;
                }
            }
            logits = all[drafts.len() * vocab..].to_vec();
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

#[cfg(test)]
mod tests {
    use super::lookup_drafts;

    #[test]
    fn lookup_drafts_follows_the_latest_longest_match() {
        // The trigram [1, 2, 3] last occurred at 5..8, so its continuation is [9, 4].
        let hist = [1, 2, 3, 4, 0, 1, 2, 3, 9, 4, 7, 1, 2, 3];
        assert_eq!(lookup_drafts(&hist, 2), &[9, 4]);
        // Capped by `max` and by the end of the history.
        assert_eq!(lookup_drafts(&hist, 1), &[9]);
        assert_eq!(lookup_drafts(&[5, 6, 7, 5, 6], 8), &[7, 5, 6]);
    }

    #[test]
    fn lookup_drafts_falls_back_to_bigrams_and_ignores_single_tokens() {
        assert_eq!(lookup_drafts(&[8, 2, 3, 6, 1, 2, 3], 3), &[6, 1, 2]);
        assert_eq!(lookup_drafts(&[4, 9, 1, 4], 3), &[] as &[u32]);
        assert_eq!(lookup_drafts(&[4], 3), &[] as &[u32]);
        assert_eq!(lookup_drafts(&[], 3), &[] as &[u32]);
    }
}
