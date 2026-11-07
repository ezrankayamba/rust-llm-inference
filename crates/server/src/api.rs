//! OpenAI-compatible endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/completions`.

use crate::http::{self, Request};
use rli_models::{Engine, Finish, GenParams, Message, SamplerConfig, Session};
use serde_json::{Value, json};
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct App {
    pub engine: Engine,
    /// One shared KV cache: requests run one at a time (they would only compete for the same
    /// cores), and a client continuing a conversation reuses its cached prefix.
    pub session: Mutex<Session>,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn error_body(msg: &str) -> Vec<u8> {
    json!({"error": {"message": msg, "type": "invalid_request_error"}})
        .to_string()
        .into_bytes()
}

/// Sampling parameters with OpenAI's defaults (temperature 1, top_p 1, no penalty).
/// `top_k`, `repeat_penalty` and `seed` are accepted as extensions.
pub fn gen_params(req: &Value, room: usize) -> Result<GenParams, String> {
    let f = |k: &str| req.get(k).and_then(Value::as_f64);
    let u = |k: &str| req.get(k).and_then(Value::as_u64);
    let max_tokens = u("max_completion_tokens")
        .or_else(|| u("max_tokens"))
        .map_or(room, |n| (n as usize).min(room));
    let temperature = f("temperature").unwrap_or(1.0) as f32;
    if !(0.0..=2.0).contains(&temperature) {
        return Err("temperature must be between 0 and 2".into());
    }
    Ok(GenParams {
        max_tokens,
        sampler: SamplerConfig {
            temperature,
            top_p: f("top_p").unwrap_or(1.0) as f32,
            top_k: u("top_k").unwrap_or(0) as usize,
            repeat_penalty: f("repeat_penalty").unwrap_or(1.0) as f32,
            repeat_last_n: 64,
            seed: u("seed").unwrap_or_else(|| NEXT_ID.fetch_add(1, Ordering::Relaxed) ^ now()),
        },
    })
}

pub fn parse_messages(req: &Value) -> Result<Vec<Message>, String> {
    let msgs = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("`messages` must be an array")?;
    if msgs.is_empty() {
        return Err("`messages` must not be empty".into());
    }
    msgs.iter()
        .map(|m| {
            let role = m
                .get("role")
                .and_then(Value::as_str)
                .ok_or("each message needs a string `role`")?;
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                // Content parts: concatenate the text parts.
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| {
                        (p.get("type")?.as_str()? == "text")
                            .then(|| p.get("text")?.as_str())
                            .flatten()
                    })
                    .collect::<Vec<_>>()
                    .join(""),
                _ => return Err("each message needs `content`".to_string()),
            };
            Ok(Message::new(role, &content))
        })
        .collect()
}

fn finish_reason(f: Finish) -> &'static str {
    match f {
        Finish::Stop | Finish::Cancelled => "stop",
        Finish::Length => "length",
    }
}

pub fn handle(app: &App, req: Request, w: &mut impl Write) -> std::io::Result<()> {
    let json_ct = "application/json";
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/healthz") => http::respond(w, 200, "text/plain", b"ok"),
        ("GET", "/v1/models") => {
            let body = json!({"object": "list", "data": [{
                "id": app.engine.model_name, "object": "model", "created": 0, "owned_by": "rli"
            }]});
            http::respond(w, 200, json_ct, body.to_string().as_bytes())
        }
        ("POST", "/v1/chat/completions") => complete(app, &req.body, true, w),
        ("POST", "/v1/completions") => complete(app, &req.body, false, w),
        (_, "/healthz" | "/v1/models" | "/v1/chat/completions" | "/v1/completions") => {
            http::respond(w, 405, json_ct, &error_body("method not allowed"))
        }
        _ => http::respond(w, 404, json_ct, &error_body("not found")),
    }
}

fn complete(app: &App, body: &[u8], chat: bool, w: &mut impl Write) -> std::io::Result<()> {
    let bad = |w: &mut _, msg: &str| http::respond(w, 400, "application/json", &error_body(msg));
    let req: Value = match serde_json::from_slice(body) {
        Ok(v @ Value::Object(_)) => v,
        _ => return bad(w, "request body must be a JSON object"),
    };
    let e = &app.engine;
    let tokens = if chat {
        match parse_messages(&req) {
            Ok(m) => e.chat_tokens(&m),
            Err(msg) => return bad(w, &msg),
        }
    } else {
        match req.get("prompt").and_then(Value::as_str) {
            Some(p) => e.tokenizer.encode(p, e.tokenizer.add_bos, false),
            None => return bad(w, "`prompt` must be a string"),
        }
    };
    // A poisoned lock only means an earlier request panicked mid-generation; the session's
    // prefix check re-validates the cache, so it is safe to keep using.
    let mut sess = app.session.lock().unwrap_or_else(|p| p.into_inner());
    if tokens.len() >= sess.max_ctx {
        return bad(
            w,
            &format!(
                "prompt is {} tokens; context size is {}",
                tokens.len(),
                sess.max_ctx
            ),
        );
    }
    let params = match gen_params(&req, sess.max_ctx - tokens.len()) {
        Ok(p) => p,
        Err(msg) => return bad(w, &msg),
    };
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let id = format!(
        "{}-{}",
        if chat { "chatcmpl" } else { "cmpl" },
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    let (created, model) = (now(), e.model_name.as_str());
    let object = if chat {
        "chat.completion"
    } else {
        "text_completion"
    };

    if !stream {
        let out = e.generate(&mut sess, &tokens, &params, |_| true);
        let choice = if chat {
            json!({"index": 0, "message": {"role": "assistant", "content": out.text}, "finish_reason": finish_reason(out.finish)})
        } else {
            json!({"index": 0, "text": out.text, "finish_reason": finish_reason(out.finish)})
        };
        let body = json!({
            "id": id, "object": object, "created": created, "model": model, "choices": [choice],
            "usage": {"prompt_tokens": out.prompt_tokens, "completion_tokens": out.completion_tokens,
                      "total_tokens": out.prompt_tokens + out.completion_tokens},
        });
        return http::respond(w, 200, "application/json", body.to_string().as_bytes());
    }

    let chunk_object = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let chunk = |delta: Value, finish: Option<&str>| {
        let choice = if chat {
            json!({"index": 0, "delta": delta, "finish_reason": finish})
        } else {
            json!({"index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")), "finish_reason": finish})
        };
        json!({"id": id, "object": chunk_object, "created": created, "model": model, "choices": [choice]}).to_string()
    };
    http::start_sse(w)?;
    if chat {
        http::sse_event(w, &chunk(json!({"role": "assistant", "content": ""}), None))?;
    }
    // A failed write means the client went away: stop generating.
    let out = e.generate(&mut sess, &tokens, &params, |t| {
        http::sse_event(w, &chunk(json!({"content": t}), None)).is_ok()
    });
    if out.finish == Finish::Cancelled {
        return Ok(());
    }
    http::sse_event(w, &chunk(json!({}), Some(finish_reason(out.finish))))?;
    http::sse_event(w, "[DONE]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_params_use_openai_defaults_and_clamp_to_context() {
        let p = gen_params(&json!({"max_tokens": 10_000, "seed": 3}), 100).unwrap();
        assert_eq!(p.max_tokens, 100);
        assert_eq!(
            (
                p.sampler.temperature,
                p.sampler.top_p,
                p.sampler.top_k,
                p.sampler.seed
            ),
            (1.0, 1.0, 0, 3)
        );
        assert!(gen_params(&json!({"temperature": 5}), 10).is_err());
    }

    #[test]
    fn messages_accept_string_and_text_part_content() {
        let m = parse_messages(&json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]}
        ]}))
        .unwrap();
        assert_eq!(m, [Message::new("user", "hi"), Message::new("user", "ab")]);
        assert!(parse_messages(&json!({"messages": []})).is_err());
        assert!(parse_messages(&json!({"messages": [{"content": "x"}]})).is_err());
    }
}
