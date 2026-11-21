//! OpenAI-compatible endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/completions`.
//! `/` serves a small chat page for trying a model by hand.

use crate::http::{self, Request};
use crate::tool_stream::{Delta, ToolStream};
use rli_models::tools::{self, Grammar, Tool};
use rli_models::{
    Engine, Finish, GenOutput, GenParams, Message, RenderOpts, SamplerConfig, Session,
};
use serde_json::{Value, json};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct App {
    pub engine: Engine,
    /// One shared KV cache: requests run one at a time (they would only compete for the same
    /// cores), and a client continuing a conversation reuses its cached prefix.
    pub session: Mutex<Session>,
    /// The session's context size, so requests can be validated while another one generates.
    pub max_ctx: usize,
}

impl App {
    /// Waits for the session. A poisoned lock only means an earlier request panicked
    /// mid-generation; the session's prefix check re-validates the cache, so it is safe to keep
    /// using. Hold it only around generation: nothing else may stall other clients.
    fn session(&self) -> MutexGuard<'_, Session> {
        self.session.lock().unwrap_or_else(|p| p.into_inner())
    }
}

const CHAT_PAGE: &str = include_str!("chat.html");

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Roles are written into the prompt as they are, so only known ones are accepted.
const ROLES: [&str; 6] = [
    "system",
    "developer",
    "user",
    "assistant",
    "tool",
    "function",
];

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub fn error_body(msg: &str) -> Vec<u8> {
    json!({"error": {"message": msg, "type": "invalid_request_error", "param": null, "code": null}})
        .to_string()
        .into_bytes()
}

/// Bounds the matching work per token, which grows with these. OpenAI allows 4 stops, but some
/// local clients send more.
const MAX_STOPS: usize = 16;
const MAX_STOP_BYTES: usize = 256;

/// `stop`: absent, null, a string or an array of strings.
fn parse_stop(req: &Value) -> Result<Vec<String>, String> {
    let values = match req.get("stop") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(a)) => a.as_slice(),
        Some(v) => std::slice::from_ref(v),
    };
    if values.len() > MAX_STOPS {
        return Err(format!("`stop` allows at most {MAX_STOPS} strings"));
    }
    values
        .iter()
        .map(|v| match v.as_str() {
            Some(s) if !s.is_empty() && s.len() <= MAX_STOP_BYTES => Ok(s.to_owned()),
            _ => Err(format!(
                "`stop` must be strings of 1 to {MAX_STOP_BYTES} bytes"
            )),
        })
        .collect()
}

/// A field that is absent or null, or else must be a number. A value of the wrong type is an
/// error, not the default: `"max_tokens": "100"` must not mean the whole context.
fn number(req: &Value, k: &str) -> Result<Option<f64>, String> {
    match req.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or(format!("`{k}` must be a number")),
    }
}

fn whole(req: &Value, k: &str) -> Result<Option<u64>, String> {
    match req.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or(format!("`{k}` must be a non-negative integer")),
    }
}

fn boolean(req: &Value, k: &str) -> Result<bool, String> {
    match req.get(k) {
        None | Some(Value::Null) => Ok(false),
        Some(v) => v.as_bool().ok_or(format!("`{k}` must be a boolean")),
    }
}

/// Sampling parameters with OpenAI's defaults (temperature 1, top_p 1, no penalty).
/// `top_k`, `repeat_penalty` and `seed` are accepted as extensions.
pub fn gen_params(req: &Value, room: usize) -> Result<GenParams, String> {
    let max_tokens = match whole(req, "max_completion_tokens")? {
        Some(n) => Some(n),
        None => whole(req, "max_tokens")?,
    };
    let max_tokens = match max_tokens {
        Some(0) => return Err("`max_tokens` must be at least 1".into()),
        Some(n) => usize::try_from(n).unwrap_or(usize::MAX).min(room),
        None => room,
    };
    if whole(req, "n")?.is_some_and(|n| n != 1) {
        return Err("`n` must be 1".into());
    }
    let temperature = number(req, "temperature")?.unwrap_or(1.0) as f32;
    if !(0.0..=2.0).contains(&temperature) {
        return Err("temperature must be between 0 and 2".into());
    }
    let top_p = number(req, "top_p")?.unwrap_or(1.0) as f32;
    if !(0.0..=1.0).contains(&top_p) {
        return Err("top_p must be between 0 and 1".into());
    }
    // 0 makes every penalized logit infinite, and the sampler NaN.
    let repeat_penalty = number(req, "repeat_penalty")?.unwrap_or(1.0) as f32;
    if !(repeat_penalty.is_finite() && repeat_penalty > 0.0) {
        return Err("repeat_penalty must be a positive number".into());
    }
    Ok(GenParams {
        stop: parse_stop(req)?,
        max_tokens,
        sampler: SamplerConfig {
            temperature,
            top_p,
            top_k: whole(req, "top_k")?.map_or(0, |k| usize::try_from(k).unwrap_or(usize::MAX)),
            repeat_penalty,
            repeat_last_n: 64,
            seed: whole(req, "seed")?
                .unwrap_or_else(|| NEXT_ID.fetch_add(1, Ordering::Relaxed) ^ now()),
        },
        tools: None,
    })
}

/// The request's tools, if it has any and `tool_choice` lets the model call them. `None` when it
/// has none; `Some(empty)` for `tool_choice: "none"`, which keeps the request a tools request
/// (thinking stays off) but shows the model no tools.
pub fn parse_tool_request(req: &Value, engine: &Engine) -> Result<Option<Vec<Tool>>, String> {
    let Some(v) = req.get("tools").filter(|v| !v.is_null()) else {
        if req
            .get("tool_choice")
            .is_some_and(|c| !c.is_null() && c != "none" && c != "auto")
        {
            return Err("`tool_choice` needs `tools`".into());
        }
        return Ok(None);
    };
    if !engine.chat_defaults.tools {
        return Err("this model's chat template has no tool-call format".into());
    }
    let tools = tools::parse_tools(v)?;
    let show = match req.get("tool_choice") {
        None | Some(Value::Null) => true,
        Some(c) if c == "auto" => true,
        Some(c) if c == "none" => false,
        Some(_) => return Err("`tool_choice` must be \"auto\" or \"none\"".into()),
    };
    match req.get("parallel_tool_calls") {
        None | Some(Value::Null | Value::Bool(true)) => {}
        Some(Value::Bool(false)) => {
            return Err("`parallel_tool_calls: false` is not supported".into());
        }
        Some(_) => return Err("`parallel_tool_calls` must be a boolean".into()),
    }
    // The template's tool-response turns aren't rendered: a client can't send results back.
    let msgs = req.get("messages").and_then(Value::as_array);
    if msgs.is_some_and(|m| {
        m.iter().any(|m| {
            m.get("role").and_then(Value::as_str) == Some("tool")
                || m.get("tool_calls")
                    .is_some_and(|c| c.as_array().is_none_or(|c| !c.is_empty()))
        })
    }) {
        return Err("messages with tool calls or tool results are not supported".into());
    }
    Ok(Some(if show { tools } else { Vec::new() }))
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
            if !ROLES.contains(&role) {
                return Err(format!("unknown message role {role:?}"));
            }
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
                // An assistant turn that only called tools, replayed by the client.
                Some(Value::Null) if role == "assistant" => String::new(),
                _ => return Err("each message needs `content`".to_string()),
            };
            Ok(Message::new(role, &content))
        })
        .collect()
}

fn usage(out: &GenOutput) -> Value {
    json!({
        "prompt_tokens": out.prompt_tokens,
        "completion_tokens": out.completion_tokens,
        "total_tokens": out.prompt_tokens + out.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": out.prompt_tokens - out.prompt_evaluated},
    })
}

/// A non-standard extension, named as llama.cpp's server names it. `prompt_n` counts the prompt
/// tokens evaluated, not those reused from the cache.
fn timings(out: &GenOutput) -> Value {
    let per_second = |n: usize, secs: f64| if secs > 0.0 { n as f64 / secs } else { 0.0 };
    json!({
        "prompt_n": out.prompt_evaluated,
        "prompt_ms": out.prefill_secs * 1000.0,
        "prompt_per_second": per_second(out.prompt_evaluated, out.prefill_secs),
        "predicted_n": out.completion_tokens,
        "predicted_ms": out.decode_secs * 1000.0,
        "predicted_per_second": per_second(out.completion_tokens, out.decode_secs),
    })
}

/// A reply to a tools request: its `<tool_call>` blocks as `tool_calls`, the rest as `content`
/// (null when there is none).
/// Splits a reply into Qwen3's `<think>` reasoning and the rest, which holds the answer.
fn split_thinking(text: &str) -> (Option<&str>, &str) {
    match text.rfind("</think>") {
        Some(end) => {
            let thought = text[..end]
                .trim_start()
                .trim_start_matches("<think>")
                .trim();
            (
                (!thought.is_empty()).then_some(thought),
                &text[end + "</think>".len()..],
            )
        }
        None => (None, text),
    }
}

fn tool_choice(out: &GenOutput) -> Value {
    // Calls are read from the answer only: a model may sketch one while it thinks.
    let (thought, answer) = split_thinking(&out.text);
    let (content, calls) = tools::extract_calls(answer);
    let finish = match out.finish {
        Finish::Stop if !calls.is_empty() => "tool_calls",
        f => finish_reason(f),
    };
    let mut message =
        json!({"role": "assistant", "content": (!content.is_empty()).then_some(content)});
    if let Some(t) = thought {
        message["reasoning_content"] = json!(t);
    }
    if !calls.is_empty() {
        let calls: Vec<Value> = calls
            .into_iter()
            .map(|c| {
                json!({
                    "id": format!("call_{}", NEXT_ID.fetch_add(1, Ordering::Relaxed)),
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                })
            })
            .collect();
        message["tool_calls"] = Value::Array(calls);
    }
    json!({"index": 0, "message": message, "finish_reason": finish})
}

/// A streamed piece of a tools reply as OpenAI's chunk deltas: a call is its name, then its
/// arguments, as clients accumulate them.
fn delta_json(d: Delta) -> Vec<Value> {
    match d {
        Delta::Reasoning(s) => vec![json!({"reasoning_content": s})],
        Delta::Content(s) => vec![json!({"content": s})],
        Delta::Call(i, c) => vec![
            json!({"tool_calls": [{
                "index": i,
                "id": format!("call_{}", NEXT_ID.fetch_add(1, Ordering::Relaxed)),
                "type": "function",
                "function": {"name": c.name, "arguments": ""},
            }]}),
            json!({"tool_calls": [{"index": i, "function": {"arguments": c.arguments}}]}),
        ],
    }
}

fn finish_reason(f: Finish) -> &'static str {
    match f {
        Finish::Stop | Finish::Cancelled => "stop",
        Finish::Length => "length",
    }
}

/// `gone` reports that the client has disconnected, which cancels its generation.
pub fn handle(
    app: &App,
    req: Request,
    w: &mut impl Write,
    gone: impl Fn() -> bool,
) -> std::io::Result<()> {
    let json_ct = "application/json";
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => http::respond(w, 200, "text/html; charset=utf-8", CHAT_PAGE.as_bytes()),
        ("GET", "/healthz") => http::respond(w, 200, "text/plain", b"ok"),
        ("GET", "/v1/models") => {
            let body = json!({"object": "list", "data": [{
                "id": app.engine.model_name, "object": "model", "created": 0, "owned_by": "rli"
            }]});
            http::respond(w, 200, json_ct, body.to_string().as_bytes())
        }
        ("POST", "/v1/chat/completions") => complete(app, &req.body, true, w, gone),
        ("POST", "/v1/completions") => complete(app, &req.body, false, w, gone),
        (_, "/" | "/healthz" | "/v1/models" | "/v1/chat/completions" | "/v1/completions") => {
            http::respond(w, 405, json_ct, &error_body("method not allowed"))
        }
        _ => http::respond(w, 404, json_ct, &error_body("not found")),
    }
}

fn complete(
    app: &App,
    body: &[u8],
    chat: bool,
    w: &mut impl Write,
    gone: impl Fn() -> bool,
) -> std::io::Result<()> {
    let bad = |w: &mut _, msg: &str| http::respond(w, 400, "application/json", &error_body(msg));
    let req: Value = match serde_json::from_slice(body) {
        Ok(v @ Value::Object(_)) => v,
        _ => return bad(w, "request body must be a JSON object"),
    };
    let e = &app.engine;
    let tool_req = match parse_tool_request(&req, e) {
        Ok(t) if chat || t.is_none() => t,
        Ok(_) => return bad(w, "`tools` needs /v1/chat/completions"),
        Err(msg) => return bad(w, &msg),
    };
    // `chat_template_kwargs.enable_thinking`, as vLLM and llama.cpp's server take it.
    let thinking = match req.pointer("/chat_template_kwargs/enable_thinking") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(b)) => Some(*b),
        Some(_) => {
            return bad(
                w,
                "`chat_template_kwargs.enable_thinking` must be a boolean",
            );
        }
    };
    // Off for tools requests unless asked for: it costs a reply's worth of tokens.
    let think = thinking.unwrap_or(tool_req.is_none());
    let tokens = if chat {
        let opts = RenderOpts {
            tools: tool_req.iter().flatten().map(Tool::to_json).collect(),
            no_think: e.chat_defaults.thinking && !think,
        };
        match parse_messages(&req) {
            Ok(m) => e.chat_tokens_with(&m, &opts),
            Err(msg) => return bad(w, &msg),
        }
    } else {
        match req.get("prompt").and_then(Value::as_str) {
            Some(p) => e.tokenizer.encode(p, e.tokenizer.add_bos, false),
            None => return bad(w, "`prompt` must be a string"),
        }
    };
    if tokens.len() >= app.max_ctx {
        return bad(
            w,
            &format!(
                "prompt is {} tokens; context size is {}",
                tokens.len(),
                app.max_ctx
            ),
        );
    }
    let mut params = match gen_params(&req, app.max_ctx - tokens.len()) {
        Ok(p) => p,
        Err(msg) => return bad(w, &msg),
    };
    // A call the model writes can only be one the schemas allow.
    params.tools = tool_req
        .as_deref()
        .filter(|t| !t.is_empty())
        .map(|t| Arc::new(Grammar::new(t)));
    let stream = match boolean(&req, "stream") {
        Ok(s) => s,
        Err(msg) => return bad(w, &msg),
    };
    let include_usage = req
        .pointer("/stream_options/include_usage")
        .and_then(Value::as_bool)
        .unwrap_or(false);
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
        let out = e.generate(&mut app.session(), &tokens, &params, gone, |_| true);
        if out.finish == Finish::Cancelled {
            return Ok(());
        }
        let choice = if params.tools.is_some() {
            tool_choice(&out)
        } else if chat {
            json!({"index": 0, "message": {"role": "assistant", "content": out.text}, "finish_reason": finish_reason(out.finish)})
        } else {
            json!({"index": 0, "text": out.text, "finish_reason": finish_reason(out.finish)})
        };
        let body = json!({
            "id": id, "object": object, "created": created, "model": model, "choices": [choice],
            "usage": usage(&out), "timings": timings(&out),
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
    let mut calls = params
        .tools
        .is_some()
        .then(|| ToolStream::new(e.chat_defaults.thinking && think));
    let send = |w: &mut _, d: Delta| {
        delta_json(d)
            .into_iter()
            .try_for_each(|d| http::sse_event(w, &chunk(d, None)))
    };
    // A failed write means the client went away: stop generating.
    let out = e.generate(
        &mut app.session(),
        &tokens,
        &params,
        gone,
        |t| match &mut calls {
            Some(c) => c.push(t).into_iter().all(|d| send(w, d).is_ok()),
            None => http::sse_event(w, &chunk(json!({"content": t}), None)).is_ok(),
        },
    );
    if out.finish == Finish::Cancelled {
        return Ok(());
    }
    let finish = match calls {
        Some(mut c) => {
            c.finish().into_iter().try_for_each(|d| send(w, d))?;
            match out.finish {
                Finish::Stop if c.calls > 0 => "tool_calls",
                f => finish_reason(f),
            }
        }
        None => finish_reason(out.finish),
    };
    http::sse_event(w, &chunk(json!({}), Some(finish)))?;
    if include_usage {
        let last = json!({"id": id, "object": chunk_object, "created": created, "model": model,
            "choices": [], "usage": usage(&out), "timings": timings(&out)});
        http::sse_event(w, &last.to_string())?;
    }
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
    fn gen_params_reject_values_of_the_wrong_type_or_range() {
        for bad in [
            json!({"max_tokens": -1}),
            json!({"max_tokens": "100"}),
            json!({"max_tokens": 0}),
            json!({"max_completion_tokens": 2.5}),
            json!({"temperature": "hot"}),
            json!({"top_p": 1.5}),
            json!({"repeat_penalty": 0}),
            json!({"repeat_penalty": -1}),
            json!({"seed": -3}),
            json!({"n": 2}),
        ] {
            assert!(gen_params(&bad, 10).is_err(), "{bad}");
        }
        let p = gen_params(&json!({"max_tokens": null, "n": 1, "top_p": 0}), 10).unwrap();
        assert_eq!((p.max_tokens, p.sampler.top_p), (10, 0.0));
    }

    #[test]
    fn messages_accept_null_content_on_assistant_tool_calls_only() {
        let m = parse_messages(&json!({"messages": [
            {"role": "assistant", "content": null, "tool_calls": []}
        ]}))
        .unwrap();
        assert_eq!(m, [Message::new("assistant", "")]);
        assert!(parse_messages(&json!({"messages": [{"role": "user", "content": null}]})).is_err());
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

    fn reply(text: &str, finish: Finish) -> GenOutput {
        GenOutput {
            text: text.into(),
            prompt_tokens: 1,
            prompt_evaluated: 1,
            completion_tokens: 1,
            finish,
            prefill_secs: 0.0,
            decode_secs: 0.0,
        }
    }

    #[test]
    fn tool_reply_carries_calls_and_null_content() {
        let text = "<tool_call>\n{\"name\": \"open_form\", \"arguments\": {\"form_id\": \"loan_application\"}}\n</tool_call>\n\
                    <tool_call>\n{\"name\": \"fill_field\", \"arguments\": {\"value\": \"Asha Juma\"}}\n</tool_call>";
        let c = tool_choice(&reply(text, Finish::Stop));
        assert_eq!(c["finish_reason"], "tool_calls");
        assert!(c["message"]["content"].is_null());
        let calls = c["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "open_form");
        assert_eq!(
            calls[1]["function"]["arguments"],
            r#"{"value":"Asha Juma"}"#
        );
        assert_ne!(calls[0]["id"], calls[1]["id"]);
    }

    #[test]
    fn tool_reply_keeps_reasoning_apart_and_ignores_calls_in_it() {
        let text = "<think>\nThey want home. <tool_call>{\"name\": \"x\", \"arguments\": {}}</tool_call>\n</think>\n\n\
                    <tool_call>\n{\"name\": \"navigate\", \"arguments\": {\"page\": \"home\"}}\n</tool_call>";
        let c = tool_choice(&reply(text, Finish::Stop));
        let calls = c["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "navigate");
        assert!(c["message"]["content"].is_null());
        assert!(
            c["message"]["reasoning_content"]
                .as_str()
                .unwrap()
                .starts_with("They want home.")
        );
    }

    #[test]
    fn tool_reply_without_calls_is_plain_text() {
        let c = tool_choice(&reply("Sawa.", Finish::Stop));
        assert_eq!(c["finish_reason"], "stop");
        assert_eq!(c["message"]["content"], "Sawa.");
        assert!(c["message"].get("tool_calls").is_none());
        let broken = tool_choice(&reply("<tool_call>{oops</tool_call>", Finish::Stop));
        assert_eq!(broken["message"]["content"], "<tool_call>{oops</tool_call>");
    }

    /// A streamed tools reply, joined, is the reply `tool_choice` gives for the whole text,
    /// however the text is split.
    #[test]
    fn streamed_tool_reply_joins_to_the_whole_reply() {
        let texts = [
            (false, "Sawa."),
            (false, "  \n"),
            (
                false,
                "Let me.\n<tool_call>\n{\"name\": \"open_form\", \"arguments\": {\"form_id\": \"loan\"}}\n</tool_call>\n\
                 and <tool_call>{oops</tool_call> then\n<tool_call>{\"name\": \"b\", \"arguments\": {\"v\": \"é\"}}</tool_call>\n",
            ),
            (
                false,
                "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call><tool_call>{\"name\"",
            ),
            (
                true,
                "<think>\nHmm <tool_call>{\"name\": \"x\", \"arguments\": {}}</tool_call>\n</think>\n\n\
                 <tool_call>\n{\"name\": \"navigate\", \"arguments\": {\"page\": \"home\"}}\n</tool_call>",
            ),
            (true, "<think>\n\n</think>\n\nJust text <tool"),
        ];
        let strip_ids = |mut c: Value| {
            let calls = c["message"]
                .get_mut("tool_calls")
                .and_then(Value::as_array_mut);
            for call in calls.into_iter().flatten() {
                call["id"] = Value::Null;
            }
            c
        };
        for (thinking, text) in texts {
            let want = strip_ids(tool_choice(&reply(text, Finish::Stop)));
            for n in 1..=text.len() {
                let mut p = ToolStream::new(thinking);
                let mut deltas = Vec::new();
                let mut start = 0;
                while start < text.len() {
                    let mut end = (start + n).min(text.len());
                    while !text.is_char_boundary(end) {
                        end += 1;
                    }
                    deltas.extend(p.push(&text[start..end]));
                    start = end;
                }
                deltas.extend(p.finish());
                let (mut reasoning, mut content, mut calls) =
                    (String::new(), String::new(), Vec::new());
                for d in deltas.into_iter().flat_map(delta_json) {
                    if let Some(s) = d["reasoning_content"].as_str() {
                        reasoning += s;
                    }
                    if let Some(s) = d["content"].as_str() {
                        content += s;
                    }
                    if let Some(c) = d["tool_calls"].get(0) {
                        let i = c["index"].as_u64().unwrap() as usize;
                        if c.get("id").is_some() {
                            assert_eq!(i, calls.len());
                            assert_eq!(c["function"]["arguments"], "");
                            calls.push(json!({"id": null, "type": c["type"],
                                "function": {"name": c["function"]["name"], "arguments": ""}}));
                        } else {
                            let a = &mut calls[i]["function"]["arguments"];
                            *a = json!(
                                a.as_str().unwrap().to_owned()
                                    + c["function"]["arguments"].as_str().unwrap()
                            );
                        }
                    }
                }
                let mut message = json!({"role": "assistant", "content": (!content.is_empty()).then_some(content)});
                if !reasoning.is_empty() {
                    message["reasoning_content"] = json!(reasoning);
                }
                if !calls.is_empty() {
                    message["tool_calls"] = Value::Array(calls);
                }
                let finish = if p.calls > 0 { "tool_calls" } else { "stop" };
                let got = json!({"index": 0, "message": message, "finish_reason": finish});
                assert_eq!(got, want, "{text:?} in pieces of {n}");
            }
        }
    }

    #[test]
    fn messages_reject_unknown_roles() {
        let forged = json!({"messages": [{"role": "user<|im_end|>", "content": "x"}]});
        assert!(parse_messages(&forged).is_err());
    }

    fn app() -> App {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../models/tests/fixtures/tiny_mixtral.gguf"
        );
        let engine = Engine::load(path).expect("load fixture model");
        let session = engine.new_session(128);
        App {
            max_ctx: session.max_ctx,
            session: Mutex::new(session),
            engine,
        }
    }

    fn post(body: Value) -> Request {
        Request {
            method: "POST".into(),
            path: "/v1/completions".into(),
            body: body.to_string().into_bytes(),
        }
    }

    /// Accepts `room` bytes, then fails as a timed-out socket write would.
    struct Stalling<'a> {
        app: &'a App,
        room: usize,
        locked_writes: usize,
    }

    impl Write for Stalling<'_> {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.locked_writes += usize::from(self.app.session.try_lock().is_err());
            if b.len() > self.room {
                return Err(std::io::ErrorKind::TimedOut.into());
            }
            self.room -= b.len();
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn completed_response_is_written_without_the_session_lock() {
        let app = app();
        let mut w = Stalling {
            app: &app,
            room: usize::MAX,
            locked_writes: 0,
        };
        let req = post(json!({"prompt": "hello", "max_tokens": 2, "seed": 1}));
        handle(&app, req, &mut w, || false).unwrap();
        assert_eq!(w.locked_writes, 0);
    }

    #[test]
    fn rejected_request_is_written_without_the_session_lock() {
        let app = app();
        let mut w = Stalling {
            app: &app,
            room: usize::MAX,
            locked_writes: 0,
        };
        let req = post(json!({"prompt": "hello", "temperature": 5}));
        handle(&app, req, &mut w, || false).unwrap();
        assert_eq!(w.locked_writes, 0);
    }

    #[test]
    fn stream_headers_are_written_without_the_session_lock() {
        let app = app();
        let held = app.session.lock().unwrap();
        // The session is busy, so a client must still get headers before its turn comes: run
        // only up to the failure on the first write, which the held lock must not block.
        let mut w = Stalling {
            app: &app,
            room: 0,
            locked_writes: 0,
        };
        let req = post(json!({"prompt": "hello", "stream": true}));
        assert!(handle(&app, req, &mut w, || false).is_err());
        drop(held);
    }

    #[test]
    fn stalled_stream_stops_generating_and_frees_the_session() {
        let app = app();
        // Room for the SSE headers and a few events, far less than a full generation needs.
        let mut w = Stalling {
            app: &app,
            room: 1024,
            locked_writes: 0,
        };
        let req = post(json!({"prompt": "hello", "stream": true, "seed": 1}));
        handle(&app, req, &mut w, || false).unwrap();
        let sess = app.session.try_lock().expect("session released");
        assert!(sess.tokens.len() < sess.max_ctx, "{}", sess.tokens.len());
    }

    #[test]
    fn disconnected_client_gets_no_response_and_no_generation() {
        let app = app();
        let mut w = Vec::new();
        let req = post(json!({"prompt": "hello", "seed": 1}));
        handle(&app, req, &mut w, || true).unwrap();
        assert!(w.is_empty());
        assert!(app.session.try_lock().unwrap().tokens.is_empty());
    }
}
