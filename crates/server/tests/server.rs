//! Runs the real `rli-server` binary on a tiny fixture model and talks HTTP to it.

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};

struct Server {
    child: Child,
    addr: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start() -> Server {
    let model = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/tests/fixtures/tiny_mixtral.gguf"
    );
    let mut child = Command::new(env!("CARGO_BIN_EXE_rli-server"))
        .args(["-m", model, "--port", "0", "--ctx", "128"])
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn server");
    // Skip any warnings (e.g. a GPU fallback) before the listening line.
    let mut stderr = BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    while !line.contains("http://") {
        line.clear();
        assert_ne!(stderr.read_line(&mut line).unwrap(), 0, "server exited");
    }
    let addr = line
        .trim()
        .rsplit("http://")
        .next()
        .expect("listening line")
        .to_owned();
    Server { child, addr }
}

/// Sends one request and returns (status, body).
fn request(addr: &str, method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    let body = body.map(Value::to_string).unwrap_or_default();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    let (head, body) = resp.split_once("\r\n\r\n").unwrap();
    (
        head.split_whitespace().nth(1).unwrap().parse().unwrap(),
        body.to_owned(),
    )
}

#[test]
fn serves_openai_compatible_endpoints() {
    let srv = start();
    let a = srv.addr.as_str();

    assert_eq!(request(a, "GET", "/healthz", None), (200, "ok".into()));
    let (st, models) = request(a, "GET", "/v1/models", None);
    assert_eq!(st, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&models).unwrap()["object"],
        "list"
    );

    let chat = json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 5, "temperature": 0});
    let (st, body) = request(a, "POST", "/v1/chat/completions", Some(&chat));
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(completion_tokens <= 5);
    let full_text = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();

    // Streaming the same greedy request yields the same text, chunked, ending in [DONE].
    let mut stream_req = chat.clone();
    stream_req["stream"] = json!(true);
    let (st, sse) = request(a, "POST", "/v1/chat/completions", Some(&stream_req));
    assert_eq!(st, 200);
    let events: Vec<&str> = sse
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"));
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    let streamed: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(streamed, full_text);
    assert!(chunks.last().unwrap()["choices"][0]["finish_reason"].is_string());

    let (st, body) = request(
        a,
        "POST",
        "/v1/completions",
        Some(&json!({"prompt": "he", "max_tokens": 3, "temperature": 0})),
    );
    assert_eq!(st, 200, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["object"],
        "text_completion"
    );

    assert_eq!(
        request(
            a,
            "POST",
            "/v1/chat/completions",
            Some(&json!({"messages": "nope"}))
        )
        .0,
        400
    );
    assert_eq!(request(a, "GET", "/v1/chat/completions", None).0, 405);
    assert_eq!(request(a, "GET", "/nope", None).0, 404);
    let huge = "x".repeat(1000);
    let too_long = json!({"messages": [{"role": "user", "content": huge}]});
    assert_eq!(
        request(a, "POST", "/v1/chat/completions", Some(&too_long)).0,
        400
    );
}

fn body_text(path: &str, body: &str) -> String {
    let v: Value = serde_json::from_str(body).unwrap();
    let choice = &v["choices"][0];
    let text = if path.contains("chat") {
        &choice["message"]["content"]
    } else {
        &choice["text"]
    };
    text.as_str().unwrap().to_owned()
}

/// Joins an SSE body's text, and returns it with the last event's finish reason.
fn stream_text(path: &str, body: &str) -> (String, String) {
    let chunks: Vec<Value> = body
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let text = chunks
        .iter()
        .filter_map(|c| {
            let choice = &c["choices"][0];
            if path.contains("chat") {
                choice["delta"]["content"].as_str()
            } else {
                choice["text"].as_str()
            }
        })
        .collect();
    let finish = chunks.last().unwrap()["choices"][0]["finish_reason"]
        .as_str()
        .unwrap()
        .to_owned();
    (text, finish)
}

#[test]
fn stop_ends_the_reply_before_the_match_on_both_endpoints() {
    let srv = start();
    let a = srv.addr.as_str();
    for (path, base) in [
        (
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": "hello"}]}),
        ),
        ("/v1/completions", json!({"prompt": "hello"})),
    ] {
        let with = |extra: Value| {
            let mut v = base.clone();
            v.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            v
        };
        let full_req = with(json!({"max_tokens": 20, "temperature": 0}));
        // Twice: the first evaluates the prompt as a batch, every later one finds it cached and
        // re-runs only its last token. The two paths round differently (half-precision prompt
        // matmuls on Metal), which is enough to change this random model's text.
        request(a, "POST", path, Some(&full_req));
        let (st, full) = request(a, "POST", path, Some(&full_req));
        assert_eq!(st, 200, "{full}");
        let full = body_text(path, &full);
        // Long enough to span tokens, and away from the start so text comes before it.
        let chars: Vec<char> = full.chars().collect();
        assert!(chars.len() >= 8, "{full:?}");
        let stop: String = chars[4..7].iter().collect();
        let want = &full[..full.find(&stop).unwrap()];

        for stop_value in [json!(stop), json!(["never appears", stop])] {
            let req = with(json!({"max_tokens": 20, "temperature": 0, "stop": stop_value}));
            let (st, body) = request(a, "POST", path, Some(&req));
            assert_eq!(st, 200, "{body}");
            assert_eq!(body_text(path, &body), want, "{path}");
            let v: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v["choices"][0]["finish_reason"], "stop");

            let req = with(
                json!({"max_tokens": 20, "temperature": 0, "stop": stop_value, "stream": true}),
            );
            let (st, body) = request(a, "POST", path, Some(&req));
            assert_eq!(st, 200, "{body}");
            let (text, finish) = stream_text(path, &body);
            assert_eq!(text, want, "{path} streamed");
            assert_eq!(finish, "stop");
        }
    }
}

#[test]
fn stop_rejects_invalid_values() {
    let srv = start();
    let long = "x".repeat(257);
    let many: Vec<String> = (0..17).map(|i| i.to_string()).collect();
    for stop in [
        json!(""),
        json!(5),
        json!(["ok", ""]),
        json!(["ok", 5]),
        json!(long),
        json!(many),
    ] {
        let req = json!({"prompt": "hello", "max_tokens": 1, "stop": stop});
        let (st, body) = request(&srv.addr, "POST", "/v1/completions", Some(&req));
        assert_eq!(st, 400, "{stop}: {body}");
    }
}

#[test]
fn stop_accepts_null_and_an_empty_list() {
    let srv = start();
    for stop in [json!(null), json!([])] {
        let req = json!({"prompt": "hello", "max_tokens": 1, "stop": stop});
        let (st, body) = request(&srv.addr, "POST", "/v1/completions", Some(&req));
        assert_eq!(st, 200, "{stop}: {body}");
    }
}

#[test]
fn root_serves_the_chat_page() {
    let srv = start();
    let mut s = TcpStream::connect(&srv.addr).unwrap();
    write!(s, "GET / HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    let (head, body) = resp.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(head.contains("Content-Type: text/html"), "{head}");
    assert!(body.contains("<title>rli chat</title>"));
}

#[test]
fn root_rejects_other_methods() {
    let srv = start();
    let (st, _) = request(&srv.addr, "POST", "/", None);
    assert_eq!(st, 405);
}

fn last_data_chunks(body: &str) -> Vec<Value> {
    body.split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

#[test]
fn stream_reports_usage_and_timings_only_when_asked() {
    let srv = start();
    let a = srv.addr.as_str();
    for (path, base) in [
        (
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": "hello"}]}),
        ),
        ("/v1/completions", json!({"prompt": "hello"})),
    ] {
        let with = |extra: Value| {
            let mut v = base.clone();
            v.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            v
        };
        let common = json!({"max_tokens": 6, "temperature": 0});
        let (_, plain) = request(a, "POST", path, Some(&with(common.clone())));
        let plain: Value = serde_json::from_str(&plain).unwrap();

        let silent = with(json!({"max_tokens": 6, "temperature": 0, "stream": true}));
        let (_, body) = request(a, "POST", path, Some(&silent));
        assert!(
            last_data_chunks(&body)
                .iter()
                .all(|c| c.get("usage").is_none()),
            "{path}: usage sent without being asked for"
        );

        let asked = with(json!({"max_tokens": 6, "temperature": 0, "stream": true,
            "stream_options": {"include_usage": true}}));
        let (_, body) = request(a, "POST", path, Some(&asked));
        assert!(body.trim_end().ends_with("data: [DONE]"), "{path}");
        let chunks = last_data_chunks(&body);
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"], json!([]), "{path}");
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
            assert_eq!(last["usage"][key], plain["usage"][key], "{path}: {key}");
        }
        assert_eq!(
            last["timings"]["predicted_n"],
            last["usage"]["completion_tokens"]
        );
        assert!(last["timings"]["prompt_per_second"].as_f64().unwrap() > 0.0);
        assert!(last["timings"]["predicted_per_second"].as_f64().unwrap() > 0.0);
    }
}

#[test]
fn usage_reports_cached_prompt_tokens() {
    let srv = start();
    let req = json!({"prompt": "hello there", "max_tokens": 2, "temperature": 0});
    let ask = || {
        let (st, body) = request(&srv.addr, "POST", "/v1/completions", Some(&req));
        assert_eq!(st, 200, "{body}");
        serde_json::from_str::<Value>(&body).unwrap()
    };
    let first = ask();
    assert_eq!(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    assert_eq!(
        first["timings"]["prompt_n"],
        first["usage"]["prompt_tokens"]
    );

    // The last prompt token is always evaluated again, for its logits.
    let again = ask();
    let prompt = again["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(
        again["usage"]["prompt_tokens_details"]["cached_tokens"],
        prompt - 1
    );
    assert_eq!(again["timings"]["prompt_n"], 1);
}

#[test]
fn chunked_body_gets_411_as_openai_json() {
    let srv = start();
    let mut s = TcpStream::connect(&srv.addr).unwrap();
    s.write_all(
        b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
    )
    .unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    let (head, body) = resp.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("HTTP/1.1 411"), "{head}");
    let v: Value = serde_json::from_str(body).unwrap();
    assert!(v["error"]["message"].is_string(), "{body}");
}

#[test]
fn wrongly_typed_max_tokens_is_rejected() {
    let srv = start();
    let (st, body) = request(
        &srv.addr,
        "POST",
        "/v1/completions",
        Some(&json!({"prompt": "hi", "max_tokens": "5"})),
    );
    assert_eq!(st, 400, "{body}");
}
