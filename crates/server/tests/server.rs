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
    let mut line = String::new();
    BufReader::new(child.stderr.take().unwrap())
        .read_line(&mut line)
        .unwrap();
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
