//! A tools reply as it streams: reasoning, text and `<tool_call>` blocks told apart while the
//! text arrives in pieces, giving what the whole reply gives once it is complete.

use rli_models::tools::{self, CALL_CLOSE, CALL_OPEN, ToolCall};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Debug, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    /// A call and its index among the reply's calls.
    Call(usize, ToolCall),
}

/// Trims text that arrives in pieces as `str::trim` trims the whole: whitespace is held back
/// until something follows it.
#[derive(Default)]
struct Trimmed {
    started: bool,
    held: String,
}

impl Trimmed {
    fn push(&mut self, s: &str) -> Option<String> {
        let s = if self.started { s } else { s.trim_start() };
        if s.is_empty() {
            return None;
        }
        self.started = true;
        let body = s.trim_end();
        if body.is_empty() {
            self.held += s;
            return None;
        }
        let out = std::mem::take(&mut self.held) + body;
        self.held = s[body.len()..].to_owned();
        Some(out)
    }
}

#[derive(Clone, Copy)]
enum State {
    /// Before reasoning: waiting to see whether it opens with `<think>`.
    Opening,
    Reasoning,
    Text,
    /// Inside a `<tool_call>` block, whose text is held until it closes.
    Call,
}

pub struct ToolStream {
    state: State,
    /// Text not yet given out.
    buf: String,
    reasoning: Trimmed,
    content: Trimmed,
    pub calls: usize,
}

/// The length of `s` less any end of it that could be the start of `marker`.
fn safe_len(s: &str, marker: &str) -> usize {
    let k = (1..marker.len())
        .rev()
        .find(|&k| s.ends_with(&marker[..k]))
        .unwrap_or(0);
    s.len() - k
}

impl ToolStream {
    /// `thinking`: the reply starts with reasoning, which ends at the first `</think>`. Unlike
    /// the whole reply's split, reasoning cut off before `</think>` has already gone out as
    /// reasoning, and a later `</think>` is text.
    pub fn new(thinking: bool) -> Self {
        Self {
            state: if thinking {
                State::Opening
            } else {
                State::Text
            },
            buf: String::new(),
            reasoning: Trimmed::default(),
            content: Trimmed::default(),
            calls: 0,
        }
    }

    fn reason(&mut self, s: &str, out: &mut Vec<Delta>) {
        out.extend(self.reasoning.push(s).map(Delta::Reasoning));
    }

    fn say(&mut self, s: &str, out: &mut Vec<Delta>) {
        out.extend(self.content.push(s).map(Delta::Content));
    }

    pub fn push(&mut self, text: &str) -> Vec<Delta> {
        self.buf += text;
        let mut out = Vec::new();
        loop {
            let buf = std::mem::take(&mut self.buf);
            let (state, rest) = match self.state {
                State::Opening => {
                    let t = buf.trim_start();
                    if let Some(after) = t.strip_prefix(THINK_OPEN) {
                        (State::Reasoning, after.to_owned())
                    } else if THINK_OPEN.starts_with(t) {
                        self.buf = buf;
                        break;
                    } else {
                        (State::Reasoning, buf)
                    }
                }
                State::Reasoning => match buf.find(THINK_CLOSE) {
                    Some(i) => {
                        self.reason(&buf[..i], &mut out);
                        (State::Text, buf[i + THINK_CLOSE.len()..].to_owned())
                    }
                    None => {
                        let n = safe_len(&buf, THINK_CLOSE);
                        self.reason(&buf[..n], &mut out);
                        self.buf = buf[n..].to_owned();
                        break;
                    }
                },
                State::Text => match buf.find(CALL_OPEN) {
                    Some(i) => {
                        self.say(&buf[..i], &mut out);
                        (State::Call, buf[i + CALL_OPEN.len()..].to_owned())
                    }
                    None => {
                        let n = safe_len(&buf, CALL_OPEN);
                        self.say(&buf[..n], &mut out);
                        self.buf = buf[n..].to_owned();
                        break;
                    }
                },
                State::Call => match buf.find(CALL_CLOSE) {
                    Some(i) => {
                        match tools::parse_call(&buf[..i]) {
                            Some(c) => {
                                out.push(Delta::Call(self.calls, c));
                                self.calls += 1;
                            }
                            // Malformed: the block stays in the text, as `extract_calls` keeps it.
                            None => {
                                let block = format!("{CALL_OPEN}{}{CALL_CLOSE}", &buf[..i]);
                                self.say(&block, &mut out);
                            }
                        }
                        (State::Text, buf[i + CALL_CLOSE.len()..].to_owned())
                    }
                    None => {
                        self.buf = buf;
                        break;
                    }
                },
            };
            self.state = state;
            self.buf = rest;
        }
        out
    }

    /// Gives out what is held at the end of the reply. An unclosed call is text.
    pub fn finish(&mut self) -> Vec<Delta> {
        let buf = std::mem::take(&mut self.buf);
        let mut out = Vec::new();
        match self.state {
            State::Opening | State::Reasoning => self.reason(&buf, &mut out),
            State::Text => self.say(&buf, &mut out),
            State::Call => self.say(&format!("{CALL_OPEN}{buf}"), &mut out),
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reply fed in pieces `n` bytes long (at char boundaries), as (reasoning, content,
    /// calls).
    fn run(text: &str, thinking: bool, n: usize) -> (String, String, Vec<ToolCall>) {
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
        let (mut r, mut c, mut calls) = (String::new(), String::new(), Vec::new());
        for d in deltas {
            match d {
                Delta::Reasoning(s) => r += &s,
                Delta::Content(s) => c += &s,
                Delta::Call(i, call) => {
                    assert_eq!(i, calls.len());
                    calls.push(call);
                }
            }
        }
        (r, c, calls)
    }

    fn call(name: &str) -> ToolCall {
        ToolCall {
            name: name.into(),
            arguments: "{}".into(),
        }
    }

    #[test]
    fn markers_split_across_pieces_never_leak_into_content() {
        let text = "Sawa. <tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call> then \
                    <tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>\n done <tool";
        for n in 1..=text.len() {
            let (r, c, calls) = run(text, false, n);
            assert_eq!(r, "");
            assert_eq!(c, "Sawa.  then \n done <tool", "pieces of {n}");
            assert_eq!(calls, [call("a"), call("b")], "pieces of {n}");
        }
    }

    #[test]
    fn a_malformed_or_unclosed_call_is_content() {
        for n in 1..=8 {
            assert_eq!(
                run("x <tool_call>{oops</tool_call> y", false, n),
                (
                    String::new(),
                    "x <tool_call>{oops</tool_call> y".into(),
                    vec![]
                )
            );
            assert_eq!(
                run("x <tool_call>{\"name\": \"a\"", false, n).1,
                "x <tool_call>{\"name\": \"a\""
            );
        }
    }

    #[test]
    fn reasoning_streams_apart_and_its_calls_are_not_calls() {
        let text = "\n<think>\nHmm <tool_call>{\"name\": \"x\", \"arguments\": {}}</tool_call>\n\
                    </think>\n\nOk <tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>";
        for n in 1..=text.len() {
            let (r, c, calls) = run(text, true, n);
            assert_eq!(
                r, "Hmm <tool_call>{\"name\": \"x\", \"arguments\": {}}</tool_call>",
                "pieces of {n}"
            );
            assert_eq!(c, "Ok", "pieces of {n}");
            assert_eq!(calls, [call("a")], "pieces of {n}");
        }
        // Reasoning that a template opened in the prompt, so the reply has no `<think>`.
        assert_eq!(run("why</think>so", true, 2).0, "why");
        assert_eq!(run("why</think>so", true, 2).1, "so");
    }
}
