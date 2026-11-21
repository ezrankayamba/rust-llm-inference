//! OpenAI-style function tools for templates in the Hermes format Qwen2.5 and Qwen3 were trained
//! on: definitions go into the system turn inside `<tools></tools>`, and the model answers with
//! `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks.
//!
//! Only a subset of JSON Schema is accepted: an object whose properties are all required strings,
//! optionally limited to an enum. Anything else is an error rather than ignored, so a client
//! never believes a constraint holds that the server doesn't know about.

use crate::tokenizer::Tokenizer;
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub strict: Option<bool>,
    /// In the order of the schema's `required` list: that order survives JSON parsing, where an
    /// object's keys don't.
    pub params: Vec<Param>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    /// `Some` when the value must be one of these strings.
    pub values: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    /// The arguments object as compact JSON, as OpenAI's `function.arguments` carries it.
    pub arguments: String,
}

/// Properties per function: constrained decoding tracks the keys written in a 64-bit mask.
pub const MAX_PARAMS: usize = 64;

/// Opens a tool call in the model's output.
pub const CALL_OPEN: &str = "<tool_call>";
/// Closes a tool call.
pub const CALL_CLOSE: &str = "</tool_call>";

fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>, String> {
    v.as_object().ok_or(format!("{what} must be an object"))
}

/// Rejects keys of `m` outside `allowed`.
fn only_keys(m: &Map<String, Value>, allowed: &[&str], what: &str) -> Result<(), String> {
    match m.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(format!(
            "{what}: unsupported key {k:?} (supported: {})",
            allowed.join(", ")
        )),
        None => Ok(()),
    }
}

fn string_field(m: &Map<String, Value>, k: &str, what: &str) -> Result<Option<String>, String> {
    match m.get(k) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(format!("{what}.{k} must be a string")),
    }
}

/// OpenAI's rule for function names. It also keeps them out of the way of the JSON they are
/// written into and of special tokens.
fn valid_name(s: &str) -> bool {
    (1..=64).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Parses a request's `tools` array.
pub fn parse_tools(v: &Value) -> Result<Vec<Tool>, String> {
    let list = v.as_array().ok_or("`tools` must be an array")?;
    if list.is_empty() {
        return Err("`tools` must not be empty".into());
    }
    let mut tools: Vec<Tool> = Vec::with_capacity(list.len());
    for (i, t) in list.iter().enumerate() {
        let what = format!("tools[{i}]");
        let t = object(t, &what)?;
        only_keys(t, &["type", "function"], &what)?;
        if t.get("type").and_then(Value::as_str) != Some("function") {
            return Err(format!("{what}.type must be \"function\""));
        }
        let what = format!("{what}.function");
        let f = object(t.get("function").unwrap_or(&Value::Null), &what)?;
        only_keys(f, &["name", "description", "parameters", "strict"], &what)?;
        let name = string_field(f, "name", &what)?.unwrap_or_default();
        if !valid_name(&name) {
            return Err(format!(
                "{what}.name must be 1 to 64 letters, digits, '_' or '-'"
            ));
        }
        if tools.iter().any(|t| t.name == name) {
            return Err(format!("{what}: duplicate function name {name:?}"));
        }
        let strict = match f.get("strict") {
            None => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(_) => return Err(format!("{what}.strict must be a boolean")),
        };
        let what_p = format!("{what}.parameters");
        let params = parse_parameters(f.get("parameters"), &what_p)?;
        tools.push(Tool {
            name,
            description: string_field(f, "description", &what)?,
            strict,
            params,
        });
    }
    Ok(tools)
}

fn parse_parameters(v: Option<&Value>, what: &str) -> Result<Vec<Param>, String> {
    let p = object(v.ok_or(format!("{what} is required"))?, what)?;
    only_keys(
        p,
        &["type", "properties", "required", "additionalProperties"],
        what,
    )?;
    if p.get("type").and_then(Value::as_str) != Some("object") {
        return Err(format!("{what}.type must be \"object\""));
    }
    if p.get("additionalProperties") != Some(&Value::Bool(false)) {
        return Err(format!("{what}.additionalProperties must be false"));
    }
    let props = object(
        p.get("properties").unwrap_or(&Value::Null),
        &format!("{what}.properties"),
    )?;
    let required = p
        .get("required")
        .and_then(Value::as_array)
        .ok_or(format!("{what}.required must be an array"))?;
    let mut params: Vec<Param> = Vec::with_capacity(required.len());
    for r in required {
        let name = r
            .as_str()
            .ok_or(format!("{what}.required must hold strings"))?;
        if params.iter().any(|p| p.name == name) {
            return Err(format!("{what}.required lists {name:?} twice"));
        }
        let what = format!("{what}.properties.{name}");
        let prop = props
            .get(name)
            .ok_or(format!("{what} is required but not defined"))?;
        params.push(Param {
            name: name.to_owned(),
            values: parse_property(prop, &what)?,
        });
    }
    if params.len() > MAX_PARAMS {
        return Err(format!("{what}: at most {MAX_PARAMS} properties"));
    }
    if let Some(k) = props.keys().find(|k| !params.iter().any(|p| &p.name == *k)) {
        return Err(format!(
            "{what}.required must list every property; {k:?} is missing"
        ));
    }
    Ok(params)
}

fn parse_property(v: &Value, what: &str) -> Result<Option<Vec<String>>, String> {
    let p = object(v, what)?;
    only_keys(p, &["type", "enum"], what)?;
    if p.get("type").and_then(Value::as_str) != Some("string") {
        return Err(format!("{what}.type must be \"string\""));
    }
    let Some(e) = p.get("enum") else {
        return Ok(None);
    };
    let e = e
        .as_array()
        .filter(|e| !e.is_empty())
        .ok_or(format!("{what}.enum must be a non-empty array"))?;
    let mut values: Vec<String> = Vec::with_capacity(e.len());
    for v in e {
        let s = v.as_str().ok_or(format!("{what}.enum must hold strings"))?;
        if values.iter().any(|x| x == s) {
            return Err(format!("{what}.enum lists {s:?} twice"));
        }
        values.push(s.to_owned());
    }
    Ok(Some(values))
}

/// A JSON string literal, escaped as Python's `json.dumps(ensure_ascii=False)` escapes it.
fn json_str(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

impl Tool {
    /// The tool as the template writes it: `tool | tojson`, which in Hugging Face's renderer is
    /// `json.dumps` with its default `", "` and `": "` separators and the client's key order.
    /// Properties follow `required`, the only order the request keeps.
    pub fn to_json(&self) -> String {
        let mut f = format!("{{\"name\": {}", json_str(&self.name));
        if let Some(d) = &self.description {
            f += &format!(", \"description\": {}", json_str(d));
        }
        let props: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                let mut s = format!("{}: {{\"type\": \"string\"", json_str(&p.name));
                if let Some(values) = &p.values {
                    let v: Vec<String> = values.iter().map(|v| json_str(v)).collect();
                    s += &format!(", \"enum\": [{}]", v.join(", "));
                }
                s + "}"
            })
            .collect();
        let required: Vec<String> = self.params.iter().map(|p| json_str(&p.name)).collect();
        f += &format!(
            ", \"parameters\": {{\"type\": \"object\", \"properties\": {{{}}}, \"required\": [{}], \"additionalProperties\": false}}",
            props.join(", "),
            required.join(", ")
        );
        if let Some(s) = self.strict {
            f += &format!(", \"strict\": {s}");
        }
        format!("{{\"type\": \"function\", \"function\": {f}}}}}")
    }
}

/// A `<tool_call>` block's JSON, if it is `{"name": string, "arguments": object}`.
fn parse_call(body: &str) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(body.trim()).ok()?;
    let m = v.as_object()?;
    let name = m.get("name")?.as_str()?;
    let args = m.get("arguments")?;
    args.as_object()?;
    Some(ToolCall {
        name: name.to_owned(),
        arguments: args.to_string(),
    })
}

/// Splits generated text into the text outside `<tool_call>` blocks (trimmed) and the calls.
/// A block that is unclosed or doesn't hold a call's JSON stays in the text as it was.
pub fn extract_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut rest = String::new();
    let mut s = text;
    while let Some(open) = s.find(CALL_OPEN) {
        let after = &s[open + CALL_OPEN.len()..];
        let Some(close) = after.find(CALL_CLOSE) else {
            break;
        };
        match parse_call(&after[..close]) {
            Some(c) => {
                calls.push(c);
                rest += &s[..open];
            }
            None => rest += &s[..open + CALL_OPEN.len() + close + CALL_CLOSE.len()],
        }
        s = &after[close + CALL_CLOSE.len()..];
    }
    rest += s;
    (rest.trim().to_owned(), calls)
}

// Constrained decoding. Inside a `<tool_call>` block the reply must be
//   ["\n"] {"name": NAME, "arguments": {KEY: VALUE, ...}} ["\n"]
// with a known function name, each of its keys exactly once (in any order), enum values from
// their list and other values JSON strings; the space after ':' and ',' is optional. It is
// checked a byte at a time, since tokens span several JSON characters and the boundaries of the
// literals in it. Names, keys and enum values are JSON-quoted literals in byte tries: each ends
// in its closing quote, so no literal is a prefix of another.

#[derive(Debug, Default)]
struct Trie {
    nodes: Vec<TrieNode>,
}

#[derive(Debug, Default)]
struct TrieNode {
    next: Vec<(u8, u32)>,
    /// The literal that ends here.
    leaf: Option<u16>,
    /// The literals (by index, the first 64) that pass through here.
    reach: u64,
}

impl Trie {
    fn new(words: &[String]) -> Self {
        let mut t = Self {
            nodes: vec![TrieNode::default()],
        };
        for (i, w) in words.iter().enumerate() {
            let bit = if i < 64 { 1u64 << i } else { 0 };
            let mut n = 0usize;
            t.nodes[0].reach |= bit;
            for &b in w.as_bytes() {
                n = match t.nodes[n].next.iter().find(|(c, _)| *c == b) {
                    Some(&(_, m)) => m as usize,
                    None => {
                        t.nodes.push(TrieNode::default());
                        let m = t.nodes.len() - 1;
                        t.nodes[n].next.push((b, m as u32));
                        m
                    }
                };
                t.nodes[n].reach |= bit;
            }
            t.nodes[n].leaf = Some(i as u16);
        }
        t
    }

    fn step(&self, node: u32, b: u8) -> Option<u32> {
        let n = &self.nodes[node as usize];
        n.next.iter().find(|(c, _)| *c == b).map(|&(_, m)| m)
    }
}

#[derive(Debug)]
struct ToolGrammar {
    keys: Trie,
    /// Per key: the trie of its enum values, if it has an enum.
    values: Vec<Option<Trie>>,
    /// Every key's bit.
    all: u64,
}

/// What a tool call may contain, for one request's tools.
#[derive(Debug)]
pub struct Grammar {
    names: Trie,
    tools: Vec<ToolGrammar>,
}

impl Grammar {
    pub fn new(tools: &[Tool]) -> Self {
        let quoted = |s: &str| json_str(s);
        Self {
            names: Trie::new(&tools.iter().map(|t| quoted(&t.name)).collect::<Vec<_>>()),
            tools: tools
                .iter()
                .map(|t| ToolGrammar {
                    keys: Trie::new(&t.params.iter().map(|p| quoted(&p.name)).collect::<Vec<_>>()),
                    values: t
                        .params
                        .iter()
                        .map(|p| {
                            p.values.as_ref().map(|v| {
                                Trie::new(&v.iter().map(|v| quoted(v)).collect::<Vec<_>>())
                            })
                        })
                        .collect(),
                    all: if t.params.len() == 64 {
                        u64::MAX
                    } else {
                        (1u64 << t.params.len()) - 1
                    },
                })
                .collect(),
        }
    }
}

/// Fixed text between the parts that vary; 0 is an optional space.
const LITS: [&[u8]; 6] = [
    b"{\"name\":\0",
    b",\0\"arguments\":\0{",
    b":\0",
    b"\"",
    b",\0",
    b"}}",
];
const HEAD: u8 = 0;
const ARGS: u8 = 1;
const COLON: u8 = 2;
const QUOTE: u8 = 3;
const COMMA: u8 = 4;
const CLOSE: u8 = 5;

/// Where a call is: `tool` is the function, `used` the keys written, `key` the one being given
/// a value.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum St {
    /// Just after `<tool_call>`: an optional newline, then `{`.
    Start,
    Lit {
        lit: u8,
        pos: u8,
        tool: u16,
        used: u64,
        key: u16,
    },
    Name {
        node: u32,
    },
    Key {
        tool: u16,
        used: u64,
        node: u32,
    },
    Enum {
        tool: u16,
        used: u64,
        key: u16,
        node: u32,
    },
    /// In a free string value. `esc`: 0 plain, 1 after `\`, 2 to 5 in `\u` digits.
    Str {
        tool: u16,
        used: u64,
        esc: u8,
    },
    /// After a value: `,` and another key, or `}` once every key has one.
    After {
        tool: u16,
        used: u64,
    },
    /// The call is complete: an optional newline, then `</tool_call>`.
    End,
    EndNl,
}

impl Grammar {
    fn lit(&self, lit: u8, pos: usize, tool: u16, used: u64, key: u16) -> St {
        if pos < LITS[lit as usize].len() {
            return St::Lit {
                lit,
                pos: pos as u8,
                tool,
                used,
                key,
            };
        }
        let t = &self.tools[tool as usize];
        match lit {
            HEAD => St::Name { node: 0 },
            ARGS if t.all == 0 => self.lit(CLOSE, 0, tool, used, key),
            ARGS | COMMA => St::Key {
                tool,
                used,
                node: 0,
            },
            COLON if t.values[key as usize].is_some() => St::Enum {
                tool,
                used,
                key,
                node: 0,
            },
            COLON => self.lit(QUOTE, 0, tool, used, key),
            QUOTE => St::Str { tool, used, esc: 0 },
            _ => St::End,
        }
    }

    fn step(&self, s: St, b: u8) -> Option<St> {
        Some(match s {
            St::Start if b == b'\n' => self.lit(HEAD, 0, 0, 0, 0),
            St::Start => return self.step(self.lit(HEAD, 0, 0, 0, 0), b),
            St::Lit {
                lit,
                pos,
                tool,
                used,
                key,
            } => {
                let text = LITS[lit as usize];
                let pos = pos as usize;
                if text[pos] == 0 {
                    // The optional space, or past it to whatever follows, which may be the
                    // next part of the call when the space ends the literal.
                    let past = self.lit(lit, pos + 1, tool, used, key);
                    return if b == b' ' {
                        Some(past)
                    } else {
                        self.step(past, b)
                    };
                }
                if text[pos] != b {
                    return None;
                }
                self.lit(lit, pos + 1, tool, used, key)
            }
            St::Name { node } => {
                let n = self.names.step(node, b)?;
                match self.names.nodes[n as usize].leaf {
                    Some(tool) => self.lit(ARGS, 0, tool, 0, 0),
                    None => St::Name { node: n },
                }
            }
            St::Key { tool, used, node } => {
                let keys = &self.tools[tool as usize].keys;
                let n = keys.step(node, b)?;
                let node = &keys.nodes[n as usize];
                if node.reach & !used == 0 {
                    return None;
                }
                match node.leaf {
                    Some(k) => self.lit(COLON, 0, tool, used | 1 << k, k),
                    None => St::Key {
                        tool,
                        used,
                        node: n,
                    },
                }
            }
            St::Enum {
                tool,
                used,
                key,
                node,
            } => {
                let values = self.tools[tool as usize].values[key as usize].as_ref()?;
                let n = values.step(node, b)?;
                match values.nodes[n as usize].leaf {
                    Some(_) => St::After { tool, used },
                    None => St::Enum {
                        tool,
                        used,
                        key,
                        node: n,
                    },
                }
            }
            St::Str { tool, used, esc } => {
                let esc = match (esc, b) {
                    (0, b'"') => return Some(St::After { tool, used }),
                    (0, b'\\') => 1,
                    (0, ..0x20) => return None,
                    (0, _) => 0,
                    (1, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => 0,
                    (1, b'u') => 2,
                    (2..=5, b) if b.is_ascii_hexdigit() => (esc + 1) % 6,
                    _ => return None,
                };
                St::Str { tool, used, esc }
            }
            St::After { tool, used } => {
                let done = used == self.tools[tool as usize].all;
                match b {
                    b'}' if done => self.lit(CLOSE, 1, tool, used, 0),
                    b',' if !done => self.lit(COMMA, 1, tool, used, 0),
                    _ => return None,
                }
            }
            St::End if b == b'\n' => St::EndNl,
            St::End | St::EndNl => return None,
        })
    }
}

/// A vocabulary's tokens as bytes, indexed for [`Constraint`]. Built once per model.
pub struct TokenTable {
    /// Empty for special tokens.
    bytes: Vec<Vec<u8>>,
    /// Text tokens by their first byte.
    by_first: Vec<Vec<u32>>,
    /// Tokens that stay inside a JSON string: no quote, backslash or control byte.
    in_string: Vec<u64>,
    /// The other text tokens.
    not_in_string: Vec<u32>,
    open: Option<u32>,
    close: Option<u32>,
}

impl TokenTable {
    pub fn new(tok: &Tokenizer) -> Self {
        let bytes = (0..tok.vocab_size() as u32)
            .map(|id| {
                let mut b = Vec::new();
                if !tok.is_special(id) {
                    tok.decode_bytes(id, &mut b);
                }
                b
            })
            .collect();
        let special = |s| tok.token_id(s).filter(|&id| tok.is_special(id));
        Self::from_bytes(bytes, special(CALL_OPEN), special(CALL_CLOSE))
    }

    /// From each token's bytes (empty for special tokens) and the ids that open and close a call.
    pub fn from_bytes(bytes: Vec<Vec<u8>>, open: Option<u32>, close: Option<u32>) -> Self {
        let mut t = Self {
            by_first: vec![Vec::new(); 256],
            in_string: vec![0; bytes.len().div_ceil(64)],
            not_in_string: Vec::new(),
            bytes: Vec::new(),
            open,
            close,
        };
        for (id, b) in bytes.iter().enumerate() {
            let id = id as u32;
            if let Some(&first) = b.first() {
                t.by_first[first as usize].push(id);
                if b.iter().all(|&c| c >= 0x20 && c != b'"' && c != b'\\') {
                    t.in_string[id as usize / 64] |= 1 << (id % 64);
                } else {
                    t.not_in_string.push(id);
                }
            }
        }
        t.bytes = bytes;
        t
    }

    /// Whether this vocabulary marks tool calls with tokens of their own.
    pub fn has_call_tokens(&self) -> bool {
        self.open.is_some() && self.close.is_some()
    }
}

/// Masks the logits inside `<tool_call>` blocks to what [`Grammar`] allows, one generation's
/// worth. Outside a block nothing changes.
pub struct Constraint<'a> {
    g: &'a Grammar,
    t: &'a TokenTable,
    state: Option<St>,
    /// The tokens allowed in each state seen, as a bit per token.
    allowed: HashMap<St, Vec<u64>>,
}

impl<'a> Constraint<'a> {
    pub fn new(g: &'a Grammar, t: &'a TokenTable) -> Self {
        Self {
            g,
            t,
            state: None,
            allowed: HashMap::new(),
        }
    }

    fn walk(&self, mut s: St, bytes: &[u8]) -> Option<St> {
        for &b in bytes {
            s = self.g.step(s, b)?;
        }
        Some(s)
    }

    fn compute(&self, s: St) -> Vec<u64> {
        fn set(bits: &mut [u64], id: u32) {
            bits[id as usize / 64] |= 1 << (id % 64);
        }
        let mut bits = vec![0u64; self.t.in_string.len()];
        let first: Vec<bool> = (0..=255u8).map(|b| self.g.step(s, b).is_some()).collect();
        let ok = |id: &u32| {
            let b = &self.t.bytes[*id as usize];
            first[b[0] as usize] && self.walk(s, b).is_some()
        };
        if let St::Str { esc: 0, .. } = s {
            // Most of the vocabulary stays in the string: only the rest needs walking.
            bits.copy_from_slice(&self.t.in_string);
            for &id in self.t.not_in_string.iter().filter(|id| ok(id)) {
                set(&mut bits, id);
            }
        } else {
            for (b, ids) in self.t.by_first.iter().enumerate() {
                if first[b] {
                    for &id in ids.iter().filter(|id| ok(id)) {
                        set(&mut bits, id);
                    }
                }
            }
        }
        if matches!(s, St::End | St::EndNl)
            && let Some(close) = self.t.close
        {
            set(&mut bits, close);
        }
        bits
    }

    /// Whether a call is being written.
    pub fn in_call(&self) -> bool {
        self.state.is_some()
    }

    /// Sets every token the grammar rejects to -inf, inside a call.
    pub fn mask(&mut self, logits: &mut [f32]) {
        let Some(s) = self.state else { return };
        if !self.allowed.contains_key(&s) {
            let bits = self.compute(s);
            self.allowed.insert(s, bits);
        }
        let bits = &self.allowed[&s];
        // Never a dead end: a state always has a byte to go on with, and every byte is a token
        // in byte-level vocabularies; one without them is left unconstrained.
        if bits.iter().all(|&w| w == 0) {
            return;
        }
        for (i, l) in logits.iter_mut().enumerate() {
            if bits.get(i / 64).is_none_or(|w| w >> (i % 64) & 1 == 0) {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// Moves past the token chosen.
    pub fn advance(&mut self, id: u32) {
        self.state = match self.state {
            None => (Some(id) == self.t.open).then_some(St::Start),
            Some(_) if Some(id) == self.t.close => None,
            // A token the mask didn't allow (only when it had nothing to allow): stop checking.
            Some(s) => self
                .t
                .bytes
                .get(id as usize)
                .filter(|b| !b.is_empty())
                .and_then(|b| self.walk(s, b)),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn example() -> Value {
        json!([{"type": "function", "function": {
            "name": "fill_field",
            "description": "Fill one field of a form.",
            "parameters": {
                "type": "object",
                "properties": {
                    "form_id": {"type": "string", "enum": ["loan_application"]},
                    "field_id": {"type": "string", "enum": ["amount", "full_name", "phone"]},
                    "value": {"type": "string"}
                },
                "required": ["form_id", "field_id", "value"],
                "additionalProperties": false
            },
            "strict": true
        }}])
    }

    #[test]
    fn parses_the_supported_subset_in_required_order() {
        let t = &parse_tools(&example()).unwrap()[0];
        assert_eq!(t.name, "fill_field");
        let names: Vec<&str> = t.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["form_id", "field_id", "value"]);
        assert_eq!(t.params[2].values, None);
        assert_eq!(
            t.params[0].values.as_deref(),
            Some(&["loan_application".to_owned()][..])
        );
    }

    #[test]
    fn renders_like_python_json_dumps() {
        let t = &parse_tools(&example()).unwrap()[0];
        assert_eq!(
            t.to_json(),
            r#"{"type": "function", "function": {"name": "fill_field", "description": "Fill one field of a form.", "parameters": {"type": "object", "properties": {"form_id": {"type": "string", "enum": ["loan_application"]}, "field_id": {"type": "string", "enum": ["amount", "full_name", "phone"]}, "value": {"type": "string"}}, "required": ["form_id", "field_id", "value"], "additionalProperties": false}, "strict": true}}"#
        );
    }

    #[test]
    fn rejects_everything_outside_the_subset() {
        let edits: [(&str, Value); 13] = [
            ("/0/type", json!("retrieval")),
            ("/0/function/name", json!("bad name")),
            ("/0/function/strict", json!("yes")),
            ("/0/function/parameters/type", json!("array")),
            ("/0/function/parameters/additionalProperties", json!(true)),
            (
                "/0/function/parameters/required",
                json!(["form_id", "field_id"]),
            ),
            (
                "/0/function/parameters/required",
                json!(["form_id", "field_id", "value", "x"]),
            ),
            (
                "/0/function/parameters/required",
                json!(["value", "value", "form_id", "field_id"]),
            ),
            (
                "/0/function/parameters/properties/value",
                json!({"type": "number"}),
            ),
            (
                "/0/function/parameters/properties/value",
                json!({"type": "string", "description": "d"}),
            ),
            (
                "/0/function/parameters/properties/value",
                json!({"type": "string", "enum": []}),
            ),
            (
                "/0/function/parameters/properties/value",
                json!({"type": "string", "enum": [1]}),
            ),
            (
                "/0/function/parameters/properties/value",
                json!({"type": "string", "enum": ["a", "a"]}),
            ),
        ];
        for (path, v) in edits {
            let mut t = example();
            *t.pointer_mut(path).unwrap() = v.clone();
            assert!(parse_tools(&t).is_err(), "{path} = {v}");
        }
        let mut extra = example();
        extra[0]["function"]["parameters"]["minProperties"] = json!(1);
        assert!(parse_tools(&extra).is_err());
        let mut dup = example();
        dup.as_array_mut().unwrap().push(example()[0].clone());
        assert!(parse_tools(&dup).is_err());
        assert!(parse_tools(&json!([])).is_err());
        assert!(parse_tools(&json!({})).is_err());
    }

    #[test]
    fn extracts_one_call_and_the_text_around_it() {
        let text = "Sure.\n<tool_call>\n{\"name\": \"go\", \"arguments\": {\"page\": \"home\"}}\n</tool_call>";
        let (content, calls) = extract_calls(text);
        assert_eq!(content, "Sure.");
        assert_eq!(
            calls,
            [ToolCall {
                name: "go".into(),
                arguments: r#"{"page":"home"}"#.into()
            }]
        );
    }

    #[test]
    fn extracts_several_calls() {
        let text = "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n\
                    <tool_call>\n{\"name\": \"b\", \"arguments\": {\"x\": \"1\"}}\n</tool_call>";
        let (content, calls) = extract_calls(text);
        assert_eq!(content, "");
        let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
    }

    #[test]
    fn leaves_malformed_or_unclosed_calls_in_the_text() {
        for text in [
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {}\n</tool_call>",
            "<tool_call>{\"name\": 3, \"arguments\": {}}</tool_call>",
            "<tool_call>{\"name\": \"a\", \"arguments\": \"{}\"}</tool_call>",
            "<tool_call>[]</tool_call>",
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}",
            "</tool_call><tool_call>",
        ] {
            let (content, calls) = extract_calls(text);
            assert!(calls.is_empty(), "{text}");
            assert_eq!(content, text.trim(), "{text}");
        }
        // A good call after a bad one is still found.
        let (content, calls) = extract_calls(
            "<tool_call>oops</tool_call><tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>",
        );
        assert_eq!(
            (content.as_str(), calls.len()),
            ("<tool_call>oops</tool_call>", 1)
        );
    }

    fn tools2() -> Vec<Tool> {
        let mut t = parse_tools(&example()).unwrap();
        t.extend(
            parse_tools(&json!([{"type": "function", "function": {"name": "go_back",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": false}}}]))
            .unwrap(),
        );
        t
    }

    /// Walks `text` through the grammar from the start of a call.
    fn accepts(g: &Grammar, text: &str) -> bool {
        let mut s = St::Start;
        for &b in text.as_bytes() {
            match g.step(s, b) {
                Some(n) => s = n,
                None => return false,
            }
        }
        matches!(s, St::End | St::EndNl)
    }

    #[test]
    fn grammar_accepts_calls_in_any_key_order_and_spacing() {
        let g = Grammar::new(&tools2());
        for text in [
            "\n{\"name\": \"fill_field\", \"arguments\": {\"form_id\": \"loan_application\", \"field_id\": \"amount\", \"value\": \"500000\"}}\n",
            "{\"name\":\"fill_field\",\"arguments\":{\"value\":\"Asha \\\"A\\\" Juma \\u00e9 ü\",\"field_id\":\"full_name\",\"form_id\":\"loan_application\"}}",
            "{\"name\": \"go_back\", \"arguments\": {}}",
        ] {
            assert!(accepts(&g, text), "{text}");
        }
    }

    #[test]
    fn grammar_rejects_unknown_names_keys_values_and_missing_or_repeated_keys() {
        let g = Grammar::new(&tools2());
        for text in [
            "{\"name\": \"delete_all\", \"arguments\": {}}",
            "{\"name\": \"go_back\", \"arguments\": {\"x\": \"1\"}}",
            "{\"name\": \"fill_field\", \"arguments\": {\"form_id\": \"other\", \"field_id\": \"amount\", \"value\": \"1\"}}",
            "{\"name\": \"fill_field\", \"arguments\": {\"form_id\": \"loan_applicationx\", \"field_id\": \"amount\", \"value\": \"1\"}}",
            "{\"name\": \"fill_field\", \"arguments\": {\"form_id\": \"loan_application\", \"value\": \"1\"}}",
            "{\"name\": \"fill_field\", \"arguments\": {\"value\": \"1\", \"value\": \"2\", \"form_id\": \"loan_application\", \"field_id\": \"amount\"}}",
            "{\"name\": \"fill_field\", \"arguments\": {\"form_id\": \"loan_application\", \"field_id\": \"amount\", \"value\": 5}}",
            "{\"name\": \"fill_field\", \"arguments\": {\"form_id\": \"loan_application\", \"field_id\": \"amount\", \"value\": \"a\nb\"}}",
            "{\"name\": \"go_back\", \"arguments\": {}, \"extra\": 1}",
            "{\"name\": \"go_back\"}",
        ] {
            assert!(!accepts(&g, text), "{text}");
        }
    }

    /// A vocabulary of every byte plus multi-byte tokens that straddle the literals' boundaries
    /// or spell what the schema forbids, and the two call tokens.
    fn vocab() -> (Vec<Vec<u8>>, u32, u32) {
        let mut v: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for t in [
            "{\"",
            "name",
            "\":",
            " \"",
            "fill",
            "_field",
            "fill_field\"",
            "\", \"",
            "arguments",
            "\": {\"",
            "form_id",
            "field",
            "_id",
            "\": \"",
            "loan",
            "_app",
            "loan_application\"",
            "lication\", \"",
            "amount\"",
            "full_name",
            "phone\"}",
            "value",
            "Asha",
            " Juma",
            "\"}}",
            "\"}}\n",
            "}}",
            "\"delete_all\"",
            "evil",
            "\"evil\": \"",
            "other\"",
            "\", \"value\": \"x",
            "go_back",
            "{}}",
            "\": {}}",
            "\\u00",
            "\\\"",
        ] {
            v.push(t.as_bytes().to_vec());
        }
        v.push(Vec::new()); // a stop token
        v.push(Vec::new()); // <tool_call>
        v.push(Vec::new()); // </tool_call>
        let n = v.len() as u32;
        (v, n - 2, n - 1)
    }

    /// Generates with the constraint and `pick` choosing among the logits it leaves, from
    /// `<tool_call>` until `</tool_call>` or `steps` tokens. Returns the call's text if it closed.
    fn run(
        g: &Grammar,
        t: &TokenTable,
        steps: usize,
        mut pick: impl FnMut(&[f32]) -> u32,
    ) -> Option<String> {
        let (bytes, open, close) = vocab();
        let mut c = Constraint::new(g, t);
        c.advance(open);
        let mut out = Vec::new();
        for _ in 0..steps {
            let mut logits = vec![0.0f32; bytes.len()];
            c.mask(&mut logits);
            assert!(
                logits.iter().any(|l| l.is_finite()),
                "dead end after {:?}",
                String::from_utf8_lossy(&out)
            );
            let id = pick(&logits);
            assert!(logits[id as usize].is_finite(), "picked a masked token");
            if id == close {
                c.advance(id);
                assert!(!c.in_call());
                return Some(String::from_utf8(out).expect("utf-8"));
            }
            out.extend(&bytes[id as usize]);
            c.advance(id);
            assert!(c.in_call());
        }
        None
    }

    /// The call parsed back and checked against the schemas, independently of the grammar.
    fn check_call(tools: &[Tool], text: &str) {
        let v: Value = serde_json::from_str(text.trim()).unwrap_or_else(|e| panic!("{e}: {text}"));
        let m = v.as_object().unwrap();
        assert_eq!(m.len(), 2, "{text}");
        let tool = tools
            .iter()
            .find(|t| v["name"] == t.name.as_str())
            .unwrap_or_else(|| panic!("{text}"));
        let args = v["arguments"].as_object().unwrap();
        assert_eq!(args.len(), tool.params.len(), "{text}");
        for p in &tool.params {
            let a = args
                .get(&p.name)
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("{text}"));
            if let Some(values) = &p.values {
                assert!(values.iter().any(|v| v == a), "{text}");
            }
        }
    }

    #[test]
    fn random_and_adversarial_logits_only_produce_valid_calls() {
        let tools = tools2();
        let g = Grammar::new(&tools);
        let (bytes, open, close) = vocab();
        let t = TokenTable::from_bytes(bytes.clone(), Some(open), Some(close));
        let mut closed = 0;
        for seed in 1..=300u64 {
            let mut r = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            // Random scores, the forbidden tokens far ahead of the rest, the close token and
            // closing quotes favoured often enough that strings end.
            let text = run(&g, &t, 400, |logits| {
                let mut best = (f32::NEG_INFINITY, 0);
                for (i, &l) in logits.iter().enumerate() {
                    r ^= r << 13;
                    r ^= r >> 7;
                    r ^= r << 17;
                    let tok = &bytes[i];
                    let forbidden = [
                        &b"\"delete_all\""[..],
                        b"evil",
                        b"\"evil\": \"",
                        b"other\"",
                        b"\", \"value\": \"x",
                    ];
                    let bias = if forbidden.contains(&tok.as_slice()) {
                        100.0
                    } else if i as u32 == close || tok.contains(&b'"') {
                        2.0
                    } else {
                        0.0
                    };
                    let score = l + bias + (r % 1000) as f32 / 1000.0;
                    if score > best.0 {
                        best = (score, i as u32);
                    }
                }
                best.1
            });
            if let Some(text) = text {
                check_call(&tools, &text);
                closed += 1;
            }
        }
        assert!(closed > 250, "only {closed} calls closed");
    }

    #[test]
    fn outside_a_call_nothing_is_masked() {
        let g = Grammar::new(&tools2());
        let (bytes, open, close) = vocab();
        let t = TokenTable::from_bytes(bytes.clone(), Some(open), Some(close));
        let mut c = Constraint::new(&g, &t);
        let mut logits = vec![1.0f32; bytes.len()];
        c.mask(&mut logits);
        assert!(logits.iter().all(|&l| l == 1.0));
        c.advance(b'x' as u32);
        assert!(!c.in_call());
        c.advance(open);
        assert!(c.in_call());
        c.mask(&mut logits);
        // In a call: only a newline or `{` (and the tokens starting `{"`) may come first.
        let allowed: Vec<&[u8]> = (0..bytes.len())
            .filter(|&i| logits[i].is_finite())
            .map(|i| bytes[i].as_slice())
            .collect();
        assert!(
            allowed.iter().all(|b| b[0] == b'\n' || b[0] == b'{'),
            "{allowed:?}"
        );
        assert!(!allowed.is_empty());
    }
}
