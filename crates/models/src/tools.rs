//! OpenAI-style function tools for templates in the Hermes format Qwen2.5 and Qwen3 were trained
//! on: definitions go into the system turn inside `<tools></tools>`, and the model answers with
//! `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks.
//!
//! Only a subset of JSON Schema is accepted: an object whose properties are all required strings,
//! optionally limited to an enum. Anything else is an error rather than ignored, so a client
//! never believes a constraint holds that the server doesn't know about.

use serde_json::{Map, Value};

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
}
