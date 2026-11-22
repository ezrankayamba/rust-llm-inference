//! Chat prompt formatting. GGUF ships a Jinja template; rather than embed a Jinja engine, the
//! template text is fingerprinted to pick one of the common formats below, and any default system
//! text it carries is read from it, so models get what their authors wrote without per-model code.

use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<|im_start|>role\n...<|im_end|>` (Qwen, SmolLM, many fine-tunes).
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>`.
    Llama3,
    /// `[INST] ... [/INST]` with `<<SYS>>` and `<s>` between turns (Llama 2).
    Llama2,
    /// `[INST] ... [/INST]` with one BOS and no system block (Mistral, Mixtral).
    Mistral,
    /// `<|user|>\n...</s>\n<|assistant|>\n` (TinyLlama chat, Zephyr).
    Zephyr,
    /// `User: ...\n\nAssistant: ...<｜end▁of▁sentence｜>` (DeepSeek V2).
    DeepSeek2,
    /// `<｜User｜>...<｜Assistant｜>...<｜end▁of▁sentence｜>` after the system text (DeepSeek V3,
    /// R1 and V3.1, which differ in their think tags: [`ChatDefaults::think`]).
    DeepSeek3,
}

/// DeepSeek's end of a reply in its templates.
const DS_EOS: &str = "<｜end▁of▁sentence｜>";

/// How a DeepSeek V3-family template opens a reply.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Think {
    /// No think tags (V3).
    #[default]
    None,
    /// The reply opens a think block (R1), and replies in the history keep only the text after
    /// their last `</think>`.
    Open,
    /// The reply opens with a closed one, `</think>` (V3.1 without `thinking`), in the history
    /// too, where replies keep the text after their first `</think>`.
    Closed,
}

/// A rendered prompt and the byte ranges of `text` that hold client-supplied text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    /// Message content: no control token parses here (user-defined ones still do, so a reply
    /// containing `<think>` replays as generated).
    pub untrusted: Vec<Range<usize>>,
    /// Tool definitions: no special token at all parses here, `<tool_call>` included.
    pub opaque: Vec<Range<usize>>,
}

impl Rendered {
    fn push_untrusted(&mut self, content: &str) {
        let start = self.text.len();
        self.text += content;
        self.untrusted.push(start..self.text.len());
    }

    fn push_opaque(&mut self, content: &str) {
        let start = self.text.len();
        self.text += content;
        self.opaque.push(start..self.text.len());
    }
}

/// What a request adds to the conversation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderOpts {
    /// Tool definitions as JSON (`Tool::to_json`), written into the system turn. Only ChatML
    /// templates with a Hermes tools block (`ChatDefaults::tools`) take them.
    pub tools: Vec<String>,
    /// Opens the reply with an empty think block, as Qwen3's template does for
    /// `enable_thinking=false`, so the model answers without reasoning first.
    pub no_think: bool,
}

const TOOLS_HEAD: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
     You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const TOOLS_TAIL: &str = "\n</tools>\n\nFor each function call, return a json object with function name and \
     arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
     {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// What a template adds when the conversation doesn't say otherwise.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatDefaults {
    /// ChatML system prompt used when the first message isn't a system message.
    pub system: Option<String>,
    /// Llama 3.x always writes a system turn with this cutoff and today's date.
    pub knowledge_cutoff: Option<String>,
    /// The template has Qwen's (Hermes) tools block: `<tools>` definitions, `<tool_call>` replies.
    pub tools: bool,
    /// The template can open a reply with an empty think block (Qwen3's `enable_thinking`).
    pub thinking: bool,
    /// DeepSeek V3-family think tags.
    pub think: Think,
}

impl ChatDefaults {
    pub fn from_jinja(template: ChatTemplate, jinja: &str) -> Self {
        Self {
            system: (template == ChatTemplate::ChatMl)
                .then(|| chatml_default_system(jinja))
                .flatten(),
            knowledge_cutoff: (template == ChatTemplate::Llama3)
                .then(|| llama3_cutoff(jinja))
                .flatten(),
            tools: template == ChatTemplate::ChatMl
                && jinja.contains("<tools>")
                && jinja.contains("<tool_call>"),
            thinking: template == ChatTemplate::ChatMl && jinja.contains("enable_thinking"),
            think: match template {
                ChatTemplate::DeepSeek3 if jinja.contains("<｜Assistant｜><think>") => {
                    Think::Open
                }
                ChatTemplate::DeepSeek3 if jinja.contains("{{'</think>'}}") => Think::Closed,
                _ => Think::None,
            },
        }
    }
}

/// The first string literal holding a whole system turn, e.g.
/// `'<|im_start|>system\nYou are helpful.<|im_end|>\n'`. Templates that insert the message there
/// close the literal first (`'<|im_start|>system\n' + message`), so they don't match.
fn chatml_default_system(jinja: &str) -> Option<String> {
    const OPEN: &str = "<|im_start|>system";
    jinja.match_indices(OPEN).find_map(|(pos, _)| {
        let quote = jinja[..pos]
            .chars()
            .next_back()
            .filter(|c| matches!(c, '\'' | '"'))?;
        let rest = &jinja[pos + OPEN.len()..];
        let rest = rest
            .strip_prefix("\\n")
            .or_else(|| rest.strip_prefix('\n'))?;
        let body = &rest[..rest.find("<|im_end|>")?];
        let text = unescape_literal(body, quote)?;
        (!text.is_empty()).then_some(text)
    })
}

/// A Jinja string literal's value, or `None` if the literal ends before `body` does.
fn unescape_literal(body: &str, quote: char) -> Option<String> {
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                other => out.push(other),
            },
            c if c == quote => return None,
            c => out.push(c),
        }
    }
    Some(out)
}

fn llama3_cutoff(jinja: &str) -> Option<String> {
    const KEY: &str = "Cutting Knowledge Date: ";
    let rest = &jinja[jinja.find(KEY)? + KEY.len()..];
    let end = rest.find(['\\', '\n', '\'', '"'])?;
    (end > 0).then(|| rest[..end].to_owned())
}

/// `%d %b %Y` in UTC, the format Llama 3.x templates ask for. The standard library has no
/// calendar; this is the usual days-to-civil conversion over 400-year eras.
pub fn format_date(unix_secs: u64) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    // Days since 0000-03-01, so a leap day falls at the end of a year.
    let z = unix_secs / 86_400 + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!("{day:02} {} {year}", MONTHS[month as usize - 1])
}

impl ChatTemplate {
    /// The format a template's text fingerprints as, if any.
    pub fn detect(jinja: &str) -> Option<Self> {
        match jinja {
            t if t.contains("<|start_header_id|>") => Some(Self::Llama3),
            t if t.contains("<|im_start|>") => Some(Self::ChatMl),
            t if t.contains("<|user|>") => Some(Self::Zephyr),
            t if t.contains("<<SYS>>") => Some(Self::Llama2),
            t if t.contains("[INST]") => Some(Self::Mistral),
            t if t.contains("<｜User｜>") => Some(Self::DeepSeek3),
            t if t.contains("'User: '") && t.contains("'Assistant:'") => Some(Self::DeepSeek2),
            _ => None,
        }
    }

    /// Renders the conversation, ending with the assistant turn opener. BOS is not included:
    /// the tokenizer adds it when the model requests one. `today` goes in Llama 3.x's date line.
    pub fn render(self, messages: &[Message], defaults: &ChatDefaults, today: &str) -> Rendered {
        self.render_with(messages, defaults, today, &RenderOpts::default())
    }

    /// [`render`](Self::render) with tools and a think switch, which only ChatML applies.
    pub fn render_with(
        self,
        messages: &[Message],
        defaults: &ChatDefaults,
        today: &str,
        opts: &RenderOpts,
    ) -> Rendered {
        let mut r = Rendered::default();
        match self {
            Self::ChatMl => {
                let mut messages = messages;
                let first_is_system = messages
                    .first()
                    .is_some_and(|m| matches!(m.role.as_str(), "system" | "developer"));
                if !opts.tools.is_empty() {
                    // Qwen's tools branch: the system text (the client's, else the default),
                    // then the definitions, all in one system turn.
                    r.text += "<|im_start|>system\n";
                    if first_is_system {
                        r.push_untrusted(&messages[0].content);
                        r.text += "\n\n";
                        messages = &messages[1..];
                    } else if let Some(sys) = &defaults.system {
                        r.text += sys;
                        r.text += "\n\n";
                    }
                    r.text += TOOLS_HEAD;
                    for t in &opts.tools {
                        r.text += "\n";
                        r.push_opaque(t);
                    }
                    r.text += TOOLS_TAIL;
                } else if let Some(sys) = &defaults.system
                    && !first_is_system
                {
                    r.text += &format!("<|im_start|>system\n{sys}<|im_end|>\n");
                }
                for m in messages {
                    r.text += &format!("<|im_start|>{}\n", m.role);
                    r.push_untrusted(&m.content);
                    r.text += "<|im_end|>\n";
                }
                r.text += "<|im_start|>assistant\n";
                if opts.no_think {
                    r.text += "<think>\n\n</think>\n\n";
                }
            }
            Self::Llama3 => {
                let mut messages = messages;
                if let Some(cutoff) = &defaults.knowledge_cutoff {
                    r.text += &format!(
                        "<|start_header_id|>system<|end_header_id|>\n\n\
                         Cutting Knowledge Date: {cutoff}\nToday Date: {today}\n\n"
                    );
                    if let Some((m, rest)) = messages.split_first()
                        && m.role == "system"
                    {
                        r.push_untrusted(m.content.trim());
                        messages = rest;
                    }
                    r.text += "<|eot_id|>";
                }
                for m in messages {
                    r.text += &format!("<|start_header_id|>{}<|end_header_id|>\n\n", m.role);
                    r.push_untrusted(m.content.trim());
                    r.text += "<|eot_id|>";
                }
                r.text += "<|start_header_id|>assistant<|end_header_id|>\n\n";
            }
            Self::Zephyr => {
                for m in messages {
                    r.text += &format!("<|{}|>\n", m.role);
                    r.push_untrusted(&m.content);
                    r.text += "</s>\n";
                }
                r.text += "<|assistant|>\n";
            }
            Self::Llama2 => {
                let mut system: Option<&str> = None;
                let mut first = true;
                for m in messages {
                    match m.role.as_str() {
                        "system" | "developer" => system = Some(&m.content),
                        "user" => {
                            if !first {
                                r.text += "<s>";
                            }
                            r.text += "[INST] ";
                            if let Some(sys) = system.take() {
                                r.text += "<<SYS>>\n";
                                r.push_untrusted(sys);
                                r.text += "\n<</SYS>>\n\n";
                            }
                            r.push_untrusted(m.content.trim());
                            r.text += " [/INST]";
                            first = false;
                        }
                        _ => {
                            r.text += " ";
                            r.push_untrusted(m.content.trim());
                            r.text += "</s>";
                        }
                    }
                }
            }
            Self::Mistral => {
                let mut system: Option<&str> = None;
                for m in messages {
                    match m.role.as_str() {
                        "system" | "developer" => system = Some(&m.content),
                        "user" => {
                            r.text += "[INST] ";
                            // On the first user turn, not the last, so the prompt prefix stays
                            // the same from turn to turn and the KV cache keeps matching it.
                            if let Some(sys) = system.take() {
                                r.push_untrusted(sys);
                                r.text += "\n\n";
                            }
                            r.push_untrusted(&m.content);
                            r.text += " [/INST]";
                        }
                        // Replies are generated with a leading space, and the one-space form
                        // replays to the same tokens whether or not a client stripped it.
                        _ => {
                            r.text += " ";
                            r.push_untrusted(m.content.trim());
                            r.text += "</s>";
                        }
                    }
                }
                if let Some(sys) = system {
                    r.text += "[INST] ";
                    r.push_untrusted(sys);
                    r.text += " [/INST]";
                }
            }
            Self::DeepSeek2 => {
                for m in messages {
                    match m.role.as_str() {
                        "user" => {
                            r.text += "User: ";
                            r.push_untrusted(&m.content);
                            r.text += "\n\n";
                        }
                        "assistant" => {
                            r.text += "Assistant: ";
                            r.push_untrusted(&m.content);
                            r.text += DS_EOS;
                        }
                        "system" | "developer" => {
                            r.push_untrusted(&m.content);
                            r.text += "\n\n";
                        }
                        _ => {}
                    }
                }
                r.text += "Assistant:";
            }
            Self::DeepSeek3 => {
                // Every system message goes first, joined by a blank line.
                let system = |m: &&Message| matches!(m.role.as_str(), "system" | "developer");
                for (i, m) in messages.iter().filter(system).enumerate() {
                    if i > 0 {
                        r.text += "\n\n";
                    }
                    r.push_untrusted(&m.content);
                }
                let opener = match defaults.think {
                    Think::Closed => "<｜Assistant｜></think>",
                    _ => "<｜Assistant｜>",
                };
                for m in messages {
                    match m.role.as_str() {
                        "user" => {
                            r.text += "<｜User｜>";
                            r.push_untrusted(&m.content);
                        }
                        "assistant" => {
                            let c = m.content.as_str();
                            let c = match defaults.think {
                                Think::None => c,
                                Think::Open => c.rsplit("</think>").next().unwrap_or(c),
                                Think::Closed => c.split_once("</think>").map_or(c, |(_, t)| t),
                            };
                            r.text += opener;
                            r.push_untrusted(c);
                            r.text += DS_EOS;
                        }
                        _ => {}
                    }
                }
                r.text += opener;
                if defaults.think == Think::Open {
                    r.text += "<think>\n";
                }
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(t: ChatTemplate, msgs: &[Message]) -> Rendered {
        t.render(msgs, &ChatDefaults::default(), "")
    }

    /// Expected texts from the models' own templates (tokenizer_config.json) through jinja2,
    /// with an empty BOS.
    #[test]
    fn deepseek_formats_match_their_templates() {
        let msgs = [
            Message::new("system", "Be brief."),
            Message::new("user", "Hi"),
            Message::new("assistant", "<think>x</think>Hello!"),
            Message::new("system", "Also kind."),
            Message::new("user", "Bye"),
        ];
        for (jinja, want) in [
            (
                "{{ 'User: ' + message['content'] + '\\n\\n' }}{{ 'Assistant:' }}",
                "Be brief.\n\nUser: Hi\n\nAssistant: <think>x</think>Hello!<｜end▁of▁sentence｜>\
                 Also kind.\n\nUser: Bye\n\nAssistant:",
            ),
            (
                "{{'<｜User｜>' + message['content']}}{{'<｜Assistant｜>'}}",
                "Be brief.\n\nAlso kind.<｜User｜>Hi<｜Assistant｜><think>x</think>Hello!\
                 <｜end▁of▁sentence｜><｜User｜>Bye<｜Assistant｜>",
            ),
            (
                "{{'<｜User｜>' + message['content']}}{{'<｜Assistant｜><think>\\n'}}",
                "Be brief.\n\nAlso kind.<｜User｜>Hi<｜Assistant｜>Hello!<｜end▁of▁sentence｜>\
                 <｜User｜>Bye<｜Assistant｜><think>\n",
            ),
            (
                "{{'<｜User｜>' + message['content']}}{%- if not thinking %}{{'</think>'}}",
                "Be brief.\n\nAlso kind.<｜User｜>Hi<｜Assistant｜></think>Hello!\
                 <｜end▁of▁sentence｜><｜User｜>Bye<｜Assistant｜></think>",
            ),
        ] {
            let t = ChatTemplate::detect(jinja).expect("detected");
            let got = t
                .render(&msgs, &ChatDefaults::from_jinja(t, jinja), "")
                .text;
            assert_eq!(got, want, "{jinja}");
        }
    }

    #[test]
    fn detects_formats_from_jinja_markers() {
        for (jinja, want) in [
            ("{{'<|im_start|>' + m.role}}", Some(ChatTemplate::ChatMl)),
            ("<|start_header_id|>", Some(ChatTemplate::Llama3)),
            ("{{ '<|user|>\n' }}", Some(ChatTemplate::Zephyr)),
            ("[INST] <<SYS>>", Some(ChatTemplate::Llama2)),
            ("[INST] x [/INST]", Some(ChatTemplate::Mistral)),
            ("{{ message }}", None),
        ] {
            assert_eq!(ChatTemplate::detect(jinja), want, "{jinja}");
        }
    }

    #[test]
    fn mistral_renders_turns_with_one_bos_and_no_system_block() {
        let msgs = [
            Message::new("user", "U1"),
            Message::new("assistant", " A1 "),
            Message::new("user", "U2"),
        ];
        assert_eq!(
            plain(ChatTemplate::Mistral, &msgs).text,
            "[INST] U1 [/INST] A1</s>[INST] U2 [/INST]"
        );
    }

    #[test]
    fn mistral_puts_system_at_the_start_of_the_first_user_turn() {
        for role in ["system", "developer"] {
            let msgs = [
                Message::new(role, "S"),
                Message::new("user", "U1"),
                Message::new("assistant", "A1"),
                Message::new("user", "U2"),
            ];
            assert_eq!(
                plain(ChatTemplate::Mistral, &msgs).text,
                "[INST] S\n\nU1 [/INST] A1</s>[INST] U2 [/INST]"
            );
        }
    }

    #[test]
    fn mistral_gives_a_lone_system_message_its_own_turn() {
        let msgs = [Message::new("system", "S")];
        assert_eq!(plain(ChatTemplate::Mistral, &msgs).text, "[INST] S [/INST]");
    }

    #[test]
    fn llama2_treats_developer_as_system() {
        let msgs = [Message::new("developer", "S"), Message::new("user", "U")];
        assert_eq!(
            plain(ChatTemplate::Llama2, &msgs).text,
            "[INST] <<SYS>>\nS\n<</SYS>>\n\nU [/INST]"
        );
    }

    #[test]
    fn chatml_renders_turns_and_generation_prompt() {
        let msgs = [
            Message::new("system", "Be brief."),
            Message::new("user", "Hi"),
        ];
        assert_eq!(
            plain(ChatTemplate::ChatMl, &msgs).text,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn llama2_folds_system_into_first_user_turn() {
        let msgs = [
            Message::new("system", "S"),
            Message::new("user", "U1"),
            Message::new("assistant", "A1"),
            Message::new("user", "U2"),
        ];
        assert_eq!(
            plain(ChatTemplate::Llama2, &msgs).text,
            "[INST] <<SYS>>\nS\n<</SYS>>\n\nU1 [/INST] A1</s><s>[INST] U2 [/INST]"
        );
    }

    #[test]
    fn render_marks_only_message_content_as_untrusted() {
        let msgs = [
            Message::new("system", "S"),
            Message::new("user", " U "),
            Message::new("assistant", "A"),
            Message::new("user", "V"),
        ];
        for (template, want) in [
            (ChatTemplate::ChatMl, vec!["S", " U ", "A", "V"]),
            (ChatTemplate::Llama3, vec!["S", "U", "A", "V"]),
            (ChatTemplate::Zephyr, vec!["S", " U ", "A", "V"]),
            (ChatTemplate::Llama2, vec!["S", "U", "A", "V"]),
            (ChatTemplate::Mistral, vec!["S", " U ", "A", "V"]),
        ] {
            let r = plain(template, &msgs);
            let got: Vec<&str> = r.untrusted.iter().map(|x| &r.text[x.clone()]).collect();
            assert_eq!(got, want, "{template:?}");
        }
    }

    #[test]
    fn chatml_default_reads_a_literal_with_real_or_escaped_newlines() {
        for jinja in [
            "{{ '<|im_start|>system\nYou are helpful.<|im_end|>\n' }}",
            "{{ '<|im_start|>system\\nYou are helpful.<|im_end|>\\n' }}",
            "{{ \"<|im_start|>system\\nYou are helpful.<|im_end|>\\n\" }}",
        ] {
            assert_eq!(
                chatml_default_system(jinja).as_deref(),
                Some("You are helpful."),
                "{jinja}"
            );
        }
    }

    #[test]
    fn chatml_default_keeps_quotes_and_escapes_in_the_text() {
        let jinja = r#"{{ '<|im_start|>system\nIt\'s "Bob".\nHi.<|im_end|>' }}"#;
        assert_eq!(
            chatml_default_system(jinja).as_deref(),
            Some("It's \"Bob\".\nHi.")
        );
    }

    #[test]
    fn chatml_default_ignores_templates_that_insert_the_message() {
        let jinja = "{{ '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}";
        assert_eq!(chatml_default_system(jinja), None);
    }

    #[test]
    fn chatml_default_skips_an_inserting_branch_to_find_the_literal() {
        let jinja = "{{ '<|im_start|>system\\n' }}{{ m }}<|im_end|>\
                     {{ '<|im_start|>system\\nDefault.<|im_end|>' }}";
        assert_eq!(chatml_default_system(jinja).as_deref(), Some("Default."));
    }

    #[test]
    fn default_extraction_tolerates_broken_templates() {
        for jinja in [
            "",
            "<|im_start|>system",
            "'<|im_start|>system",
            "'<|im_start|>system\n",
            "'<|im_start|>system\nno end",
            "'<|im_start|>system\n<|im_end|>",
            "'<|im_start|>system\ntrailing\\<|im_end|>",
            "Cutting Knowledge Date: ",
            "Cutting Knowledge Date: \\n",
            "é'<|im_start|>system\nü<|im_end|>",
        ] {
            let d = ChatDefaults::from_jinja(ChatTemplate::ChatMl, jinja);
            let e = ChatDefaults::from_jinja(ChatTemplate::Llama3, jinja);
            assert_eq!(e.system, None, "{jinja}");
            assert_eq!(d.knowledge_cutoff, None, "{jinja}");
        }
    }

    #[test]
    fn llama3_cutoff_stops_at_the_end_of_the_line() {
        let jinja = r#"{{- "Cutting Knowledge Date: December 2023\n" }}"#;
        assert_eq!(llama3_cutoff(jinja).as_deref(), Some("December 2023"));
        assert_eq!(llama3_cutoff("no date here"), None);
    }

    #[test]
    fn chatml_adds_the_default_unless_a_system_message_comes_first() {
        let d = ChatDefaults {
            system: Some("D".into()),
            knowledge_cutoff: None,
            ..Default::default()
        };
        let user = [Message::new("user", "U")];
        assert_eq!(
            ChatTemplate::ChatMl.render(&user, &d, "").text,
            "<|im_start|>system\nD<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
        );
        for role in ["system", "developer"] {
            let msgs = [Message::new(role, "S"), Message::new("user", "U")];
            assert!(
                !ChatTemplate::ChatMl
                    .render(&msgs, &d, "")
                    .text
                    .contains('D')
            );
        }
    }

    #[test]
    fn chatml_default_is_not_marked_untrusted() {
        let d = ChatDefaults {
            system: Some("D".into()),
            knowledge_cutoff: None,
            ..Default::default()
        };
        let r = ChatTemplate::ChatMl.render(&[Message::new("user", "U")], &d, "");
        let got: Vec<&str> = r.untrusted.iter().map(|x| &r.text[x.clone()]).collect();
        assert_eq!(got, ["U"]);
    }

    #[test]
    fn llama3_header_holds_the_date_and_the_first_system_message() {
        let d = ChatDefaults {
            system: None,
            knowledge_cutoff: Some("December 2023".into()),
            ..Default::default()
        };
        let head = "<|start_header_id|>system<|end_header_id|>\n\n\
                    Cutting Knowledge Date: December 2023\nToday Date: 09 Nov 2026\n\n";
        let tail = "<|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|>\
                    <|start_header_id|>assistant<|end_header_id|>\n\n";
        let user = Message::new("user", "U");
        let bare = ChatTemplate::Llama3.render(std::slice::from_ref(&user), &d, "09 Nov 2026");
        assert_eq!(bare.text, format!("{head}<|eot_id|>{tail}"));
        let sys = Message::new("system", " S ");
        let with = ChatTemplate::Llama3.render(&[sys, user], &d, "09 Nov 2026");
        assert_eq!(with.text, format!("{head}S<|eot_id|>{tail}"));
    }

    #[test]
    fn format_date_handles_epoch_leap_days_and_year_ends() {
        for (secs, want) in [
            (0, "01 Jan 1970"),
            (1_709_164_800, "29 Feb 2024"),
            (1_709_251_200, "01 Mar 2024"),
            (1_721_952_000, "26 Jul 2024"),
            (1_735_689_599, "31 Dec 2024"),
            (1_735_689_600, "01 Jan 2025"),
        ] {
            assert_eq!(format_date(secs), want);
        }
    }

    fn tools_opts() -> RenderOpts {
        RenderOpts {
            tools: vec![r#"{"type": "function", "function": {"name": "go"}}"#.into()],
            no_think: true,
        }
    }

    #[test]
    fn chatml_renders_qwen_tools_block_and_empty_think() {
        let msgs = [Message::new("system", "S"), Message::new("user", "U")];
        let r =
            ChatTemplate::ChatMl.render_with(&msgs, &ChatDefaults::default(), "", &tools_opts());
        assert_eq!(
            r.text,
            "<|im_start|>system\nS\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\
             {\"type\": \"function\", \"function\": {\"name\": \"go\"}}\n</tools>\n\n\
             For each function call, return a json object with function name and arguments within \
             <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": \
             <args-json-object>}\n</tool_call><|im_end|>\n\
             <|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let opaque: Vec<&str> = r.opaque.iter().map(|x| &r.text[x.clone()]).collect();
        assert_eq!(
            opaque,
            [r#"{"type": "function", "function": {"name": "go"}}"#]
        );
        let untrusted: Vec<&str> = r.untrusted.iter().map(|x| &r.text[x.clone()]).collect();
        assert_eq!(untrusted, ["S", "U"]);
    }

    #[test]
    fn chatml_tools_block_takes_the_default_system_text_or_none() {
        let user = [Message::new("user", "U")];
        let bare =
            ChatTemplate::ChatMl.render_with(&user, &ChatDefaults::default(), "", &tools_opts());
        assert!(
            bare.text.starts_with("<|im_start|>system\n# Tools\n\n"),
            "{}",
            bare.text
        );
        let d = ChatDefaults {
            system: Some("D".into()),
            ..Default::default()
        };
        let with = ChatTemplate::ChatMl.render_with(&user, &d, "", &tools_opts());
        assert!(
            with.text
                .starts_with("<|im_start|>system\nD\n\n# Tools\n\n"),
            "{}",
            with.text
        );
    }

    #[test]
    fn detects_qwen3_tools_and_thinking_switch() {
        let qwen3 = "{%- if tools %}<tools></tools><tool_call>{%- if enable_thinking is defined %}<|im_start|>";
        let d = ChatDefaults::from_jinja(ChatTemplate::ChatMl, qwen3);
        assert!(d.tools && d.thinking);
        let smol = "{{'<|im_start|>' + message['role']}}";
        let d = ChatDefaults::from_jinja(ChatTemplate::ChatMl, smol);
        assert!(!d.tools && !d.thinking);
    }
}
