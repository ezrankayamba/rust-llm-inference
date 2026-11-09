//! Chat prompt formatting. GGUF ships a Jinja template; rather than embed a Jinja engine, the
//! template text is fingerprinted to pick one of the common formats below.

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
}

/// A rendered prompt and the byte ranges of `text` that hold message content.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    pub untrusted: Vec<Range<usize>>,
}

impl Rendered {
    fn push_untrusted(&mut self, content: &str) {
        let start = self.text.len();
        self.text += content;
        self.untrusted.push(start..self.text.len());
    }
}

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

impl ChatTemplate {
    /// The format a template's text fingerprints as, if any.
    pub fn detect(jinja: &str) -> Option<Self> {
        match jinja {
            t if t.contains("<|start_header_id|>") => Some(Self::Llama3),
            t if t.contains("<|im_start|>") => Some(Self::ChatMl),
            t if t.contains("<|user|>") => Some(Self::Zephyr),
            t if t.contains("<<SYS>>") => Some(Self::Llama2),
            t if t.contains("[INST]") => Some(Self::Mistral),
            _ => None,
        }
    }

    /// Renders the conversation, ending with the assistant turn opener. BOS is not included:
    /// the tokenizer adds it when the model requests one.
    pub fn render(self, messages: &[Message]) -> Rendered {
        let mut r = Rendered::default();
        match self {
            Self::ChatMl => {
                for m in messages {
                    r.text += &format!("<|im_start|>{}\n", m.role);
                    r.push_untrusted(&m.content);
                    r.text += "<|im_end|>\n";
                }
                r.text += "<|im_start|>assistant\n";
            }
            Self::Llama3 => {
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
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            ChatTemplate::Mistral.render(&msgs).text,
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
                ChatTemplate::Mistral.render(&msgs).text,
                "[INST] S\n\nU1 [/INST] A1</s>[INST] U2 [/INST]"
            );
        }
    }

    #[test]
    fn mistral_gives_a_lone_system_message_its_own_turn() {
        let msgs = [Message::new("system", "S")];
        assert_eq!(ChatTemplate::Mistral.render(&msgs).text, "[INST] S [/INST]");
    }

    #[test]
    fn llama2_treats_developer_as_system() {
        let msgs = [Message::new("developer", "S"), Message::new("user", "U")];
        assert_eq!(
            ChatTemplate::Llama2.render(&msgs).text,
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
            ChatTemplate::ChatMl.render(&msgs).text,
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
            ChatTemplate::Llama2.render(&msgs).text,
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
            let r = template.render(&msgs);
            let got: Vec<&str> = r.untrusted.iter().map(|x| &r.text[x.clone()]).collect();
            assert_eq!(got, want, "{template:?}");
        }
    }
}
