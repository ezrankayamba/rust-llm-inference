//! Chat prompt formatting. GGUF ships a Jinja template; rather than embed a Jinja engine, the
//! template text is fingerprinted to pick one of the common formats below.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<|im_start|>role\n...<|im_end|>` (Qwen, SmolLM, many fine-tunes).
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>`.
    Llama3,
    /// `[INST] ... [/INST]` with optional `<<SYS>>` (Llama 2, Mistral, Mixtral).
    Llama2,
    /// `<|user|>\n...</s>\n<|assistant|>\n` (TinyLlama chat, Zephyr).
    Zephyr,
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
    pub fn detect(jinja: Option<&str>) -> Self {
        match jinja {
            Some(t) if t.contains("<|start_header_id|>") => Self::Llama3,
            Some(t) if t.contains("<|im_start|>") => Self::ChatMl,
            Some(t) if t.contains("<|user|>") => Self::Zephyr,
            Some(t) if t.contains("[INST]") => Self::Llama2,
            _ => Self::ChatMl,
        }
    }

    /// Renders the conversation, ending with the assistant turn opener. BOS is not included:
    /// the tokenizer adds it when the model requests one.
    pub fn render(self, messages: &[Message]) -> String {
        let mut s = String::new();
        match self {
            Self::ChatMl => {
                for m in messages {
                    s += &format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content);
                }
                s += "<|im_start|>assistant\n";
            }
            Self::Llama3 => {
                for m in messages {
                    s += &format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        m.role,
                        m.content.trim()
                    );
                }
                s += "<|start_header_id|>assistant<|end_header_id|>\n\n";
            }
            Self::Zephyr => {
                for m in messages {
                    s += &format!("<|{}|>\n{}</s>\n", m.role, m.content);
                }
                s += "<|assistant|>\n";
            }
            Self::Llama2 => {
                let mut system: Option<&str> = None;
                let mut first = true;
                for m in messages {
                    match m.role.as_str() {
                        "system" => system = Some(&m.content),
                        "user" => {
                            if !first {
                                s += "<s>";
                            }
                            let content = match system.take() {
                                Some(sys) => {
                                    format!("<<SYS>>\n{sys}\n<</SYS>>\n\n{}", m.content.trim())
                                }
                                None => m.content.trim().to_owned(),
                            };
                            s += &format!("[INST] {content} [/INST]");
                            first = false;
                        }
                        _ => s += &format!(" {}</s>", m.content.trim()),
                    }
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_formats_from_jinja_markers() {
        assert_eq!(
            ChatTemplate::detect(Some("{{'<|im_start|>' + m.role}}")),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::detect(Some("<|start_header_id|>")),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::detect(Some("{{ '<|user|>\n' }}")),
            ChatTemplate::Zephyr
        );
        assert_eq!(ChatTemplate::detect(Some("[INST]")), ChatTemplate::Llama2);
        assert_eq!(ChatTemplate::detect(None), ChatTemplate::ChatMl);
    }

    #[test]
    fn chatml_renders_turns_and_generation_prompt() {
        let msgs = [
            Message::new("system", "Be brief."),
            Message::new("user", "Hi"),
        ];
        assert_eq!(
            ChatTemplate::ChatMl.render(&msgs),
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
            ChatTemplate::Llama2.render(&msgs),
            "[INST] <<SYS>>\nS\n<</SYS>>\n\nU1 [/INST] A1</s><s>[INST] U2 [/INST]"
        );
    }
}
