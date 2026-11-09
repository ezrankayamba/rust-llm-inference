//! Hyperparameters read from `{arch}.*` GGUF metadata.

use rli_gguf::{Error, Gguf, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// Llama 2/3, Mistral, TinyLlama, SmolLM; Mixtral when `n_expert > 0`.
    Llama,
    Qwen2,
    Qwen3,
    Qwen2Moe,
    Qwen3Moe,
}

impl Arch {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "llama" | "mistral" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "qwen2moe" => Arch::Qwen2Moe,
            "qwen3moe" => Arch::Qwen3Moe,
            other => return Err(Error::Format(format!("unsupported architecture {other:?}"))),
        })
    }

    /// Qwen checkpoints keep HF's rotate-half layout; llama GGUFs are permuted to adjacent pairs.
    pub fn neox_rope(self) -> bool {
        !matches!(self, Arch::Llama)
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: Arch,
    pub arch_name: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_ctx_train: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rope_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// Renormalize the selected experts' router weights to sum to 1 (Mixtral, Qwen3-MoE).
    pub norm_topk: bool,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch_name = g.str("general.architecture")?.to_owned();
        let arch = Arch::parse(&arch_name)?;
        let key = |k: &str| format!("{arch_name}.{k}");
        let opt = |k: &str| g.usize(&key(k)).ok();

        let n_embd = g.usize(&key("embedding_length"))?;
        let n_head = g.usize(&key("attention.head_count"))?;
        let head_dim =
            opt("attention.key_length").unwrap_or(n_embd.checked_div(n_head).unwrap_or(0));
        let n_expert = opt("expert_count").unwrap_or(0);
        let n_ff = match arch {
            Arch::Qwen2Moe | Arch::Qwen3Moe => opt("expert_feed_forward_length")
                .or_else(|| opt("feed_forward_length"))
                .ok_or_else(|| Error::Missing(key("expert_feed_forward_length")))?,
            _ => g.usize(&key("feed_forward_length"))?,
        };
        let cfg = Self {
            arch,
            n_embd,
            n_layer: g.usize(&key("block_count"))?,
            n_head,
            n_head_kv: opt("attention.head_count_kv").unwrap_or(n_head),
            head_dim,
            n_ff,
            n_ctx_train: opt("context_length").unwrap_or(2048),
            rms_eps: g
                .f32(&key("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            rope_theta: g.f32(&key("rope.freq_base")).unwrap_or(10000.0),
            rope_dim: opt("rope.dimension_count").unwrap_or(head_dim),
            n_expert,
            n_expert_used: opt("expert_used_count").unwrap_or(0),
            norm_topk: match arch {
                Arch::Qwen2Moe => false,
                Arch::Qwen3Moe => g
                    .get(&key("expert_weights_norm"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                _ => true,
            },
            arch_name,
        };
        cfg.validate(g.tensors.len())?;
        Ok(cfg)
    }

    /// Rejects values that would divide by zero or size buffers from nonsense. Tensor shapes are
    /// checked against these once the tensors are read.
    fn validate(&self, n_tensors: usize) -> Result<()> {
        let bad = |why: String| Err(Error::Format(format!("{}: {why}", self.arch_name)));
        for (name, v) in [
            ("embedding_length", self.n_embd),
            ("block_count", self.n_layer),
            ("attention.head_count", self.n_head),
            ("attention.head_count_kv", self.n_head_kv),
            ("head dimension", self.head_dim),
            ("feed-forward length", self.n_ff),
            ("context_length", self.n_ctx_train),
        ] {
            if v == 0 {
                return bad(format!("{name} is 0"));
            }
        }
        if self.n_layer > n_tensors {
            return bad(format!(
                "{} layers but only {n_tensors} tensors",
                self.n_layer
            ));
        }
        if !self.n_head.is_multiple_of(self.n_head_kv) {
            return bad(format!(
                "{} heads don't divide into {} KV heads",
                self.n_head, self.n_head_kv
            ));
        }
        if self.n_head.checked_mul(self.head_dim).is_none() {
            return bad("attention size overflows".into());
        }
        if !self.rope_dim.is_multiple_of(2) || self.rope_dim > self.head_dim {
            return bad(format!(
                "rope dimension {} doesn't fit head dimension {}",
                self.rope_dim, self.head_dim
            ));
        }
        if self.n_expert_used > self.n_expert {
            return bad(format!(
                "{} experts used of {}",
                self.n_expert_used, self.n_expert
            ));
        }
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
}
