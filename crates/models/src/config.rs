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
    /// DeepSeek V2 and V3 (GGUF `deepseek2`): multi-head latent attention (`Config::mla`) and
    /// MoE with shared experts after `n_dense_lead` dense layers.
    Deepseek2,
}

impl Arch {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "llama" | "mistral" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "qwen2moe" => Arch::Qwen2Moe,
            "qwen3moe" => Arch::Qwen3Moe,
            "deepseek2" => Arch::Deepseek2,
            other => return Err(Error::Format(format!("unsupported architecture {other:?}"))),
        })
    }

    /// Qwen checkpoints keep HF's rotate-half layout; llama GGUFs are permuted to adjacent pairs,
    /// and DeepSeek's rotated dims are adjacent pairs in the checkpoint itself.
    pub fn neox_rope(self) -> bool {
        !matches!(self, Arch::Llama | Arch::Deepseek2)
    }
}

/// How router logits become expert weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gating {
    /// A softmax over every expert.
    Softmax,
    /// Each expert's sigmoid (DeepSeek V3), chosen by score plus a per-expert bias.
    Sigmoid,
}

/// Multi-head latent attention (DeepSeek V2/V3). Keys and values come from one latent vector of
/// `kv_lora` values per token, which is what the cache holds, with the `rope` rotated key dims
/// shared by every head: per head, a query's first `nope` dims meet the latent through the
/// head's `k_b`, its last `rope` dims the shared rotated key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mla {
    /// Rank of the low-rank query projection, 0 for a full one (V2-Lite).
    pub q_lora: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_head: usize,
}

/// YaRN context extension. DeepSeek applies it to the rotation frequencies only and moves its
/// magnitude correction into the attention scale (as llama.cpp does).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub factor: f32,
    pub orig_ctx: usize,
    pub log_mul: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Yarn {
    /// The factor each rotation frequency of a `dims`-wide rotation is multiplied by: 1 for the
    /// fast dims, `1 / factor` for the slow ones, a linear ramp between (ggml's
    /// `rope_yarn_ramp`).
    pub fn freq_scales(&self, dims: usize, theta: f32) -> Vec<f32> {
        let corr = |beta: f32| {
            dims as f32 * (self.orig_ctx as f32 / (beta * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * theta.ln())
        };
        let lo = corr(self.beta_fast).floor().max(0.0);
        let hi = corr(self.beta_slow).ceil().min(dims as f32 - 1.0);
        (0..dims / 2)
            .map(|i| {
                let y = (i as f32 - lo) / (hi - lo).max(0.001);
                let mix = 1.0 - y.clamp(0.0, 1.0);
                (1.0 - mix) / self.factor + mix
            })
            .collect()
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
    /// Dense layers before the first MoE layer (DeepSeek), and their FFN width.
    pub n_dense_lead: usize,
    pub n_ff_dense: usize,
    pub gating: Gating,
    /// Multiplies the routed experts' weights (DeepSeek's `routed_scaling_factor`).
    pub expert_scale: f32,
    /// Experts are chosen within the best `n_group_used` of `n_group` equal groups.
    pub n_group: usize,
    pub n_group_used: usize,
    pub mla: Option<Mla>,
    pub yarn: Option<Yarn>,
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
            Arch::Qwen2Moe | Arch::Qwen3Moe | Arch::Deepseek2 => opt("expert_feed_forward_length")
                .or_else(|| opt("feed_forward_length"))
                .ok_or_else(|| Error::Missing(key("expert_feed_forward_length")))?,
            _ => g.usize(&key("feed_forward_length"))?,
        };
        let deepseek = arch == Arch::Deepseek2;
        let mla = if deepseek {
            // Conversions since llama.cpp's MLA change (2025) store the absorbed shapes, one
            // KV head of the latent plus rotated dims, and the per-head sizes under `_mla`.
            let nope_rope = opt("attention.key_length_mla").unwrap_or(head_dim);
            let rope = opt("rope.dimension_count").unwrap_or(0);
            Some(Mla {
                q_lora: opt("attention.q_lora_rank").unwrap_or(0),
                kv_lora: g.usize(&key("attention.kv_lora_rank"))?,
                nope: nope_rope.saturating_sub(rope),
                rope,
                v_head: opt("attention.value_length_mla")
                    .or_else(|| opt("attention.value_length"))
                    .unwrap_or(0),
            })
        } else {
            None
        };
        let head_dim = mla.map_or(head_dim, |m| m.nope + m.rope);
        let yarn = match g.get(&key("rope.scaling.type")).and_then(|v| v.as_str()) {
            Some("yarn") if deepseek => Some(Yarn {
                factor: g.f32(&key("rope.scaling.factor"))?,
                orig_ctx: g.usize(&key("rope.scaling.original_context_length"))?,
                log_mul: g
                    .f32(&key("rope.scaling.yarn_log_multiplier"))
                    .unwrap_or(0.0),
                beta_fast: g.f32(&key("rope.scaling.yarn_beta_fast")).unwrap_or(32.0),
                beta_slow: g.f32(&key("rope.scaling.yarn_beta_slow")).unwrap_or(1.0),
            }),
            _ => None,
        };
        let cfg = Self {
            arch,
            n_embd,
            n_layer: g.usize(&key("block_count"))?,
            n_head,
            // An MLA cache has one row per token; this is only for checks of other archs.
            n_head_kv: if deepseek {
                n_head
            } else {
                opt("attention.head_count_kv").unwrap_or(n_head)
            },
            head_dim,
            n_ff,
            n_ctx_train: opt("context_length").unwrap_or(2048),
            rms_eps: g
                .f32(&key("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            rope_theta: g.f32(&key("rope.freq_base")).unwrap_or(10000.0),
            rope_dim: opt("rope.dimension_count").unwrap_or(head_dim),
            n_dense_lead: opt("leading_dense_block_count").unwrap_or(0),
            n_ff_dense: opt("feed_forward_length").unwrap_or(n_ff),
            // Files from before llama.cpp stored this are V2, which takes a softmax.
            gating: match opt("expert_gating_func") {
                Some(2) => Gating::Sigmoid,
                Some(1) | None => Gating::Softmax,
                Some(other) => {
                    return Err(Error::Format(format!(
                        "{arch_name}: expert_gating_func {other} is not supported"
                    )));
                }
            },
            expert_scale: g.f32(&key("expert_weights_scale")).unwrap_or(1.0),
            n_group: opt("expert_group_count").unwrap_or(1).max(1),
            n_group_used: opt("expert_group_used_count").unwrap_or(1).max(1),
            mla,
            yarn,
            n_expert,
            n_expert_used: opt("expert_used_count").unwrap_or(0),
            norm_topk: match arch {
                Arch::Qwen2Moe => false,
                Arch::Deepseek2 => g
                    .get(&key("expert_weights_norm"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
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
        for (name, v) in [
            ("attention.layer_norm_rms_epsilon", self.rms_eps),
            ("rope.freq_base", self.rope_theta),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return bad(format!("{name} is {v}"));
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
        if let Some(m) = &self.mla {
            for (name, v) in [
                ("attention.kv_lora_rank", m.kv_lora),
                ("query and key dims outside the rotation", m.nope),
                ("rope.dimension_count", m.rope),
                ("attention.value_length", m.v_head),
            ] {
                if v == 0 {
                    return bad(format!("{name} is 0"));
                }
            }
            // A cache row, a head's query and its output, sized as products of these.
            let sizes = [
                m.kv_lora.checked_add(m.rope),
                self.n_head.checked_mul(m.v_head),
            ];
            if sizes.iter().any(Option::is_none) || m.kv_lora > 1 << 16 {
                return bad("latent attention size overflows".into());
            }
        }
        if let Some(y) = &self.yarn
            && !(y.factor.is_finite() && y.factor >= 1.0 && y.orig_ctx > 0 && y.log_mul.is_finite())
        {
            return bad(format!("bad YaRN parameters {y:?}"));
        }
        if self.n_expert_used > self.n_expert {
            return bad(format!(
                "{} experts used of {}",
                self.n_expert_used, self.n_expert
            ));
        }
        if self.n_expert > 0 {
            let groups = self.n_expert.is_multiple_of(self.n_group)
                && self.n_group_used <= self.n_group
                && self.n_expert_used <= self.n_group_used * (self.n_expert / self.n_group);
            if !groups {
                return bad(format!(
                    "{} experts don't make {} groups of which {} hold {} used",
                    self.n_expert, self.n_group, self.n_group_used, self.n_expert_used
                ));
            }
            if !self.expert_scale.is_finite() || self.n_dense_lead > self.n_layer {
                return bad("bad expert weights scale or dense layer count".into());
            }
        }
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Values a token keeps per layer in the KV cache (for MLA one latent row of keys and values).
    pub fn kv_dim(&self) -> usize {
        match &self.mla {
            Some(m) => m.kv_lora + m.rope,
            None => self.n_head_kv * self.head_dim,
        }
    }

    /// Width of the attention output, the input of `attn_output`.
    pub fn att_dim(&self) -> usize {
        self.mla.map_or(self.q_dim(), |m| self.n_head * m.v_head)
    }

    /// Multiplies attention scores: `1 / sqrt(head_dim)`, times YaRN's magnitude correction
    /// squared (once for the queries, once for the keys) where the rotation leaves it out.
    pub fn kq_scale(&self) -> f32 {
        let m = self.yarn.map_or(1.0, |y| 1.0 + y.log_mul * y.factor.ln());
        m * m / (self.head_dim as f32).sqrt()
    }
}
