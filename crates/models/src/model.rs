//! Decoder-only transformer (Llama family, Qwen2/3, and their MoE variants) over a mapped GGUF.

use crate::config::Config;
use rayon::prelude::*;
use rli_core::ops::{self, Rope};
use rli_core::{GgmlType, QMatrix, simd};
use rli_gguf::{Error, Gguf, Result};
use std::any::Any;

/// A weight matrix identified by its location in the mapped file, so CPU kernels and
/// accelerator backends address the same bytes without copies.
#[derive(Clone, Copy, Debug)]
pub struct Weight {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    /// Absolute byte offset in the GGUF file.
    pub offset: usize,
}

impl Weight {
    pub fn n_bytes(&self) -> usize {
        self.ty.row_bytes(self.cols) * self.rows
    }

    pub(crate) fn rows_slice(&self, start: usize, n: usize) -> Weight {
        Weight {
            rows: n,
            offset: self.offset + start * self.ty.row_bytes(self.cols),
            ..*self
        }
    }
}

/// A backend that runs the first [`Accelerator::layers`] decoder layers of a token's forward
/// pass (and the output head when it covers every layer); the CPU runs the rest.
pub trait Accelerator: Send + Sync {
    fn name(&self) -> &str;
    fn layers(&self) -> usize;
    /// Per-session state, e.g. the KV cache of the accelerated layers.
    fn new_state(&self, max_ctx: usize) -> Box<dyn Any + Send>;
    /// Runs the accelerated layers on the embedding `x` (in place) at position `pos`. When it
    /// covers all layers and `logits` is given, it also fills the logits and returns `true`.
    fn forward(
        &self,
        model: &Model,
        state: &mut (dyn Any + Send),
        x: &mut [f32],
        pos: usize,
        logits: Option<&mut [f32]>,
    ) -> bool;
}

pub(crate) struct Expert {
    pub(crate) gate: Weight,
    pub(crate) up: Weight,
    pub(crate) down: Weight,
}

pub(crate) enum Ffn {
    Dense(Expert),
    Moe {
        router: Weight,
        experts: Vec<Expert>,
        shared: Option<(Weight, Expert)>,
    },
}

pub(crate) struct Layer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) wq: Weight,
    pub(crate) wk: Weight,
    pub(crate) wv: Weight,
    pub(crate) wo: Weight,
    pub(crate) bq: Option<Vec<f32>>,
    pub(crate) bk: Option<Vec<f32>>,
    pub(crate) bv: Option<Vec<f32>>,
    pub(crate) q_norm: Option<Vec<f32>>,
    pub(crate) k_norm: Option<Vec<f32>>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) ffn: Ffn,
}

pub struct Model {
    /// Declared before `gguf`: fields drop in order, and a backend may hold zero-copy views of
    /// the mapping that must be released before it is unmapped.
    accel: Option<Box<dyn Accelerator>>,
    gguf: Gguf,
    pub cfg: Config,
    embd: Weight,
    pub(crate) output: Weight,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) rope_freqs: Option<Vec<f32>>,
    pub(crate) layers: Vec<Layer>,
    /// Keep activations in f32 for CPU matvecs (slow; used to validate against references).
    pub precise: bool,
}

/// Per-conversation state: KV cache (grown on demand, so memory tracks tokens actually used)
/// and the token history it corresponds to.
pub struct Session {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub tokens: Vec<u32>,
    pub max_ctx: usize,
    /// The accelerator's per-session state (its layers' KV cache), created on first use.
    accel: Option<Box<dyn Any + Send>>,
}

impl Session {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Drops cached positions `>= n`. (Accelerator caches are indexed by position, so stale
    /// entries are simply overwritten.)
    pub fn truncate(&mut self, n: usize, kv_dim: usize) {
        self.tokens.truncate(n);
        for c in self.k.iter_mut().chain(self.v.iter_mut()) {
            c.truncate(n * kv_dim);
        }
    }

    pub fn kv_bytes(&self) -> usize {
        self.k.iter().chain(&self.v).map(|c| c.len() * 4).sum()
    }
}

fn weight(g: &Gguf, name: &str) -> Result<Weight> {
    let t = g
        .tensor(name)
        .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
    let cols = t.dims[0] as usize;
    Ok(Weight {
        ty: t.ty()?,
        rows: t.n_elements() / cols,
        cols,
        offset: t.offset,
    })
}

/// Splits stacked 3-D expert tensors (`[n_expert][rows][cols]`) into per-expert views.
fn split_experts(gate: Weight, up: Weight, down: Weight, n_expert: usize) -> Vec<Expert> {
    let (n_ff, n_embd) = (gate.rows / n_expert, down.rows / n_expert);
    (0..n_expert)
        .map(|e| Expert {
            gate: gate.rows_slice(e * n_ff, n_ff),
            up: up.rows_slice(e * n_ff, n_ff),
            down: down.rows_slice(e * n_embd, n_embd),
        })
        .collect()
}

impl Model {
    pub fn load(gguf: Gguf) -> Result<Self> {
        let cfg = Config::from_gguf(&gguf)?;
        let g = &gguf;
        let opt_vec = |name: &str| g.tensor(name).map(|_| g.f32_vec(name)).transpose();
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let w = |s: &str| weight(g, &p(s));
            let ffn = if cfg.n_expert > 0 {
                let experts = if g.tensor(&p("ffn_gate_exps.weight")).is_some() {
                    split_experts(
                        w("ffn_gate_exps.weight")?,
                        w("ffn_up_exps.weight")?,
                        w("ffn_down_exps.weight")?,
                        cfg.n_expert,
                    )
                } else {
                    // Older Mixtral conversions store one tensor per expert.
                    (0..cfg.n_expert)
                        .map(|e| {
                            Ok(Expert {
                                gate: w(&format!("ffn_gate.{e}.weight"))?,
                                up: w(&format!("ffn_up.{e}.weight"))?,
                                down: w(&format!("ffn_down.{e}.weight"))?,
                            })
                        })
                        .collect::<Result<_>>()?
                };
                let shared = match g.tensor(&p("ffn_gate_inp_shexp.weight")) {
                    Some(_) => Some((
                        w("ffn_gate_inp_shexp.weight")?,
                        Expert {
                            gate: w("ffn_gate_shexp.weight")?,
                            up: w("ffn_up_shexp.weight")?,
                            down: w("ffn_down_shexp.weight")?,
                        },
                    )),
                    None => None,
                };
                Ffn::Moe {
                    router: w("ffn_gate_inp.weight")?,
                    experts,
                    shared,
                }
            } else {
                Ffn::Dense(Expert {
                    gate: w("ffn_gate.weight")?,
                    up: w("ffn_up.weight")?,
                    down: w("ffn_down.weight")?,
                })
            };
            layers.push(Layer {
                attn_norm: g.f32_vec(&p("attn_norm.weight"))?,
                wq: w("attn_q.weight")?,
                wk: w("attn_k.weight")?,
                wv: w("attn_v.weight")?,
                wo: w("attn_output.weight")?,
                bq: opt_vec(&p("attn_q.bias"))?,
                bk: opt_vec(&p("attn_k.bias"))?,
                bv: opt_vec(&p("attn_v.bias"))?,
                q_norm: opt_vec(&p("attn_q_norm.weight"))?,
                k_norm: opt_vec(&p("attn_k_norm.weight"))?,
                ffn_norm: g.f32_vec(&p("ffn_norm.weight"))?,
                ffn,
            });
        }
        let embd = weight(g, "token_embd.weight")?;
        let output = if g.tensor("output.weight").is_some() {
            weight(g, "output.weight")?
        } else {
            embd
        };
        let output_norm = g.f32_vec("output_norm.weight")?;
        let rope_freqs = opt_vec("rope_freqs.weight")?;
        if cfg.n_expert > 0 && cfg.n_expert_used == 0 {
            return Err(Error::Missing(format!(
                "{}.expert_used_count",
                cfg.arch_name
            )));
        }
        Ok(Self {
            gguf,
            cfg,
            embd,
            output,
            output_norm,
            rope_freqs,
            layers,
            accel: None,
            precise: false,
        })
    }

    pub fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    /// Runs the first `accel.layers()` layers on `accel`. Sessions created earlier must not be
    /// reused afterwards (their KV cache for those layers lives on the CPU).
    pub fn set_accelerator(&mut self, accel: Box<dyn Accelerator>) {
        self.accel = Some(accel);
    }

    pub fn accelerator(&self) -> Option<(&str, usize)> {
        self.accel.as_ref().map(|a| (a.name(), a.layers()))
    }

    pub fn vocab_size(&self) -> usize {
        self.output.rows
    }

    pub fn new_session(&self, max_ctx: usize) -> Session {
        let n = self.cfg.n_layer;
        Session {
            k: vec![Vec::new(); n],
            v: vec![Vec::new(); n],
            tokens: Vec::new(),
            max_ctx,
            accel: None,
        }
    }

    pub(crate) fn mat(&self, w: &Weight) -> QMatrix<'_> {
        QMatrix::new(
            w.ty,
            w.rows,
            w.cols,
            &self.gguf.mapped_bytes()[w.offset..w.offset + w.n_bytes()],
        )
    }

    fn matvec(&self, w: &Weight, x: &[f32], y: &mut [f32]) {
        self.mat(w).matvec_with(x, y, self.precise);
    }

    fn ffn(&self, e: &Expert, x: &[f32], out: &mut [f32]) {
        let mut gate = vec![0.0; e.gate.rows];
        let mut up = vec![0.0; e.up.rows];
        self.matvec(&e.gate, x, &mut gate);
        self.matvec(&e.up, x, &mut up);
        ops::swiglu(&mut gate, &up);
        self.matvec(&e.down, &gate, out);
    }

    /// Runs one token at position `sess.len()` and appends it to the session. Returns logits when
    /// `want_logits` (prompt tokens other than the last don't need the output head).
    pub fn forward(&self, sess: &mut Session, token: u32, want_logits: bool) -> Option<Vec<f32>> {
        let c = &self.cfg;
        let pos = sess.len();
        let (hd, kv_dim) = (c.head_dim, c.kv_dim());
        let group = c.n_head / c.n_head_kv;
        let rope = Rope {
            head_dim: hd,
            rope_dim: c.rope_dim,
            theta: c.rope_theta,
            neox: c.arch.neox_rope(),
            freq_factors: self.rope_freqs.as_deref(),
        };

        let mut x = vec![0.0; c.n_embd];
        self.mat(&self.embd)
            .dequantize_row_into(token as usize, &mut x);
        let mut logits = if want_logits {
            vec![0.0; self.output.rows]
        } else {
            Vec::new()
        };
        let mut first_cpu_layer = 0;
        if let Some(a) = &self.accel {
            let state = sess.accel.get_or_insert_with(|| a.new_state(sess.max_ctx));
            let done = a.forward(
                self,
                state.as_mut(),
                &mut x,
                pos,
                want_logits.then_some(&mut logits[..]),
            );
            first_cpu_layer = a.layers();
            if done {
                sess.tokens.push(token);
                return Some(logits);
            }
        }
        let mut xb = vec![0.0; c.n_embd];
        let mut q = vec![0.0; c.q_dim()];
        let mut k = vec![0.0; kv_dim];
        let mut v = vec![0.0; kv_dim];
        let mut att = vec![0.0; c.q_dim()];
        let mut tmp = vec![0.0; c.n_embd];
        let scale = 1.0 / (hd as f32).sqrt();

        for (l, layer) in self.layers.iter().enumerate().skip(first_cpu_layer) {
            ops::rmsnorm(&mut xb, &x, &layer.attn_norm, c.rms_eps);
            self.matvec(&layer.wq, &xb, &mut q);
            self.matvec(&layer.wk, &xb, &mut k);
            self.matvec(&layer.wv, &xb, &mut v);
            for (t, b) in [
                (&mut q, &layer.bq),
                (&mut k, &layer.bk),
                (&mut v, &layer.bv),
            ] {
                if let Some(b) = b {
                    ops::add_assign(t, b);
                }
            }
            for (t, norm) in [(&mut q, &layer.q_norm), (&mut k, &layer.k_norm)] {
                if let Some(w) = norm {
                    for h in t.chunks_exact_mut(hd) {
                        let src = h.to_vec();
                        ops::rmsnorm(h, &src, w, c.rms_eps);
                    }
                }
            }
            rope.apply(&mut q, c.n_head, pos);
            rope.apply(&mut k, c.n_head_kv, pos);
            sess.k[l].extend_from_slice(&k);
            sess.v[l].extend_from_slice(&v);

            let (kc, vc) = (&sess.k[l], &sess.v[l]);
            let n_pos = pos + 1;
            att.par_chunks_mut(hd)
                .zip(q.par_chunks(hd))
                .enumerate()
                .for_each(|(h, (out, qh))| {
                    let off = (h / group) * hd;
                    let mut scores: Vec<f32> = (0..n_pos)
                        .map(|t| {
                            simd::dot(qh, &kc[t * kv_dim + off..t * kv_dim + off + hd]) * scale
                        })
                        .collect();
                    ops::softmax(&mut scores);
                    out.fill(0.0);
                    for (t, &p) in scores.iter().enumerate() {
                        for (o, &vv) in out
                            .iter_mut()
                            .zip(&vc[t * kv_dim + off..t * kv_dim + off + hd])
                        {
                            *o += p * vv;
                        }
                    }
                });
            self.matvec(&layer.wo, &att, &mut tmp);
            ops::add_assign(&mut x, &tmp);

            ops::rmsnorm(&mut xb, &x, &layer.ffn_norm, c.rms_eps);
            match &layer.ffn {
                Ffn::Dense(e) => self.ffn(e, &xb, &mut tmp),
                Ffn::Moe {
                    router,
                    experts,
                    shared,
                } => self.moe(router, experts, shared.as_ref(), &xb, &mut tmp),
            }
            ops::add_assign(&mut x, &tmp);
        }
        sess.tokens.push(token);

        if !want_logits {
            return None;
        }
        ops::rmsnorm(&mut xb, &x, &self.output_norm, c.rms_eps);
        self.matvec(&self.output, &xb, &mut logits);
        Some(logits)
    }

    fn moe(
        &self,
        router: &Weight,
        experts: &[Expert],
        shared: Option<&(Weight, Expert)>,
        x: &[f32],
        out: &mut [f32],
    ) {
        let c = &self.cfg;
        let mut probs = vec![0.0; experts.len()];
        self.matvec(router, x, &mut probs);
        ops::softmax(&mut probs);
        let mut order: Vec<usize> = (0..experts.len()).collect();
        order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let chosen = &order[..c.n_expert_used];
        let norm = if c.norm_topk {
            chosen.iter().map(|&e| probs[e]).sum::<f32>()
        } else {
            1.0
        };

        out.fill(0.0);
        let mut y = vec![0.0; out.len()];
        for &e in chosen {
            self.ffn(&experts[e], x, &mut y);
            let w = probs[e] / norm;
            for (o, &v) in out.iter_mut().zip(&y) {
                *o += w * v;
            }
        }
        if let Some((gate_inp, e)) = shared {
            let mut g = [0.0];
            self.matvec(gate_inp, x, &mut g);
            let gate = 1.0 / (1.0 + (-g[0]).exp());
            self.ffn(e, x, &mut y);
            for (o, &v) in out.iter_mut().zip(&y) {
                *o += gate * v;
            }
        }
    }
}
