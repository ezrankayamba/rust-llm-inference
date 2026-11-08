//! Decoder-only transformer (Llama family, Qwen2/3, and their MoE variants) over a mapped GGUF.

use crate::config::Config;
use rayon::prelude::*;
use rli_core::half::f32_to_f16;
use rli_core::ops::{self, Rope};
use rli_core::{GgmlType, QMatrix, simd};
use rli_gguf::{Error, Gguf, Result};
use std::any::Any;

/// Prompt tokens per CPU pass: amortizes weight reads while keeping scratch buffers (and the
/// quantized activations each weight block is multiplied against) cache-sized.
const CPU_CHUNK: usize = 64;

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
    /// Like [`Accelerator::forward`] for `xs.len() / n_embd` consecutive tokens (row-major
    /// embeddings, in place) at positions `pos..`; `logits` are for the last token. The default
    /// runs them one at a time.
    fn forward_batch(
        &self,
        model: &Model,
        state: &mut (dyn Any + Send),
        xs: &mut [f32],
        pos: usize,
        logits: Option<&mut [f32]>,
    ) -> bool {
        forward_each(self, model, state, xs, pos, logits)
    }
}

/// [`Accelerator::forward_batch`] as one [`Accelerator::forward`] call per token.
pub(crate) fn forward_each<A: Accelerator + ?Sized>(
    accel: &A,
    model: &Model,
    state: &mut (dyn Any + Send),
    xs: &mut [f32],
    pos: usize,
    mut logits: Option<&mut [f32]>,
) -> bool {
    let n = xs.len() / model.cfg.n_embd;
    let mut done = false;
    for (i, x) in xs.chunks_exact_mut(model.cfg.n_embd).enumerate() {
        let l = if i + 1 == n { logits.take() } else { None };
        done = accel.forward(model, state, x, pos + i, l);
    }
    done
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

/// One layer's cached keys or values for the CPU layers: f16 (half the memory), or f32 when the
/// session was created in precise mode (the f16 rounding moves logits by ~0.01).
#[derive(Clone)]
enum KvCache {
    F32(Vec<f32>),
    /// Little-endian IEEE halves.
    F16(Vec<u8>),
}

impl KvCache {
    fn new(half: bool) -> Self {
        if half {
            KvCache::F16(Vec::new())
        } else {
            KvCache::F32(Vec::new())
        }
    }

    fn push(&mut self, x: &[f32]) {
        match self {
            KvCache::F32(c) => c.extend_from_slice(x),
            KvCache::F16(c) => c.extend(x.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes())),
        }
    }

    /// Keeps the first `n` values.
    fn truncate(&mut self, n: usize) {
        match self {
            KvCache::F32(c) => c.truncate(n),
            KvCache::F16(c) => c.truncate(2 * n),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            KvCache::F32(c) => 4 * c.len(),
            KvCache::F16(c) => c.len(),
        }
    }

    /// `q · cache[at..at + q.len()]`.
    fn dot(&self, at: usize, q: &[f32]) -> f32 {
        match self {
            KvCache::F32(c) => simd::dot(q, &c[at..at + q.len()]),
            KvCache::F16(c) => simd::dot_f16(&c[2 * at..2 * (at + q.len())], q),
        }
    }

    /// `out = Σ_t w[t] · cache[at + t * stride..][..out.len()]`.
    fn weighted_sum(&self, at: usize, stride: usize, w: &[f32], out: &mut [f32]) {
        match self {
            KvCache::F32(c) => {
                out.fill(0.0);
                for (t, &p) in w.iter().enumerate() {
                    let row = &c[at + t * stride..][..out.len()];
                    out.iter_mut().zip(row).for_each(|(o, &v)| *o += p * v);
                }
            }
            KvCache::F16(c) => simd::weighted_rows_f16(out, w, &c[2 * at..], 2 * stride),
        }
    }
}

/// Per-conversation state: KV cache (grown on demand, so memory tracks tokens actually used)
/// and the token history it corresponds to.
pub struct Session {
    k: Vec<KvCache>,
    v: Vec<KvCache>,
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

    /// Bytes held by the CPU layers' KV cache.
    pub fn kv_bytes(&self) -> usize {
        self.k.iter().chain(&self.v).map(KvCache::bytes).sum()
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

    /// The CPU layers' KV cache is f16, or f32 if `precise` is set now (it is fixed for the
    /// session's lifetime).
    pub fn new_session(&self, max_ctx: usize) -> Session {
        let n = self.cfg.n_layer;
        let half = !self.precise;
        Session {
            k: vec![KvCache::new(half); n],
            v: vec![KvCache::new(half); n],
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

    fn matmul(&self, w: &Weight, xs: &[f32], ys: &mut [f32], n: usize) {
        self.mat(w).matmul_with(xs, ys, n, self.precise);
    }

    /// SwiGLU FFN of `n` rows of `x` into `out`.
    fn ffn(&self, e: &Expert, x: &[f32], out: &mut [f32], n: usize) {
        let mut gate = vec![0.0; n * e.gate.rows];
        let mut up = vec![0.0; n * e.up.rows];
        self.matmul(&e.gate, x, &mut gate, n);
        self.matmul(&e.up, x, &mut up, n);
        ops::swiglu(&mut gate, &up);
        self.matmul(&e.down, &gate, out, n);
    }

    /// Runs one token at position `sess.len()` and appends it to the session. Returns logits when
    /// `want_logits` (prompt tokens other than the last don't need the output head).
    pub fn forward(&self, sess: &mut Session, token: u32, want_logits: bool) -> Option<Vec<f32>> {
        self.forward_batch(sess, &[token], want_logits)
    }

    /// Runs `tokens` at positions `sess.len()..` and appends them to the session; the logits
    /// (when wanted) are for the last token. An accelerator may process them as one batch.
    pub fn forward_batch(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        want_logits: bool,
    ) -> Option<Vec<f32>> {
        let n_embd = self.cfg.n_embd;
        let pos = sess.len();
        if tokens.is_empty() {
            return None;
        }
        let embd = self.mat(&self.embd);
        let mut xs = vec![0.0; tokens.len() * n_embd];
        for (x, &t) in xs.chunks_exact_mut(n_embd).zip(tokens) {
            embd.dequantize_row_into(t as usize, x);
        }
        let mut logits = if want_logits {
            vec![0.0; self.output.rows]
        } else {
            Vec::new()
        };
        let mut first_cpu_layer = 0;
        if let Some(a) = &self.accel {
            let state = sess.accel.get_or_insert_with(|| a.new_state(sess.max_ctx));
            let done = a.forward_batch(
                self,
                state.as_mut(),
                &mut xs,
                pos,
                want_logits.then_some(&mut logits[..]),
            );
            first_cpu_layer = a.layers();
            if done {
                sess.tokens.extend_from_slice(tokens);
                return Some(logits);
            }
        }
        for (i, chunk) in xs.chunks_mut(CPU_CHUNK * n_embd).enumerate() {
            self.cpu_layers(sess, chunk, pos + i * CPU_CHUNK, first_cpu_layer);
        }
        sess.tokens.extend_from_slice(tokens);
        if !want_logits {
            return None;
        }
        let x = &xs[(tokens.len() - 1) * n_embd..];
        let mut xb = vec![0.0; n_embd];
        ops::rmsnorm(&mut xb, x, &self.output_norm, self.cfg.rms_eps);
        self.matvec(&self.output, &xb, &mut logits);
        Some(logits)
    }

    /// Runs layers `first..` on the CPU for the token rows in `xs` (positions `pos..`, in
    /// place), appending to their KV cache. Rows are processed together, so each weight matrix
    /// is read once per call; one row is the decode case.
    fn cpu_layers(&self, sess: &mut Session, xs: &mut [f32], pos: usize, first: usize) {
        let c = &self.cfg;
        let (hd, kv_dim, q_dim, n_embd) = (c.head_dim, c.kv_dim(), c.q_dim(), c.n_embd);
        let n = xs.len() / n_embd;
        let group = c.n_head / c.n_head_kv;
        let rope = Rope {
            head_dim: hd,
            rope_dim: c.rope_dim,
            theta: c.rope_theta,
            neox: c.arch.neox_rope(),
            freq_factors: self.rope_freqs.as_deref(),
        };
        let mut xb = vec![0.0; n * n_embd];
        let mut q = vec![0.0; n * q_dim];
        let mut k = vec![0.0; n * kv_dim];
        let mut v = vec![0.0; n * kv_dim];
        let mut att = vec![0.0; n * q_dim];
        let mut tmp = vec![0.0; n * n_embd];
        let scale = 1.0 / (hd as f32).sqrt();
        let norm_rows = |out: &mut [f32], x: &[f32], w: &[f32]| {
            for (o, r) in out.chunks_exact_mut(n_embd).zip(x.chunks_exact(n_embd)) {
                ops::rmsnorm(o, r, w, c.rms_eps);
            }
        };

        for (l, layer) in self.layers.iter().enumerate().skip(first) {
            norm_rows(&mut xb, xs, &layer.attn_norm);
            self.matmul(&layer.wq, &xb, &mut q, n);
            self.matmul(&layer.wk, &xb, &mut k, n);
            self.matmul(&layer.wv, &xb, &mut v, n);
            for (t, b, dim) in [
                (&mut q, &layer.bq, q_dim),
                (&mut k, &layer.bk, kv_dim),
                (&mut v, &layer.bv, kv_dim),
            ] {
                if let Some(b) = b {
                    t.chunks_exact_mut(dim).for_each(|r| ops::add_assign(r, b));
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
            for i in 0..n {
                rope.apply(&mut q[i * q_dim..(i + 1) * q_dim], c.n_head, pos + i);
                rope.apply(&mut k[i * kv_dim..(i + 1) * kv_dim], c.n_head_kv, pos + i);
            }
            sess.k[l].push(&k);
            sess.v[l].push(&v);

            // Causal attention, parallel over (token, head): token i sees positions 0..=pos + i.
            let (kc, vc) = (&sess.k[l], &sess.v[l]);
            att.par_chunks_mut(hd)
                .zip(q.par_chunks(hd))
                .enumerate()
                .for_each_init(Vec::new, |scores, (j, (out, qh))| {
                    let (i, h) = (j / c.n_head, j % c.n_head);
                    let n_pos = pos + i + 1;
                    let off = (h / group) * hd;
                    scores.clear();
                    scores.extend((0..n_pos).map(|t| kc.dot(t * kv_dim + off, qh) * scale));
                    ops::softmax(scores);
                    vc.weighted_sum(off, kv_dim, scores, out);
                });
            self.matmul(&layer.wo, &att, &mut tmp, n);
            ops::add_assign(xs, &tmp);

            norm_rows(&mut xb, xs, &layer.ffn_norm);
            match &layer.ffn {
                Ffn::Dense(e) => self.ffn(e, &xb, &mut tmp, n),
                Ffn::Moe {
                    router,
                    experts,
                    shared,
                } => self.moe(router, experts, shared.as_ref(), &xb, &mut tmp, n),
            }
            ops::add_assign(xs, &tmp);
        }
    }

    /// MoE FFN of `n` rows: routes each row, then runs every used expert once, over the rows
    /// that picked it, so its weights are read once per call as in the dense case.
    fn moe(
        &self,
        router: &Weight,
        experts: &[Expert],
        shared: Option<&(Weight, Expert)>,
        xs: &[f32],
        out: &mut [f32],
        n: usize,
    ) {
        let c = &self.cfg;
        let mut probs = vec![0.0; n * experts.len()];
        self.matmul(router, xs, &mut probs, n);
        let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); experts.len()];
        let mut order: Vec<usize> = (0..experts.len()).collect();
        for (t, p) in probs.chunks_exact_mut(experts.len()).enumerate() {
            ops::softmax(p);
            order.sort_unstable_by(|&a, &b| p[b].total_cmp(&p[a]));
            let chosen = &order[..c.n_expert_used];
            let norm = if c.norm_topk {
                chosen.iter().map(|&e| p[e]).sum::<f32>()
            } else {
                1.0
            };
            for &e in chosen {
                rows[e].push((t, p[e] / norm));
            }
        }

        out.fill(0.0);
        let (mut gathered, mut y) = (Vec::new(), Vec::new());
        for (e, rows) in experts.iter().zip(&rows).filter(|(_, r)| !r.is_empty()) {
            gathered.clear();
            for &(t, _) in rows {
                gathered.extend_from_slice(&xs[t * c.n_embd..(t + 1) * c.n_embd]);
            }
            y.resize(gathered.len(), 0.0);
            self.ffn(e, &gathered, &mut y, rows.len());
            for (&(t, w), r) in rows.iter().zip(y.chunks_exact(c.n_embd)) {
                axpy(&mut out[t * c.n_embd..(t + 1) * c.n_embd], w, r);
            }
        }
        if let Some((gate_inp, e)) = shared {
            let mut g = vec![0.0; n];
            self.matmul(gate_inp, xs, &mut g, n);
            y.resize(out.len(), 0.0);
            self.ffn(e, xs, &mut y, n);
            for ((o, r), &g) in out
                .chunks_exact_mut(c.n_embd)
                .zip(y.chunks_exact(c.n_embd))
                .zip(&g)
            {
                axpy(o, 1.0 / (1.0 + (-g).exp()), r);
            }
        }
    }
}

fn axpy(out: &mut [f32], w: f32, x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += w * v;
    }
}
