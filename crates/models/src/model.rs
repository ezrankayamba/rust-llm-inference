//! Decoder-only transformer (Llama family, Qwen2/3, and their MoE variants) over a mapped GGUF.

use crate::config::Config;
use rayon::prelude::*;
use rli_core::half::f32_to_f16;
use rli_core::ops::{self, Rope};
use rli_core::{GgmlType, QMatrix, qdot, simd};
use rli_gguf::{Error, Gguf, Result};
use std::any::Any;
use std::collections::HashMap;

/// Prompt tokens per CPU pass. The x86 block kernels decode each group of weight rows once per
/// pass, which cost about as much as multiplying them by 64 tokens: 128 made prompts 21-24% faster
/// than 64 there (no change on NEON), and 256 gained 2-6% on one model and lost as much on another.
const CPU_CHUNK: usize = 128;
/// Values of a SwiGLU activation per parallel task.
const SWIGLU_CHUNK: usize = 1024;
/// Tokens per [`Model::forward_batch`] call when a caller wants to stop between calls
/// ([`Model::prompt_slice`]), unless an accelerator that runs every layer takes more. A multiple
/// of the CPU's chunk size and a divisor of the GPU backends', so slicing a prompt moves no
/// chunk boundaries.
pub const PROMPT_SLICE: usize = 256;
const _: () = assert!(PROMPT_SLICE.is_multiple_of(CPU_CHUNK));

/// A weight matrix identified by its location in the mapped file, so CPU kernels and
/// accelerator backends address the same bytes without copies.
#[derive(Clone, Copy, Debug)]
pub struct Weight {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    /// Absolute byte offset in the GGUF file.
    pub offset: usize,
    /// Multiplies `W x`: an NVFP4 tensor's second-level scale (its `.scale` tensor, one per
    /// expert for stacked experts), 1 for every other weight.
    pub scale: f32,
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
    /// Most prompt tokens one [`Accelerator::forward_batch`] pass takes, when that is more
    /// than [`PROMPT_SLICE`] (a multiple of it).
    fn prompt_slice(&self) -> usize {
        PROMPT_SLICE
    }
    /// Per-session state, e.g. the KV cache of the accelerated layers.
    fn new_state(&self, max_ctx: usize) -> Box<dyn Any + Send>;
    /// Most tokens of one [`Accelerator::forward_batch`] pass it fills logits for (when it
    /// covers every layer).
    fn logit_rows(&self) -> usize {
        1
    }
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
    /// embeddings, in place) at positions `pos..`; `logits` are for the last
    /// `logits.len() / n_vocab` tokens (at most [`Accelerator::logit_rows`]). The default runs
    /// them one at a time.
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

/// The byte strides between consecutive experts' gate, up and down matrices, if every expert
/// sits at the same stride (slices of stacked tensors) with the same shape and type.
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) fn expert_strides(experts: &[Expert]) -> Option<[usize; 3]> {
    fn pick(e: &Expert, i: usize) -> &Weight {
        [&e.gate, &e.up, &e.down][i]
    }
    let mut strides = [0; 3];
    for (i, stride) in strides.iter_mut().enumerate() {
        let first = pick(&experts[0], i);
        *stride = match experts.get(1) {
            Some(e) => pick(e, i).offset.checked_sub(first.offset)?,
            None => 0,
        };
        let even = experts.iter().enumerate().all(|(j, e)| {
            let m = pick(e, i);
            m.offset == first.offset + j * *stride
                && (m.ty, m.rows, m.cols) == (first.ty, first.rows, first.cols)
        });
        if !even {
            return None;
        }
    }
    Some(strides)
}

/// [`Accelerator::forward_batch`] as one [`Accelerator::forward`] call per token.
pub(crate) fn forward_each<A: Accelerator + ?Sized>(
    accel: &A,
    model: &Model,
    state: &mut (dyn Any + Send),
    xs: &mut [f32],
    pos: usize,
    logits: Option<&mut [f32]>,
) -> bool {
    let n = xs.len() / model.cfg.n_embd;
    let vocab = model.output.rows;
    let mut rows = logits.map(|l| l.chunks_exact_mut(vocab));
    let first = n - rows.as_ref().map_or(0, |r| r.len());
    let mut done = false;
    for (i, x) in xs.chunks_exact_mut(model.cfg.n_embd).enumerate() {
        let l = rows
            .as_mut()
            .filter(|_| i >= first)
            .and_then(Iterator::next);
        done = accel.forward(model, state, x, pos + i, l);
    }
    done
}

/// How many of the rows `start..end` of an `n`-row batch are among its last `rows`.
pub(crate) fn rows_in(start: usize, end: usize, n: usize, rows: usize) -> usize {
    end.saturating_sub(start.max(n - rows))
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
    /// Quantize the CPU layers' prompt activations with one scale per 256 values, as llama.cpp
    /// does, instead of one per 32 (`--coarse-acts`): faster for K-quants on ARM (13% per
    /// matmul on an M4), less accurate. Decode is not affected, nor is `precise`.
    pub coarse_acts: bool,
    /// Interleaved copies of weights ([`Model::repack`]), by file offset and row count.
    packed: HashMap<(usize, usize), Vec<u8>>,
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

    /// `out[t] = q · cache[at + t * stride..][..q.len()]`.
    fn dots(&self, at: usize, stride: usize, q: &[f32], out: &mut [f32]) {
        match self {
            KvCache::F32(c) => {
                for (t, o) in out.iter_mut().enumerate() {
                    *o = simd::dot(q, &c[at + t * stride..][..q.len()]);
                }
            }
            KvCache::F16(c) => simd::dot_f16_rows(q, &c[2 * at..], 2 * stride, out),
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
    /// The draft model's session, when generation speculates with one (`Engine::new_session`).
    pub draft: Option<Box<Session>>,
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
    let mut w = Weight {
        ty: t.ty()?,
        rows: t.n_elements() / cols,
        cols,
        offset: t.offset,
        scale: 1.0,
    };
    if let Some(s) = tensor_scales(g, name, &w, 1)? {
        w.scale = s[0];
    }
    Ok(w)
}

/// The `n` values of the `.scale` tensor that goes with weight `name` (llama.cpp's name for an
/// NVFP4 tensor's per-tensor scale), if there is one. The GPU kernels apply it to NVFP4 weights
/// only, so on another type it is an error unless it is 1.
fn tensor_scales(g: &Gguf, name: &str, w: &Weight, n: usize) -> Result<Option<Vec<f32>>> {
    let Some(base) = name.strip_suffix(".weight") else {
        return Ok(None);
    };
    let s = opt_vector(g, &format!("{base}.scale"), n)?;
    if let Some(s) = &s {
        if let Some(v) = s.iter().find(|v| !v.is_finite()) {
            return Err(Error::Format(format!("{base}.scale: {v} is not finite")));
        }
        if w.ty != GgmlType::NVFP4 && s.iter().any(|&v| v != 1.0) {
            return Err(Error::Format(format!(
                "{base}.scale: a tensor scale on a {:?} weight",
                w.ty
            )));
        }
    }
    Ok(s)
}

/// An f32 copy of a vector tensor that must have `len` elements. Checked first, so a bad file
/// can't make this decode a huge tensor.
fn vector(g: &Gguf, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = g
        .tensor(name)
        .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
    if t.n_elements() != len {
        return Err(Error::Format(format!(
            "tensor {name}: {} elements, expected {len}",
            t.n_elements()
        )));
    }
    g.f32_vec(name)
}

fn opt_vector(g: &Gguf, name: &str, len: usize) -> Result<Option<Vec<f32>>> {
    g.tensor(name).map(|_| vector(g, name, len)).transpose()
}

/// The kernels size their inputs and outputs from the config and the weights from these shapes,
/// and GPU kernels have no bounds checks, so the two must agree.
fn check_shape(name: &str, w: &Weight, rows: usize, cols: usize) -> Result<()> {
    if (w.rows, w.cols) != (rows, cols) {
        return Err(Error::Format(format!(
            "tensor {name}: {} x {}, expected {rows} x {cols}",
            w.rows, w.cols
        )));
    }
    Ok(())
}

fn check_expert(name: &str, e: &Expert, n_ff: usize, n_embd: usize) -> Result<()> {
    check_shape(&format!("{name} gate"), &e.gate, n_ff, n_embd)?;
    check_shape(&format!("{name} up"), &e.up, n_ff, n_embd)?;
    check_shape(&format!("{name} down"), &e.down, n_embd, n_ff)
}

/// A stacked expert tensor, with its per-expert `.scale` values if it has them. Nothing is
/// allocated from `n_expert` without them: it is checked against the shapes only later.
fn stacked(g: &Gguf, name: &str, n_expert: usize) -> Result<(Weight, Option<Vec<f32>>)> {
    let t = g
        .tensor(name)
        .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
    let cols = t.dims[0] as usize;
    let w = Weight {
        ty: t.ty()?,
        rows: t.n_elements() / cols,
        cols,
        offset: t.offset,
        scale: 1.0,
    };
    Ok((w, tensor_scales(g, name, &w, n_expert)?))
}

/// Splits stacked 3-D expert tensors (`[n_expert][rows][cols]`) into per-expert views.
fn split_experts(
    (gate, gs): (Weight, Option<Vec<f32>>),
    (up, us): (Weight, Option<Vec<f32>>),
    (down, ds): (Weight, Option<Vec<f32>>),
    n_expert: usize,
) -> Result<Vec<Expert>> {
    let (n_ff, n_embd) = (gate.rows / n_expert, down.rows / n_expert);
    if n_ff == 0
        || n_embd == 0
        || up.rows != gate.rows
        || gate.rows != n_ff * n_expert
        || down.rows != n_embd * n_expert
    {
        return Err(Error::Format(format!(
            "expert tensors of {} and {} rows don't split into {n_expert} experts",
            gate.rows, down.rows
        )));
    }
    let at = |s: &Option<Vec<f32>>, e: usize| s.as_ref().map_or(1.0, |s| s[e]);
    Ok((0..n_expert)
        .map(|e| Expert {
            gate: Weight {
                scale: at(&gs, e),
                ..gate.rows_slice(e * n_ff, n_ff)
            },
            up: Weight {
                scale: at(&us, e),
                ..up.rows_slice(e * n_ff, n_ff)
            },
            down: Weight {
                scale: at(&ds, e),
                ..down.rows_slice(e * n_embd, n_embd)
            },
        })
        .collect())
}

impl Model {
    pub fn load(gguf: Gguf) -> Result<Self> {
        let cfg = Config::from_gguf(&gguf)?;
        let g = &gguf;
        let (n_embd, q_dim, kv_dim) = (cfg.n_embd, cfg.q_dim(), cfg.kv_dim());
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let w = |s: &str| weight(g, &p(s));
            let ffn = if cfg.n_expert > 0 {
                let experts = if g.tensor(&p("ffn_gate_exps.weight")).is_some() {
                    let s = |n: &str| stacked(g, &p(n), cfg.n_expert);
                    split_experts(
                        s("ffn_gate_exps.weight")?,
                        s("ffn_up_exps.weight")?,
                        s("ffn_down_exps.weight")?,
                        cfg.n_expert,
                    )?
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
                let experts: Vec<Expert> = experts;
                for e in &experts {
                    check_expert(&p("expert"), e, cfg.n_ff, n_embd)?;
                }
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
                if let Some((gate_inp, e)) = &shared {
                    check_shape(&p("shared gate input"), gate_inp, 1, n_embd)?;
                    check_expert(&p("shared expert"), e, e.gate.rows, n_embd)?;
                }
                let router = w("ffn_gate_inp.weight")?;
                check_shape(&p("router"), &router, cfg.n_expert, n_embd)?;
                Ffn::Moe {
                    router,
                    experts,
                    shared,
                }
            } else {
                let e = Expert {
                    gate: w("ffn_gate.weight")?,
                    up: w("ffn_up.weight")?,
                    down: w("ffn_down.weight")?,
                };
                check_expert(&p("ffn"), &e, cfg.n_ff, n_embd)?;
                Ffn::Dense(e)
            };
            let (wq, wk, wv, wo) = (
                w("attn_q.weight")?,
                w("attn_k.weight")?,
                w("attn_v.weight")?,
                w("attn_output.weight")?,
            );
            check_shape(&p("attn_q"), &wq, q_dim, n_embd)?;
            check_shape(&p("attn_k"), &wk, kv_dim, n_embd)?;
            check_shape(&p("attn_v"), &wv, kv_dim, n_embd)?;
            check_shape(&p("attn_output"), &wo, n_embd, q_dim)?;
            layers.push(Layer {
                attn_norm: vector(g, &p("attn_norm.weight"), n_embd)?,
                wq,
                wk,
                wv,
                wo,
                bq: opt_vector(g, &p("attn_q.bias"), q_dim)?,
                bk: opt_vector(g, &p("attn_k.bias"), kv_dim)?,
                bv: opt_vector(g, &p("attn_v.bias"), kv_dim)?,
                q_norm: opt_vector(g, &p("attn_q_norm.weight"), cfg.head_dim)?,
                k_norm: opt_vector(g, &p("attn_k_norm.weight"), cfg.head_dim)?,
                ffn_norm: vector(g, &p("ffn_norm.weight"), n_embd)?,
                ffn,
            });
        }
        let embd = weight(g, "token_embd.weight")?;
        let output = if g.tensor("output.weight").is_some() {
            weight(g, "output.weight")?
        } else {
            embd
        };
        check_shape("token_embd", &embd, embd.rows, n_embd)?;
        check_shape("output", &output, embd.rows, n_embd)?;
        let output_norm = vector(g, "output_norm.weight", n_embd)?;
        let rope_freqs = opt_vector(g, "rope_freqs.weight", cfg.rope_dim / 2)?;
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
            coarse_acts: false,
            packed: HashMap::new(),
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
            draft: None,
        }
    }

    /// Rows of the embedding matrix, which must match the tokenizer's vocabulary.
    pub fn n_vocab(&self) -> usize {
        self.embd.rows
    }

    pub(crate) fn mat(&self, w: &Weight) -> QMatrix<'_> {
        QMatrix::new(
            w.ty,
            w.rows,
            w.cols,
            &self.gguf.mapped_bytes()[w.offset..w.offset + w.n_bytes()],
        )
    }

    /// Copies the weights of the layers that run on the CPU into the layout the CPU's decode
    /// kernel reads fastest, where it has one (Q4_K on x86 with AVX2), and returns the bytes
    /// taken: slightly more than those weights' own size, in memory that can't be paged back
    /// to the file. One token at a time then reads the copies; prompts and `precise` still read
    /// the mapping. Call it after attaching an accelerator.
    pub fn repack(&mut self) -> usize {
        let mut ws = vec![self.output];
        for layer in &self.layers[self.accelerator().map_or(0, |(_, n)| n)..] {
            ws.extend([layer.wq, layer.wk, layer.wv, layer.wo]);
            let expert = |e: &Expert| [e.gate, e.up, e.down];
            match &layer.ffn {
                Ffn::Dense(e) => ws.extend(expert(e)),
                Ffn::Moe {
                    router,
                    experts,
                    shared,
                } => {
                    ws.push(*router);
                    ws.extend(experts.iter().flat_map(expert));
                    if let Some((gate, e)) = shared {
                        ws.push(*gate);
                        ws.extend(expert(e));
                    }
                }
            }
        }
        for w in ws {
            if qdot::packs(w.ty) && w.rows >= qdot::PACK_ROWS {
                let m = self.mat(&w);
                let packed = crate::pool().install(|| qdot::pack_rows(m.data, m.row_bytes()));
                self.packed.insert((w.offset, w.rows), packed);
            }
        }
        self.packed.values().map(Vec::len).sum()
    }

    fn matvec(&self, w: &Weight, x: &[f32], y: &mut [f32]) {
        match self.packed.get(&(w.offset, w.rows)) {
            Some(p) if !self.precise => self.mat(w).matvec_packed(p, x, y),
            _ => self.mat(w).matvec_with(x, y, self.precise),
        }
        apply_scale(w, y);
    }

    fn matmul(&self, w: &Weight, xs: &[f32], ys: &mut [f32], n: usize) {
        if n == 1 {
            return self.matvec(w, xs, ys);
        }
        if self.coarse_acts && !self.precise {
            self.mat(w).matmul_coarse(xs, ys, n);
        } else {
            self.mat(w).matmul_with(xs, ys, n, self.precise);
        }
        apply_scale(w, ys);
    }

    /// SwiGLU FFN of `n` rows of `x` into `out`.
    fn ffn(&self, e: &Expert, x: &[f32], out: &mut [f32], n: usize) {
        let mut gate = vec![0.0; n * e.gate.rows];
        let mut up = vec![0.0; n * e.up.rows];
        self.matmul(&e.gate, x, &mut gate, n);
        self.matmul(&e.up, x, &mut up, n);
        // An `exp` per value: left on one thread it was 4% of a prompt's single-threaded time,
        // which is 15% of it on four.
        gate.par_chunks_mut(SWIGLU_CHUNK)
            .zip(up.par_chunks(SWIGLU_CHUNK))
            .for_each(|(g, u)| ops::swiglu(g, u));
        self.matmul(&e.down, &gate, out, n);
    }

    /// Runs one token at position `sess.len()` and appends it to the session. Returns logits when
    /// `want_logits` (prompt tokens other than the last don't need the output head).
    pub fn forward(&self, sess: &mut Session, token: u32, want_logits: bool) -> Option<Vec<f32>> {
        self.forward_batch(sess, &[token], want_logits)
    }

    /// Tokens per [`Model::forward_batch`] call for a caller that wants to stop between calls:
    /// the largest pass of an accelerator that runs every layer, else [`PROMPT_SLICE`].
    pub fn prompt_slice(&self) -> usize {
        match &self.accel {
            Some(a) if a.layers() == self.layers.len() => a.prompt_slice(),
            _ => PROMPT_SLICE,
        }
    }

    /// Runs `tokens` at positions `sess.len()..` and appends them to the session; the logits
    /// (when wanted) are for the last token. An accelerator may process them as one batch.
    pub fn forward_batch(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        want_logits: bool,
    ) -> Option<Vec<f32>> {
        self.forward_rows(sess, tokens, usize::from(want_logits))
    }

    /// Most rows [`Model::forward_rows`] gives logits for in one pass.
    pub fn logit_rows(&self) -> usize {
        match &self.accel {
            Some(a) if a.layers() == self.layers.len() => a.logit_rows(),
            _ => usize::MAX,
        }
    }

    /// Like [`Model::forward_batch`], with the logits of the last `rows` tokens (row-major,
    /// `rows * n_vocab`; none when `rows` is 0), `rows` at most [`Model::logit_rows`].
    pub fn forward_rows(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        rows: usize,
    ) -> Option<Vec<f32>> {
        assert!(rows <= tokens.len() && rows <= self.logit_rows());
        // A pass that runs wholly on the accelerator has nothing for the pool to do, and
        // handing it over costs a thread wake-up per token.
        if self
            .accel
            .as_ref()
            .is_some_and(|a| a.layers() == self.layers.len())
        {
            return self.forward_batch_in_pool(sess, tokens, rows);
        }
        crate::pool().install(|| self.forward_batch_in_pool(sess, tokens, rows))
    }

    fn forward_batch_in_pool(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        rows: usize,
    ) -> Option<Vec<f32>> {
        let want_logits = rows > 0;
        let n_embd = self.cfg.n_embd;
        let pos = sess.len();
        if tokens.is_empty() {
            return None;
        }
        let embd = self.mat(&self.embd);
        let mut xs = vec![0.0; tokens.len() * n_embd];
        for (x, &t) in xs.chunks_exact_mut(n_embd).zip(tokens) {
            embd.dequantize_row_into(t as usize, x);
            apply_scale(&self.embd, x);
        }
        let mut logits = vec![0.0; rows * self.output.rows];
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
        let n = tokens.len();
        for (i, chunk) in xs.chunks_mut(CPU_CHUNK * n_embd).enumerate() {
            let start = i * CPU_CHUNK;
            let tail = rows_in(start, start + chunk.len() / n_embd, n, rows);
            self.cpu_layers(sess, chunk, pos + start, first_cpu_layer, tail);
        }
        sess.tokens.extend_from_slice(tokens);
        if !want_logits {
            return None;
        }
        let x = &xs[(n - rows) * n_embd..];
        let mut xb = vec![0.0; rows * n_embd];
        for (o, r) in xb.chunks_exact_mut(n_embd).zip(x.chunks_exact(n_embd)) {
            ops::rmsnorm(o, r, &self.output_norm, self.cfg.rms_eps);
        }
        self.matmul(&self.output, &xb, &mut logits, rows);
        Some(logits)
    }

    /// Runs layers `first..` on the CPU for the token rows in `xs` (positions `pos..`, in
    /// place), appending to their KV cache. Rows are processed together, so each weight matrix
    /// is read once per call; one row is the decode case. Only the last `tail` rows are needed
    /// after the last layer, which then computes just its keys and values for the
    /// others: a prompt saves most of a layer.
    fn cpu_layers(
        &self,
        sess: &mut Session,
        xs: &mut [f32],
        pos: usize,
        first: usize,
        tail: usize,
    ) {
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
        // The per-token steps between matmuls run on the pool too: left on one thread they
        // were 7% of a four-thread prompt, with the other three waiting.
        let norm_rows = |out: &mut [f32], x: &[f32], w: &[f32]| {
            out.par_chunks_exact_mut(n_embd)
                .zip(x.par_chunks_exact(n_embd))
                .for_each(|(o, r)| ops::rmsnorm(o, r, w, c.rms_eps));
        };

        for (l, layer) in self.layers.iter().enumerate().skip(first) {
            // The rows whose queries, attention and FFN this layer computes: `lo..`.
            let m = if l + 1 == self.layers.len() { tail } else { n };
            let lo = n - m;
            let (q, att, tmp) = (
                &mut q[..m * q_dim],
                &mut att[..m * q_dim],
                &mut tmp[..m * n_embd],
            );
            norm_rows(&mut xb, xs, &layer.attn_norm);
            if m > 0 {
                self.matmul(&layer.wq, &xb[lo * n_embd..], q, m);
            }
            self.matmul(&layer.wk, &xb, &mut k, n);
            self.matmul(&layer.wv, &xb, &mut v, n);
            for (t, b, dim) in [
                (&mut *q, &layer.bq, q_dim),
                (&mut k[..], &layer.bk, kv_dim),
                (&mut v[..], &layer.bv, kv_dim),
            ] {
                if let Some(b) = b {
                    t.chunks_exact_mut(dim).for_each(|r| ops::add_assign(r, b));
                }
            }
            for (t, norm) in [(&mut *q, &layer.q_norm), (&mut k[..], &layer.k_norm)] {
                if let Some(w) = norm {
                    for h in t.chunks_exact_mut(hd) {
                        let src = h.to_vec();
                        ops::rmsnorm(h, &src, w, c.rms_eps);
                    }
                }
            }
            q.par_chunks_exact_mut(q_dim)
                .enumerate()
                .for_each(|(i, q)| rope.apply(q, c.n_head, pos + lo + i));
            k.par_chunks_exact_mut(kv_dim)
                .enumerate()
                .for_each(|(i, k)| rope.apply(k, c.n_head_kv, pos + i));
            sess.k[l].push(&k);
            sess.v[l].push(&v);
            if m == 0 {
                break;
            }

            // Causal attention, parallel over (token, head): token i sees positions 0..=pos + i.
            let (kc, vc) = (&sess.k[l], &sess.v[l]);
            att.par_chunks_mut(hd)
                .zip(q.par_chunks(hd))
                .enumerate()
                .for_each_init(Vec::new, |scores, (j, (out, qh))| {
                    let (i, h) = (lo + j / c.n_head, j % c.n_head);
                    let n_pos = pos + i + 1;
                    let off = (h / group) * hd;
                    scores.resize(n_pos, 0.0);
                    kc.dots(off, kv_dim, qh, scores);
                    scores.iter_mut().for_each(|s| *s *= scale);
                    ops::softmax(scores);
                    vc.weighted_sum(off, kv_dim, scores, out);
                });
            let xs = &mut xs[lo * n_embd..];
            self.matmul(&layer.wo, att, tmp, m);
            ops::add_assign(xs, tmp);

            let xb = &mut xb[..m * n_embd];
            norm_rows(xb, xs, &layer.ffn_norm);
            match &layer.ffn {
                Ffn::Dense(e) => self.ffn(e, xb, tmp, m),
                Ffn::Moe {
                    router,
                    experts,
                    shared,
                } => self.moe(router, experts, shared.as_ref(), xb, tmp, m),
            }
            ops::add_assign(xs, tmp);
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

/// Multiplies a product of `w` by its tensor scale.
fn apply_scale(w: &Weight, y: &mut [f32]) {
    if w.scale != 1.0 {
        y.iter_mut().for_each(|v| *v *= w.scale);
    }
}

fn axpy(out: &mut [f32], w: f32, x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += w * v;
    }
}
