//! CUDA accelerator: the same layer program as `metal.rs`, but for a discrete GPU. Only the
//! offloaded layers' weights are copied to VRAM (partial offload for low-VRAM cards); the rest
//! run on the CPU from the file mapping. One stream sync per token: MoE layers pick their
//! experts on the GPU (experts stored as separate tensors are routed on the CPU instead, with a
//! sync per layer). Prompts run in chunks of up to [`BATCH_CHUNK`] tokens per pass (matmul
//! instead of matvec, one sync per chunk), unless an offloaded layer is MoE.

use crate::backend::BackendError;
use crate::model::{Accelerator, Expert, Ffn, Model, Weight, expert_strides, forward_each};
use rli_core::cuda::{
    ATTN_THREADS, At, AttnParams, Batch, Gpu, GpuBuf, GpuIds, MOE_MAX_EXPERTS, RopeArgs,
};
use rli_core::ops;
use std::any::Any;
use std::sync::Mutex;

/// Prompt tokens per batched pass.
const BATCH_CHUNK: usize = 256;
const _: () = assert!(crate::model::PROMPT_SLICE.is_multiple_of(BATCH_CHUNK));
/// Most f32s of attention scores scratch a session allocates for batched prompts (128 MB). A
/// chunk whose scores need more runs its attention a few tokens at a time.
const SCORE_BUDGET: usize = 1 << 25;

struct GpuLayer {
    attn_norm: GpuBuf,
    ffn_norm: GpuBuf,
    bq: Option<GpuBuf>,
    bk: Option<GpuBuf>,
    bv: Option<GpuBuf>,
    q_norm: Option<GpuBuf>,
    k_norm: Option<GpuBuf>,
    /// Set when the layer's experts can be picked by id on the GPU.
    expert_strides: Option<[usize; 3]>,
}

/// Activation buffers for a pass over up to `n` tokens, reused across passes.
struct Scratch {
    x: GpuBuf,
    xb: GpuBuf,
    q: GpuBuf,
    att: GpuBuf,
    tmp: GpuBuf,
    gate: GpuBuf,
    up: GpuBuf,
    router: GpuBuf,
    shared_gate: GpuBuf,
    /// The chosen experts of an MoE layer, and their weights (then the shared expert's gate).
    route_ids: GpuIds,
    route_w: GpuBuf,
    logits: GpuBuf,
}

impl Scratch {
    fn new(gpu: &Gpu, c: &crate::config::Config, max_ff: usize, logits: usize, n: usize) -> Self {
        // An MoE token holds its chosen experts' activations side by side, and their outputs
        // plus a shared expert's.
        let k = c.n_expert_used;
        Self {
            x: gpu.alloc(n * c.n_embd),
            xb: gpu.alloc(n * c.n_embd),
            q: gpu.alloc(n * c.q_dim()),
            att: gpu.alloc(n * c.q_dim()),
            tmp: gpu.alloc(n.max(k + 1) * c.n_embd),
            gate: gpu.alloc(n.max(k) * max_ff),
            up: gpu.alloc(n.max(k) * max_ff),
            router: gpu.alloc(c.n_expert.max(1)),
            shared_gate: gpu.alloc(1),
            route_ids: gpu.alloc_ids(k),
            route_w: gpu.alloc(k + 1),
            logits: gpu.alloc(logits),
        }
    }
}

/// Per-session KV cache of the offloaded layers, preallocated in VRAM for `max_ctx` positions.
struct Kv {
    k: Vec<GpuBuf>,
    v: Vec<GpuBuf>,
    scores: GpuBuf,
    max_ctx: usize,
}

struct Cuda {
    gpu: Gpu,
    layers: Vec<GpuLayer>,
    output_norm: GpuBuf,
    rope_freqs: Option<GpuBuf>,
    covers_all: bool,
    /// No offloaded layer is MoE, so prompts can run as matmuls.
    batchable: bool,
    max_ff: usize,
    kv_dim: usize,
    n_head: usize,
    scratch: Mutex<Scratch>,
    /// Scratch for [`BATCH_CHUNK`] tokens, allocated by the first batched prompt.
    batch: Mutex<Option<Scratch>>,
}

fn experts_of(f: &Ffn) -> Vec<&Expert> {
    match f {
        Ffn::Dense(e) => vec![e],
        Ffn::Moe {
            experts, shared, ..
        } => experts
            .iter()
            .chain(shared.as_ref().map(|(_, e)| e))
            .collect(),
    }
}

fn mv(b: &mut Batch<'_>, w: &Weight, x: At<'_>, y: At<'_>) {
    mm(b, w, x, y, 1);
}

/// `w` applied to each of `n` tokens.
fn mm(b: &mut Batch<'_>, w: &Weight, x: At<'_>, y: At<'_>, n: usize) {
    b.matmul(w.ty, w.rows, w.cols, w.offset, x, y, n);
}

fn ffn(b: &mut Batch<'_>, s: &Scratch, e: &Expert, out: At<'_>, n: usize) {
    mm(b, &e.gate, At(&s.xb, 0), At(&s.gate, 0), n);
    mm(b, &e.up, At(&s.xb, 0), At(&s.up, 0), n);
    b.swiglu(At(&s.gate, 0), At(&s.up, 0), n * e.gate.rows);
    mm(b, &e.down, At(&s.gate, 0), out, n);
}

/// Queues one token's MoE FFN over `s.xb`, adding into `s.x`, once the router's (and shared
/// gate's) outputs are queued: the experts are picked on the GPU, so nothing waits, and the k
/// chosen ones run as one id-indexed matvec per matrix.
fn moe_decode(
    c: &crate::config::Config,
    b: &mut Batch<'_>,
    s: &Scratch,
    experts: &[Expert],
    shared: Option<&(Weight, Expert)>,
    [gate_stride, up_stride, down_stride]: [usize; 3],
) {
    let (ne, k, n_embd) = (experts.len(), c.n_expert_used, c.n_embd);
    let ids = &s.route_ids;
    b.moe_topk(
        &s.router,
        &s.shared_gate,
        ids,
        &s.route_w,
        ne,
        k,
        c.norm_topk,
        shared.is_some(),
    );
    let e = &experts[0];
    let ff = e.gate.rows;
    let (xb, gate, up, tmp) = (At(&s.xb, 0), At(&s.gate, 0), At(&s.up, 0), At(&s.tmp, 0));
    for (m, stride, out) in [(&e.gate, gate_stride, gate), (&e.up, up_stride, up)] {
        b.matvec_id(
            m.ty, m.rows, m.cols, m.offset, stride, ne, ids, xb, 0, out, ff, k,
        );
    }
    b.swiglu(gate, up, k * ff);
    let d = &e.down;
    b.matvec_id(
        d.ty,
        d.rows,
        d.cols,
        d.offset,
        down_stride,
        ne,
        ids,
        gate,
        ff,
        tmp,
        n_embd,
        k,
    );
    if let Some((_, e)) = shared {
        ffn(b, s, e, At(&s.tmp, k * n_embd), 1);
    }
    let m = k + usize::from(shared.is_some());
    b.moe_combine(At(&s.x, 0), &s.tmp, &s.route_w, m, n_embd);
}

impl Cuda {
    /// Runs the offloaded layers over the tokens of `xs` (row-major embeddings, in place) at
    /// positions `pos..`; `s` must hold that many tokens. More than one token needs
    /// `self.batchable`. Returns whether it filled `logits` (the last token's).
    fn pass(
        &self,
        model: &Model,
        kv: &Kv,
        s: &mut Scratch,
        xs: &mut [f32],
        pos: usize,
        logits: Option<&mut [f32]>,
    ) -> bool {
        let c = &model.cfg;
        let (hd, kv_dim, n_embd) = (c.head_dim, c.kv_dim(), c.n_embd);
        let n = xs.len() / n_embd;
        assert!(
            pos + n <= kv.max_ctx,
            "position {} beyond the GPU KV cache ({})",
            pos + n - 1,
            kv.max_ctx
        );
        self.gpu.write(&mut s.x, xs);
        let s = &*s;

        let x_at = At(&s.x, 0);
        let rope = |n_heads| RopeArgs {
            n_tok: n,
            n_heads,
            head_dim: hd,
            rope_dim: c.rope_dim,
            pos,
            theta: c.rope_theta,
            neox: c.arch.neox_rope(),
            freq_factors: self.rope_freqs.as_ref(),
        };
        let mut b = self.gpu.batch();
        for (l, (layer, g)) in model.layers.iter().zip(&self.layers).enumerate() {
            let (k_at, v_at) = (At(&kv.k[l], pos * kv_dim), At(&kv.v[l], pos * kv_dim));
            b.rmsnorm(x_at, At(&s.xb, 0), &g.attn_norm, n, n_embd, c.rms_eps);
            mm(&mut b, &layer.wq, At(&s.xb, 0), At(&s.q, 0), n);
            mm(&mut b, &layer.wk, At(&s.xb, 0), k_at, n);
            mm(&mut b, &layer.wv, At(&s.xb, 0), v_at, n);
            for (t, bias, len) in [
                (At(&s.q, 0), &g.bq, c.q_dim()),
                (k_at, &g.bk, kv_dim),
                (v_at, &g.bv, kv_dim),
            ] {
                if let Some(bias) = bias {
                    b.add_bias(t, bias, n * len);
                }
            }
            if let Some(w) = &g.q_norm {
                b.rmsnorm(At(&s.q, 0), At(&s.q, 0), w, n * c.n_head, hd, c.rms_eps);
            }
            if let Some(w) = &g.k_norm {
                b.rmsnorm(k_at, k_at, w, n * c.n_head_kv, hd, c.rms_eps);
            }
            b.rope(At(&s.q, 0), &rope(c.n_head));
            b.rope(k_at, &rope(c.n_head_kv));
            let p = AttnParams {
                n_head: c.n_head as u32,
                n_kv: c.n_head_kv as u32,
                head_dim: hd as u32,
                n_tok: n as u32,
                n_pos: (pos + 1) as u32,
                kv_dim: kv_dim as u32,
                max_ctx: kv.max_ctx as u32,
                scale: 1.0 / (hd as f32).sqrt(),
            };
            b.attention(&s.q, &kv.k[l], &kv.v[l], &s.att, &kv.scores, &p);
            mm(&mut b, &layer.wo, At(&s.att, 0), At(&s.tmp, 0), n);
            b.add(x_at, At(&s.tmp, 0), n * n_embd);
            b.rmsnorm(x_at, At(&s.xb, 0), &g.ffn_norm, n, n_embd, c.rms_eps);
            match &layer.ffn {
                Ffn::Dense(e) => {
                    ffn(&mut b, s, e, At(&s.tmp, 0), n);
                    b.add(x_at, At(&s.tmp, 0), n * n_embd);
                }
                Ffn::Moe {
                    router,
                    experts,
                    shared,
                } => {
                    assert_eq!(n, 1, "MoE layers run one token at a time");
                    mv(&mut b, router, At(&s.xb, 0), At(&s.router, 0));
                    if let Some(strides) = g.expert_strides {
                        if let Some((gi, _)) = shared {
                            mv(&mut b, gi, At(&s.xb, 0), At(&s.shared_gate, 0));
                        }
                        moe_decode(c, &mut b, s, experts, shared.as_ref(), strides);
                        continue;
                    }
                    // Routing needs the router output on the CPU: one sync per MoE layer.
                    let mut shared_gate = [0.0f32];
                    if let Some((gi, _)) = shared {
                        mv(&mut b, gi, At(&s.xb, 0), At(&s.shared_gate, 0));
                        self.gpu.read(&s.shared_gate, &mut shared_gate);
                    }
                    let mut probs = vec![0.0; experts.len()];
                    self.gpu.read(&s.router, &mut probs);
                    ops::softmax(&mut probs);
                    let mut order: Vec<usize> = (0..experts.len()).collect();
                    order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
                    let chosen = &order[..c.n_expert_used];
                    let norm = if c.norm_topk {
                        chosen.iter().map(|&e| probs[e]).sum::<f32>()
                    } else {
                        1.0
                    };
                    for &e in chosen {
                        ffn(&mut b, s, &experts[e], At(&s.tmp, 0), 1);
                        b.axpy(x_at, At(&s.tmp, 0), probs[e] / norm, n_embd);
                    }
                    if let Some((_, e)) = shared {
                        ffn(&mut b, s, e, At(&s.tmp, 0), 1);
                        let gate = 1.0 / (1.0 + (-shared_gate[0]).exp());
                        b.axpy(x_at, At(&s.tmp, 0), gate, n_embd);
                    }
                }
            }
        }
        let want_logits = self.covers_all && logits.is_some();
        if want_logits {
            let last = At(&s.x, (n - 1) * n_embd);
            b.rmsnorm(last, At(&s.xb, 0), &self.output_norm, 1, n_embd, c.rms_eps);
            mv(&mut b, &model.output, At(&s.xb, 0), At(&s.logits, 0));
        }
        b.finish();
        self.gpu.read(&s.x, xs);
        if want_logits && let Some(out) = logits {
            self.gpu.read(&s.logits, out);
        }
        want_logits
    }
}

impl Accelerator for Cuda {
    fn name(&self) -> &str {
        self.gpu.name()
    }

    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn new_state(&self, max_ctx: usize) -> Box<dyn Any + Send> {
        let n = self.layers.len();
        Box::new(Kv {
            k: (0..n)
                .map(|_| self.gpu.alloc(max_ctx * self.kv_dim))
                .collect(),
            v: (0..n)
                .map(|_| self.gpu.alloc(max_ctx * self.kv_dim))
                .collect(),
            scores: self.gpu.alloc(self.n_head * max_ctx),
            max_ctx,
        })
    }

    fn forward(
        &self,
        model: &Model,
        state: &mut (dyn Any + Send),
        x: &mut [f32],
        pos: usize,
        logits: Option<&mut [f32]>,
    ) -> bool {
        let kv: &mut Kv = state
            .downcast_mut()
            .expect("state created by this accelerator");
        let mut s = self.scratch.lock().unwrap_or_else(|p| p.into_inner());
        self.pass(model, kv, &mut s, x, pos, logits)
    }

    fn forward_batch(
        &self,
        model: &Model,
        state: &mut (dyn Any + Send),
        xs: &mut [f32],
        pos: usize,
        mut logits: Option<&mut [f32]>,
    ) -> bool {
        let n_embd = model.cfg.n_embd;
        if !self.batchable || xs.len() == n_embd {
            return forward_each(self, model, state, xs, pos, logits);
        }
        let kv: &mut Kv = state
            .downcast_mut()
            .expect("state created by this accelerator");
        let per_token = self.n_head * kv.max_ctx;
        let score_rows = (SCORE_BUDGET / per_token).clamp(1, BATCH_CHUNK);
        if kv.scores.len() < score_rows * per_token {
            kv.scores = self.gpu.alloc(score_rows * per_token);
        }
        let mut guard = self.batch.lock().unwrap_or_else(|p| p.into_inner());
        let s = guard.get_or_insert_with(|| {
            let logits = if self.covers_all {
                model.output.rows
            } else {
                0
            };
            Scratch::new(&self.gpu, &model.cfg, self.max_ff, logits, BATCH_CHUNK)
        });
        let n_chunks = xs.len().div_ceil(BATCH_CHUNK * n_embd);
        let mut done = false;
        for (i, chunk) in xs.chunks_mut(BATCH_CHUNK * n_embd).enumerate() {
            let l = if i + 1 == n_chunks {
                logits.take()
            } else {
                None
            };
            done = self.pass(model, kv, s, chunk, pos + i * BATCH_CHUNK, l);
        }
        done
    }
}

pub fn attach(model: &mut Model, layers: usize) -> Result<(), BackendError> {
    let c = model.cfg.clone();
    let n = layers.min(c.n_layer);
    let covers_all = n == c.n_layer;
    let mut weights: Vec<&Weight> = Vec::new();
    for layer in &model.layers[..n] {
        weights.extend([&layer.wq, &layer.wk, &layer.wv, &layer.wo]);
        if let Ffn::Moe { router, shared, .. } = &layer.ffn {
            weights.push(router);
            weights.extend(shared.as_ref().map(|(g, _)| g));
        }
        for e in experts_of(&layer.ffn) {
            weights.extend([&e.gate, &e.up, &e.down]);
        }
    }
    if covers_all {
        weights.push(&model.output);
    }
    if let Some(w) = weights
        .iter()
        .find(|w| !Gpu::supports(w.ty) || w.cols % 32 != 0)
    {
        return Err(BackendError(format!(
            "{:?} weights ({} cols) have no CUDA kernel; run more layers on the CPU (lower --gpu-layers)",
            w.ty, w.cols
        )));
    }
    if c.head_dim > ATTN_THREADS as usize {
        return Err(BackendError(format!(
            "head_dim {} is beyond the CUDA attention kernel's {ATTN_THREADS}",
            c.head_dim
        )));
    }
    let ranges: Vec<(usize, usize)> = weights.iter().map(|w| (w.offset, w.n_bytes())).collect();
    let gpu = Gpu::new(model.gguf().mapped_bytes(), &ranges).map_err(BackendError)?;
    let max_ff = weights
        .iter()
        .filter(|w| w.cols == c.n_embd)
        .map(|w| w.rows)
        .max()
        .unwrap_or(0)
        .max(c.n_ff);
    let opt = |v: &Option<Vec<f32>>| v.as_ref().map(|v| gpu.upload(v));
    let gpu_layers = model.layers[..n]
        .iter()
        .map(|l| GpuLayer {
            attn_norm: gpu.upload(&l.attn_norm),
            ffn_norm: gpu.upload(&l.ffn_norm),
            bq: opt(&l.bq),
            bk: opt(&l.bk),
            bv: opt(&l.bv),
            q_norm: opt(&l.q_norm),
            k_norm: opt(&l.k_norm),
            expert_strides: match &l.ffn {
                Ffn::Moe { experts, .. } if experts.len() <= MOE_MAX_EXPERTS => {
                    expert_strides(experts)
                }
                _ => None,
            },
        })
        .collect();
    let logits = if covers_all { model.output.rows } else { 0 };
    let scratch = Scratch::new(&gpu, &c, max_ff, logits, 1);
    let batchable = model.layers[..n]
        .iter()
        .all(|l| matches!(l.ffn, Ffn::Dense(_)));
    let accel = Cuda {
        output_norm: gpu.upload(&model.output_norm),
        rope_freqs: opt(&model.rope_freqs),
        layers: gpu_layers,
        covers_all,
        batchable,
        max_ff,
        kv_dim: c.kv_dim(),
        n_head: c.n_head,
        scratch: Mutex::new(scratch),
        batch: Mutex::new(None),
        gpu,
    };
    model.set_accelerator(Box::new(accel));
    Ok(())
}
