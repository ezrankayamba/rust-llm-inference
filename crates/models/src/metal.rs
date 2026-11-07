//! Metal accelerator: runs whole decoder layers on the GPU, reading weights in place from the
//! model's file mapping (unified memory, no copies). All offloaded layers of a token are
//! recorded into one command buffer, so a dense model costs one CPU/GPU round trip per token;
//! MoE layers add one sync each, for top-k routing on the CPU.
//!
//! Prompts are processed in chunks of up to [`BATCH_CHUNK`] tokens per pass (matmul instead of
//! matvec, causal batched attention). MoE models process prompts token by token for now.

use crate::backend::BackendError;
use crate::config::Config;
use crate::model::{Accelerator, Expert, Ffn, Model, Weight, forward_each};
use rli_core::metal::{At, AttnParams, Batch, BatchAttnParams, Gpu, GpuBuf, RopeArgs};
use rli_core::ops;
use std::any::Any;
use std::sync::Mutex;

struct GpuLayer {
    attn_norm: GpuBuf,
    ffn_norm: GpuBuf,
    bq: Option<GpuBuf>,
    bk: Option<GpuBuf>,
    bv: Option<GpuBuf>,
    q_norm: Option<GpuBuf>,
    k_norm: Option<GpuBuf>,
}

/// Most prompt tokens processed per GPU pass; bounds the batch scratch buffers.
const BATCH_CHUNK: usize = 256;

/// Activation buffers for up to `cap` tokens, reused across calls.
struct Scratch {
    cap: usize,
    x: GpuBuf,
    xb: GpuBuf,
    q: GpuBuf,
    att: GpuBuf,
    tmp: GpuBuf,
    gate: GpuBuf,
    up: GpuBuf,
    router: GpuBuf,
    shared_gate: GpuBuf,
    logits: GpuBuf,
}

/// Per-session KV cache of the offloaded layers, preallocated for `max_ctx` positions.
struct Kv {
    k: Vec<GpuBuf>,
    v: Vec<GpuBuf>,
    scores: GpuBuf,
    max_ctx: usize,
}

impl Scratch {
    fn new(gpu: &Gpu, c: &Config, max_ff: usize, logits: usize, cap: usize) -> Self {
        Scratch {
            cap,
            x: gpu.alloc(cap * c.n_embd),
            xb: gpu.alloc(cap * c.n_embd),
            q: gpu.alloc(cap * c.q_dim()),
            att: gpu.alloc(cap * c.q_dim()),
            tmp: gpu.alloc(cap * c.n_embd),
            gate: gpu.alloc(cap * max_ff),
            up: gpu.alloc(cap * max_ff),
            router: gpu.alloc(c.n_expert.max(1)),
            shared_gate: gpu.alloc(1),
            logits: gpu.alloc(logits),
        }
    }
}

struct Metal {
    gpu: Gpu,
    /// Prompts can run as batches (no MoE layers, and head_dim fits `attention_batch`).
    batchable: bool,
    max_ff: usize,
    layers: Vec<GpuLayer>,
    output_norm: GpuBuf,
    rope_freqs: Option<GpuBuf>,
    covers_all: bool,
    kv_dim: usize,
    n_head: usize,
    scratch: Mutex<Scratch>,
    /// Allocated on the first batched call.
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

fn mm(b: &mut Batch<'_>, w: &Weight, x: At<'_>, y: At<'_>, n: usize) {
    b.matmul(w.ty, w.rows, w.cols, w.offset, x, y, n);
}

/// SwiGLU FFN of `n` rows of `s.xb` into `out`.
fn ffn(b: &mut Batch<'_>, s: &Scratch, e: &Expert, out: At<'_>, n: usize) {
    mm(b, &e.gate, At(&s.xb, 0), At(&s.gate, 0), n);
    mm(b, &e.up, At(&s.xb, 0), At(&s.up, 0), n);
    b.swiglu(At(&s.gate, 0), At(&s.up, 0), n * e.gate.rows);
    mm(b, &e.down, At(&s.gate, 0), out, n);
}

impl Metal {
    /// Runs the offloaded layers on the `n` token rows in `s.x` (positions `pos..pos + n`) and,
    /// when `want_logits`, the output head on the last row into `s.logits`. Waits for the GPU.
    fn run(&self, model: &Model, kv: &Kv, s: &Scratch, n: usize, pos: usize, want_logits: bool) {
        assert!(n >= 1 && n <= s.cap);
        assert!(
            pos + n <= kv.max_ctx,
            "positions up to {} beyond the GPU KV cache ({})",
            pos + n,
            kv.max_ctx
        );
        let c = &model.cfg;
        let (hd, kv_dim, n_embd) = (c.head_dim, c.kv_dim(), c.n_embd);
        let x_at = At(&s.x, 0);
        let rope = |n_heads| RopeArgs {
            n_tokens: n,
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
            for (t, bias, dim) in [
                (At(&s.q, 0), &g.bq, c.q_dim()),
                (k_at, &g.bk, kv_dim),
                (v_at, &g.bv, kv_dim),
            ] {
                if let Some(bias) = bias {
                    b.add_bcast(t, bias.into(), dim, n * dim);
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
            if n == 1 {
                let p = AttnParams {
                    n_head: c.n_head as u32,
                    n_kv: c.n_head_kv as u32,
                    head_dim: hd as u32,
                    n_pos: (pos + 1) as u32,
                    kv_dim: kv_dim as u32,
                    max_ctx: kv.max_ctx as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                };
                b.attention(
                    At(&s.q, 0),
                    At(&kv.k[l], 0),
                    At(&kv.v[l], 0),
                    At(&s.att, 0),
                    &kv.scores,
                    &p,
                );
            } else {
                let p = BatchAttnParams {
                    n_head: c.n_head as u32,
                    n_kv: c.n_head_kv as u32,
                    head_dim: hd as u32,
                    pos: pos as u32,
                    kv_dim: kv_dim as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                };
                b.attention_batch(At(&s.q, 0), &kv.k[l], &kv.v[l], At(&s.att, 0), n, &p);
            }
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
                    // Routing needs the router output on the CPU: one sync per MoE layer.
                    mm(&mut b, router, At(&s.xb, 0), At(&s.router, 0), 1);
                    if let Some((gi, _)) = shared {
                        mm(&mut b, gi, At(&s.xb, 0), At(&s.shared_gate, 0), 1);
                    }
                    b.finish();
                    // SAFETY: the batch completed; nothing is in flight.
                    let mut probs = unsafe { s.router.as_slice()[..experts.len()].to_vec() };
                    let shared_gate = unsafe { s.shared_gate.as_slice()[0] };
                    ops::softmax(&mut probs);
                    let mut order: Vec<usize> = (0..experts.len()).collect();
                    order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
                    let chosen = &order[..c.n_expert_used];
                    let norm = if c.norm_topk {
                        chosen.iter().map(|&e| probs[e]).sum::<f32>()
                    } else {
                        1.0
                    };
                    b = self.gpu.batch();
                    for &e in chosen {
                        ffn(&mut b, s, &experts[e], At(&s.tmp, 0), 1);
                        b.axpy(x_at, At(&s.tmp, 0), probs[e] / norm, n_embd);
                    }
                    if let Some((_, e)) = shared {
                        ffn(&mut b, s, e, At(&s.tmp, 0), 1);
                        b.axpy(
                            x_at,
                            At(&s.tmp, 0),
                            1.0 / (1.0 + (-shared_gate).exp()),
                            n_embd,
                        );
                    }
                }
            }
        }
        if want_logits {
            let last = At(&s.x, (n - 1) * n_embd);
            b.rmsnorm(last, At(&s.xb, 0), &self.output_norm, 1, n_embd, c.rms_eps);
            mm(&mut b, &model.output, At(&s.xb, 0), At(&s.logits, 0), 1);
        }
        b.finish();
    }

    /// Uploads `xs` (rows of n_embd), runs them, and copies the rows (and logits) back.
    fn pass(
        &self,
        model: &Model,
        kv: &Kv,
        s: &Scratch,
        xs: &mut [f32],
        pos: usize,
        logits: Option<&mut [f32]>,
    ) -> bool {
        let n = xs.len() / model.cfg.n_embd;
        let want_logits = self.covers_all && logits.is_some();
        // SAFETY: callers hold the scratch lock and no GPU work is in flight.
        unsafe { s.x.as_mut_slice()[..xs.len()].copy_from_slice(xs) };
        self.run(model, kv, s, n, pos, want_logits);
        // SAFETY: `run` waited for the GPU.
        unsafe {
            xs.copy_from_slice(&s.x.as_slice()[..xs.len()]);
            if want_logits && let Some(out) = logits {
                out.copy_from_slice(&s.logits.as_slice()[..out.len()]);
            }
        }
        want_logits
    }
}

impl Accelerator for Metal {
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
        let kv: &Kv = state
            .downcast_ref()
            .expect("state created by this accelerator");
        let s = self.scratch.lock().unwrap_or_else(|p| p.into_inner());
        self.pass(model, kv, &s, x, pos, logits)
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
        let kv: &Kv = state
            .downcast_ref()
            .expect("state created by this accelerator");
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
    // SAFETY: the GPU views the model's own mapping, and `Model` drops its accelerator before
    // the mapping (field order), so the view never outlives it.
    let gpu = unsafe { Gpu::new(model.gguf().mapped_bytes()) }.map_err(BackendError)?;
    if let Some(w) = weights
        .iter()
        .find(|w| !gpu.supports(w.ty) || w.cols % 32 != 0)
    {
        return Err(BackendError(format!(
            "{:?} weights ({} cols) have no Metal kernel; run more layers on the CPU (lower --gpu-layers)",
            w.ty, w.cols
        )));
    }
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
        })
        .collect();
    let logits = if covers_all { model.output.rows } else { 0 };
    let scratch = Scratch::new(&gpu, &c, max_ff, logits, 1);
    let has_moe = model.layers[..n]
        .iter()
        .any(|l| matches!(l.ffn, Ffn::Moe { .. }));
    let batchable = !has_moe && c.head_dim.is_multiple_of(32) && c.head_dim <= 256;
    let accel = Metal {
        batchable,
        max_ff,
        batch: Mutex::new(None),
        output_norm: gpu.upload(&model.output_norm),
        rope_freqs: opt(&model.rope_freqs),
        layers: gpu_layers,
        covers_all,
        kv_dim: c.kv_dim(),
        n_head: c.n_head,
        scratch: Mutex::new(scratch),
        gpu,
    };
    model.set_accelerator(Box::new(accel));
    Ok(())
}
