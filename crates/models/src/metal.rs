//! Metal accelerator: runs whole decoder layers on the GPU, reading weights in place from the
//! model's file mapping (unified memory, no copies). All offloaded layers of a token are
//! recorded into one command buffer, so a dense model costs one CPU/GPU round trip per token;
//! MoE layers add one sync each, for top-k routing on the CPU.

use crate::backend::BackendError;
use crate::model::{Accelerator, Expert, Ffn, Model, Weight};
use rli_core::metal::{At, AttnParams, Batch, Gpu, GpuBuf, RopeArgs};
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

/// Activation buffers, reused across tokens.
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
    logits: GpuBuf,
}

/// Per-session KV cache of the offloaded layers, preallocated for `max_ctx` positions.
struct Kv {
    k: Vec<GpuBuf>,
    v: Vec<GpuBuf>,
    scores: GpuBuf,
    max_ctx: usize,
}

struct Metal {
    gpu: Gpu,
    layers: Vec<GpuLayer>,
    output_norm: GpuBuf,
    rope_freqs: Option<GpuBuf>,
    covers_all: bool,
    kv_dim: usize,
    n_head: usize,
    scratch: Mutex<Scratch>,
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

impl Metal {
    fn ffn(&self, b: &mut Batch<'_>, s: &Scratch, e: &Expert, out: At<'_>) {
        b.matvec(
            e.gate.ty,
            e.gate.rows,
            e.gate.cols,
            e.gate.offset,
            At(&s.xb, 0),
            At(&s.gate, 0),
        );
        b.matvec(
            e.up.ty,
            e.up.rows,
            e.up.cols,
            e.up.offset,
            At(&s.xb, 0),
            At(&s.up, 0),
        );
        b.swiglu(At(&s.gate, 0), At(&s.up, 0), e.gate.rows);
        b.matvec(
            e.down.ty,
            e.down.rows,
            e.down.cols,
            e.down.offset,
            At(&s.gate, 0),
            out,
        );
    }
}

fn mv(b: &mut Batch<'_>, w: &Weight, x: At<'_>, y: At<'_>) {
    b.matvec(w.ty, w.rows, w.cols, w.offset, x, y);
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
        let kv: &mut Kv = state
            .downcast_mut()
            .expect("state created by this accelerator");
        assert!(
            pos < kv.max_ctx,
            "position {pos} beyond the GPU KV cache ({})",
            kv.max_ctx
        );
        let c = &model.cfg;
        let (hd, kv_dim, n_embd) = (c.head_dim, c.kv_dim(), c.n_embd);
        let s = self.scratch.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: no GPU work is in flight outside this locked section.
        unsafe { s.x.as_mut_slice()[..n_embd].copy_from_slice(x) };

        let x_at = At(&s.x, 0);
        let rope = |n_heads| RopeArgs {
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
            b.rmsnorm(x_at, At(&s.xb, 0), &g.attn_norm, 1, n_embd, c.rms_eps);
            mv(&mut b, &layer.wq, At(&s.xb, 0), At(&s.q, 0));
            mv(&mut b, &layer.wk, At(&s.xb, 0), k_at);
            mv(&mut b, &layer.wv, At(&s.xb, 0), v_at);
            for (t, bias, n) in [
                (At(&s.q, 0), &g.bq, c.q_dim()),
                (k_at, &g.bk, kv_dim),
                (v_at, &g.bv, kv_dim),
            ] {
                if let Some(bias) = bias {
                    b.add(t, bias.into(), n);
                }
            }
            if let Some(w) = &g.q_norm {
                b.rmsnorm(At(&s.q, 0), At(&s.q, 0), w, c.n_head, hd, c.rms_eps);
            }
            if let Some(w) = &g.k_norm {
                b.rmsnorm(k_at, k_at, w, c.n_head_kv, hd, c.rms_eps);
            }
            b.rope(At(&s.q, 0), &rope(c.n_head));
            b.rope(k_at, &rope(c.n_head_kv));
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
            mv(&mut b, &layer.wo, At(&s.att, 0), At(&s.tmp, 0));
            b.add(x_at, At(&s.tmp, 0), n_embd);
            b.rmsnorm(x_at, At(&s.xb, 0), &g.ffn_norm, 1, n_embd, c.rms_eps);
            match &layer.ffn {
                Ffn::Dense(e) => {
                    self.ffn(&mut b, &s, e, At(&s.tmp, 0));
                    b.add(x_at, At(&s.tmp, 0), n_embd);
                }
                Ffn::Moe {
                    router,
                    experts,
                    shared,
                } => {
                    // Routing needs the router output on the CPU: one sync per MoE layer.
                    mv(&mut b, router, At(&s.xb, 0), At(&s.router, 0));
                    if let Some((gi, _)) = shared {
                        mv(&mut b, gi, At(&s.xb, 0), At(&s.shared_gate, 0));
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
                        self.ffn(&mut b, &s, &experts[e], At(&s.tmp, 0));
                        b.axpy(x_at, At(&s.tmp, 0), probs[e] / norm, n_embd);
                    }
                    if let Some((_, e)) = shared {
                        self.ffn(&mut b, &s, e, At(&s.tmp, 0));
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
        let want_logits = self.covers_all && logits.is_some();
        if want_logits {
            b.rmsnorm(x_at, At(&s.xb, 0), &self.output_norm, 1, n_embd, c.rms_eps);
            mv(&mut b, &model.output, At(&s.xb, 0), At(&s.logits, 0));
        }
        b.finish();
        // SAFETY: the batch completed; nothing is in flight.
        unsafe {
            x.copy_from_slice(&s.x.as_slice()[..n_embd]);
            if want_logits && let Some(out) = logits {
                out.copy_from_slice(&s.logits.as_slice()[..out.len()]);
            }
        }
        want_logits
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
    let scratch = Scratch {
        x: gpu.alloc(c.n_embd),
        xb: gpu.alloc(c.n_embd),
        q: gpu.alloc(c.q_dim()),
        att: gpu.alloc(c.q_dim()),
        tmp: gpu.alloc(c.n_embd),
        gate: gpu.alloc(max_ff),
        up: gpu.alloc(max_ff),
        router: gpu.alloc(c.n_expert.max(1)),
        shared_gate: gpu.alloc(1),
        logits: gpu.alloc(if covers_all { model.output.rows } else { 0 }),
    };
    let accel = Metal {
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
