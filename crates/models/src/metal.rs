//! Metal accelerator: runs whole decoder layers on the GPU, reading weights in place from the
//! model's file mapping (unified memory, no copies). All offloaded layers of a token are
//! recorded into one command buffer, so a dense model costs one CPU/GPU round trip per token;
//! MoE layers add one sync each, for top-k routing on the CPU.
//!
//! Prompts are processed in chunks of up to [`BATCH_CHUNK`] tokens per pass (matmul instead of
//! matvec, causal batched attention). An MoE layer syncs once per chunk: the CPU routes every
//! token, then each used expert runs as one matmul over the rows gathered for it.

use crate::backend::BackendError;
use crate::config::Config;
use crate::model::{Accelerator, Expert, Ffn, Model, Weight, forward_each};
use rli_core::metal::{At, AttnParams, Batch, BatchAttnParams, Gpu, GpuBuf, KvBuf, RopeArgs};
use rli_core::ops;
use std::any::Any;
use std::ops::Range;
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
/// Positions a session's GPU KV cache holds at first; it doubles as needed, up to `max_ctx`.
const KV_INITIAL: usize = 256;

/// Activation buffers for up to `cap` tokens, reused across calls.
struct Scratch {
    cap: usize,
    x: GpuBuf,
    xb: GpuBuf,
    q: GpuBuf,
    /// This pass's K and V rows in f32, before they are stored into the cache.
    k: GpuBuf,
    v: GpuBuf,
    att: GpuBuf,
    tmp: GpuBuf,
    gate: GpuBuf,
    up: GpuBuf,
    router: GpuBuf,
    shared_gate: GpuBuf,
    /// Rows gathered for one expert.
    gx: GpuBuf,
    /// Per-expert token indices (u32 bits) in the first half, their weights in the second.
    route: GpuBuf,
    logits: GpuBuf,
}

/// Per-session KV cache of the offloaded layers: `cap` positions allocated, grown on demand up
/// to `max_ctx`. f16 unless the model is in precise mode.
struct Kv {
    k: Vec<KvBuf>,
    v: Vec<KvBuf>,
    /// Decode attention scratch, `n_head * cap`.
    scores: GpuBuf,
    cap: usize,
    max_ctx: usize,
}

impl Kv {
    fn new(m: &Metal, cap: usize, max_ctx: usize) -> Self {
        let alloc = || {
            (0..m.layers.len())
                .map(|_| m.gpu.alloc_kv(cap * m.kv_dim, m.kv_half))
                .collect()
        };
        Kv {
            k: alloc(),
            v: alloc(),
            scores: m.gpu.alloc(m.n_head * cap),
            cap,
            max_ctx,
        }
    }

    /// Makes room for positions `0..need`, keeping the cached rows.
    fn reserve(&mut self, m: &Metal, need: usize) {
        assert!(
            need <= self.max_ctx,
            "positions up to {need} beyond the GPU KV cache ({})",
            self.max_ctx
        );
        if need <= self.cap {
            return;
        }
        let grown = Kv::new(m, need.max(2 * self.cap).min(self.max_ctx), self.max_ctx);
        for (new, old) in grown
            .k
            .iter()
            .zip(&self.k)
            .chain(grown.v.iter().zip(&self.v))
        {
            // SAFETY: every pass waits for the GPU, so none is in flight between calls.
            unsafe { new.copy_from(old) };
        }
        *self = grown;
    }
}

impl Scratch {
    fn new(gpu: &Gpu, c: &Config, max_ff: usize, logits: usize, cap: usize) -> Self {
        Scratch {
            cap,
            x: gpu.alloc(cap * c.n_embd),
            xb: gpu.alloc(cap * c.n_embd),
            q: gpu.alloc(cap * c.q_dim()),
            k: gpu.alloc(cap * c.kv_dim()),
            v: gpu.alloc(cap * c.kv_dim()),
            att: gpu.alloc(cap * c.q_dim()),
            tmp: gpu.alloc(cap * c.n_embd),
            gate: gpu.alloc(cap * max_ff),
            up: gpu.alloc(cap * max_ff),
            router: gpu.alloc(cap * c.n_expert.max(1)),
            shared_gate: gpu.alloc(cap),
            gx: gpu.alloc(if c.n_expert > 0 { cap * c.n_embd } else { 0 }),
            // Each token is in n_expert_used expert lists, plus the shared expert's list.
            route: gpu.alloc(2 * cap * (c.n_expert_used + 1)),
            logits: gpu.alloc(logits),
        }
    }
}

struct Metal {
    gpu: Gpu,
    /// Prompts can run as batches (head_dim fits `attention_batch`).
    batchable: bool,
    max_ff: usize,
    layers: Vec<GpuLayer>,
    output_norm: GpuBuf,
    rope_freqs: Option<GpuBuf>,
    covers_all: bool,
    kv_dim: usize,
    n_head: usize,
    /// KV cache stored as f16 (chosen at attach: f32 when the model is in precise mode).
    kv_half: bool,
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

/// SwiGLU FFN of `n` rows of `x` into `out`.
fn ffn(b: &mut Batch<'_>, s: &Scratch, e: &Expert, x: At<'_>, out: At<'_>, n: usize) {
    mm(b, &e.gate, x, At(&s.gate, 0), n);
    mm(b, &e.up, x, At(&s.up, 0), n);
    b.swiglu(At(&s.gate, 0), At(&s.up, 0), n * e.gate.rows);
    mm(b, &e.down, At(&s.gate, 0), out, n);
}

impl Metal {
    /// Runs the offloaded layers on the `n` token rows in `s.x` (positions `pos..pos + n`) and,
    /// when `want_logits`, the output head on the last row into `s.logits`. Waits for the GPU.
    fn run(&self, model: &Model, kv: &Kv, s: &Scratch, n: usize, pos: usize, want_logits: bool) {
        assert!(n >= 1 && n <= s.cap);
        assert!(pos + n <= kv.cap, "KV cache not reserved");
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
            let (k_at, v_at) = (At(&s.k, 0), At(&s.v, 0));
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
            b.store_kv(k_at, &kv.k[l], pos * kv_dim, n * kv_dim);
            b.store_kv(v_at, &kv.v[l], pos * kv_dim, n * kv_dim);
            if n == 1 {
                let p = AttnParams {
                    n_head: c.n_head as u32,
                    n_kv: c.n_head_kv as u32,
                    head_dim: hd as u32,
                    n_pos: (pos + 1) as u32,
                    kv_dim: kv_dim as u32,
                    max_ctx: kv.cap as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                };
                b.attention(
                    At(&s.q, 0),
                    &kv.k[l],
                    &kv.v[l],
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
                    ffn(&mut b, s, e, At(&s.xb, 0), At(&s.tmp, 0), n);
                    b.add(x_at, At(&s.tmp, 0), n * n_embd);
                }
                Ffn::Moe {
                    router,
                    experts,
                    shared,
                } => {
                    b = self.moe(model, b, s, router, experts, shared.as_ref(), n);
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

    /// Queues an MoE FFN over `n` rows of `s.xb`, adding into `s.x`. Routing needs the router
    /// output on the CPU, so this waits for `b` once and returns a fresh batch to continue in.
    #[allow(clippy::too_many_arguments)]
    fn moe<'g>(
        &'g self,
        model: &Model,
        b: Batch<'g>,
        s: &Scratch,
        router: &Weight,
        experts: &[Expert],
        shared: Option<&(Weight, Expert)>,
        n: usize,
    ) -> Batch<'g> {
        let (c, ne, n_embd) = (&model.cfg, experts.len(), model.cfg.n_embd);
        let mut b = b;
        mm(&mut b, router, At(&s.xb, 0), At(&s.router, 0), n);
        if let Some((gi, _)) = shared {
            mm(&mut b, gi, At(&s.xb, 0), At(&s.shared_gate, 0), n);
        }
        b.finish();
        // SAFETY: the batch completed; nothing is in flight.
        let (router_out, gates) = unsafe {
            (
                s.router.as_slice()[..n * ne].to_vec(),
                s.shared_gate.as_slice()[..n].to_vec(),
            )
        };
        // Softmax over all experts, top-k, optionally renormalized over the chosen k.
        let mut lists: Vec<Vec<(u32, f32)>> = vec![Vec::new(); ne];
        for (t, row) in router_out.chunks_exact(ne).enumerate() {
            let mut probs = row.to_vec();
            ops::softmax(&mut probs);
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            let chosen = &order[..c.n_expert_used];
            let norm = if c.norm_topk {
                chosen.iter().map(|&e| probs[e]).sum::<f32>()
            } else {
                1.0
            };
            for &e in chosen {
                lists[e].push((t as u32, probs[e] / norm));
            }
        }
        let (mut idx, mut wts): (Vec<u32>, Vec<f32>) = lists.iter().flatten().copied().unzip();
        if shared.is_some() {
            idx.extend(0..n as u32);
            wts.extend(gates.iter().map(|g| 1.0 / (1.0 + (-g).exp())));
        }
        let w0 = s.route.len() / 2;
        // SAFETY: nothing is in flight.
        unsafe {
            s.route.write_u32(0, &idx);
            s.route.as_mut_slice()[w0..w0 + wts.len()].copy_from_slice(&wts);
        }
        let mut b = self.gpu.batch();
        let x_at = At(&s.x, 0);
        let mut off = 0;
        for (e, list) in experts.iter().zip(&lists) {
            let m = list.len();
            if m == 0 {
                continue;
            }
            b.gather_rows(At(&s.xb, 0), At(&s.gx, 0), At(&s.route, off), m, n_embd);
            ffn(&mut b, s, e, At(&s.gx, 0), At(&s.tmp, 0), m);
            let (i, w) = (At(&s.route, off), At(&s.route, w0 + off));
            b.scatter_axpy(x_at, At(&s.tmp, 0), i, w, m, n_embd);
            off += m;
        }
        if let Some((_, e)) = shared {
            ffn(&mut b, s, e, At(&s.xb, 0), At(&s.tmp, 0), n);
            let (i, w) = (At(&s.route, off), At(&s.route, w0 + off));
            b.scatter_axpy(x_at, At(&s.tmp, 0), i, w, n, n_embd);
        }
        b
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
        Box::new(Kv::new(self, KV_INITIAL.min(max_ctx), max_ctx))
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
        kv.reserve(self, pos + 1);
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
        let kv: &mut Kv = state
            .downcast_mut()
            .expect("state created by this accelerator");
        kv.reserve(self, pos + xs.len() / n_embd);
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

/// The matrices the GPU reads when the first `n` layers are offloaded (plus the output head
/// when that is all of them).
fn offloaded_weights(model: &Model, n: usize) -> Vec<&Weight> {
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
    if n == model.cfg.n_layer {
        weights.push(&model.output);
    }
    weights
}

/// The file byte range covering `weights`.
fn span(weights: &[&Weight]) -> Range<usize> {
    let start = weights.iter().map(|w| w.offset).min().unwrap_or(0);
    let end = weights
        .iter()
        .map(|w| w.offset + w.rows * w.ty.row_bytes(w.cols))
        .max()
        .unwrap_or(0);
    start..end.max(start)
}

/// The most leading layers whose weights fit in one Metal buffer (the whole model when it
/// fits), or `None` without a Metal device.
pub fn fitting_layers(model: &Model) -> Option<usize> {
    let max = rli_core::metal::max_buffer_len()?;
    let fits = |n| rli_core::metal::wrapped_len(&span(&offloaded_weights(model, n))) <= max;
    Some(
        (0..=model.cfg.n_layer)
            .rev()
            .find(|&n| fits(n))
            .unwrap_or(0),
    )
}

pub fn attach(model: &mut Model, layers: usize) -> Result<(), BackendError> {
    let c = model.cfg.clone();
    let n = layers.min(c.n_layer);
    let covers_all = n == c.n_layer;
    let weights = offloaded_weights(model, n);
    let mapped = model.gguf().mapped_bytes();
    // SAFETY: the GPU views the model's own mapping, and `Model` drops its accelerator before
    // the mapping (field order), so the view never outlives it.
    let gpu = unsafe { Gpu::new(mapped, span(&weights)) }
        .map_err(|e| BackendError(format!("{e}; offload fewer layers (--gpu-layers)")))?;
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
    let batchable = c.head_dim.is_multiple_of(32) && c.head_dim <= 256;
    let accel = Metal {
        batchable,
        kv_half: !model.precise,
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
