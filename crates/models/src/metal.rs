//! Metal accelerator: runs whole decoder layers on the GPU, reading weights in place from the
//! model's file mapping (unified memory, no copies). All offloaded layers of a token are
//! recorded into one command buffer, so decoding costs one CPU/GPU round trip per token. MoE
//! decode routes on the GPU (top-k, then matvecs that read their expert id from a buffer); only
//! experts stored as separate tensors (old Mixtral files) add a sync per layer for CPU routing.
//!
//! Prompts are processed in chunks of up to [`BATCH_CHUNK`] tokens per pass (matmul instead of
//! matvec, causal batched attention). An MoE layer syncs once per chunk: the CPU routes every
//! token, then each used expert runs as one matmul over the rows gathered for it.

use crate::backend::BackendError;
use crate::config::Config;
use crate::model::{
    Accelerator, Expert, Ffn, Model, Weight, expert_strides, forward_each, rows_in,
};
use rli_core::metal::{
    At, AttnParams, Batch, BatchAttnParams, Gpu, GpuBuf, KvBuf, RopeArgs, attention_part_len,
};
use rli_core::{GgmlType, ops};
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
    /// Byte distance between consecutive experts' gate, up and down matrices, when the experts
    /// are evenly spaced slices of stacked tensors (needed for routing on the GPU).
    expert_strides: Option<[usize; 3]>,
}

/// Most prompt tokens processed per GPU pass; bounds the batch scratch buffers.
const BATCH_CHUNK: usize = 512;
const _: () = assert!(BATCH_CHUNK.is_multiple_of(crate::model::PROMPT_SLICE));
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
    /// A chunk's rows gathered for the experts that chose them, sorted by expert.
    gx: GpuBuf,
    /// Three equal parts (see `Metal::moe`): each gathered row's token, each token's gathered
    /// rows (both u32 bits), and the weights. GPU-routed decode keeps ids and weights in halves.
    route: GpuBuf,
    /// Room for this many rows of logits.
    logit_rows: usize,
    logits: GpuBuf,
}

/// Per-session KV cache of the offloaded layers: `cap` positions allocated, grown on demand up
/// to `max_ctx`. f16 unless the model is in precise mode.
struct Kv {
    k: Vec<KvBuf>,
    v: Vec<KvBuf>,
    /// Decode attention partials for up to `cap` positions.
    part: GpuBuf,
    cap: usize,
    max_ctx: usize,
}

impl Kv {
    fn new(m: &Metal, cap: usize, max_ctx: usize) -> Self {
        let alloc = || {
            (0..m.layers.len())
                .map(|_| {
                    m.gpu
                        .alloc_kv(cap.next_multiple_of(8) * m.kv_dim, m.kv_half)
                })
                .collect()
        };
        Kv {
            k: alloc(),
            v: alloc(),
            part: m.gpu.alloc(attention_part_len(m.n_head, m.head_dim, cap)),
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
    /// `vocab` is 0 when the offload doesn't reach the output head.
    fn new(
        gpu: &Gpu,
        c: &Config,
        max_ff: usize,
        vocab: usize,
        cap: usize,
        logit_rows: usize,
    ) -> Self {
        // An MoE layer keeps a row per token and chosen expert.
        let pairs = cap * c.n_expert_used;
        Scratch {
            cap,
            x: gpu.alloc(cap * c.n_embd),
            xb: gpu.alloc(cap * c.n_embd),
            q: gpu.alloc(cap * c.q_dim()),
            k: gpu.alloc(cap * c.kv_dim()),
            v: gpu.alloc(cap * c.kv_dim()),
            att: gpu.alloc(cap * c.q_dim()),
            // GPU-routed decode keeps a row per chosen expert, plus the shared one.
            tmp: gpu.alloc(cap.max(pairs).max(c.n_expert_used + 1) * c.n_embd),
            gate: gpu.alloc(cap.max(pairs) * max_ff),
            up: gpu.alloc(cap.max(pairs) * max_ff),
            router: gpu.alloc(cap * c.n_expert.max(1)),
            shared_gate: gpu.alloc(cap),
            gx: gpu.alloc(pairs * c.n_embd),
            // Each token is in n_expert_used expert lists, plus the shared expert's list.
            route: gpu.alloc(3 * cap * (c.n_expert_used + 1)),
            logit_rows,
            logits: gpu.alloc(logit_rows * vocab),
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
    head_dim: usize,
    /// KV cache stored as f16 and prompt matmuls on half tiles (chosen at attach: f32 when the
    /// model is in precise mode).
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

/// `resid += x W^T` for `n` rows.
fn mm_add(b: &mut Batch<'_>, w: &Weight, x: At<'_>, resid: At<'_>, n: usize) {
    b.matmul_add(w.ty, w.rows, w.cols, w.offset, x, resid, n);
}

/// SwiGLU FFN of `n` rows of `x` into `out`.
fn ffn(b: &mut Batch<'_>, s: &Scratch, e: &Expert, x: At<'_>, out: At<'_>, n: usize) {
    mm(b, &e.gate, x, At(&s.gate, 0), n);
    b.free = true;
    mm(b, &e.up, x, At(&s.up, 0), n);
    b.free = false;
    b.swiglu(At(&s.gate, 0), At(&s.up, 0), n * e.gate.rows);
    mm(b, &e.down, At(&s.gate, 0), out, n);
}

impl Metal {
    /// Runs the offloaded layers on the `n` token rows in `s.x` (positions `pos..pos + n`) and
    /// the output head on the last `rows` of them into `s.logits`. Waits for the GPU.
    fn run(&self, model: &Model, kv: &Kv, s: &Scratch, n: usize, pos: usize, rows: usize) {
        assert!(n >= 1 && n <= s.cap && rows <= n.min(s.logit_rows));
        assert!(rows == 0 || self.covers_all);
        assert!(pos + n <= kv.cap, "KV cache not reserved");
        let c = &model.cfg;
        let (hd, kv_dim, n_embd) = (c.head_dim, c.kv_dim(), c.n_embd);
        let rope = |n_heads, n_tokens, pos| RopeArgs {
            n_tokens,
            n_heads,
            head_dim: hd,
            rope_dim: c.rope_dim,
            pos,
            theta: c.rope_theta,
            neox: c.arch.neox_rope(),
            freq_factors: self.rope_freqs.as_ref(),
        };
        let mut b = self.gpu.batch();
        b.half = self.kv_half;
        for (l, (layer, g)) in model.layers.iter().zip(&self.layers).enumerate() {
            if l == 1 && n == 1 {
                b.start(); // a token's first layer runs while the others are queued
            }
            let (k_at, v_at) = (At(&s.k, 0), At(&s.v, 0));
            // The rows whose queries, attention and FFN this layer computes, `lo..`: all of
            // them, but after the last layer of the model only the rows the logits come from
            // are read, so there the others get just their keys and values.
            let m = if self.covers_all && l + 1 == self.layers.len() {
                rows
            } else {
                n
            };
            let lo = n - m;
            let x_at = At(&s.x, lo * n_embd);
            b.rmsnorm(
                At(&s.x, 0),
                At(&s.xb, 0),
                &g.attn_norm,
                n,
                n_embd,
                c.rms_eps,
            );
            // Q, K and V read the same input and write different buffers, so they (and their
            // biases, which one token's matvec adds itself) run together.
            let qkv = [
                (&layer.wk, k_at, &g.bk, n, 0),
                (&layer.wv, v_at, &g.bv, n, 0),
                (&layer.wq, At(&s.q, 0), &g.bq, m, lo),
            ];
            let qkv = &qkv[..if m == 0 { 2 } else { 3 }];
            for (i, &(w, out, bias, rows, at)) in qkv.iter().enumerate() {
                b.free = i > 0;
                let x = At(&s.xb, at * n_embd);
                match bias {
                    Some(bias) if rows == 1 => {
                        b.matvec_bias(w.ty, w.rows, w.cols, w.offset, x, out, bias.into());
                    }
                    _ => mm(&mut b, w, x, out, rows),
                }
            }
            b.free = false;
            for &(w, out, bias, rows, _) in qkv {
                if let Some(bias) = bias.as_ref().filter(|_| rows > 1) {
                    b.add_bcast(out, bias.into(), w.rows, rows * w.rows);
                    b.free = true; // the first waits for the matmuls
                }
            }
            b.free = false;
            if let Some(w) = g.q_norm.as_ref().filter(|_| m > 0) {
                b.rmsnorm(At(&s.q, 0), At(&s.q, 0), w, m * c.n_head, hd, c.rms_eps);
            }
            if let Some(w) = &g.k_norm {
                b.rmsnorm(k_at, k_at, w, n * c.n_head_kv, hd, c.rms_eps);
            }
            // Three dispatches that touch different buffers, so they run together; rotating K
            // straight into the cache saves a fourth that would have to wait for them.
            if m > 0 {
                b.rope(At(&s.q, 0), &rope(c.n_head, m, pos + lo));
                b.free = true;
            }
            b.store_kv(v_at, &kv.v[l], pos * kv_dim, n * kv_dim);
            b.free = true;
            if c.rope_dim == hd {
                b.rope_store(k_at, &kv.k[l], pos * kv_dim, &rope(c.n_head_kv, n, pos));
                b.free = false;
            } else {
                b.rope(k_at, &rope(c.n_head_kv, n, pos));
                b.free = false;
                b.store_kv(k_at, &kv.k[l], pos * kv_dim, n * kv_dim);
            }
            if m == 0 {
                break;
            }
            // From here on the layer's rows are the `m` from `lo`.
            let n = m;
            if n == 1 {
                let p = AttnParams {
                    n_head: c.n_head as u32,
                    n_kv: c.n_head_kv as u32,
                    head_dim: hd as u32,
                    n_pos: (pos + lo + 1) as u32,
                    kv_dim: kv_dim as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                };
                b.attention(At(&s.q, 0), &kv.k[l], &kv.v[l], At(&s.att, 0), &kv.part, &p);
            } else {
                // The queries are rows `lo..`: after the last layer, only those the logits
                // come from.
                let p = BatchAttnParams {
                    n_head: c.n_head as u32,
                    n_kv: c.n_head_kv as u32,
                    head_dim: hd as u32,
                    pos: (pos + lo) as u32,
                    kv_dim: kv_dim as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                };
                b.attention_batch(At(&s.q, 0), &kv.k[l], &kv.v[l], At(&s.att, 0), n, &p);
            }
            mm_add(&mut b, &layer.wo, At(&s.att, 0), x_at, n);
            b.rmsnorm(x_at, At(&s.xb, 0), &g.ffn_norm, n, n_embd, c.rms_eps);
            match &layer.ffn {
                Ffn::Dense(e)
                    if n == 1 && Batch::has_matvec_swiglu(e.gate.ty) && e.up.ty == e.gate.ty =>
                {
                    let (g, u) = (&e.gate, &e.up);
                    b.matvec_swiglu(
                        g.ty,
                        g.rows,
                        g.cols,
                        g.offset,
                        u.offset,
                        At(&s.xb, 0),
                        At(&s.gate, 0),
                    );
                    mm_add(&mut b, &e.down, At(&s.gate, 0), x_at, n);
                }
                Ffn::Dense(e) => {
                    ffn(&mut b, s, e, At(&s.xb, 0), At(&s.tmp, 0), n);
                    b.add(x_at, At(&s.tmp, 0), n * n_embd);
                }
                Ffn::Moe {
                    router,
                    experts,
                    shared,
                } => match g.expert_strides {
                    Some(strides) if n == 1 => self.moe_decode(
                        model,
                        &mut b,
                        s,
                        x_at,
                        router,
                        experts,
                        shared.as_ref(),
                        strides,
                    ),
                    _ => b = self.moe(model, b, s, x_at, router, experts, shared.as_ref(), n),
                },
            }
        }
        if rows > 0 {
            let last = At(&s.x, (n - rows) * n_embd);
            b.rmsnorm(
                last,
                At(&s.xb, 0),
                &self.output_norm,
                rows,
                n_embd,
                c.rms_eps,
            );
            mm(&mut b, &model.output, At(&s.xb, 0), At(&s.logits, 0), rows);
        }
        b.finish();
    }

    /// Queues one token's MoE FFN over `s.xb`, adding into the row at `x`, routed on the GPU so the
    /// batch needn't wait: the k chosen experts run as one id-indexed matvec per matrix.
    #[allow(clippy::too_many_arguments)]
    fn moe_decode(
        &self,
        model: &Model,
        b: &mut Batch<'_>,
        s: &Scratch,
        x: At<'_>,
        router: &Weight,
        experts: &[Expert],
        shared: Option<&(Weight, Expert)>,
        [gate_stride, up_stride, down_stride]: [usize; 3],
    ) {
        let (c, ne, n_embd) = (&model.cfg, experts.len(), model.cfg.n_embd);
        let k = c.n_expert_used;
        mm(b, router, At(&s.xb, 0), At(&s.router, 0), 1);
        if let Some((gi, _)) = shared {
            mm(b, gi, At(&s.xb, 0), At(&s.shared_gate, 0), 1);
        }
        let (ids, w) = (At(&s.route, 0), At(&s.route, s.route.len() / 2));
        let (rx, sg) = (At(&s.router, 0), At(&s.shared_gate, 0));
        b.moe_topk(rx, sg, ids, w, ne, k, c.norm_topk, shared.is_some());
        let e = &experts[0];
        let ff = e.gate.rows;
        let (xb, gate, up, tmp) = (At(&s.xb, 0), At(&s.gate, 0), At(&s.up, 0), At(&s.tmp, 0));
        let (g, u) = (&e.gate, &e.up);
        if Batch::has_matvec_swiglu(g.ty) && g.ty == u.ty {
            let (at, strides) = ([g.offset, u.offset], [gate_stride, up_stride]);
            b.matvec_swiglu_id(g.ty, ff, g.cols, at, strides, ne, ids, xb, gate, ff, k);
        } else {
            for (m, stride, out) in [(&e.gate, gate_stride, gate), (&e.up, up_stride, up)] {
                b.matvec_id(
                    m.ty, m.rows, m.cols, m.offset, stride, ne, ids, xb, 0, out, ff, k,
                );
            }
            b.swiglu(gate, up, k * ff);
        }
        let d = &e.down;
        // The chosen experts' outputs, weighted, into the residual stream: one dispatch where
        // the type has the kernel, else a matvec per slot and the sum.
        let fused = Batch::has_matvec_swiglu(d.ty);
        if fused {
            b.matvec_id_add(
                d.ty,
                d.rows,
                d.cols,
                d.offset,
                down_stride,
                ne,
                ids,
                w,
                gate,
                ff,
                x,
                k,
            );
        } else {
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
        }
        if let Some((_, e)) = shared {
            ffn(b, s, e, xb, At(&s.tmp, k * n_embd), 1);
        }
        let n_shared = usize::from(shared.is_some());
        if !fused {
            b.moe_combine(x, tmp, w, k + n_shared, n_embd);
        } else if n_shared == 1 {
            let (src, w) = (At(&s.tmp, k * n_embd), At(w.0, w.1 + k));
            b.moe_combine(x, src, w, 1, n_embd);
        }
    }

    /// Queues an MoE FFN over `n` rows of `s.xb`, adding into the rows at `x`. Routing needs the router
    /// output on the CPU, so this waits for `b` once and returns a fresh batch to continue in.
    #[allow(clippy::too_many_arguments)]
    fn moe<'g>(
        &'g self,
        model: &Model,
        b: Batch<'g>,
        s: &Scratch,
        x: At<'_>,
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
        let (k, ff) = (c.n_expert_used, experts[0].gate.rows);
        let mut lists: Vec<Vec<(u32, f32)>> = vec![Vec::new(); ne];
        for (t, row) in router_out.chunks_exact(ne).enumerate() {
            let mut probs = row.to_vec();
            ops::softmax(&mut probs);
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            let chosen = &order[..k];
            let norm = if c.norm_topk {
                chosen.iter().map(|&e| probs[e]).sum::<f32>()
            } else {
                1.0
            };
            for &e in chosen {
                lists[e].push((t as u32, probs[e] / norm));
            }
        }
        // The (token, expert) pairs sorted by expert: pair p is a row of the scratch buffers.
        // `idx[p]` is its token; token t's k pairs and their weights are at `t * k..`.
        let pairs = n * k;
        let mut idx: Vec<u32> = Vec::with_capacity(pairs + n);
        let (mut ids, mut wts) = (vec![0u32; pairs], vec![0.0f32; pairs]);
        let mut filled = vec![0; n];
        for &(t, w) in lists.iter().flatten() {
            let slot = t as usize * k + filled[t as usize];
            (ids[slot], wts[slot]) = (idx.len() as u32, w);
            filled[t as usize] += 1;
            idx.push(t);
        }
        if shared.is_some() {
            idx.extend(0..n as u32);
            wts.extend(gates.iter().map(|g| 1.0 / (1.0 + (-g).exp())));
        }
        let r = s.route.len() / 3;
        // SAFETY: nothing is in flight.
        unsafe {
            s.route.write_u32(0, &idx);
            s.route.write_u32(r, &ids);
            s.route.as_mut_slice()[2 * r..2 * r + wts.len()].copy_from_slice(&wts);
        }
        let mut b = self.gpu.batch();
        b.half = self.kv_half;
        // A dispatch that waits costs more than a few rows' matmul, and there are three per
        // expert: so one gather, every expert's gate and up side by side, one activation,
        // every down, one combine. Each expert reads and writes its own rows.
        b.gather_rows(At(&s.xb, 0), At(&s.gx, 0), At(&s.route, 0), pairs, n_embd);
        let mut runs = Vec::new(); // each used expert and its rows: (expert, first, count)
        for (e, list) in experts.iter().zip(&lists).filter(|(_, l)| !l.is_empty()) {
            assert_eq!(e.gate.rows, ff);
            let off = runs.last().map_or(0, |&(_, off, m)| off + m);
            runs.push((e, off, list.len()));
        }
        for &(e, off, m) in &runs {
            let x = At(&s.gx, off * n_embd);
            mm(&mut b, &e.gate, x, At(&s.gate, off * ff), m);
            b.free = true;
            mm(&mut b, &e.up, x, At(&s.up, off * ff), m);
        }
        b.free = false;
        b.swiglu(At(&s.gate, 0), At(&s.up, 0), pairs * ff);
        for &(e, off, m) in &runs {
            let (x, y) = (At(&s.gate, off * ff), At(&s.tmp, off * n_embd));
            mm(&mut b, &e.down, x, y, m);
            b.free = true;
        }
        b.free = false;
        let (ids, w) = (At(&s.route, r), At(&s.route, 2 * r));
        b.moe_combine_rows(x, At(&s.tmp, 0), ids, w, n, n_embd, k);
        if let Some((_, e)) = shared {
            ffn(&mut b, s, e, At(&s.xb, 0), At(&s.tmp, 0), n);
            let (i, w) = (At(&s.route, pairs), At(&s.route, 2 * r + pairs));
            b.scatter_axpy(x, At(&s.tmp, 0), i, w, n, n_embd);
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
        let logits = logits.filter(|_| self.covers_all);
        let rows = logits.as_ref().map_or(0, |l| l.len() / model.output.rows);
        // SAFETY: callers hold the scratch lock and no GPU work is in flight.
        unsafe { s.x.as_mut_slice()[..xs.len()].copy_from_slice(xs) };
        self.run(model, kv, s, n, pos, rows);
        // SAFETY: `run` waited for the GPU.
        unsafe {
            xs.copy_from_slice(&s.x.as_slice()[..xs.len()]);
            if let Some(out) = logits {
                out.copy_from_slice(&s.logits.as_slice()[..out.len()]);
                return true;
            }
        }
        false
    }
}

impl Accelerator for Metal {
    fn name(&self) -> &str {
        self.gpu.name()
    }

    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn prompt_slice(&self) -> usize {
        BATCH_CHUNK
    }

    fn logit_rows(&self) -> usize {
        usize::MAX
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
        logits: Option<&mut [f32]>,
    ) -> bool {
        let n_embd = model.cfg.n_embd;
        if !self.batchable || xs.len() == n_embd {
            return forward_each(self, model, state, xs, pos, logits);
        }
        let kv: &mut Kv = state
            .downcast_mut()
            .expect("state created by this accelerator");
        let n = xs.len() / n_embd;
        kv.reserve(self, pos + n);
        let vocab = model.output.rows;
        let mut logits = logits.filter(|_| self.covers_all);
        let rows = logits.as_ref().map_or(0, |l| l.len() / vocab);
        let mut guard = self.batch.lock().unwrap_or_else(|p| p.into_inner());
        // Sized for the longest chunk seen: an MoE layer's rows per token and chosen expert
        // come to 92 MB for 512 tokens of Qwen3-30B-A3B. Logits likewise, for the most rows
        // asked for (152k floats a row).
        let need = n.min(BATCH_CHUNK);
        let need_rows = rows.clamp(1, BATCH_CHUNK);
        if guard
            .as_ref()
            .is_none_or(|s| s.cap < need || s.logit_rows < need_rows)
        {
            let (cap, have) = guard.as_ref().map_or((0, 0), |s| (s.cap, s.logit_rows));
            let vocab = if self.covers_all { vocab } else { 0 };
            *guard = Some(Scratch::new(
                &self.gpu,
                &model.cfg,
                self.max_ff,
                vocab,
                need.max(cap),
                need_rows.max(have),
            ));
        }
        let s = guard.as_ref().expect("allocated above");
        let mut done = false;
        for (i, chunk) in xs.chunks_mut(BATCH_CHUNK * n_embd).enumerate() {
            let start = i * BATCH_CHUNK;
            let r = rows_in(start, start + chunk.len() / n_embd, n, rows);
            let l = match logits.take() {
                Some(l) if r > 0 => {
                    let (this, rest) = l.split_at_mut(r * vocab);
                    logits = Some(rest);
                    Some(this)
                }
                l => {
                    logits = l;
                    None
                }
            };
            done = self.pass(model, kv, s, chunk, pos + start, l);
        }
        done
    }
}

/// The matrices one layer's offload reads.
fn layer_weights(layer: &crate::model::Layer) -> Vec<&Weight> {
    let mut weights = vec![&layer.wq, &layer.wk, &layer.wv, &layer.wo];
    if let Ffn::Moe { router, shared, .. } = &layer.ffn {
        weights.push(router);
        weights.extend(shared.as_ref().map(|(g, _)| g));
    }
    for e in experts_of(&layer.ffn) {
        weights.extend([&e.gate, &e.up, &e.down]);
    }
    weights
}

/// The matrices the GPU reads when the first `n` layers are offloaded (plus the output head
/// when that is all of them).
fn offloaded_weights(model: &Model, n: usize) -> Vec<&Weight> {
    let mut weights: Vec<&Weight> = model.layers[..n].iter().flat_map(layer_weights).collect();
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

/// The file ranges the GPU wraps when the first `n` layers are offloaded: one per layer (so a
/// stacked expert tensor is never split across buffers) and the output head with all of them.
fn offload_units(model: &Model, n: usize) -> Vec<Range<usize>> {
    let mut units: Vec<Range<usize>> = model.layers[..n]
        .iter()
        .map(|l| span(&layer_weights(l)))
        .collect();
    if n == model.cfg.n_layer {
        units.push(span(&[&model.output]));
    }
    units
}

/// Room left in Metal's recommended working set for the KV cache, scratch buffers and the
/// window server when choosing how many layers to offload.
const WORKING_SET_MARGIN: usize = 1 << 30;

/// The most leading layers whose weights fit Metal's recommended working set (less
/// [`WORKING_SET_MARGIN`]) in buffers of at most `maxBufferLength`, or `None` without a device.
pub fn fitting_layers(model: &Model) -> Option<usize> {
    use rli_core::metal::{buffer_spans, max_buffer_len, working_set_len, wrapped_len};
    let (max, budget) = (
        max_buffer_len()?,
        working_set_len()?.saturating_sub(WORKING_SET_MARGIN),
    );
    let fits = |n| {
        buffer_spans(&offload_units(model, n), max)
            .is_ok_and(|spans| spans.iter().map(wrapped_len).sum::<usize>() <= budget)
    };
    Some(
        (0..=model.cfg.n_layer)
            .rev()
            .find(|&n| fits(n))
            .unwrap_or(0),
    )
}

pub fn attach(model: &mut Model, layers: usize) -> Result<(), BackendError> {
    attach_with(model, layers, None)
}

/// [`attach`] with the weights wrapped in buffers of at most `max_buffer` bytes, so tests can
/// split a small model across several buffers.
#[doc(hidden)]
pub fn attach_capped(
    model: &mut Model,
    layers: usize,
    max_buffer: usize,
) -> Result<(), BackendError> {
    attach_with(model, layers, Some(max_buffer))
}

fn attach_with(model: &mut Model, layers: usize, cap: Option<usize>) -> Result<(), BackendError> {
    let c = model.cfg.clone();
    let n = layers.min(c.n_layer);
    let covers_all = n == c.n_layer;
    let weights = offloaded_weights(model, n);
    let units = offload_units(model, n);
    let mapped = model.gguf().mapped_bytes();
    // SAFETY: the GPU views the model's own mapping, and `Model` drops its accelerator before
    // the mapping (field order), so the view never outlives it.
    let mut gpu = match cap {
        None => unsafe { Gpu::new(mapped, &units) },
        Some(max) => unsafe { Gpu::new_capped(mapped, &units, max) },
    }
    .map_err(BackendError)?;
    for w in weights.iter().filter(|w| w.ty == GgmlType::NVFP4) {
        gpu.set_scale(w.offset, w.scale);
    }
    if let Some(w) = weights
        .iter()
        .find(|w| !gpu.supports(w.ty) || w.cols % 32 != 0)
    {
        return Err(BackendError(format!(
            "{:?} weights ({} cols) have no Metal kernel; run more layers on the CPU (lower --gpu-layers)",
            w.ty, w.cols
        )));
    }
    // `moe_topk` marks the experts it took in one 64-bit word per lane.
    if c.n_expert > 32 * 64 {
        return Err(BackendError(format!(
            "{} experts; Metal routes at most 2048",
            c.n_expert
        )));
    }
    // The widest FFN among the layers. Not the output head, which has n_embd columns too:
    // counting its rows made the gate and up scratch a chunk of vocabulary-sized rows each
    // (622 MB on Qwen3-30B-A3B).
    let max_ff = model.layers[..n]
        .iter()
        .flat_map(layer_weights)
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
            // Experts of a type without id kernels (NVFP4, BF16) take the CPU-routed path.
            expert_strides: match &l.ffn {
                Ffn::Moe { experts, .. }
                    if [experts[0].gate, experts[0].up, experts[0].down]
                        .iter()
                        .all(|w| gpu.has_matvec_id(w.ty)) =>
                {
                    expert_strides(experts)
                }
                _ => None,
            },
        })
        .collect();
    let logits = if covers_all { model.output.rows } else { 0 };
    let scratch = Scratch::new(&gpu, &c, max_ff, logits, 1, 1);
    let batchable = c.head_dim.is_multiple_of(8) && c.head_dim <= 256;
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
        head_dim: c.head_dim,
        scratch: Mutex::new(scratch),
        gpu,
    };
    model.set_accelerator(Box::new(accel));
    Ok(())
}
