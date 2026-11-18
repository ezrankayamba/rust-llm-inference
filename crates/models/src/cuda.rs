//! CUDA accelerator: the same layer program as `metal.rs`, but for a discrete GPU. Only the
//! offloaded layers' weights are copied to VRAM (partial offload for low-VRAM cards); the rest
//! run on the CPU from the file mapping. One stream sync per token: MoE layers pick their
//! experts on the GPU (experts stored as separate tensors are routed on the CPU instead, with a
//! sync per layer). Prompts run in chunks of up to [`BATCH_CHUNK`] tokens per pass (matmul
//! instead of matvec, one sync per chunk), unless an offloaded layer is routed on the CPU.

use crate::backend::BackendError;
use crate::model::{Accelerator, Expert, Ffn, Model, Weight, expert_strides, forward_each};
use rli_core::cuda::{
    ATTN_THREADS, At, AttnParams, Batch, GEMM_MIN_TOKENS, Gpu, GpuBuf, GpuHalf, GpuIds, GpuQ,
    Graph, MMQ_MIN_TOKENS, MOE_MAX_EXPERTS, RopeArgs,
};
use rli_core::{GgmlType, ops};
use std::any::Any;
use std::sync::Mutex;

/// Prompt tokens per batched pass. A matrix is decoded once per pass, whatever the tokens:
/// 512 against 256 took prompts from 12.4k to 14.1k tok/s on Llama-3.2-1B, 2720 to 3540 on
/// Mistral-7B and 3115 to 3940 on Qwen3-30B-A3B (A100).
const BATCH_CHUNK: usize = 512;
const _: () = assert!(BATCH_CHUNK.is_multiple_of(crate::model::PROMPT_SLICE));
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
    /// A chunk's (token, expert) pairs grouped by expert, and where each expert's begin.
    perm: GpuIds,
    starts: GpuIds,
    route_w: GpuBuf,
    /// A matmul's input, quantized to int8.
    xq: GpuQ,
    /// For the f16 GEMM of long prompts: a matmul's input, and its matrix written out.
    xh: GpuHalf,
    wh: GpuHalf,
    logits: GpuBuf,
}

impl Scratch {
    /// `max_ff`: the widest feed-forward activation; `max_w`: the most weights in a matrix.
    fn new(
        gpu: &Gpu,
        c: &crate::config::Config,
        (max_ff, max_w): (usize, usize),
        logits: usize,
        n: usize,
    ) -> Self {
        // An MoE token holds its k chosen experts' activations side by side; their outputs are
        // followed by the shared expert's, one row per token.
        let k = c.n_expert_used;
        let (gemm, half) = (n >= GEMM_MIN_TOKENS, n >= MMQ_MIN_TOKENS);
        Self {
            x: gpu.alloc(n * c.n_embd),
            xb: gpu.alloc(n * c.n_embd),
            q: gpu.alloc(n * c.q_dim()),
            att: gpu.alloc(n * c.q_dim()),
            tmp: gpu.alloc(n * (k + 1) * c.n_embd),
            // A dense FFN keeps its up activations after its gate ones.
            gate: gpu.alloc(n * k.max(2) * max_ff),
            up: gpu.alloc(n * k.max(1) * max_ff),
            router: gpu.alloc(n * c.n_expert.max(1)),
            shared_gate: gpu.alloc(n),
            route_ids: gpu.alloc_ids(n * k),
            perm: gpu.alloc_ids(n * k),
            starts: gpu.alloc_ids(c.n_expert + 1),
            route_w: gpu.alloc(n * (k + 1)),
            xq: gpu.alloc_q(n * c.n_embd.max(c.q_dim()).max(k.max(1) * max_ff)),
            xh: gpu.alloc_half(if half {
                n * c.n_embd.max(c.q_dim()).max(k.max(1) * max_ff)
            } else {
                0
            }),
            wh: gpu.alloc_half(if gemm { max_w } else { 0 }),
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
    /// The launches of one decoded token. Declared before `gpu`, so that it is dropped while
    /// the device context is still there.
    graph: Mutex<Option<Graph>>,
    gpu: Gpu,
    layers: Vec<GpuLayer>,
    output_norm: GpuBuf,
    rope_freqs: Option<GpuBuf>,
    covers_all: bool,
    /// No offloaded layer is routed on the CPU, so prompts can run as batches.
    batchable: bool,
    /// The widest feed-forward activation and the largest matrix, for [`Scratch::new`].
    sizes: (usize, usize),
    kv_dim: usize,
    n_head: usize,
    head_dim: usize,
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

/// Fewest tokens of a chunk whose expert matrices run grouped by expert. Qwen3-30B-A3B (8 of
/// 128 experts a token) on an A100: 16 tokens 621 tok/s a pair at a time and 594 grouped,
/// 20 tokens 650 and 686, 32 tokens 740 and 997.
const MOE_GROUP_TOKENS: usize = 20;
/// The same for the down projection on f16 tensor cores. It has the most rows and the
/// fewest columns, so a matvec per pair is mostly its per-row work: Qwen3-30B-A3B at 16
/// tokens 661 tok/s -> 727 grouped, at 12 tokens 608 -> 625, at 8 tokens 502 -> 475.
const MOE_GROUP_DOWN_TOKENS: usize = 12;

/// Most tokens of a chunk whose f32 router runs as a matvec per token (a block of 8 warps
/// per row and token) and not as the matmul, whose warp walks a row's columns alone:
/// Qwen3-30B-A3B prompts 9% faster at 8 tokens, 6-7% at 16 and 32, level at 128, 4.5% slower
/// at 512.
const ROUTER_MATVEC_TOKENS: usize = 64;

/// Whether any of `ws` runs in int8, so that their input is wanted quantized.
fn wants_q(b: &Batch<'_>, ws: &[&Weight], int8: bool) -> bool {
    int8 && ws.iter().any(|w| b.has_int8(w.ty, w.cols, w.offset))
}

/// What the kernel that produced a matmul's input wrote besides the f32: nothing, the int8
/// activations in `Scratch::xq`, or f16 in `Scratch::xh`. Whatever is missing, the matmul
/// makes with a launch of its own.
#[derive(Clone, Copy, PartialEq)]
enum Fed {
    No,
    Q,
    Half,
}

impl Fed {
    fn q(self, s: &Scratch) -> Option<&GpuQ> {
        (self == Fed::Q).then_some(&s.xq)
    }

    fn half(self, s: &Scratch) -> Option<&GpuHalf> {
        (self == Fed::Half).then_some(&s.xh)
    }
}

/// What [`mm_each`] will read the input of `ws` as, for `n` tokens.
fn fed(b: &Batch<'_>, ws: &[&Weight], n: usize, int8: bool) -> Fed {
    let float = |w: &&&Weight| matches!(w.ty, GgmlType::F32 | GgmlType::F16);
    let dense = |w: &&&Weight| b.has_gemm_dense(w.ty, w.cols, w.offset, n);
    let rest: Vec<_> = ws.iter().filter(|w| !dense(w)).collect();
    if int8 && (rest.is_empty() || (!rest.iter().any(|w| float(w)) && b.has_gemm(ws[0].cols, n))) {
        Fed::Half
    } else if wants_q(b, ws, int8) {
        Fed::Q
    } else {
        Fed::No
    }
}

/// [`fed`] for [`mm_add`].
fn fed_add(b: &Batch<'_>, w: &Weight, n: usize, int8: bool) -> Fed {
    let dense = int8 && b.has_gemm_dense(w.ty, w.cols, w.offset, n);
    let gemm = int8 && !matches!(w.ty, GgmlType::F32 | GgmlType::F16) && b.has_gemm(w.cols, n);
    let q = n < MMQ_MIN_TOKENS || (b.has_matmul_q_add() && !gemm);
    if !dense && q && wants_q(b, &[w], int8) {
        Fed::Q
    } else {
        fed(b, &[w], n, int8)
    }
}

/// Each of `ws` applied to the `n` tokens at `x`, into the matching `ys`. With `int8`, the
/// matrices that have an int8 kernel run over `s.xq`: `x` quantized, by the kernel that
/// produced it (`fed`) or else here.
#[allow(clippy::too_many_arguments)]
fn mm_each(
    b: &mut Batch<'_>,
    s: &Scratch,
    ws: &[&Weight],
    x: At<'_>,
    ys: &[At<'_>],
    n: usize,
    int8: bool,
    fed: Fed,
) {
    // A long prompt's matmuls run on f16 tensor cores, whatever quantized type the weights
    // are. F32 and F16 weights are the reference path and keep f32 activations.
    let float = |w: &&Weight| matches!(w.ty, GgmlType::F32 | GgmlType::F16);
    // Shorter ones too, for the matrices with a kernel that decodes the weights on the way
    // into the tensor cores. Its blocks add into y, which the launch that converts x zeroes.
    let dense = |w: &Weight| int8 && b.has_gemm_dense(w.ty, w.cols, w.offset, n);
    let (half, rest): (Vec<_>, Vec<_>) = ws.iter().zip(ys).partition(|(w, _)| dense(w));
    if !half.is_empty() {
        let zs: Vec<_> = half.iter().map(|&(w, &y)| (y, n * w.rows)).collect();
        let convert = if fed == Fed::Half { 0 } else { n * ws[0].cols };
        b.to_f16_zero(x, convert, &s.xh, &zs);
        for &(w, &y) in &half {
            b.gemm_dense(w.ty, w.rows, w.cols, w.offset, &s.xh, y, n, true);
        }
    }
    if rest.is_empty() {
        return;
    }
    let (ws, ys): (Vec<&Weight>, Vec<At<'_>>) = rest.into_iter().map(|(&w, &y)| (w, y)).unzip();
    let (ws, ys) = (&ws[..], &ys[..]);
    if int8 && !ws.iter().any(float) && b.has_gemm(ws[0].cols, n) {
        if fed != Fed::Half {
            b.to_f16(x, n * ws[0].cols, &s.xh);
        }
        for (w, &y) in ws.iter().zip(ys) {
            b.dequant(w.ty, w.rows, w.cols, w.offset, &s.wh);
            b.gemm(w.rows, w.cols, &s.wh, &s.xh, y, n);
        }
        return;
    }
    if fed != Fed::Q && wants_q(b, ws, int8) {
        b.quantize(x, n * ws[0].cols, &s.xq);
    }
    // Where the int8 matmul adds its parts into y, one launch zeroes every result first
    // (the launch per matrix it would make was 2-3% of a 16-token prompt).
    let adds = n >= MMQ_MIN_TOKENS && b.has_matmul_q_add();
    if adds {
        let q = |(w, _): &(&&Weight, &At<'_>)| wants_q(b, &[**w], int8);
        let zs: Vec<_> = ws
            .iter()
            .zip(ys)
            .filter(q)
            .map(|(w, &y)| (y, n * w.rows))
            .collect();
        if !zs.is_empty() {
            b.to_f16_zero(x, 0, &s.xh, &zs);
        }
    }
    for (w, &y) in ws.iter().zip(ys) {
        if wants_q(b, &[w], int8) {
            b.matmul_q(w.ty, w.rows, w.cols, w.offset, &s.xq, y, n, adds);
        } else {
            mm(b, w, x, y, n);
        }
    }
}

/// `x += w * src` for the `n` tokens at `src`, in one launch where the int8 matvec runs;
/// otherwise through `s.tmp`. `int8` and `quantized` as in [`mm_each`].
#[allow(clippy::too_many_arguments)]
fn mm_add(
    b: &mut Batch<'_>,
    s: &Scratch,
    w: &Weight,
    src: At<'_>,
    x: At<'_>,
    n: usize,
    int8: bool,
    fed: Fed,
) {
    if int8 && b.has_gemm_dense(w.ty, w.cols, w.offset, n) {
        if fed != Fed::Half {
            b.to_f16(src, n * w.cols, &s.xh);
        }
        b.gemm_dense(w.ty, w.rows, w.cols, w.offset, &s.xh, x, n, true);
        return;
    }
    let gemm = int8 && !matches!(w.ty, GgmlType::F32 | GgmlType::F16) && b.has_gemm(w.cols, n);
    if (n < MMQ_MIN_TOKENS || (b.has_matmul_q_add() && !gemm)) && wants_q(b, &[w], int8) {
        if fed != Fed::Q {
            b.quantize(src, n * w.cols, &s.xq);
        }
        b.matmul_q(w.ty, w.rows, w.cols, w.offset, &s.xq, x, n, true);
    } else if int8
        && !matches!(w.ty, GgmlType::F32 | GgmlType::F16)
        && b.has_gemm_add(w.rows, w.cols, n)
    {
        // As in `mm_each`, by the kernel that adds.
        if fed != Fed::Half {
            b.to_f16(src, n * w.cols, &s.xh);
        }
        b.dequant(w.ty, w.rows, w.cols, w.offset, &s.wh);
        b.gemm_add(w.rows, w.cols, &s.wh, &s.xh, x, n);
    } else {
        let tmp = At(&s.tmp, 0);
        mm_each(b, s, &[w], src, &[tmp], n, int8, fed);
        b.add(x, tmp, n * w.rows);
    }
}

/// Queues the FFN `e` over the `n` tokens in `s.xb` (`quantized`: in `s.xq` too), into `out`,
/// or added to `out` with `add`.
#[allow(clippy::too_many_arguments)]
fn ffn(
    b: &mut Batch<'_>,
    s: &Scratch,
    e: &Expert,
    out: At<'_>,
    n: usize,
    int8: bool,
    fed: Fed,
    add: bool,
) {
    let (g, u, ff) = (&e.gate, &e.up, e.gate.rows);
    let (xb, gate, up) = (At(&s.xb, 0), At(&s.gate, 0), At(&s.gate, n * ff));
    // One token's gate and up are one launch when they are alike.
    if n == 1 && wants_q(b, &[g], int8) && wants_q(b, &[u], int8) && g.ty == u.ty {
        if fed != Fed::Q {
            b.quantize(xb, g.cols, &s.xq);
        }
        b.matvec_q_pair(g.ty, ff, g.cols, [g.offset, u.offset], &s.xq, gate);
    } else {
        mm_each(b, s, &[g, u], xb, &[gate, up], n, int8, fed);
    }
    let q = match add {
        true => fed_add(b, &e.down, n, int8),
        false => self::fed(b, &[&e.down], n, int8),
    };
    b.swiglu(gate, up, n * ff, q.q(s), q.half(s));
    if add {
        mm_add(b, s, &e.down, gate, out, n, int8, q);
    } else {
        mm_each(b, s, &[&e.down], gate, &[out], n, int8, q);
    }
}

/// Queues the MoE FFN of `n` tokens over `s.xb`, adding into the rows at `x`. The experts are picked on
/// the GPU, so nothing waits, and every token's k chosen ones run as one id-indexed matvec per
/// matrix: in int8 where the matrix has that kernel and `int8` is set, over `s.xq`, which holds
/// `s.xb` quantized if `quantized`.
#[allow(clippy::too_many_arguments)]
fn moe(
    c: &crate::config::Config,
    b: &mut Batch<'_>,
    s: &Scratch,
    x: At<'_>,
    router: &Weight,
    experts: &[Expert],
    shared: Option<&(Weight, Expert)>,
    [gate_stride, up_stride, down_stride]: [usize; 3],
    n: usize,
    int8: bool,
    quantized: bool,
) {
    let (ne, k, n_embd) = (experts.len(), c.n_expert_used, c.n_embd);
    let (xb, ids) = (At(&s.xb, 0), &s.route_ids);
    if n <= ROUTER_MATVEC_TOKENS && matches!(router.ty, GgmlType::F32 | GgmlType::F16) {
        // One matrix, so the ids (whatever the last layer left) are multiplied by no stride.
        let (r, o) = (router, At(&s.router, 0));
        b.matvec_id(
            r.ty, r.rows, r.cols, r.offset, 0, 1, ids, xb, r.cols, 1, o, r.rows, n,
        );
    } else {
        mm(b, router, xb, At(&s.router, 0), n);
    }
    if let Some((gi, _)) = shared {
        mm(b, gi, xb, At(&s.shared_gate, 0), n);
    }
    b.moe_topk(
        &s.router,
        &s.shared_gate,
        ids,
        &s.route_w,
        ne,
        k,
        c.norm_topk,
        shared.is_some(),
        n,
    );
    let e = &experts[0];
    let ff = e.gate.rows;
    let (gate, up, tmp) = (At(&s.gate, 0), At(&s.up, 0), At(&s.tmp, 0));
    if !quantized && wants_q(b, &[&e.gate, &e.up], int8) {
        b.quantize(xb, n * n_embd, &s.xq);
    }
    // In decode, stacks of one type and stride run as one launch. Over a prompt chunk that was
    // 1.5% slower than two.
    // A long chunk's (token, expert) pairs are grouped by expert, whose weights are then
    // decoded once for all of its tokens.
    // Q4_K stacks then go through f16 tensor cores, where the GPU has them.
    let grouped = n >= MOE_GROUP_TOKENS;
    // The down projection is grouped from fewer tokens: see MOE_GROUP_DOWN_TOKENS.
    let d = &e.down;
    let down_half =
        n >= MOE_GROUP_DOWN_TOKENS && int8 && b.has_gemm_id(d.ty, d.cols, d.offset, down_stride);
    if (grouped || down_half) && wants_q(b, &[&e.gate, &e.up, &e.down], int8) {
        b.moe_group(ids, &s.perm, &s.starts, n * k, ne);
    }
    let groups = (&s.perm, &s.starts);
    let half = |b: &Batch<'_>, m: &Weight, stride| {
        grouped && int8 && b.has_gemm_id(m.ty, m.cols, m.offset, stride)
    };
    if half(b, &e.gate, gate_stride) || half(b, &e.up, up_stride) {
        b.to_f16(xb, n * n_embd, &s.xh);
    }
    let paired = n < MMQ_MIN_TOKENS
        && e.gate.ty == e.up.ty
        && gate_stride == up_stride
        && wants_q(b, &[&e.gate, &e.up], int8);
    let stacks = [(&e.gate, gate_stride, gate), (&e.up, up_stride, up)];
    for (m, stride, out) in &stacks[..2 - paired as usize] {
        let (m, stride, out) = (*m, *stride, *out);
        let (ty, at) = (m.ty, m.offset);
        if half(b, m, stride) {
            b.gemm_id(ty, ff, n_embd, at, stride, ne, groups, &s.xh, k, out, n * k);
        } else if grouped && wants_q(b, &[m], int8) {
            b.matmul_q_id(ty, ff, n_embd, at, stride, ne, groups, &s.xq, k, out, n * k);
        } else if wants_q(b, &[m], int8) {
            let xq = &s.xq;
            let second = paired.then_some((e.up.offset, up));
            b.matvec_q_id(
                ty,
                ff,
                n_embd,
                at,
                stride,
                ne,
                ids,
                xq,
                n_embd,
                k,
                out,
                ff,
                n * k,
                false,
                second,
            );
        } else {
            b.matvec_id(
                ty,
                ff,
                n_embd,
                at,
                stride,
                ne,
                ids,
                xb,
                n_embd,
                k,
                out,
                ff,
                n * k,
            );
        }
    }
    let (ty, at) = (d.ty, d.offset);
    if down_half {
        b.swiglu(gate, up, n * k * ff, None, None);
        b.to_f16(gate, n * k * ff, &s.xh);
        b.gemm_id(
            ty,
            n_embd,
            ff,
            at,
            down_stride,
            ne,
            groups,
            &s.xh,
            1,
            tmp,
            n * k,
        );
    } else if wants_q(b, &[d], int8) {
        b.swiglu(gate, up, n * k * ff, Some(&s.xq), None);
        let xq = &s.xq;
        if grouped {
            b.matmul_q_id(
                ty,
                n_embd,
                ff,
                at,
                down_stride,
                ne,
                groups,
                xq,
                1,
                tmp,
                n * k,
            );
        } else {
            b.matvec_q_id(
                ty,
                n_embd,
                ff,
                at,
                down_stride,
                ne,
                ids,
                xq,
                ff,
                1,
                tmp,
                n_embd,
                n * k,
                false,
                None,
            );
        }
    } else {
        b.swiglu(gate, up, n * k * ff, None, None);
        b.matvec_id(
            ty,
            n_embd,
            ff,
            at,
            down_stride,
            ne,
            ids,
            gate,
            ff,
            1,
            tmp,
            n_embd,
            n * k,
        );
    }
    let w_stride = k + usize::from(shared.is_some());
    b.moe_combine(x, tmp, At(&s.route_w, 0), w_stride, k, n_embd, n);
    if let Some((_, e)) = shared {
        let out = At(&s.tmp, n * k * n_embd);
        ffn(b, s, e, out, n, false, Fed::No, false);
        b.moe_combine(x, out, At(&s.route_w, k), w_stride, 1, n_embd, n);
    }
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
        // Matrices with an int8 kernel multiply in int8, as the CPU does; `precise` keeps every
        // product in f32.
        let int8 = !model.precise;

        let rope = RopeArgs {
            n_tok: n,
            n_heads: c.n_head,
            n_heads2: c.n_head_kv,
            head_dim: hd,
            rope_dim: c.rope_dim,
            pos,
            theta: c.rope_theta,
            neox: c.arch.neox_rope(),
            freq_factors: self.rope_freqs.as_ref(),
        };
        let want_logits = self.covers_all && logits.is_some();
        let queue = |b: &mut Batch<'_>| {
            for (l, (layer, g)) in model.layers.iter().zip(&self.layers).enumerate() {
                let (k_at, v_at) = (At(&kv.k[l], pos * kv_dim), At(&kv.v[l], pos * kv_dim));
                let (xb, eps) = (At(&s.xb, 0), c.rms_eps);
                // After the last layer of the model only the row the logits come from is read:
                // there the rows before `lo` get their keys and values (and, since the kernels
                // take queries and keys together, their attention), but no output projection
                // and no FFN, which are nine tenths of a layer.
                let m = if self.covers_all && l + 1 == self.layers.len() {
                    usize::from(want_logits)
                } else {
                    n
                };
                let lo = n - m;
                let x_at = At(&s.x, lo * n_embd);
                // With no row to finish, the queries are left as they were: nothing reads
                // what the rotation and the attention below make of them.
                let qkv = [&layer.wq, &layer.wk, &layer.wv];
                let outs = [At(&s.q, 0), k_at, v_at];
                let skip = usize::from(m == 0);
                let (qkv, outs) = (&qkv[skip..], &outs[skip..]);
                // A kernel whose result feeds int8 matrices quantizes it on the way out.
                let f = fed(b, qkv, n, int8);
                b.rmsnorm(
                    At(&s.x, 0),
                    xb,
                    &g.attn_norm,
                    n,
                    n_embd,
                    eps,
                    f.q(s),
                    f.half(s),
                );
                mm_each(b, s, qkv, xb, outs, n, int8, f);
                for (t, bias, len) in [
                    (At(&s.q, 0), &g.bq, c.q_dim()),
                    (k_at, &g.bk, kv_dim),
                    (v_at, &g.bv, kv_dim),
                ] {
                    if let Some(bias) = bias {
                        b.add_bias(t, bias, n * len);
                    }
                }
                match (&g.q_norm, &g.k_norm) {
                    (Some(wq), Some(wk)) if Batch::has_norm_rope(hd) => {
                        b.norm_rope(At(&s.q, 0), k_at, wq, wk, eps, &rope);
                    }
                    (wq, wk) => {
                        if let Some(w) = wq {
                            b.rmsnorm(
                                At(&s.q, 0),
                                At(&s.q, 0),
                                w,
                                n * c.n_head,
                                hd,
                                eps,
                                None,
                                None,
                            );
                        }
                        if let Some(w) = wk {
                            b.rmsnorm(k_at, k_at, w, n * c.n_head_kv, hd, eps, None, None);
                        }
                        b.rope(At(&s.q, 0), k_at, &rope);
                    }
                }
                let p = AttnParams {
                    n_head: c.n_head as u32,
                    n_kv: c.n_head_kv as u32,
                    head_dim: hd as u32,
                    n_tok: n as u32,
                    n_pos: (pos + 1) as u32,
                    kv_dim: kv_dim as u32,
                    max_ctx: kv.max_ctx as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                    half: int8,
                };
                if m == 0 {
                    break;
                }
                // The attention's quantized copy starts at row 0, so it serves only when the
                // output projection takes every row.
                let f = match lo == 0 && hd.is_multiple_of(32) {
                    true => fed_add(b, &layer.wo, m, int8),
                    false => Fed::No,
                };
                let (k, v) = (&kv.k[l], &kv.v[l]);
                b.attention(&s.q, k, v, &s.att, &kv.scores, &p, f.q(s), f.half(s));
                // From here on the layer's rows are the `m` from `lo`.
                let n = m;
                let att = At(&s.att, lo * c.q_dim());
                mm_add(b, s, &layer.wo, att, x_at, n, int8, f);
                let f = match &layer.ffn {
                    Ffn::Dense(e) => fed(b, &[&e.gate, &e.up], n, int8),
                    Ffn::Moe { experts, .. } => {
                        let e = &experts[0];
                        match g.expert_strides.is_some() && wants_q(b, &[&e.gate, &e.up], int8) {
                            true => Fed::Q,
                            false => Fed::No,
                        }
                    }
                };
                let quant = f == Fed::Q;
                b.rmsnorm(x_at, xb, &g.ffn_norm, n, n_embd, eps, f.q(s), f.half(s));
                match &layer.ffn {
                    Ffn::Dense(e) => ffn(b, s, e, x_at, n, int8, f, true),
                    Ffn::Moe {
                        router,
                        experts,
                        shared,
                    } => {
                        if let Some(strides) = g.expert_strides {
                            let shared = shared.as_ref();
                            moe(
                                c, b, s, x_at, router, experts, shared, strides, n, int8, quant,
                            );
                            continue;
                        }
                        assert_eq!(n, 1, "CPU-routed MoE layers run one token at a time");
                        // Routing needs the router output on the CPU: one sync per MoE layer.
                        mv(b, router, At(&s.xb, 0), At(&s.router, 0));
                        let mut shared_gate = [0.0f32];
                        if let Some((gi, _)) = shared {
                            mv(b, gi, At(&s.xb, 0), At(&s.shared_gate, 0));
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
                            ffn(b, s, &experts[e], At(&s.tmp, 0), 1, false, Fed::No, false);
                            b.axpy(x_at, At(&s.tmp, 0), probs[e] / norm, n_embd);
                        }
                        if let Some((_, e)) = shared {
                            ffn(b, s, e, At(&s.tmp, 0), 1, false, Fed::No, false);
                            let gate = 1.0 / (1.0 + (-shared_gate[0]).exp());
                            b.axpy(x_at, At(&s.tmp, 0), gate, n_embd);
                        }
                    }
                }
            }
            if want_logits {
                let last = At(&s.x, (n - 1) * n_embd);
                let (xb, out) = (At(&s.xb, 0), [&model.output]);
                let f = fed(b, &out, 1, int8);
                b.rmsnorm(
                    last,
                    xb,
                    &self.output_norm,
                    1,
                    n_embd,
                    c.rms_eps,
                    f.q(s),
                    f.half(s),
                );
                mm_each(b, s, &out, xb, &[At(&s.logits, 0)], 1, int8, f);
            }
        };
        // One token's launches are replayed as a graph, which costs less per launch and leaves
        // the CPU out of it. A layer routed on the CPU reads back mid-pass, so it can't be.
        if n == 1 && self.batchable && !rli_core::cuda::profiling() {
            let mut graph = self.graph.lock().unwrap_or_else(|p| p.into_inner());
            self.gpu.run_graph(&mut graph, queue);
        } else {
            let mut b = self.gpu.batch();
            queue(&mut b);
            b.finish();
        }
        // The caller has no use for x once it has the logits.
        if want_logits && let Some(out) = logits {
            self.gpu.read(&s.logits, out);
        } else {
            self.gpu.read(&s.x, xs);
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

    fn prompt_slice(&self) -> usize {
        BATCH_CHUNK
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
            scores: self.gpu.alloc(Batch::attention_scratch(
                self.n_head,
                self.head_dim,
                max_ctx,
            )),
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
            Scratch::new(&self.gpu, &model.cfg, self.sizes, logits, BATCH_CHUNK)
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
    // The widest feed-forward activation among the offloaded layers' experts.
    let max_ff = model.layers[..n]
        .iter()
        .flat_map(|l| experts_of(&l.ffn))
        .map(|e| e.gate.rows)
        .max()
        .unwrap_or(0);
    let opt = |v: &Option<Vec<f32>>| v.as_ref().map(|v| gpu.upload(v));
    let gpu_layers: Vec<GpuLayer> = model.layers[..n]
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
    // The largest matrix a prompt's GEMM writes out as f16.
    let max_w = model.layers[..n]
        .iter()
        .flat_map(|l| {
            let e = experts_of(&l.ffn).into_iter();
            [&l.wq, &l.wk, &l.wv, &l.wo]
                .into_iter()
                .chain(e.flat_map(|e| [&e.gate, &e.up, &e.down]))
        })
        .map(|w| w.rows * w.cols)
        .max()
        .unwrap_or(0);
    let scratch = Scratch::new(&gpu, &c, (max_ff, max_w), logits, 1);
    let batchable = model.layers[..n]
        .iter()
        .zip(&gpu_layers)
        .all(|(l, g)| matches!(l.ffn, Ffn::Dense(_)) || g.expert_strides.is_some());
    let accel = Cuda {
        output_norm: gpu.upload(&model.output_norm),
        rope_freqs: opt(&model.rope_freqs),
        layers: gpu_layers,
        covers_all,
        batchable,
        sizes: (max_ff, max_w),
        kv_dim: c.kv_dim(),
        n_head: c.n_head,
        head_dim: c.head_dim,
        scratch: Mutex::new(scratch),
        graph: Mutex::new(None),
        batch: Mutex::new(None),
        gpu,
    };
    model.set_accelerator(Box::new(accel));
    Ok(())
}
