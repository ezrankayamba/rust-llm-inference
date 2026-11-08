//! Apple Metal backend primitives. The model file mapping is wrapped as one no-copy shared
//! buffer, so on unified memory the GPU reads the same pages as the CPU: no weight copies.
//!
//! Work is recorded into a [`Batch`] (one command buffer, serial dispatch order) and submitted
//! once, so a whole decoder layer — or the whole model — costs a single CPU/GPU round trip.

use crate::quant::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::ops::Range;
use std::ptr::NonNull;

const SOURCE: &str = include_str!("metal.metal");
const MATVEC_ROWS_PER_TG: usize = 4;
const REDUCE_THREADS: usize = 256;
const ELT_THREADS: usize = 256;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

const MATVEC_KERNELS: [(GgmlType, &str); 8] = [
    (GgmlType::F32, "matvec_f32"),
    (GgmlType::F16, "matvec_f16"),
    (GgmlType::Q4_0, "matvec_q4_0"),
    (GgmlType::Q5_0, "matvec_q5_0"),
    (GgmlType::Q8_0, "matvec_q8_0"),
    (GgmlType::Q4_K, "matvec_q4_k"),
    (GgmlType::Q5_K, "matvec_q5_k"),
    (GgmlType::Q6_K, "matvec_q6_k"),
];
const MATMUL_KERNELS: [(GgmlType, &str); 8] = [
    (GgmlType::F32, "matmul_f32"),
    (GgmlType::F16, "matmul_f16"),
    (GgmlType::Q4_0, "matmul_q4_0"),
    (GgmlType::Q5_0, "matmul_q5_0"),
    (GgmlType::Q8_0, "matmul_q8_0"),
    (GgmlType::Q4_K, "matmul_q4_k"),
    (GgmlType::Q5_K, "matmul_q5_k"),
    (GgmlType::Q6_K, "matmul_q6_k"),
];
const MATVEC_ID_KERNELS: [(GgmlType, &str); 8] = [
    (GgmlType::F32, "matvec_id_f32"),
    (GgmlType::F16, "matvec_id_f16"),
    (GgmlType::Q4_0, "matvec_id_q4_0"),
    (GgmlType::Q5_0, "matvec_id_q5_0"),
    (GgmlType::Q8_0, "matvec_id_q8_0"),
    (GgmlType::Q4_K, "matvec_id_q4_k"),
    (GgmlType::Q5_K, "matvec_id_q5_k"),
    (GgmlType::Q6_K, "matvec_id_q6_k"),
];
/// Matmul output tile (`MM_R` x `MM_T` in metal.metal), computed by 128 threads.
const MATMUL_TILE_ROWS: usize = 64;
const MATMUL_TILE_TOKENS: usize = 32;
const MATMUL_THREADS: usize = 128;
const OTHER_KERNELS: [&str; 17] = [
    "moe_topk",
    "moe_combine",
    "rmsnorm",
    "add",
    "add_bcast",
    "gather_rows",
    "scatter_axpy",
    "axpy",
    "swiglu",
    "rope",
    "store_kv_f32",
    "store_kv_f16",
    "attention_part_f32",
    "attention_part_f16",
    "attention_merge",
    "attention_batch_f32",
    "attention_batch_f16",
];

/// A shared (CPU- and GPU-visible) f32 buffer.
pub struct GpuBuf {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

// SAFETY: MTLBuffer objects are thread-safe; content access is unsafe and caller-synchronized.
unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

impl GpuBuf {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// # Safety
    /// No GPU work that writes this buffer may be in flight.
    pub unsafe fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.buf.contents().as_ptr().cast(), self.len) }
    }

    /// # Safety
    /// No GPU work that reads or writes this buffer may be in flight, and no other slice of it
    /// may be alive.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.buf.contents().as_ptr().cast(), self.len) }
    }
}

impl GpuBuf {
    /// Stores u32 values (e.g. row indices for [`Batch::gather_rows`]) at element `off`.
    ///
    /// # Safety
    /// As [`GpuBuf::as_mut_slice`].
    pub unsafe fn write_u32(&self, off: usize, vals: &[u32]) {
        let dst = unsafe { &mut self.as_mut_slice()[off..off + vals.len()] };
        for (d, &v) in dst.iter_mut().zip(vals) {
            *d = f32::from_bits(v);
        }
    }
}

/// A shared buffer for one layer's K or V cache, holding f16 or (precise mode) f32 elements.
pub struct KvBuf {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    half: bool,
}

// SAFETY: as for GpuBuf.
unsafe impl Send for KvBuf {}
unsafe impl Sync for KvBuf {}

impl KvBuf {
    /// Capacity in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_half(&self) -> bool {
        self.half
    }

    fn elem(&self) -> usize {
        if self.half { 2 } else { 4 }
    }

    /// Copies all of `src` into the start of `self` (same element type, `self` at least as large).
    ///
    /// # Safety
    /// No GPU work that touches either buffer may be in flight.
    pub unsafe fn copy_from(&self, src: &KvBuf) {
        assert!(self.half == src.half && self.len >= src.len);
        let n = src.len * src.elem();
        // SAFETY: both buffers are shared-storage and at least `n` bytes; caller guarantees no
        // concurrent GPU access.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.buf.contents().as_ptr().cast::<u8>(),
                self.buf.contents().as_ptr().cast::<u8>(),
                n,
            )
        };
    }
}

/// A view into a [`GpuBuf`] starting at element `off`.
#[derive(Clone, Copy)]
pub struct At<'a>(pub &'a GpuBuf, pub usize);

impl<'a> From<&'a GpuBuf> for At<'a> {
    fn from(b: &'a GpuBuf) -> Self {
        At(b, 0)
    }
}

fn page_size() -> usize {
    // SAFETY: no preconditions.
    unsafe { getpagesize() as usize }
}

fn page_floor(x: usize) -> usize {
    x - x % page_size()
}

/// Bytes [`Gpu::new`] wraps for `span`: the span rounded out to whole pages.
pub fn wrapped_len(span: &Range<usize>) -> usize {
    (span.end - page_floor(span.start)).div_ceil(page_size()) * page_size()
}

/// The largest buffer the default Metal device can create, or `None` without a device.
pub fn max_buffer_len() -> Option<usize> {
    MTLCreateSystemDefaultDevice().map(|d| d.maxBufferLength())
}

pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    weights: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// File offset of the first wrapped byte (page-aligned) and the end of the usable span.
    weights_base: usize,
    weights_end: usize,
    matvec: Vec<(GgmlType, Pipeline)>,
    matmul: Vec<(GgmlType, Pipeline)>,
    matvec_id: Vec<(GgmlType, Pipeline)>,
    other: Vec<(&'static str, Pipeline)>,
    name: String,
}

// SAFETY: Metal devices, queues and pipeline states are documented as thread-safe.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

unsafe extern "C" {
    fn getpagesize() -> i32;
}

#[repr(C)]
struct MatvecParams {
    offset: u64,
    rows: u32,
    cols: u32,
    row_bytes: u32,
}
#[repr(C)]
struct MatmulParams {
    offset: u64,
    rows: u32,
    cols: u32,
    row_bytes: u32,
    n: u32,
}
#[repr(C)]
struct IdParams {
    offset: u64,
    rows: u32,
    cols: u32,
    row_bytes: u32,
    expert_stride: u64,
    x_stride: u32,
    y_stride: u32,
}
#[repr(C)]
struct TopkParams {
    n_expert: u32,
    k: u32,
    norm: u32,
    shared: u32,
}
#[repr(C)]
struct CombineParams {
    n: u32,
    m: u32,
}
#[repr(C)]
struct NormParams {
    n: u32,
    eps: f32,
}
#[repr(C)]
struct EltParams {
    n: u32,
    alpha: f32,
    m: u32,
}
#[repr(C)]
struct RopeParams {
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    theta: f32,
    neox: u32,
    has_ff: u32,
}
#[repr(C)]
pub struct AttnParams {
    pub n_head: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    pub n_pos: u32,
    pub kv_dim: u32,
    pub scale: f32,
}

/// Cached positions per decode-attention threadgroup (`ATTN_CHUNK` in metal.metal).
const ATTN_CHUNK: usize = 256;
const ATTN_THREADS: usize = 128;

/// f32s of scratch [`Batch::attention`] needs for `n_pos` positions.
pub fn attention_part_len(n_head: usize, head_dim: usize, n_pos: usize) -> usize {
    n_head * n_pos.div_ceil(ATTN_CHUNK) * (head_dim + 2)
}

#[repr(C)]
struct AttnMergeParams {
    head_dim: u32,
    n_chunks: u32,
}

#[repr(C)]
pub struct BatchAttnParams {
    pub n_head: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    /// Position of the chunk's first query token.
    pub pos: u32,
    pub kv_dim: u32,
    pub scale: f32,
}

/// Rotary embedding settings for [`Batch::rope`]: `n_tokens` consecutive rows of
/// `n_heads * head_dim`, the first at position `pos`.
pub struct RopeArgs<'a> {
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub pos: usize,
    pub theta: f32,
    pub neox: bool,
    pub freq_factors: Option<&'a GpuBuf>,
}

impl Gpu {
    /// Wraps the `span` bytes of `mapped` (a page-aligned memory mapping, e.g. the whole GGUF
    /// file) for GPU access, rounded out to whole pages. Weight offsets passed to [`Batch`]
    /// stay relative to `mapped` and must lie inside `span`.
    ///
    /// # Safety
    /// `mapped` must be the start of a memory mapping that stays valid and unmodified for the
    /// lifetime of the returned value; the pages up to the next page boundary must be mapped
    /// (true for any `mmap` of the file).
    pub unsafe fn new(mapped: &[u8], span: Range<usize>) -> Result<Self, String> {
        assert!(span.start <= span.end && span.end <= mapped.len());
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
        if !(mapped.as_ptr() as usize).is_multiple_of(page_size()) {
            return Err("weights mapping is not page-aligned".into());
        }
        let (base, len) = (page_floor(span.start), wrapped_len(&span));
        if len > device.maxBufferLength() {
            return Err(format!(
                "the offloaded weights ({len} bytes) exceed Metal's maxBufferLength ({}); \
                 offload fewer layers (--gpu-layers)",
                device.maxBufferLength()
            ));
        }
        // SAFETY: `base <= span.end <= mapped.len()`, so the pointer stays in the mapping.
        let ptr = NonNull::new(unsafe { mapped.as_ptr().add(base) } as *mut std::ffi::c_void)
            .ok_or("null mapping")?;
        // SAFETY: page-aligned, page-rounded, outlives `self` per the caller contract; the GPU
        // only reads it. No deallocator: the mapping is owned elsewhere.
        let weights = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .ok_or("could not wrap the model mapping as a Metal buffer")?;

        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|e| format!("Metal shader compile failed: {e}"))?;
        let pipeline = |name: &str| -> Result<Pipeline, String> {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("missing kernel {name}"))?;
            device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))
        };
        let matvec = MATVEC_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, pipeline(n)?)))
            .collect::<Result<_, String>>()?;
        let matmul = MATMUL_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, pipeline(n)?)))
            .collect::<Result<_, String>>()?;
        let matvec_id = MATVEC_ID_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, pipeline(n)?)))
            .collect::<Result<_, String>>()?;
        let other = OTHER_KERNELS
            .iter()
            .map(|&n| Ok((n, pipeline(n)?)))
            .collect::<Result<_, String>>()?;
        let queue = device
            .newCommandQueue()
            .ok_or("could not create a Metal command queue")?;
        let name = format!("metal ({})", device.name());
        Ok(Self {
            device,
            queue,
            weights,
            weights_base: base,
            weights_end: span.end,
            matvec,
            matmul,
            matvec_id,
            other,
            name,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Rebases a file offset of `bytes` weight bytes onto the wrapped span.
    fn weight_offset(&self, offset: usize, bytes: usize) -> usize {
        assert!(
            offset >= self.weights_base && offset + bytes <= self.weights_end,
            "weights out of range"
        );
        offset - self.weights_base
    }

    pub fn supports(&self, ty: GgmlType) -> bool {
        self.matvec.iter().any(|(t, _)| *t == ty)
    }

    /// A zeroed shared buffer of `n` f32s.
    pub fn alloc(&self, n: usize) -> GpuBuf {
        let buf = self
            .device
            .newBufferWithLength_options((n * 4).max(4), MTLResourceOptions::StorageModeShared)
            .expect("Metal buffer allocation failed");
        GpuBuf { buf, len: n }
    }

    /// A zeroed KV cache buffer of `n` elements, f16 when `half`.
    pub fn alloc_kv(&self, n: usize, half: bool) -> KvBuf {
        let size = if half { 2 } else { 4 };
        let buf = self
            .device
            .newBufferWithLength_options((n * size).max(4), MTLResourceOptions::StorageModeShared)
            .expect("Metal buffer allocation failed");
        KvBuf { buf, len: n, half }
    }

    pub fn upload(&self, data: &[f32]) -> GpuBuf {
        let b = self.alloc(data.len());
        // SAFETY: freshly allocated, not yet used by the GPU.
        unsafe { b.as_mut_slice().copy_from_slice(data) };
        b
    }

    pub fn batch(&self) -> Batch<'_> {
        let cmd = self.queue.commandBuffer().expect("Metal command buffer");
        let enc = cmd.computeCommandEncoder().expect("Metal compute encoder");
        Batch {
            gpu: self,
            cmd,
            enc,
            ended: false,
        }
    }

    fn other(&self, name: &str) -> &Pipeline {
        &self
            .other
            .iter()
            .find(|(n, _)| *n == name)
            .expect("kernel compiled in new()")
            .1
    }

    /// One-shot `y = W x` (convenience for tests and benchmarks; layers should use a batch).
    pub fn matvec(
        &self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: &[f32],
        y: &mut [f32],
    ) {
        let (xb, yb) = (self.upload(x), self.alloc(rows));
        let mut b = self.batch();
        b.matvec(ty, rows, cols, offset, (&xb).into(), (&yb).into());
        b.finish();
        // SAFETY: the batch has completed.
        y.copy_from_slice(unsafe { yb.as_slice() });
    }
}

impl Gpu {
    /// One-shot `Y = X W^T` for `n` row-major tokens (for tests and benchmarks).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: &[f32],
        y: &mut [f32],
        n: usize,
    ) {
        let (xb, yb) = (self.upload(x), self.alloc(n * rows));
        let mut b = self.batch();
        b.matmul(ty, rows, cols, offset, (&xb).into(), (&yb).into(), n);
        b.finish();
        // SAFETY: the batch has completed.
        y.copy_from_slice(unsafe { yb.as_slice() });
    }
}

/// Recorded GPU work; dispatches run in order. Nothing executes until [`Batch::finish`].
pub struct Batch<'a> {
    gpu: &'a Gpu,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    ended: bool,
}

/// A batch dropped unfinished (e.g. by a panic while recording) must still end its encoder:
/// Metal aborts the process otherwise, hiding the panic message.
impl Drop for Batch<'_> {
    fn drop(&mut self) {
        if !self.ended {
            self.enc.endEncoding();
        }
    }
}

impl Batch<'_> {
    fn bind(&self, index: usize, at: At<'_>) {
        // SAFETY: binding a live buffer at an in-bounds byte offset.
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&at.0.buf), at.1 * 4, index)
        };
    }

    fn bind_kv(&self, index: usize, kv: &KvBuf) {
        // SAFETY: binding a live buffer at offset 0.
        unsafe { self.enc.setBuffer_offset_atIndex(Some(&kv.buf), 0, index) };
    }

    fn params<T>(&self, index: usize, p: &T) {
        // SAFETY: setBytes copies `size_of::<T>()` bytes immediately.
        unsafe {
            self.enc.setBytes_length_atIndex(
                NonNull::from(p).cast(),
                std::mem::size_of::<T>(),
                index,
            )
        };
    }

    fn dispatch(&self, pipeline: &Pipeline, groups: usize, threads: usize) {
        self.dispatch2(pipeline, (groups, 1), threads);
    }

    fn dispatch2(&self, pipeline: &Pipeline, groups: (usize, usize), threads: usize) {
        self.enc.setComputePipelineState(pipeline);
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.0,
                height: groups.1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }

    /// `y = W x`, `W` being `rows x cols` of `ty` at byte `offset` of the model mapping.
    pub fn matvec(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: At<'_>,
        y: At<'_>,
    ) {
        assert!(
            cols.is_multiple_of(32),
            "matvec cols must be a multiple of 32"
        );
        assert!(
            x.1 + cols <= x.0.len && y.1 + rows <= y.0.len,
            "matvec buffer too small"
        );
        let row_bytes = ty.row_bytes(cols);
        let offset = self.gpu.weight_offset(offset, rows * row_bytes);
        let pipeline = &self
            .gpu
            .matvec
            .iter()
            .find(|(t, _)| *t == ty)
            .expect("unsupported type for Metal")
            .1;
        self.enc.setComputePipelineState(pipeline);
        // SAFETY: weights buffer bound read-only at offset 0; the kernel adds `offset` itself.
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.gpu.weights), 0, 0)
        };
        self.bind(1, x);
        self.bind(2, y);
        self.params(
            3,
            &MatvecParams {
                offset: offset as u64,
                rows: rows as u32,
                cols: cols as u32,
                row_bytes: row_bytes as u32,
            },
        );
        self.dispatch(
            pipeline,
            rows.div_ceil(MATVEC_ROWS_PER_TG),
            MATVEC_ROWS_PER_TG * 32,
        );
    }

    /// `Y = X W^T` for `n` tokens: `X` is `n x cols` and `Y` is `n x rows`, both row-major.
    /// `n == 1` uses the matvec kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: At<'_>,
        y: At<'_>,
        n: usize,
    ) {
        if n == 1 {
            return self.matvec(ty, rows, cols, offset, x, y);
        }
        assert!(
            cols.is_multiple_of(32),
            "matmul cols must be a multiple of 32"
        );
        assert!(
            x.1 + n * cols <= x.0.len && y.1 + n * rows <= y.0.len,
            "matmul buffer too small"
        );
        let row_bytes = ty.row_bytes(cols);
        let offset = self.gpu.weight_offset(offset, rows * row_bytes);
        let pipeline = &self
            .gpu
            .matmul
            .iter()
            .find(|(t, _)| *t == ty)
            .expect("unsupported type for Metal")
            .1;
        self.enc.setComputePipelineState(pipeline);
        // SAFETY: weights buffer bound read-only at offset 0; the kernel adds `offset` itself.
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.gpu.weights), 0, 0)
        };
        self.bind(1, x);
        self.bind(2, y);
        self.params(
            3,
            &MatmulParams {
                offset: offset as u64,
                rows: rows as u32,
                cols: cols as u32,
                row_bytes: row_bytes as u32,
                n: n as u32,
            },
        );
        self.dispatch2(
            pipeline,
            (
                rows.div_ceil(MATMUL_TILE_ROWS),
                n.div_ceil(MATMUL_TILE_TOKENS),
            ),
            MATMUL_THREADS,
        );
    }

    /// Routes one token's MoE layer on the GPU: from the `n_expert` router logits, writes the
    /// top `k` expert ids (u32 bits) to `ids` and their softmax weights (renormalized over the
    /// k when `norm`) to `w`; with `shared`, also `w[k]` = sigmoid(`shared_gate[0]`).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk(
        &mut self,
        logits: At<'_>,
        shared_gate: At<'_>,
        ids: At<'_>,
        w: At<'_>,
        n_expert: usize,
        k: usize,
        norm: bool,
        shared: bool,
    ) {
        assert!(logits.1 + n_expert <= logits.0.len && ids.1 + k <= ids.0.len);
        assert!(
            k <= n_expert && n_expert <= 32 * 64,
            "the kernel tracks 64 experts per lane"
        );
        assert!(w.1 + k + usize::from(shared) <= w.0.len && shared_gate.1 < shared_gate.0.len);
        self.bind(0, logits);
        self.bind(1, shared_gate);
        self.bind(2, ids);
        self.bind(3, w);
        self.params(
            4,
            &TopkParams {
                n_expert: n_expert as u32,
                k: k as u32,
                norm: norm.into(),
                shared: shared.into(),
            },
        );
        self.dispatch(self.gpu.other("moe_topk"), 1, 32);
    }

    /// `y[slot] = W[ids[slot]] x[slot]` for `slots` slots: `W[e]` is expert `e` of a stacked
    /// tensor whose expert 0 (`rows x cols` of `ty`) starts at byte `offset` of the model
    /// mapping, the `n_expert` experts `expert_stride` bytes apart. Slot inputs and outputs
    /// are `x_stride` and `y_stride` floats apart (`x_stride` 0: all slots read one `x`).
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_id(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        expert_stride: usize,
        n_expert: usize,
        ids: At<'_>,
        x: At<'_>,
        x_stride: usize,
        y: At<'_>,
        y_stride: usize,
        slots: usize,
    ) {
        assert!(
            cols.is_multiple_of(32),
            "matvec cols must be a multiple of 32"
        );
        assert!(slots >= 1 && ids.1 + slots <= ids.0.len);
        assert!(
            x.1 + (slots - 1) * x_stride + cols <= x.0.len,
            "matvec_id x too small"
        );
        assert!(
            y.1 + (slots - 1) * y_stride + rows <= y.0.len,
            "matvec_id y too small"
        );
        let row_bytes = ty.row_bytes(cols);
        let span = (n_expert - 1) * expert_stride + rows * row_bytes;
        let offset = self.gpu.weight_offset(offset, span);
        let pipeline = &self
            .gpu
            .matvec_id
            .iter()
            .find(|(t, _)| *t == ty)
            .expect("unsupported type for Metal")
            .1;
        self.enc.setComputePipelineState(pipeline);
        // SAFETY: weights buffer bound read-only at offset 0; the kernel adds `offset` itself.
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.gpu.weights), 0, 0)
        };
        self.bind(1, x);
        self.bind(2, y);
        self.params(
            3,
            &IdParams {
                offset: offset as u64,
                rows: rows as u32,
                cols: cols as u32,
                row_bytes: row_bytes as u32,
                expert_stride: expert_stride as u64,
                x_stride: x_stride as u32,
                y_stride: y_stride as u32,
            },
        );
        self.bind(4, ids);
        self.dispatch2(
            pipeline,
            (rows.div_ceil(MATVEC_ROWS_PER_TG), slots),
            MATVEC_ROWS_PER_TG * 32,
        );
    }

    /// `x += Σ_{r<m} w[r] · src[r]`, rows of `n`: the chosen experts' outputs, combined.
    pub fn moe_combine(&mut self, x: At<'_>, src: At<'_>, w: At<'_>, m: usize, n: usize) {
        assert!(x.1 + n <= x.0.len && src.1 + m * n <= src.0.len && w.1 + m <= w.0.len);
        self.bind(0, x);
        self.bind(1, src);
        self.bind(2, w);
        self.params(
            3,
            &CombineParams {
                n: n as u32,
                m: m as u32,
            },
        );
        self.dispatch(
            self.gpu.other("moe_combine"),
            n.div_ceil(ELT_THREADS),
            ELT_THREADS,
        );
    }

    /// RMS-normalizes `rows` consecutive rows of `n` from `x` into `out` (may alias).
    pub fn rmsnorm(&mut self, x: At<'_>, out: At<'_>, w: &GpuBuf, rows: usize, n: usize, eps: f32) {
        self.bind(0, x);
        self.bind(1, out);
        self.bind(2, w.into());
        self.params(3, &NormParams { n: n as u32, eps });
        self.dispatch(self.gpu.other("rmsnorm"), rows, REDUCE_THREADS);
    }

    fn elementwise(&mut self, kernel: &str, a: At<'_>, b: At<'_>, n: usize, alpha: f32) {
        self.bind(0, a);
        self.bind(1, b);
        self.params(
            2,
            &EltParams {
                n: n as u32,
                alpha,
                m: n as u32,
            },
        );
        self.dispatch(self.gpu.other(kernel), n.div_ceil(ELT_THREADS), ELT_THREADS);
    }

    /// Adds the length-`m` vector `b` to each of the `n / m` rows of `a`.
    pub fn add_bcast(&mut self, a: At<'_>, b: At<'_>, m: usize, n: usize) {
        assert!(n.is_multiple_of(m) && a.1 + n <= a.0.len && b.1 + m <= b.0.len);
        self.bind(0, a);
        self.bind(1, b);
        self.params(
            2,
            &EltParams {
                n: n as u32,
                alpha: 1.0,
                m: m as u32,
            },
        );
        self.dispatch(
            self.gpu.other("add_bcast"),
            n.div_ceil(ELT_THREADS),
            ELT_THREADS,
        );
    }

    /// `a += b`
    pub fn add(&mut self, a: At<'_>, b: At<'_>, n: usize) {
        self.elementwise("add", a, b, n, 1.0);
    }

    /// Row `r` of `dst` = row `idx[r]` of `src`, for `rows` rows of `dim`. `idx` holds u32 row
    /// indices (stored with [`GpuBuf::write_u32`]).
    pub fn gather_rows(&mut self, src: At<'_>, dst: At<'_>, idx: At<'_>, rows: usize, dim: usize) {
        assert!(dst.1 + rows * dim <= dst.0.len && idx.1 + rows <= idx.0.len);
        self.bind(0, src);
        self.bind(1, dst);
        self.bind(2, idx);
        self.indexed("gather_rows", 3, rows, dim);
    }

    /// Row `idx[r]` of `dst` += `w[r]` * row `r` of `src`; the `rows` indices must be distinct.
    pub fn scatter_axpy(
        &mut self,
        dst: At<'_>,
        src: At<'_>,
        idx: At<'_>,
        w: At<'_>,
        rows: usize,
        dim: usize,
    ) {
        assert!(src.1 + rows * dim <= src.0.len && idx.1 + rows <= idx.0.len);
        assert!(w.1 + rows <= w.0.len);
        self.bind(0, dst);
        self.bind(1, src);
        self.bind(2, idx);
        self.bind(3, w);
        self.indexed("scatter_axpy", 4, rows, dim);
    }

    fn indexed(&mut self, kernel: &str, params_at: usize, rows: usize, dim: usize) {
        let n = rows * dim;
        self.params(
            params_at,
            &EltParams {
                n: n as u32,
                alpha: 0.0,
                m: dim as u32,
            },
        );
        self.dispatch(self.gpu.other(kernel), n.div_ceil(ELT_THREADS), ELT_THREADS);
    }

    /// `a += alpha * b`
    pub fn axpy(&mut self, a: At<'_>, b: At<'_>, alpha: f32, n: usize) {
        self.elementwise("axpy", a, b, n, alpha);
    }

    /// `gate = silu(gate) * up`
    pub fn swiglu(&mut self, gate: At<'_>, up: At<'_>, n: usize) {
        self.elementwise("swiglu", gate, up, n, 0.0);
    }

    pub fn rope(&mut self, x: At<'_>, r: &RopeArgs<'_>) {
        self.bind(0, x);
        self.bind(1, r.freq_factors.unwrap_or(x.0).into());
        self.params(
            2,
            &RopeParams {
                n_heads: r.n_heads as u32,
                head_dim: r.head_dim as u32,
                rope_dim: r.rope_dim as u32,
                pos: r.pos as u32,
                theta: r.theta,
                neox: r.neox as u32,
                has_ff: r.freq_factors.is_some() as u32,
            },
        );
        assert!(x.1 + r.n_tokens * r.n_heads * r.head_dim <= x.0.len);
        let n = r.n_heads * r.rope_dim / 2;
        self.dispatch2(
            self.gpu.other("rope"),
            (n.div_ceil(ELT_THREADS), r.n_tokens),
            ELT_THREADS,
        );
    }

    /// Converts `n` f32s from `src` into `dst` starting at element `off`.
    pub fn store_kv(&mut self, src: At<'_>, dst: &KvBuf, off: usize, n: usize) {
        assert!(src.1 + n <= src.0.len && off + n <= dst.len);
        self.bind(0, src);
        // SAFETY: binding a live buffer at an in-bounds byte offset.
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&dst.buf), off * dst.elem(), 1)
        };
        self.params(
            2,
            &EltParams {
                n: n as u32,
                alpha: 0.0,
                m: 0,
            },
        );
        let kernel = if dst.half {
            "store_kv_f16"
        } else {
            "store_kv_f32"
        };
        self.dispatch(self.gpu.other(kernel), n.div_ceil(ELT_THREADS), ELT_THREADS);
    }

    /// Attention of one query over `p.n_pos` cached positions, split into chunks of positions
    /// that run in parallel and are then merged. `part` is scratch of at least
    /// [`attention_part_len`] f32s.
    pub fn attention(
        &mut self,
        q: At<'_>,
        k: &KvBuf,
        v: &KvBuf,
        out: At<'_>,
        part: &GpuBuf,
        p: &AttnParams,
    ) {
        let (n_head, hd, n_pos) = (p.n_head as usize, p.head_dim as usize, p.n_pos as usize);
        assert!(n_pos >= 1 && part.len >= attention_part_len(n_head, hd, n_pos));
        assert!(k.half == v.half && n_pos * p.kv_dim as usize <= k.len.min(v.len));
        assert!(q.1 + n_head * hd <= q.0.len && out.1 + n_head * hd <= out.0.len);
        let n_chunks = n_pos.div_ceil(ATTN_CHUNK);
        self.bind(0, q);
        self.bind_kv(1, k);
        self.bind_kv(2, v);
        self.bind(3, part.into());
        self.params(4, p);
        let kernel = if k.half {
            "attention_part_f16"
        } else {
            "attention_part_f32"
        };
        self.dispatch2(self.gpu.other(kernel), (n_head, n_chunks), ATTN_THREADS);
        self.bind(0, part.into());
        self.bind(1, out);
        self.params(
            2,
            &AttnMergeParams {
                head_dim: p.head_dim,
                n_chunks: n_chunks as u32,
            },
        );
        self.dispatch(self.gpu.other("attention_merge"), n_head, ATTN_THREADS);
    }

    /// Causal attention for `n` query rows at positions `p.pos..p.pos + n`; `k`/`v` are the
    /// caches (rows `0..p.pos + n` filled). Needs `head_dim % 32 == 0` and `head_dim <= 256`.
    pub fn attention_batch(
        &mut self,
        q: At<'_>,
        k: &KvBuf,
        v: &KvBuf,
        out: At<'_>,
        n: usize,
        p: &BatchAttnParams,
    ) {
        let (hd, q_dim) = (p.head_dim as usize, (p.n_head * p.head_dim) as usize);
        assert!(
            hd.is_multiple_of(32) && hd <= 256,
            "unsupported head_dim {hd}"
        );
        assert!(q.1 + n * q_dim <= q.0.len && out.1 + n * q_dim <= out.0.len);
        assert!(k.half == v.half && (p.pos as usize + n) * p.kv_dim as usize <= k.len.min(v.len));
        self.bind(0, q);
        self.bind_kv(1, k);
        self.bind_kv(2, v);
        self.bind(3, out);
        self.params(4, p);
        let kernel = if k.half {
            "attention_batch_f16"
        } else {
            "attention_batch_f32"
        };
        self.dispatch2(self.gpu.other(kernel), (p.n_head as usize, n), 32);
    }

    /// Submits the recorded work and waits for it.
    pub fn finish(mut self) {
        self.enc.endEncoding();
        self.ended = true;
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        if let Some(e) = self.cmd.error() {
            panic!("Metal command failed: {e}");
        }
    }
}
