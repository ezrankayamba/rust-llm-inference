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
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLDispatchType, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

const SOURCE: &str = include_str!("metal.metal");
const MATVEC_ROWS_PER_TG: usize = 4;
const REDUCE_THREADS: usize = 1024;
const ELT_THREADS: usize = 256;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// The K-quant matvec kernels take two rows a simdgroup, or four in the `matvec4_*` kernels
/// used from this many rows up (the output head; see `MATVEC4` in metal.metal).
const MATVEC4_MIN_ROWS: usize = 16384;
/// The types whose matvec kernels take two rows a simdgroup, with the kernel that runs two
/// matrices' rows as a SwiGLU ([`Batch::matvec_swiglu`]).
fn swiglu_kernel(ty: GgmlType) -> Option<&'static str> {
    match ty {
        GgmlType::Q4_0 => Some("matvec_swiglu_q4_0"),
        GgmlType::Q5_0 => Some("matvec_swiglu_q5_0"),
        GgmlType::Q8_0 => Some("matvec_swiglu_q8_0"),
        GgmlType::Q4_K => Some("matvec_swiglu_q4_k"),
        GgmlType::Q6_K => Some("matvec_swiglu_q6_k"),
        _ => None,
    }
}
fn swiglu_id_kernel(ty: GgmlType) -> Option<&'static str> {
    match ty {
        GgmlType::Q4_0 => Some("matvec_swiglu_id_q4_0"),
        GgmlType::Q5_0 => Some("matvec_swiglu_id_q5_0"),
        GgmlType::Q8_0 => Some("matvec_swiglu_id_q8_0"),
        GgmlType::Q4_K => Some("matvec_swiglu_id_q4_k"),
        GgmlType::Q6_K => Some("matvec_swiglu_id_q6_k"),
        _ => None,
    }
}
fn id_add_kernel(ty: GgmlType) -> Option<&'static str> {
    match ty {
        GgmlType::Q4_0 => Some("matvec_id_add_q4_0"),
        GgmlType::Q5_0 => Some("matvec_id_add_q5_0"),
        GgmlType::Q8_0 => Some("matvec_id_add_q8_0"),
        GgmlType::Q4_K => Some("matvec_id_add_q4_k"),
        GgmlType::Q6_K => Some("matvec_id_add_q6_k"),
        _ => None,
    }
}
const MATVEC_KERNELS: [(GgmlType, &str); 10] = [
    (GgmlType::F32, "matvec_f32"),
    (GgmlType::F16, "matvec_f16"),
    (GgmlType::BF16, "matvec_bf16"),
    (GgmlType::Q4_0, "matvec_q4_0"),
    (GgmlType::Q5_0, "matvec_q5_0"),
    (GgmlType::Q8_0, "matvec_q8_0"),
    (GgmlType::Q4_K, "matvec_q4_k"),
    (GgmlType::Q5_K, "matvec_q5_k"),
    (GgmlType::Q6_K, "matvec_q6_k"),
    (GgmlType::NVFP4, "matvec_nvfp4"),
];
const MATMUL_KERNELS: [(GgmlType, &str); 10] = [
    (GgmlType::F32, "matmul_f32"),
    (GgmlType::F16, "matmul_f16"),
    (GgmlType::BF16, "matmul_bf16"),
    (GgmlType::Q4_0, "matmul_q4_0"),
    (GgmlType::Q5_0, "matmul_q5_0"),
    (GgmlType::Q8_0, "matmul_q8_0"),
    (GgmlType::Q4_K, "matmul_q4_k"),
    (GgmlType::Q5_K, "matmul_q5_k"),
    (GgmlType::Q6_K, "matmul_q6_k"),
    (GgmlType::NVFP4, "matmul_nvfp4"),
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
/// Below this many tokens a matmul runs as one matvec per token.
const MATMUL_MIN_TOKENS: usize = 2;
/// Rows of a matmul output tile (`MM_R` in metal.metal), computed by 128 threads, and the
/// token counts a tile is compiled for (`matmul_*_t<tokens>`).
const MATMUL_TILE_ROWS: usize = 64;
const MATMUL_TILE_TOKENS: [usize; 4] = [8, 16, 32, 64];
const MATMUL_THREADS: usize = 128;
/// Up to this many tokens take `matmul_few_*` (where the columns are a multiple of 64): tiles of
/// 16 rows by 8, 16 or 24 tokens, the columns shared out among [`FEW_SIMDGROUPS`] simdgroups.
const MATMUL_FEW_TOKENS: usize = 24;
const FEW_ROWS: usize = 16;
const FEW_SIMDGROUPS: usize = 8;
/// Query heads per threadgroup of `attention_batch_staged_*` (`AG_HEADS` in metal.metal), and
/// the head sizes that kernel is compiled for; other sizes and f32 caches take the general one.
const ATTN_STAGED_HEADS: u32 = 4;
const ATTN_STAGED_DIMS: [usize; 5] = [16, 32, 64, 96, 128];
const OTHER_KERNELS: [&str; 44] = [
    "copy_heads",
    "matvec4_q4_k",
    "matvec4_q6_k",
    "matvec4_q8_0",
    "matvec_swiglu_q4_0",
    "matvec_swiglu_q5_0",
    "matvec_swiglu_q8_0",
    "matvec_swiglu_q4_k",
    "matvec_swiglu_q6_k",
    "matvec_swiglu_id_q4_0",
    "matvec_swiglu_id_q5_0",
    "matvec_swiglu_id_q8_0",
    "matvec_swiglu_id_q4_k",
    "matvec_swiglu_id_q6_k",
    "matvec_id_add_q4_0",
    "matvec_id_add_q5_0",
    "matvec_id_add_q8_0",
    "matvec_id_add_q4_k",
    "matvec_id_add_q6_k",
    "rope_store_f32",
    "rope_store_f16",
    "moe_topk",
    "moe_combine",
    "moe_combine_rows",
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
    "attention_batch_staged_16",
    "attention_batch_staged_32",
    "attention_batch_staged_64",
    "attention_batch_staged_96",
    "attention_batch_staged_128",
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

/// Units further apart than this go in separate buffers rather than wrapping the gap.
const MAX_GAP: usize = 64 << 20;

/// Groups `units` (file byte ranges, each kept whole) into the page-rounded spans [`Gpu::new`]
/// wraps as buffers: in file order, a span grows while it stays within `max` bytes and the next
/// unit starts within [`MAX_GAP`]. Neighbouring spans may share a page. Errors if a unit alone
/// exceeds `max`.
pub fn buffer_spans(units: &[Range<usize>], max: usize) -> Result<Vec<Range<usize>>, String> {
    let mut sorted: Vec<Range<usize>> = units.iter().filter(|u| u.start < u.end).cloned().collect();
    sorted.sort_by_key(|u| u.start);
    let mut spans: Vec<Range<usize>> = Vec::new();
    for u in sorted {
        if wrapped_len(&u) > max {
            return Err(format!(
                "a weight tensor of {} bytes exceeds the largest Metal buffer ({max} bytes)",
                u.end - u.start
            ));
        }
        match spans.last_mut() {
            Some(s)
                if u.start <= s.end + MAX_GAP
                    && wrapped_len(&(s.start..u.end.max(s.end))) <= max =>
            {
                s.end = s.end.max(u.end)
            }
            _ => spans.push(u),
        }
    }
    Ok(spans)
}

/// The largest buffer the default Metal device can create, or `None` without a device.
pub fn max_buffer_len() -> Option<usize> {
    MTLCreateSystemDefaultDevice().map(|d| d.maxBufferLength())
}

/// How much memory the default Metal device should keep resident before performance suffers
/// (`recommendedMaxWorkingSetSize`), or `None` without a device.
pub fn working_set_len() -> Option<usize> {
    MTLCreateSystemDefaultDevice().map(|d| d.recommendedMaxWorkingSetSize() as usize)
}

/// One no-copy view of the mapping: file bytes `base..end` (`base` page-aligned).
struct WeightBuf {
    base: usize,
    end: usize,
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// The weight spans, each wrapped as its own buffer (one buffer is capped at
    /// `maxBufferLength`), in file order.
    weights: Vec<WeightBuf>,
    matvec: Vec<(GgmlType, Pipeline)>,
    /// A kernel per tile size of [`MATMUL_TILE_TOKENS`].
    matmul: Vec<(GgmlType, Vec<Pipeline>)>,
    /// `matmul_h_*`: the same with half-precision tiles, for the quantized types.
    matmul_half: Vec<(GgmlType, Vec<Pipeline>)>,
    /// `matmul_few_*`, for up to 8, 16 and 24 tokens.
    matmul_few: Vec<(GgmlType, [Pipeline; 3])>,
    matvec_id: Vec<(GgmlType, Pipeline)>,
    other: Vec<(&'static str, Pipeline)>,
    /// Each pipeline's address and kernel name, for the profile.
    kernels: Vec<(usize, String)>,
    name: String,
    /// The tensor scale of each NVFP4 matrix, by file offset ([`Gpu::set_scale`]).
    scales: HashMap<usize, f32>,
}

static PROFILE: OnceLock<bool> = OnceLock::new();
static PROFILE_TIMES: Mutex<BTreeMap<String, (f64, u64)>> = Mutex::new(BTreeMap::new());

/// Whether `RLI_METAL_PROF` is set: every dispatch then runs alone in its own command buffer
/// and is timed, and the totals per kernel are printed when the device is dropped. A kernel
/// run alone has the GPU to itself, so the shares are a guide, not the cost within a pass.
pub fn profiling() -> bool {
    *PROFILE.get_or_init(|| std::env::var_os("RLI_METAL_PROF").is_some())
}

/// `RLI_METAL_PROF_REPEAT=N`: while profiling, each dispatch runs N times over in its command
/// buffer and its time is divided by N. A dispatch that follows an idle GPU is 10-40% slower
/// than one in sequence, by an amount that differs between kernels, so this is the number to
/// compare two kernels by. Results are then wrong wherever a kernel adds into its output.
fn profile_repeat() -> u32 {
    static REPEAT: OnceLock<u32> = OnceLock::new();
    *REPEAT.get_or_init(|| {
        let n = std::env::var("RLI_METAL_PROF_REPEAT").ok();
        n.and_then(|v| v.parse().ok()).unwrap_or(1).max(1)
    })
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let times = std::mem::take(&mut *PROFILE_TIMES.lock().unwrap_or_else(|p| p.into_inner()));
        let mut rows: Vec<_> = times.into_iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let total: f64 = rows.iter().map(|r| r.1.0).sum();
        for (name, (t, n)) in rows {
            eprintln!(
                "{name:28} {:9.2} ms {:5.1}% {n:7} dispatches {:8.1} us each",
                t * 1e3,
                t / total * 100.0,
                t / n as f64 * 1e6
            );
        }
    }
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
    add: u32,
    scale: f32,
}
#[repr(C)]
struct SwigluParams {
    gate: u64,
    up: u64,
    rows: u32,
    cols: u32,
    row_bytes: u32,
}
#[repr(C)]
struct IdAddParams {
    offset: u64,
    expert_stride: u64,
    rows: u32,
    cols: u32,
    row_bytes: u32,
    x_stride: u32,
    slots: u32,
}
#[repr(C)]
struct SwigluIdParams {
    gate: u64,
    up: u64,
    gate_stride: u64,
    up_stride: u64,
    rows: u32,
    cols: u32,
    row_bytes: u32,
    y_stride: u32,
}
#[repr(C)]
struct MatmulParams {
    offset: u64,
    rows: u32,
    cols: u32,
    row_bytes: u32,
    tok0: u32,
    n: u32,
    add: u32,
    nsg: u32,
    scale: f32,
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
struct CombineRowsParams {
    n: u32,
    dim: u32,
    k: u32,
}
#[repr(C)]
struct NormParams {
    n: u32,
    eps: f32,
    stride: u32,
}
#[repr(C)]
struct HeadsParams {
    m: u32,
    nh: u32,
    w: u32,
    in_row: u32,
    in_head: u32,
    in_off: u32,
    out_row: u32,
    out_head: u32,
    out_off: u32,
}

/// One side of [`Batch::copy_heads`]: a head's values start at `t * row + h * head + off`.
#[derive(Clone, Copy)]
pub struct Heads {
    pub row: usize,
    pub head: usize,
    pub off: usize,
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
pub const ATTN_CHUNK: usize = 256;
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
#[derive(Clone, Copy)]
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
    /// Wraps the file byte ranges `units` of `mapped` (each kept whole, e.g. one tensor or one
    /// layer's weights) as no-copy buffers of at most `maxBufferLength` bytes.
    ///
    /// # Safety
    /// `mapped` must be the start of a memory mapping that stays valid and unmodified for the
    /// lifetime of the returned value; the pages up to the next page boundary must be mapped
    /// (true for any `mmap` of the file).
    pub unsafe fn new(mapped: &[u8], units: &[Range<usize>]) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
        let max = device.maxBufferLength();
        // SAFETY: forwarded from the caller.
        unsafe { Self::with_device(device, mapped, units, max) }
    }

    /// [`Gpu::new`] with buffers capped at `max` bytes (below `maxBufferLength`), so tests can
    /// exercise weights split across several buffers.
    ///
    /// # Safety
    /// As for [`Gpu::new`].
    pub unsafe fn new_capped(
        mapped: &[u8],
        units: &[Range<usize>],
        max: usize,
    ) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
        let max = max.min(device.maxBufferLength());
        // SAFETY: forwarded from the caller.
        unsafe { Self::with_device(device, mapped, units, max) }
    }

    /// # Safety
    /// As for [`Gpu::new`].
    unsafe fn with_device(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        mapped: &[u8],
        units: &[Range<usize>],
        max: usize,
    ) -> Result<Self, String> {
        assert!(
            units
                .iter()
                .all(|u| u.start <= u.end && u.end <= mapped.len())
        );
        if !(mapped.as_ptr() as usize).is_multiple_of(page_size()) {
            return Err("weights mapping is not page-aligned".into());
        }
        let mut weights = Vec::new();
        for span in buffer_spans(units, max)? {
            let (base, len) = (page_floor(span.start), wrapped_len(&span));
            // SAFETY: `base <= span.end <= mapped.len()`, so the pointer stays in the mapping.
            let ptr = NonNull::new(unsafe { mapped.as_ptr().add(base) } as *mut std::ffi::c_void)
                .ok_or("null mapping")?;
            // SAFETY: page-aligned, page-rounded, outlives `self` per the caller contract; the
            // GPU only reads it. No deallocator: the mapping is owned elsewhere.
            let buf = unsafe {
                device.newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            }
            .ok_or("could not wrap the model mapping as a Metal buffer")?;
            weights.push(WeightBuf {
                base,
                end: span.end,
                buf,
            });
        }

        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|e| format!("Metal shader compile failed: {e}"))?;
        let kernels = RefCell::new(Vec::new());
        let pipeline = |name: &str| -> Result<Pipeline, String> {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("missing kernel {name}"))?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))?;
            kernels
                .borrow_mut()
                .push((Retained::as_ptr(&p) as usize, name.to_string()));
            Ok(p)
        };
        let matvec = MATVEC_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, pipeline(n)?)))
            .collect::<Result<_, String>>()?;
        let tiles = |name: &str| -> Result<Vec<Pipeline>, String> {
            MATMUL_TILE_TOKENS
                .iter()
                .map(|t| pipeline(&format!("{name}_t{t}")))
                .collect()
        };
        let matmul = MATMUL_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, tiles(n)?)))
            .collect::<Result<_, String>>()?;
        let matmul_half = MATMUL_KERNELS
            .iter()
            .filter(|(t, _)| !matches!(t, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16))
            .map(|&(t, n)| Ok((t, tiles(&n.replace("matmul_", "matmul_h_"))?)))
            .collect::<Result<_, String>>()?;
        let matmul_few = MATMUL_KERNELS
            .iter()
            .map(|&(t, n)| {
                let name = n.replace("matmul_", "matmul_few_");
                let few = [
                    pipeline(&format!("{name}_r1"))?,
                    pipeline(&format!("{name}_r2"))?,
                    pipeline(&format!("{name}_r3"))?,
                ];
                Ok((t, few))
            })
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
            matvec,
            matmul,
            matmul_half,
            matmul_few,
            matvec_id,
            other,
            kernels: kernels.into_inner(),
            name,
            scales: HashMap::new(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Copies `bytes` into a buffer of their own that weight offsets `base..base + len` read:
    /// matrices the loader made (DeepSeek's transposed `k_b`), addressed past the file's end.
    pub fn add_owned(&mut self, base: usize, bytes: &[u8]) {
        assert!(!bytes.is_empty() && self.weights.iter().all(|w| w.end <= base));
        // SAFETY: `bytes` is a live slice of `len` bytes; Metal copies it.
        let buf = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void).expect("non-null slice"),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Metal buffer allocation failed");
        let end = base + bytes.len();
        self.weights.push(WeightBuf { base, end, buf });
    }

    /// The buffer holding file bytes `offset..offset + bytes`, and `offset` rebased onto it.
    fn weight_buf(&self, offset: usize, bytes: usize) -> (&ProtocolObject<dyn MTLBuffer>, usize) {
        let w = self
            .weights
            .iter()
            .find(|w| offset >= w.base && offset + bytes <= w.end)
            .expect("weights out of range");
        (&w.buf, offset - w.base)
    }

    pub fn supports(&self, ty: GgmlType) -> bool {
        self.matvec.iter().any(|(t, _)| *t == ty)
    }

    /// Whether [`Batch::matvec_id`] has a kernel for `ty`, which a stack of experts routed on
    /// the GPU needs. NVFP4 has none: each expert would need its own tensor scale.
    pub fn has_matvec_id(&self, ty: GgmlType) -> bool {
        self.matvec_id.iter().any(|(t, _)| *t == ty)
    }

    /// Records the tensor scale of the NVFP4 matrix at file `offset`, which its kernels multiply
    /// by. Every NVFP4 matrix must have one (1 when the file has no `.scale` for it).
    pub fn set_scale(&mut self, offset: usize, scale: f32) {
        self.scales.insert(offset, scale);
    }

    /// The scale the kernels for the `ty` matrix at `offset` take.
    fn scale(&self, ty: GgmlType, offset: usize) -> f32 {
        if ty != GgmlType::NVFP4 {
            return 1.0;
        }
        *self
            .scales
            .get(&offset)
            .unwrap_or_else(|| panic!("NVFP4 matrix at {offset} has no scale set"))
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
        let (cmd, enc) = self.encoder();
        Batch {
            half: false,
            free: false,
            gpu: self,
            cmd: RefCell::new(cmd),
            enc: RefCell::new(enc),
            ended: false,
            started: Vec::new(),
        }
    }

    /// A command buffer and its encoder. Dispatches may overlap unless a barrier separates
    /// them, which [`Batch`] encodes before each one that isn't marked `free`.
    fn encoder(&self) -> (CommandBuffer, Encoder) {
        let cmd = self.queue.commandBuffer().expect("Metal command buffer");
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .expect("Metal compute encoder");
        (cmd, enc)
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
        half: bool,
    ) {
        let (xb, yb) = (self.upload(x), self.alloc(n * rows));
        let mut b = self.batch();
        b.half = half;
        b.matmul(ty, rows, cols, offset, (&xb).into(), (&yb).into(), n);
        b.finish();
        // SAFETY: the batch has completed.
        y.copy_from_slice(unsafe { yb.as_slice() });
    }
}

/// Recorded GPU work; dispatches run in order. Nothing executes until [`Batch::finish`].
pub struct Batch<'a> {
    /// Run matmuls over quantized weights with half-precision tiles (f32 sums): about 1.4x the
    /// speed of f32 tiles, for a relative error near 1e-3 per product. Off by default.
    pub half: bool,
    /// The dispatches queued while this is set read nothing an earlier one of them writes, so
    /// they may run at the same time as the dispatch before them.
    pub free: bool,
    gpu: &'a Gpu,
    /// In cells so that a profiled dispatch can submit them and start anew.
    cmd: RefCell<CommandBuffer>,
    enc: RefCell<Encoder>,
    ended: bool,
    /// Command buffers already submitted by [`Batch::start`].
    started: Vec<CommandBuffer>,
}

/// A batch dropped unfinished (e.g. by a panic while recording) must still end its encoder:
/// Metal aborts the process otherwise, hiding the panic message.
impl Drop for Batch<'_> {
    fn drop(&mut self) {
        if !self.ended {
            self.enc.get_mut().endEncoding();
        }
    }
}

impl Batch<'_> {
    fn bind(&self, index: usize, at: At<'_>) {
        // SAFETY: binding a live buffer at an in-bounds byte offset.
        unsafe {
            self.enc
                .borrow()
                .setBuffer_offset_atIndex(Some(&at.0.buf), at.1 * 4, index)
        };
    }

    fn bind_kv(&self, index: usize, kv: &KvBuf) {
        // SAFETY: binding a live buffer at offset 0.
        unsafe {
            self.enc
                .borrow()
                .setBuffer_offset_atIndex(Some(&kv.buf), 0, index)
        };
    }

    fn params<T>(&self, index: usize, p: &T) {
        // SAFETY: setBytes copies `size_of::<T>()` bytes immediately.
        unsafe {
            self.enc.borrow().setBytes_length_atIndex(
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
        if !self.free {
            self.enc
                .borrow()
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        self.enc.borrow().setComputePipelineState(pipeline);
        let size = |width, height| MTLSize {
            width,
            height,
            depth: 1,
        };
        let (grid, group) = (size(groups.0, groups.1), size(threads, 1));
        self.enc
            .borrow()
            .dispatchThreadgroups_threadsPerThreadgroup(grid, group);
        if profiling() {
            let repeat = profile_repeat();
            for _ in 1..repeat {
                let enc = self.enc.borrow();
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
            }
            // Run this dispatch alone and time it on the GPU's clock.
            let (cmd, enc) = self.gpu.encoder();
            self.enc.replace(enc).endEncoding();
            let cmd = self.cmd.replace(cmd);
            cmd.commit();
            cmd.waitUntilCompleted();
            let at = Retained::as_ptr(pipeline) as usize;
            let name = self.gpu.kernels.iter().find(|(p, _)| *p == at);
            let name = name.map_or("?", |(_, n)| n.as_str()).to_string();
            let mut times = PROFILE_TIMES.lock().unwrap_or_else(|p| p.into_inner());
            let e = times.entry(name).or_insert((0.0, 0));
            e.0 += (cmd.GPUEndTime() - cmd.GPUStartTime()) / f64::from(repeat);
            e.1 += 1;
        }
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
        self.matvec_impl(ty, rows, cols, offset, x, y, 0, y);
    }

    /// `y += W x`: a matvec that adds into the residual stream, saving the `add` after it.
    pub fn matvec_add(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: At<'_>,
        y: At<'_>,
    ) {
        self.matvec_impl(ty, rows, cols, offset, x, y, 1, y);
    }

    /// `y = W x + bias`, saving the `add_bcast` after the matvec.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_bias(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: At<'_>,
        y: At<'_>,
        bias: At<'_>,
    ) {
        assert!(bias.1 + rows <= bias.0.len);
        self.matvec_impl(ty, rows, cols, offset, x, y, 2, bias);
    }

    #[allow(clippy::too_many_arguments)]
    fn matvec_impl(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: At<'_>,
        y: At<'_>,
        add: u32,
        bias: At<'_>,
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
        let scale = self.gpu.scale(ty, offset);
        let (wbuf, offset) = self.gpu.weight_buf(offset, rows * row_bytes);
        // Rows a simdgroup takes: see `MATVEC2` and `MATVEC4` in metal.metal.
        let tall = rows >= MATVEC4_MIN_ROWS;
        let (pipeline, sg_rows) = match ty {
            GgmlType::Q4_K if tall => (self.gpu.other("matvec4_q4_k"), 4),
            GgmlType::Q6_K if tall => (self.gpu.other("matvec4_q6_k"), 4),
            GgmlType::Q8_0 if tall => (self.gpu.other("matvec4_q8_0"), 4),
            _ => {
                let p = &self
                    .gpu
                    .matvec
                    .iter()
                    .find(|(t, _)| *t == ty)
                    .expect("unsupported type for Metal")
                    .1;
                (p, if swiglu_kernel(ty).is_some() { 2 } else { 1 })
            }
        };
        self.enc.borrow().setComputePipelineState(pipeline);
        // SAFETY: weights buffer bound read-only at offset 0; the kernel adds `offset` itself.
        unsafe { self.enc.borrow().setBuffer_offset_atIndex(Some(wbuf), 0, 0) };
        self.bind(1, x);
        self.bind(2, y);
        self.bind(4, bias); // `y` again when there is none
        self.params(
            3,
            &MatvecParams {
                offset: offset as u64,
                rows: rows as u32,
                cols: cols as u32,
                row_bytes: row_bytes as u32,
                add,
                scale,
            },
        );
        self.dispatch(
            pipeline,
            rows.div_ceil(MATVEC_ROWS_PER_TG * sg_rows),
            MATVEC_ROWS_PER_TG * 32,
        );
    }

    /// Whether [`Batch::matvec_swiglu`] has a kernel for weights of type `ty`.
    pub fn has_matvec_swiglu(ty: GgmlType) -> bool {
        swiglu_kernel(ty).is_some()
    }

    /// `y = silu(G x) * (U x)` for one token: the gate and up matvecs of a SwiGLU FFN and the
    /// activation between them as one dispatch. `gate` and `up` are the offsets of two
    /// `rows x cols` matrices of type `ty`.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_swiglu(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        gate: usize,
        up: usize,
        x: At<'_>,
        y: At<'_>,
    ) {
        assert!(x.1 + cols <= x.0.len && y.1 + rows <= y.0.len);
        let row_bytes = ty.row_bytes(cols);
        let (gbuf, gate) = self.gpu.weight_buf(gate, rows * row_bytes);
        let (ubuf, up) = self.gpu.weight_buf(up, rows * row_bytes);
        // SAFETY: weight buffers bound read-only at offset 0; the kernel adds the offsets.
        unsafe {
            self.enc.borrow().setBuffer_offset_atIndex(Some(gbuf), 0, 0);
            self.enc.borrow().setBuffer_offset_atIndex(Some(ubuf), 0, 1);
        }
        self.bind(2, x);
        self.bind(3, y);
        self.params(
            4,
            &SwigluParams {
                gate: gate as u64,
                up: up as u64,
                rows: rows as u32,
                cols: cols as u32,
                row_bytes: row_bytes as u32,
            },
        );
        let kernel = swiglu_kernel(ty).expect("no SwiGLU matvec for this type");
        self.dispatch(
            self.gpu.other(kernel),
            rows.div_ceil(MATVEC_ROWS_PER_TG),
            MATVEC_ROWS_PER_TG * 32,
        );
    }

    /// `Y = X W^T` for `n` tokens: `X` is `n x cols` and `Y` is `n x rows`, both row-major.
    /// Below [`MATMUL_MIN_TOKENS`] tokens this is one matvec per token.
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
        self.matmul_impl(ty, rows, cols, offset, x, y, n, false);
    }

    /// `Y += X W^T`: [`Batch::matmul`] adding into `y`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_add(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: At<'_>,
        y: At<'_>,
        n: usize,
    ) {
        self.matmul_impl(ty, rows, cols, offset, x, y, n, true);
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_impl(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: At<'_>,
        y: At<'_>,
        n: usize,
        add: bool,
    ) {
        if n < MATMUL_MIN_TOKENS {
            for t in 0..n {
                let (xt, yt) = (At(x.0, x.1 + t * cols), At(y.0, y.1 + t * rows));
                if add {
                    self.matvec_add(ty, rows, cols, offset, xt, yt);
                } else {
                    self.matvec(ty, rows, cols, offset, xt, yt);
                }
            }
            return;
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
        let scale = self.gpu.scale(ty, offset);
        let (wbuf, offset) = self.gpu.weight_buf(offset, rows * row_bytes);
        // F32 and F16 weights keep f32 tiles: on them the activations' rounding would be the
        // only error there is.
        let half = self.half && !matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16);
        let kernels = if half {
            &self.gpu.matmul_half
        } else {
            &self.gpu.matmul
        };
        let tiles = &kernels
            .iter()
            .find(|(t, _)| *t == ty)
            .expect("unsupported type for Metal")
            .1;
        // Whole 64-token tiles, a 32-token one if the few-token kernel can take what it leaves,
        // then the rest: the few-token kernel if it is short enough, else the smallest tile
        // that holds it.
        let largest = MATMUL_TILE_TOKENS[MATMUL_TILE_TOKENS.len() - 1];
        let full = n / largest * largest;
        let can_few = cols.is_multiple_of(64);
        // 40 tokens ran 1.3x faster that way than padded into a 64-token tile; 56 (32 and 24)
        // 4% slower on Mistral-7B.
        let split = can_few && (32..=48).contains(&(n - full));
        let mid = if split { full + 32 } else { full };
        let rest = n - mid;
        // Where the 32-token tile is `MATMUL_K`, it takes over from the few-token kernel at 17
        // tokens: 2.5-3.7% faster on Mistral-7B, 2% slower on Llama-3.2-1B.
        let k_tile = half && matches!(ty, GgmlType::Q4_K | GgmlType::Q6_K);
        let few_max = if k_tile { 16 } else { MATMUL_FEW_TOKENS };
        let few = rest <= few_max && can_few;
        // (pipeline, rows and tokens of a threadgroup, its simdgroups)
        let rest_kernel = if few {
            let pair = &self.gpu.matmul_few;
            let pair = &pair.iter().find(|(t, _)| *t == ty).expect("as above").1;
            let groups = FEW_SIMDGROUPS.min(cols / 64);
            // No rest (a multiple of 64 tokens): this kernel is then not run.
            let rt = rest.div_ceil(8).max(1);
            (&pair[rt - 1], FEW_ROWS, 8 * rt, groups)
        } else {
            let tile = MATMUL_TILE_TOKENS
                .iter()
                .position(|&t| t >= rest)
                .expect("the rest is under the largest tile");
            let tokens = MATMUL_TILE_TOKENS[tile];
            (&tiles[tile], MATMUL_TILE_ROWS, tokens, MATMUL_THREADS / 32)
        };
        let full_kernel = (
            &tiles[MATMUL_TILE_TOKENS.len() - 1],
            MATMUL_TILE_ROWS,
            largest,
            MATMUL_THREADS / 32,
        );
        let mid_kernel = (&tiles[2], MATMUL_TILE_ROWS, 32, MATMUL_THREADS / 32);
        let free = self.free;
        for (tok0, end, (pipeline, tile_rows, tile_tokens, groups)) in [
            (0, full, full_kernel),
            (full, mid, mid_kernel),
            (mid, n, rest_kernel),
        ] {
            if tok0 == end {
                continue;
            }
            self.enc.borrow().setComputePipelineState(pipeline);
            // SAFETY: weights buffer bound read-only at offset 0; the kernel adds `offset`.
            unsafe { self.enc.borrow().setBuffer_offset_atIndex(Some(wbuf), 0, 0) };
            self.bind(1, x);
            self.bind(2, y);
            self.params(
                3,
                &MatmulParams {
                    offset: offset as u64,
                    rows: rows as u32,
                    cols: cols as u32,
                    row_bytes: row_bytes as u32,
                    tok0: tok0 as u32,
                    n: end as u32,
                    add: add as u32,
                    nsg: groups as u32,
                    scale,
                },
            );
            let grid = (rows.div_ceil(tile_rows), (end - tok0).div_ceil(tile_tokens));
            self.dispatch2(pipeline, grid, 32 * groups);
            // The rest writes other rows of y, so it runs beside the whole tiles, on weights
            // they have just read.
            self.free = true;
        }
        self.free = free;
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
        let (wbuf, offset) = self.gpu.weight_buf(offset, span);
        let pipeline = &self
            .gpu
            .matvec_id
            .iter()
            .find(|(t, _)| *t == ty)
            .expect("unsupported type for Metal")
            .1;
        self.enc.borrow().setComputePipelineState(pipeline);
        // SAFETY: weights buffer bound read-only at offset 0; the kernel adds `offset` itself.
        unsafe { self.enc.borrow().setBuffer_offset_atIndex(Some(wbuf), 0, 0) };
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
        // The types with a pair kernel take two rows a simdgroup.
        let sg_rows = if swiglu_kernel(ty).is_some() { 2 } else { 1 };
        self.dispatch2(
            pipeline,
            (rows.div_ceil(MATVEC_ROWS_PER_TG * sg_rows), slots),
            MATVEC_ROWS_PER_TG * 32,
        );
    }

    /// `y += Σ_slot wt[slot] · (W[ids[slot]] x[slot])`: [`Batch::matvec_id`] over `slots`
    /// slots and the weighted sum of its results, as one dispatch. Only for the types with
    /// [`Batch::has_matvec_swiglu`].
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_id_add(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        expert_stride: usize,
        n_expert: usize,
        ids: At<'_>,
        wt: At<'_>,
        x: At<'_>,
        x_stride: usize,
        y: At<'_>,
        slots: usize,
    ) {
        assert!(slots >= 1 && ids.1 + slots <= ids.0.len && wt.1 + slots <= wt.0.len);
        assert!(x.1 + (slots - 1) * x_stride + cols <= x.0.len && y.1 + rows <= y.0.len);
        let row_bytes = ty.row_bytes(cols);
        let span = (n_expert - 1) * expert_stride + rows * row_bytes;
        let (wbuf, offset) = self.gpu.weight_buf(offset, span);
        // SAFETY: weights buffer bound read-only at offset 0; the kernel adds `offset` itself.
        unsafe { self.enc.borrow().setBuffer_offset_atIndex(Some(wbuf), 0, 0) };
        self.bind(1, x);
        self.bind(2, y);
        self.params(
            3,
            &IdAddParams {
                offset: offset as u64,
                expert_stride: expert_stride as u64,
                rows: rows as u32,
                cols: cols as u32,
                row_bytes: row_bytes as u32,
                x_stride: x_stride as u32,
                slots: slots as u32,
            },
        );
        self.bind(4, ids);
        self.bind(5, wt);
        let kernel = id_add_kernel(ty).expect("no adding id matvec for this type");
        self.dispatch(
            self.gpu.other(kernel),
            rows.div_ceil(MATVEC_ROWS_PER_TG * 2),
            MATVEC_ROWS_PER_TG * 32,
        );
    }

    /// [`Batch::matvec_swiglu`] over the chosen experts of stacked gate and up tensors (see
    /// [`Batch::matvec_id`]): `y[slot] = silu(G[ids[slot]] x) * (U[ids[slot]] x)`, every slot
    /// reading the one `x`, outputs `y_stride` floats apart.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_swiglu_id(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        [gate, up]: [usize; 2],
        [gate_stride, up_stride]: [usize; 2],
        n_expert: usize,
        ids: At<'_>,
        x: At<'_>,
        y: At<'_>,
        y_stride: usize,
        slots: usize,
    ) {
        assert!(slots >= 1 && ids.1 + slots <= ids.0.len && x.1 + cols <= x.0.len);
        assert!(y.1 + (slots - 1) * y_stride + rows <= y.0.len);
        let row_bytes = ty.row_bytes(cols);
        let span = |stride| (n_expert - 1) * stride + rows * row_bytes;
        let (gbuf, gate) = self.gpu.weight_buf(gate, span(gate_stride));
        let (ubuf, up) = self.gpu.weight_buf(up, span(up_stride));
        // SAFETY: weight buffers bound read-only at offset 0; the kernel adds the offsets.
        unsafe {
            self.enc.borrow().setBuffer_offset_atIndex(Some(gbuf), 0, 0);
            self.enc.borrow().setBuffer_offset_atIndex(Some(ubuf), 0, 1);
        }
        self.bind(2, x);
        self.bind(3, y);
        self.params(
            4,
            &SwigluIdParams {
                gate: gate as u64,
                up: up as u64,
                gate_stride: gate_stride as u64,
                up_stride: up_stride as u64,
                rows: rows as u32,
                cols: cols as u32,
                row_bytes: row_bytes as u32,
                y_stride: y_stride as u32,
            },
        );
        self.bind(5, ids);
        let kernel = swiglu_id_kernel(ty).expect("no SwiGLU matvec for this type");
        self.dispatch2(
            self.gpu.other(kernel),
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

    /// Row `t` of `x` += `Σ_{j<k} w[t·k + j] · src[ids[t·k + j]]` for `n` rows of `dim` (a
    /// multiple of 4): each token's chosen experts' outputs, combined. `ids` must index rows
    /// within `src`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine_rows(
        &mut self,
        x: At<'_>,
        src: At<'_>,
        ids: At<'_>,
        w: At<'_>,
        n: usize,
        dim: usize,
        k: usize,
    ) {
        assert!(dim.is_multiple_of(4) && x.1 + n * dim <= x.0.len);
        assert!(ids.1 + n * k <= ids.0.len && w.1 + n * k <= w.0.len);
        self.bind(0, x);
        self.bind(1, src);
        self.bind(2, ids);
        self.bind(3, w);
        self.params(
            4,
            &CombineRowsParams {
                n: n as u32,
                dim: dim as u32,
                k: k as u32,
            },
        );
        self.dispatch(
            self.gpu.other("moe_combine_rows"),
            (n * dim).div_ceil(4 * ELT_THREADS),
            ELT_THREADS,
        );
    }

    /// RMS-normalizes `rows` consecutive rows of `n` from `x` into `out` (may alias).
    pub fn rmsnorm(&mut self, x: At<'_>, out: At<'_>, w: &GpuBuf, rows: usize, n: usize, eps: f32) {
        self.rmsnorm_strided(x, out, w, rows, n, n, eps);
    }

    /// [`Batch::rmsnorm`] of the first `n` values of rows `stride` apart, in `x` and `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_strided(
        &mut self,
        x: At<'_>,
        out: At<'_>,
        w: &GpuBuf,
        rows: usize,
        n: usize,
        stride: usize,
        eps: f32,
    ) {
        assert!(n <= stride && rows >= 1);
        let end = (rows - 1) * stride + n;
        assert!(x.1 + end <= x.0.len && out.1 + end <= out.0.len && n <= w.len);
        self.bind(0, x);
        self.bind(1, out);
        self.bind(2, w.into());
        let stride = stride as u32;
        self.params(
            3,
            &NormParams {
                n: n as u32,
                eps,
                stride,
            },
        );
        // A thread per two values, up to the largest threadgroup: one that reads several
        // waits on each read in turn (2048 values: 3.7 us a dispatch with 256 threads, 2.7
        // with 1024).
        let threads = (n / 2).next_multiple_of(32).clamp(32, REDUCE_THREADS);
        self.dispatch(self.gpu.other("rmsnorm"), rows, threads);
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
        // Four values a thread.
        self.dispatch(
            self.gpu.other(kernel),
            n.div_ceil(4 * ELT_THREADS),
            ELT_THREADS,
        );
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

    /// `w` values of each of `m` tokens' `nh` heads from `src` laid out as `from` into `dst`
    /// laid out as `to`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_heads(
        &mut self,
        src: At<'_>,
        from: Heads,
        dst: At<'_>,
        to: Heads,
        m: usize,
        nh: usize,
        w: usize,
    ) {
        let last = |h: Heads| (m - 1) * h.row + (nh - 1) * h.head + h.off + w;
        assert!(m >= 1 && nh >= 1 && src.1 + last(from) <= src.0.len);
        assert!(dst.1 + last(to) <= dst.0.len);
        self.bind(0, src);
        self.bind(1, dst);
        let u = |v: usize| u32::try_from(v).expect("fits u32");
        self.params(
            2,
            &HeadsParams {
                m: u(m),
                nh: u(nh),
                w: u(w),
                in_row: u(from.row),
                in_head: u(from.head),
                in_off: u(from.off),
                out_row: u(to.row),
                out_head: u(to.head),
                out_off: u(to.off),
            },
        );
        let n = m * nh * w;
        self.dispatch(
            self.gpu.other("copy_heads"),
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
        assert!(dim.is_multiple_of(4));
        self.params(
            3,
            &EltParams {
                n: (rows * dim) as u32,
                alpha: 0.0,
                m: dim as u32,
            },
        );
        self.dispatch(
            self.gpu.other("gather_rows"),
            (rows * dim).div_ceil(4 * ELT_THREADS),
            ELT_THREADS,
        );
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

    /// [`Batch::rope`] of `x`, written to the cache `dst` from element `off` instead of in
    /// place. Needs `rope_dim == head_dim`.
    pub fn rope_store(&mut self, x: At<'_>, dst: &KvBuf, off: usize, r: &RopeArgs<'_>) {
        let n = r.n_tokens * r.n_heads * r.head_dim;
        assert!(r.rope_dim == r.head_dim && x.1 + n <= x.0.len && off + n <= dst.len);
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
        // SAFETY: binding a live buffer at an in-bounds byte offset.
        unsafe {
            self.enc
                .borrow()
                .setBuffer_offset_atIndex(Some(&dst.buf), off * dst.elem(), 3)
        };
        let kernel = if dst.half {
            "rope_store_f16"
        } else {
            "rope_store_f32"
        };
        let pairs = r.n_heads * r.rope_dim / 2;
        self.dispatch2(
            self.gpu.other(kernel),
            (pairs.div_ceil(ELT_THREADS), r.n_tokens),
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
                .borrow()
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
        self.dispatch(
            self.gpu.other(kernel),
            n.div_ceil(4 * ELT_THREADS),
            ELT_THREADS,
        );
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
        self.bind(5, out);
        let kernel = if k.half {
            "attention_part_f16"
        } else {
            "attention_part_f32"
        };
        self.dispatch2(self.gpu.other(kernel), (n_head, n_chunks), ATTN_THREADS);
        if n_chunks == 1 {
            return; // the kernel wrote `out` itself
        }
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
    /// caches (rows `0..p.pos + n` filled, and allocated up to the next multiple of 8 rows:
    /// the kernel reads keys 8 at a time). Needs `head_dim % 8 == 0` and `head_dim <= 256`.
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
            hd.is_multiple_of(8) && hd <= 256,
            "unsupported head_dim {hd}"
        );
        assert!(q.1 + n * q_dim <= q.0.len && out.1 + n * q_dim <= out.0.len);
        let rows = (p.pos as usize + n).next_multiple_of(8);
        assert!(k.half == v.half && rows * p.kv_dim as usize <= k.len.min(v.len));
        self.bind(0, q);
        self.bind_kv(1, k);
        self.bind_kv(2, v);
        self.bind(3, out);
        #[repr(C)]
        struct Params {
            p: BatchAttnParams,
            n: u32,
        }
        self.params(4, &Params { p: *p, n: n as u32 });
        let tiles = n.div_ceil(8);
        if k.half && ATTN_STAGED_DIMS.contains(&hd) {
            let subs = (p.n_head / p.n_kv).div_ceil(ATTN_STAGED_HEADS) as usize;
            let grid = (p.n_kv as usize * subs, tiles);
            let threads = 32 * ATTN_STAGED_HEADS as usize;
            let kernel = format!("attention_batch_staged_{hd}");
            self.dispatch2(self.gpu.other(&kernel), grid, threads);
            return;
        }
        let kernel = if k.half {
            "attention_batch_f16"
        } else {
            "attention_batch_f32"
        };
        self.dispatch2(self.gpu.other(kernel), (p.n_head as usize, tiles), 32);
    }

    /// Submits what is recorded so far without waiting, and goes on recording: the GPU starts
    /// on it while the caller queues the rest (about 0.1 ms of a decoded token otherwise spent
    /// with the GPU idle).
    pub fn start(&mut self) {
        self.enc.get_mut().endEncoding();
        self.cmd.get_mut().commit();
        let (cmd, enc) = self.gpu.encoder();
        self.started.push(self.cmd.replace(cmd));
        *self.enc.get_mut() = enc;
    }

    /// Submits the recorded work and waits for it.
    pub fn finish(mut self) {
        self.enc.get_mut().endEncoding();
        self.ended = true;
        let last = self.cmd.get_mut();
        last.commit();
        last.waitUntilCompleted();
        // The queue runs command buffers in order, so the earlier ones are done too.
        for cmd in self.started.iter().chain([&*last]) {
            if let Some(e) = cmd.error() {
                panic!("Metal command failed: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_GAP, buffer_spans};

    #[test]
    fn buffer_spans_pack_units_in_file_order_up_to_the_cap() {
        let p = super::page_size();
        // Out of order; 0..p and p..2p fit one 2-page span, the third starts a new one.
        let units = [p..2 * p, 0..p, 2 * p + 10..3 * p];
        assert_eq!(
            buffer_spans(&units, 2 * p).unwrap(),
            vec![0..2 * p, 2 * p + 10..3 * p]
        );
        // A cap that holds everything gives one span.
        assert_eq!(buffer_spans(&units, 3 * p).unwrap(), vec![0..3 * p]);
    }

    #[test]
    fn buffer_spans_split_at_large_gaps_and_reject_oversized_units() {
        let p = super::page_size();
        let far = MAX_GAP + 2 * p;
        assert_eq!(
            buffer_spans(&[0..p, far..far + p], usize::MAX).unwrap(),
            vec![0..p, far..far + p]
        );
        assert!(buffer_spans(std::slice::from_ref(&(0..3 * p)), 2 * p).is_err());
        assert!(buffer_spans(&[], p).unwrap().is_empty());
    }
}
