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
const OTHER_KERNELS: [&str; 6] = ["rmsnorm", "add", "axpy", "swiglu", "rope", "attention"];

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

/// A view into a [`GpuBuf`] starting at element `off`.
#[derive(Clone, Copy)]
pub struct At<'a>(pub &'a GpuBuf, pub usize);

impl<'a> From<&'a GpuBuf> for At<'a> {
    fn from(b: &'a GpuBuf) -> Self {
        At(b, 0)
    }
}

pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    weights: Retained<ProtocolObject<dyn MTLBuffer>>,
    weights_len: usize,
    matvec: Vec<(GgmlType, Pipeline)>,
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
struct NormParams {
    n: u32,
    eps: f32,
}
#[repr(C)]
struct EltParams {
    n: u32,
    alpha: f32,
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
    pub max_ctx: u32,
    pub scale: f32,
}

/// Rotary embedding settings for [`Batch::rope`].
pub struct RopeArgs<'a> {
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub pos: usize,
    pub theta: f32,
    pub neox: bool,
    pub freq_factors: Option<&'a GpuBuf>,
}

impl Gpu {
    /// Wraps `mapped` (a page-aligned memory mapping, e.g. the whole GGUF file) for GPU access.
    ///
    /// # Safety
    /// `mapped` must be the start of a memory mapping that stays valid and unmodified for the
    /// lifetime of the returned value; the pages up to the next page boundary must be mapped
    /// (true for any `mmap` of the file).
    pub unsafe fn new(mapped: &[u8]) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
        let page = unsafe { getpagesize() } as usize;
        if !(mapped.as_ptr() as usize).is_multiple_of(page) {
            return Err("weights mapping is not page-aligned".into());
        }
        let len = mapped.len().div_ceil(page) * page;
        if len > device.maxBufferLength() {
            return Err(format!(
                "model ({len} bytes) exceeds Metal maxBufferLength ({})",
                device.maxBufferLength()
            ));
        }
        let ptr = NonNull::new(mapped.as_ptr() as *mut std::ffi::c_void).ok_or("null mapping")?;
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
            weights_len: mapped.len(),
            matvec,
            other,
            name,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
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

/// Recorded GPU work; dispatches run in order. Nothing executes until [`Batch::finish`].
pub struct Batch<'a> {
    gpu: &'a Gpu,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl Batch<'_> {
    fn bind(&self, index: usize, at: At<'_>) {
        // SAFETY: binding a live buffer at an in-bounds byte offset.
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&at.0.buf), at.1 * 4, index)
        };
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
        self.enc.setComputePipelineState(pipeline);
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
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
        assert!(
            offset + rows * row_bytes <= self.gpu.weights_len,
            "weights out of range"
        );
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
        self.params(2, &EltParams { n: n as u32, alpha });
        self.dispatch(self.gpu.other(kernel), n.div_ceil(ELT_THREADS), ELT_THREADS);
    }

    /// `a += b`
    pub fn add(&mut self, a: At<'_>, b: At<'_>, n: usize) {
        self.elementwise("add", a, b, n, 1.0);
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
        let n = r.n_heads * r.rope_dim / 2;
        self.dispatch(self.gpu.other("rope"), n.div_ceil(ELT_THREADS), ELT_THREADS);
    }

    /// Causal attention for one query position over `p.n_pos` cached positions.
    pub fn attention(
        &mut self,
        q: At<'_>,
        k: At<'_>,
        v: At<'_>,
        out: At<'_>,
        scores: &GpuBuf,
        p: &AttnParams,
    ) {
        assert!(scores.len >= (p.n_head * p.max_ctx) as usize);
        self.bind(0, q);
        self.bind(1, k);
        self.bind(2, v);
        self.bind(3, out);
        self.bind(4, scores.into());
        self.params(5, p);
        self.dispatch(
            self.gpu.other("attention"),
            p.n_head as usize,
            REDUCE_THREADS,
        );
    }

    /// Submits the recorded work and waits for it.
    pub fn finish(self) {
        self.enc.endEncoding();
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        if let Some(e) = self.cmd.error() {
            panic!("Metal command failed: {e}");
        }
    }
}
