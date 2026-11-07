//! CUDA backend primitives (feature `cuda`), mirroring [`crate::metal`].
//!
//! Unlike unified memory, VRAM is separate: only the weights of the offloaded layers are copied
//! to the device (one arena), so a low-VRAM card can hold part of a model while the rest stays
//! in the CPU's file mapping. The driver and NVRTC are loaded at runtime, so building with this
//! feature needs no CUDA toolkit; running needs an NVIDIA driver plus the NVRTC library.
//!
//! UNVERIFIED: this was written and type-checked without an NVIDIA GPU. The kernels mirror the
//! Metal ones (which are tested against the CPU), but have not been executed.

use crate::quant::GgmlType;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchArgs, LaunchConfig,
    PushKernelArg,
};
use std::sync::Arc;

const SOURCE: &str = include_str!("cuda.cu");
const MATVEC_ROWS_PER_BLOCK: usize = 4;
const REDUCE_THREADS: u32 = 256;
const ELT_THREADS: usize = 256;
/// Weight placement alignment in the device arena (f32 weights are read as `float*`).
const WEIGHT_ALIGN: usize = 256;

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
const OTHER_KERNELS: [&str; 5] = ["rmsnorm", "axpy", "swiglu", "rope", "attention"];

fn err(what: &str) -> impl Fn(DriverError) -> String + '_ {
    move |e| format!("{what}: {e:?}")
}

/// A device f32 buffer.
pub struct GpuBuf(CudaSlice<f32>);

impl GpuBuf {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
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

/// An uploaded byte range of the model file: `[file_off, file_off + len)` lives at `dev_off`.
struct Placed {
    file_off: usize,
    len: usize,
    dev_off: usize,
}

pub struct Gpu {
    stream: Arc<CudaStream>,
    weights: CudaSlice<u8>,
    placed: Vec<Placed>,
    matvec: Vec<(GgmlType, CudaFunction)>,
    other: Vec<(&'static str, CudaFunction)>,
    name: String,
}

impl Gpu {
    /// Opens device 0, compiles the kernels and copies the byte `ranges` (offset, len) of
    /// `mapped` (the model file) into VRAM. Matvecs may only read weights inside those ranges.
    pub fn new(mapped: &[u8], ranges: &[(usize, usize)]) -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(err("no CUDA device"))?;
        let stream = ctx.default_stream();
        let ptx = cudarc::nvrtc::compile_ptx(SOURCE)
            .map_err(|e| format!("CUDA kernel compile failed: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(err("load CUDA module"))?;
        let func = |n: &str| module.load_function(n).map_err(err(n));
        let matvec = MATVEC_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, func(n)?)))
            .collect::<Result<_, String>>()?;
        let other = OTHER_KERNELS
            .iter()
            .map(|&n| Ok((n, func(n)?)))
            .collect::<Result<_, String>>()?;

        let mut ranges = ranges.to_vec();
        ranges.sort_unstable();
        ranges.dedup();
        let mut placed = Vec::with_capacity(ranges.len());
        let mut total = 0;
        for &(file_off, len) in &ranges {
            assert!(file_off + len <= mapped.len(), "weight range out of file");
            placed.push(Placed {
                file_off,
                len,
                dev_off: total,
            });
            total = (total + len).next_multiple_of(WEIGHT_ALIGN);
        }
        let (free, _) = ctx.mem_get_info().map_err(err("query VRAM"))?;
        if total > free {
            return Err(format!(
                "offloaded weights need {total} bytes of VRAM but {free} are free; lower --gpu-layers"
            ));
        }
        let mut weights = stream
            .alloc_zeros::<u8>(total.max(1))
            .map_err(err("allocate weight arena"))?;
        for p in &placed {
            let src = &mapped[p.file_off..p.file_off + p.len];
            stream
                .memcpy_htod(src, &mut weights.slice_mut(p.dev_off..p.dev_off + p.len))
                .map_err(err("upload weights"))?;
        }
        let name = format!("cuda ({})", ctx.name().map_err(err("device name"))?);
        Ok(Self {
            stream,
            weights,
            placed,
            matvec,
            other,
            name,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn supports(ty: GgmlType) -> bool {
        MATVEC_KERNELS.iter().any(|(t, _)| *t == ty)
    }

    /// A zeroed device buffer of `n` f32s.
    pub fn alloc(&self, n: usize) -> GpuBuf {
        GpuBuf(
            self.stream
                .alloc_zeros(n.max(1))
                .expect("CUDA buffer allocation failed"),
        )
    }

    pub fn upload(&self, data: &[f32]) -> GpuBuf {
        GpuBuf(self.stream.clone_htod(data).expect("CUDA upload failed"))
    }

    /// Copies `data` into the start of `dst` (ordered after queued work on the stream).
    pub fn write(&self, dst: &mut GpuBuf, data: &[f32]) {
        self.stream
            .memcpy_htod(data, &mut dst.0.slice_mut(..data.len()))
            .expect("CUDA upload failed");
    }

    /// Copies the start of `src` into `out`, waiting for queued work.
    pub fn read(&self, src: &GpuBuf, out: &mut [f32]) {
        self.stream
            .memcpy_dtoh(&src.0.slice(..out.len()), out)
            .expect("CUDA download failed");
        self.stream.synchronize().expect("CUDA sync failed");
    }

    pub fn batch(&self) -> Batch<'_> {
        Batch { gpu: self }
    }

    fn other(&self, name: &str) -> &CudaFunction {
        &self
            .other
            .iter()
            .find(|(n, _)| *n == name)
            .expect("kernel compiled in new()")
            .1
    }

    /// Device offset of file bytes `[off, off + len)`, which must lie in one uploaded range.
    fn device_offset(&self, off: usize, len: usize) -> usize {
        let i = self.placed.partition_point(|p| p.file_off <= off);
        let p = i
            .checked_sub(1)
            .map(|i| &self.placed[i])
            .filter(|p| off + len <= p.file_off + p.len)
            .expect("weights were not uploaded to the GPU");
        p.dev_off + (off - p.file_off)
    }

    /// One-shot `y = W x` (for tests and benchmarks; layers should use a batch).
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
        self.read(&yb, y);
    }
}

fn grid(blocks: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch(mut args: LaunchArgs<'_>, cfg: LaunchConfig) {
    // SAFETY: each caller passes exactly the parameters of the kernel's signature in cuda.cu,
    // with buffer bounds asserted or sized by construction.
    unsafe { args.launch(cfg) }.expect("CUDA launch failed");
}

/// Queued GPU work on the device stream; kernels run in order. [`Batch::finish`] waits.
pub struct Batch<'a> {
    gpu: &'a Gpu,
}

impl Batch<'_> {
    /// `y = W x`, `W` being `rows x cols` of `ty` at byte `offset` of the model file.
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
            x.1 + cols <= x.0.len() && y.1 + rows <= y.0.len(),
            "matvec buffer too small"
        );
        let row_bytes = ty.row_bytes(cols);
        let w_off = self.gpu.device_offset(offset, rows * row_bytes) as u64;
        let f = &self
            .gpu
            .matvec
            .iter()
            .find(|(t, _)| *t == ty)
            .expect("unsupported type for CUDA")
            .1;
        let (x_off, y_off) = (x.1 as u64, y.1 as u64);
        let (rows_u, cols_u, rb) = (rows as u32, cols as u32, row_bytes as u32);
        let mut a = self.gpu.stream.launch_builder(f);
        a.arg(&self.gpu.weights)
            .arg(&w_off)
            .arg(&x.0.0)
            .arg(&x_off)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&rb);
        launch(
            a,
            grid(
                rows.div_ceil(MATVEC_ROWS_PER_BLOCK),
                (MATVEC_ROWS_PER_BLOCK * 32) as u32,
            ),
        );
    }

    /// RMS-normalizes `rows` consecutive rows of `n` from `x` into `out` (may alias).
    pub fn rmsnorm(&mut self, x: At<'_>, out: At<'_>, w: &GpuBuf, rows: usize, n: usize, eps: f32) {
        assert!(x.1 + rows * n <= x.0.len() && out.1 + rows * n <= out.0.len());
        let (x_off, o_off, n_u) = (x.1 as u64, out.1 as u64, n as u32);
        let mut a = self.gpu.stream.launch_builder(self.gpu.other("rmsnorm"));
        a.arg(&x.0.0)
            .arg(&x_off)
            .arg(&out.0.0)
            .arg(&o_off)
            .arg(&w.0)
            .arg(&n_u)
            .arg(&eps);
        launch(a, grid(rows, REDUCE_THREADS));
    }

    /// `a += alpha * b`
    pub fn axpy(&mut self, a: At<'_>, b: At<'_>, alpha: f32, n: usize) {
        assert!(a.1 + n <= a.0.len() && b.1 + n <= b.0.len());
        let (a_off, b_off, n_u) = (a.1 as u64, b.1 as u64, n as u32);
        let mut l = self.gpu.stream.launch_builder(self.gpu.other("axpy"));
        l.arg(&a.0.0)
            .arg(&a_off)
            .arg(&b.0.0)
            .arg(&b_off)
            .arg(&n_u)
            .arg(&alpha);
        launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// `a += b`
    pub fn add(&mut self, a: At<'_>, b: At<'_>, n: usize) {
        self.axpy(a, b, 1.0, n);
    }

    /// `gate = silu(gate) * up`
    pub fn swiglu(&mut self, gate: At<'_>, up: At<'_>, n: usize) {
        assert!(gate.1 + n <= gate.0.len() && up.1 + n <= up.0.len());
        let (g_off, u_off, n_u) = (gate.1 as u64, up.1 as u64, n as u32);
        let mut l = self.gpu.stream.launch_builder(self.gpu.other("swiglu"));
        l.arg(&gate.0.0)
            .arg(&g_off)
            .arg(&up.0.0)
            .arg(&u_off)
            .arg(&n_u);
        launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    pub fn rope(&mut self, x: At<'_>, r: &RopeArgs<'_>) {
        assert!(x.1 + r.n_heads * r.head_dim <= x.0.len());
        let x_off = x.1 as u64;
        let ff = r.freq_factors.unwrap_or(x.0);
        let has_ff = r.freq_factors.is_some() as u32;
        let (nh, hd, rd, pos) = (
            r.n_heads as u32,
            r.head_dim as u32,
            r.rope_dim as u32,
            r.pos as u32,
        );
        let neox = r.neox as u32;
        let mut l = self.gpu.stream.launch_builder(self.gpu.other("rope"));
        l.arg(&x.0.0)
            .arg(&x_off)
            .arg(&ff.0)
            .arg(&has_ff)
            .arg(&nh)
            .arg(&hd)
            .arg(&rd)
            .arg(&pos)
            .arg(&r.theta)
            .arg(&neox);
        let n = r.n_heads * r.rope_dim / 2;
        launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// Causal attention for one query position over `p.n_pos` cached positions. `q`, `k`, `v`
    /// and `out` start at element 0 of their buffers.
    pub fn attention(
        &mut self,
        q: &GpuBuf,
        k: &GpuBuf,
        v: &GpuBuf,
        out: &GpuBuf,
        scores: &GpuBuf,
        p: &AttnParams,
    ) {
        assert!(scores.len() >= (p.n_head * p.max_ctx) as usize);
        assert!(k.len() >= (p.n_pos * p.kv_dim) as usize && v.len() >= k.len());
        let mut l = self.gpu.stream.launch_builder(self.gpu.other("attention"));
        l.arg(&q.0)
            .arg(&k.0)
            .arg(&v.0)
            .arg(&out.0)
            .arg(&scores.0)
            .arg(&p.n_head)
            .arg(&p.n_kv)
            .arg(&p.head_dim)
            .arg(&p.n_pos)
            .arg(&p.kv_dim)
            .arg(&p.max_ctx)
            .arg(&p.scale);
        launch(l, grid(p.n_head as usize, REDUCE_THREADS));
    }

    /// Waits for all queued work.
    pub fn finish(self) {
        self.gpu.stream.synchronize().expect("CUDA command failed");
    }
}
