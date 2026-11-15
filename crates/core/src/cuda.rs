//! CUDA backend primitives (feature `cuda`), mirroring [`crate::metal`].
//!
//! Unlike unified memory, VRAM is separate: only the weights of the offloaded layers are copied
//! to the device (one arena), so a low-VRAM card can hold part of a model while the rest stays
//! in the CPU's file mapping. The driver and NVRTC are loaded at runtime, so building with this
//! feature needs no CUDA toolkit; running needs an NVIDIA driver plus the NVRTC library.
//!
//! CI only compiles this: it runs when someone has an NVIDIA GPU (first on an A100, 2026-11-15).

use crate::quant::GgmlType;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchArgs, LaunchConfig,
    PushKernelArg,
};
use std::sync::Arc;

const SOURCE: &str = include_str!("cuda.cu");
const MATVEC_ROWS_PER_BLOCK: usize = 4;
const REDUCE_THREADS: u32 = 256;
/// Threads per attention block, which is also the largest head_dim the kernel takes.
pub const ATTN_THREADS: u32 = 1024;
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
const OTHER_KERNELS: [&str; 8] = [
    "rmsnorm",
    "axpy",
    "add_bias",
    "swiglu",
    "rope",
    "attention",
    "moe_topk",
    "moe_combine",
];
/// Most experts `moe_topk` can choose among.
pub const MOE_MAX_EXPERTS: usize = 2048;
/// Tokens per block of the matmul kernels (`TILE` in cuda.cu).
const MATMUL_TILE: usize = 32;

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

/// A device u32 buffer: the expert ids [`Batch::moe_topk`] chooses.
pub struct GpuIds(CudaSlice<u32>);

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
    /// Query tokens, at consecutive positions.
    pub n_tok: u32,
    /// Cached positions the first token attends to; each later token sees one more.
    pub n_pos: u32,
    pub kv_dim: u32,
    pub max_ctx: u32,
    pub scale: f32,
}

/// Rotary embedding settings for [`Batch::rope`].
pub struct RopeArgs<'a> {
    /// Consecutive tokens in the buffer, `n_heads` heads each, the first at `pos`.
    pub n_tok: usize,
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
    /// Per weight type: the matvec kernel and the matmul one.
    matvec: Vec<(GgmlType, CudaFunction, CudaFunction)>,
    other: Vec<(&'static str, CudaFunction)>,
    /// One zero: the expert id of a plain matvec.
    zero_id: GpuIds,
    name: String,
}

impl Gpu {
    /// Opens device 0, compiles the kernels and copies the byte `ranges` (offset, len) of
    /// `mapped` (the model file) into VRAM. Matvecs may only read weights inside those ranges.
    pub fn new(mapped: &[u8], ranges: &[(usize, usize)]) -> Result<Self, String> {
        // cudarc panics on first use if a library is missing; probe so that becomes an error.
        // SAFETY: only attempts to load the driver/NVRTC libraries (running their initializers),
        // which cudarc would do anyway on first use.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err("CUDA driver library (libcuda) not found".into());
        }
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err("NVRTC library (libnvrtc) not found".into());
        }
        let ctx = CudaContext::new(0).map_err(err("no CUDA device"))?;
        let stream = ctx.default_stream();
        let ptx = cudarc::nvrtc::compile_ptx(SOURCE)
            .map_err(|e| format!("CUDA kernel compile failed: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(err("load CUDA module"))?;
        let func = |n: &str| module.load_function(n).map_err(err(n));
        let matvec = MATVEC_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, func(n)?, func(&n.replace("matvec", "matmul"))?)))
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
            // Ranges that touch in the file stay together, so a stack of experts keeps its
            // stride on the device.
            let last: Option<&mut Placed> = placed.last_mut();
            let p = match last {
                Some(p) if p.file_off + p.len == file_off => {
                    p.len += len;
                    p
                }
                _ => {
                    placed.push(Placed {
                        file_off,
                        len,
                        dev_off: total,
                    });
                    placed.last_mut().expect("just pushed")
                }
            };
            total = (p.dev_off + p.len).next_multiple_of(WEIGHT_ALIGN);
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
        let zero_id = GpuIds(stream.alloc_zeros(1).map_err(err("allocate"))?);
        let name = format!("cuda ({})", ctx.name().map_err(err("device name"))?);
        Ok(Self {
            stream,
            weights,
            placed,
            matvec,
            other,
            zero_id,
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

    /// A zeroed device buffer of `n` expert ids.
    pub fn alloc_ids(&self, n: usize) -> GpuIds {
        GpuIds(
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
    grid2(blocks, 1, threads)
}

fn grid2(x: usize, y: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (x as u32, y as u32, 1),
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
        self.matmul(ty, rows, cols, offset, x, y, 1);
    }

    /// [`Batch::matvec`] for `n_tok` tokens: `x` is `n_tok` rows of `cols`, `y` `n_tok` rows of
    /// `rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        x: At<'_>,
        y: At<'_>,
        n_tok: usize,
    ) {
        if n_tok == 1 {
            let zero = &self.gpu.zero_id;
            return self.matvec_id(ty, rows, cols, offset, 0, 1, zero, x, 0, y, 0, 1);
        }
        let span = rows * ty.row_bytes(cols);
        let (f, w_off, row_bytes) = self.matrix(ty, rows, cols, offset, span, true);
        assert!(
            n_tok > 0 && x.1 + n_tok * cols <= x.0.len() && y.1 + n_tok * rows <= y.0.len(),
            "matmul buffer too small"
        );
        let (x_off, y_off) = (x.1 as u64, y.1 as u64);
        let (rows_u, cols_u, nt) = (rows as u32, cols as u32, n_tok as u32);
        let mut a = self.gpu.stream.launch_builder(f);
        a.arg(&self.gpu.weights)
            .arg(&w_off)
            .arg(&x.0.0)
            .arg(&x_off)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&row_bytes)
            .arg(&nt);
        launch(
            a,
            grid2(
                rows.div_ceil(MATVEC_ROWS_PER_BLOCK),
                n_tok.div_ceil(MATMUL_TILE),
                (MATVEC_ROWS_PER_BLOCK * 32) as u32,
            ),
        );
    }

    /// `n` matvecs in one launch, over matrices of a stack of `n_stack` that lie `stride` bytes
    /// apart from `offset`: matvec `i` multiplies matrix `ids[i]` into `x` at `i * x_stride`
    /// (0: all read the same x) and writes `y` at `i * y_stride`.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_id(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        stride: usize,
        n_stack: usize,
        ids: &GpuIds,
        x: At<'_>,
        x_stride: usize,
        y: At<'_>,
        y_stride: usize,
        n: usize,
    ) {
        let span = (n_stack - 1) * stride + rows * ty.row_bytes(cols);
        let (f, w_off, row_bytes) = self.matrix(ty, rows, cols, offset, span, false);
        assert!(
            n > 0
                && ids.0.len() >= n
                && x.1 + (n - 1) * x_stride + cols <= x.0.len()
                && y.1 + (n - 1) * y_stride + rows <= y.0.len(),
            "matvec buffer too small"
        );
        let (x_off, y_off, id_stride) = (x.1 as u64, y.1 as u64, stride as u64);
        let (rows_u, cols_u) = (rows as u32, cols as u32);
        let (xs, ys) = (x_stride as u32, y_stride as u32);
        let mut a = self.gpu.stream.launch_builder(f);
        a.arg(&self.gpu.weights)
            .arg(&w_off)
            .arg(&x.0.0)
            .arg(&x_off)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&row_bytes)
            .arg(&ids.0)
            .arg(&id_stride)
            .arg(&xs)
            .arg(&ys);
        launch(
            a,
            grid2(
                rows.div_ceil(MATVEC_ROWS_PER_BLOCK),
                n,
                (MATVEC_ROWS_PER_BLOCK * 32) as u32,
            ),
        );
    }

    /// The kernel (matmul or matvec), device offset and row size of a matrix at byte `offset`
    /// of the model file, whose `span` bytes from there must all be on the device.
    fn matrix(
        &self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        span: usize,
        matmul: bool,
    ) -> (&CudaFunction, u64, u32) {
        assert!(
            rows > 0 && cols.is_multiple_of(32),
            "matvec cols must be a multiple of 32"
        );
        let row_bytes = ty.row_bytes(cols);
        let w_off = self.gpu.device_offset(offset, span) as u64;
        let (_, mv, mm) = self
            .gpu
            .matvec
            .iter()
            .find(|(t, ..)| *t == ty)
            .expect("unsupported type for CUDA");
        (if matmul { mm } else { mv }, w_off, row_bytes as u32)
    }

    /// Softmax over the first `n_expert` values of `logits` (the router's output), then the `k`
    /// largest: their ids into `ids` and weights into `w` (renormalized to sum to 1 when
    /// `norm`). With `shared`, `w[k]` is the sigmoid of the first value of `shared_gate`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk(
        &mut self,
        logits: &GpuBuf,
        shared_gate: &GpuBuf,
        ids: &GpuIds,
        w: &GpuBuf,
        n_expert: usize,
        k: usize,
        norm: bool,
        shared: bool,
    ) {
        assert!(n_expert <= MOE_MAX_EXPERTS && k <= n_expert);
        assert!(n_expert <= logits.len() && !shared_gate.is_empty());
        assert!(ids.0.len() >= k && w.len() >= k + usize::from(shared));
        let (ne, k_u, norm_u, shared_u) = (n_expert as u32, k as u32, norm as u32, shared as u32);
        let mut l = self.gpu.stream.launch_builder(self.gpu.other("moe_topk"));
        l.arg(&logits.0)
            .arg(&shared_gate.0)
            .arg(&ids.0)
            .arg(&w.0)
            .arg(&ne)
            .arg(&k_u)
            .arg(&norm_u)
            .arg(&shared_u);
        launch(l, grid(1, 32));
    }

    /// `x += sum over r < m of w[r] * (row r of src)`, rows of `n`.
    pub fn moe_combine(&mut self, x: At<'_>, src: &GpuBuf, w: &GpuBuf, m: usize, n: usize) {
        assert!(x.1 + n <= x.0.len() && src.len() >= m * n && w.len() >= m);
        let (x_off, n_u, m_u) = (x.1 as u64, n as u32, m as u32);
        let mut l = self
            .gpu
            .stream
            .launch_builder(self.gpu.other("moe_combine"));
        l.arg(&x.0.0)
            .arg(&x_off)
            .arg(&src.0)
            .arg(&w.0)
            .arg(&n_u)
            .arg(&m_u);
        launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
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

    /// Adds the `m` values of `bias` to each row of `m` in the `n` values at `a`.
    pub fn add_bias(&mut self, a: At<'_>, bias: &GpuBuf, n: usize) {
        let m = bias.len();
        assert!(a.1 + n <= a.0.len() && n.is_multiple_of(m));
        let (a_off, n_u, m_u) = (a.1 as u64, n as u32, m as u32);
        let mut l = self.gpu.stream.launch_builder(self.gpu.other("add_bias"));
        l.arg(&a.0.0).arg(&a_off).arg(&bias.0).arg(&n_u).arg(&m_u);
        launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
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
        let heads = r.n_tok * r.n_heads;
        assert!(x.1 + heads * r.head_dim <= x.0.len());
        let x_off = x.1 as u64;
        let ff = r.freq_factors.unwrap_or(x.0);
        let has_ff = r.freq_factors.is_some() as u32;
        let (nh, hd, rd, pos) = (
            heads as u32,
            r.head_dim as u32,
            r.rope_dim as u32,
            r.pos as u32,
        );
        let (neox, per_tok) = (r.neox as u32, r.n_heads as u32);
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
            .arg(&neox)
            .arg(&per_tok);
        let n = heads * r.rope_dim / 2;
        launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// Causal attention for `p.n_tok` query tokens over the cached positions. `q`, `k`, `v` and
    /// `out` start at element 0 of their buffers. `scores` is scratch of `n_head * max_ctx` per
    /// token; when it holds fewer tokens than `p.n_tok`, the tokens run in several launches.
    pub fn attention(
        &mut self,
        q: &GpuBuf,
        k: &GpuBuf,
        v: &GpuBuf,
        out: &GpuBuf,
        scores: &GpuBuf,
        p: &AttnParams,
    ) {
        let per_launch = (scores.len() / (p.n_head * p.max_ctx) as usize).min(p.n_tok as usize);
        assert!(per_launch > 0, "attention scores scratch too small");
        assert!(p.head_dim <= ATTN_THREADS, "head_dim too large for CUDA");
        let q_len = (p.n_tok * p.n_head * p.head_dim) as usize;
        assert!(q.len() >= q_len && out.len() >= q_len);
        assert!(p.n_pos + p.n_tok - 1 <= p.max_ctx);
        assert!(k.len() >= ((p.n_pos + p.n_tok - 1) * p.kv_dim) as usize && v.len() >= k.len());
        for tok0 in (0..p.n_tok).step_by(per_launch) {
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
                .arg(&p.scale)
                .arg(&tok0);
            let n = per_launch.min((p.n_tok - tok0) as usize);
            launch(l, grid2(p.n_head as usize, n, ATTN_THREADS));
        }
    }

    /// Waits for all queued work.
    pub fn finish(self) {
        self.gpu.stream.synchronize().expect("CUDA command failed");
    }
}
