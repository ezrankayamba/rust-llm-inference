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
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DriverError, LaunchArgs,
    LaunchConfig, PushKernelArg, sys,
};
use std::sync::Arc;

const SOURCE: &str = include_str!("cuda.cu");
const MATVEC_ROWS_PER_BLOCK: usize = 4;
const REDUCE_THREADS: u32 = 256;
/// Threads per attention block, which is also the largest head_dim the kernel takes.
pub const ATTN_THREADS: u32 = 1024;
/// The decode attention kernel's cached positions per block, threads per block, most query
/// heads per KV head and most blocks of context (`AP_T`, `AP_G` and `AP_PARTS` in the
/// source; 65536 positions).
const ATTN_PART: usize = 32;
const ATTN_PART_THREADS: u32 = 256;
const ATTN_PART_HEADS: u32 = 8;
const ATTN_PART_MAX: usize = 2048;
/// Fewest cached positions it runs for; under that its second launch costs more than the
/// one-launch kernel loses (A100: 13-20 against 10-17 microseconds at 128 positions, 19-26
/// against 37-55 at 512).
const ATTN_PART_MIN: u32 = 128;
const ELT_THREADS: usize = 256;
/// Weight placement alignment in the device arena (f32 weights are read as `float*`).
const WEIGHT_ALIGN: usize = 256;
/// Bytes past the last weight that kernels may read (and ignore): the Q6_K matvec takes its
/// unaligned blocks as whole 16-byte words.
const WEIGHT_SLACK: usize = 16;

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
/// Weight types with an int8 matmul kernel, which needs whole 256-weight super-blocks.
const MMQ_KERNELS: [(GgmlType, &str); 2] =
    [(GgmlType::Q4_K, "mmq_q4_k"), (GgmlType::Q6_K, "mmq_q6_k")];
/// Tokens per block of the int8 matmul kernels (`MQ_T` in cuda.cu).
const MMQ_TILE: usize = 64;
/// Fewest tokens the int8 matmul kernels take; fewer run as an int8 matvec each.
pub const MMQ_MIN_TOKENS: usize = 8;
/// Rows, tokens and threads of a block of the tensor-core matmuls (`MA_R`, `MA_T`, `MA_NW`
/// warps in cuda.cu), and their compile's cap on registers per thread.
const MMA_TILE: (usize, usize, u32) = (64, 128, 256);
const MMA_REGS: usize = 96;
const OTHER_KERNELS: [&str; 11] = [
    "quant_x",
    "rmsnorm",
    "axpy",
    "add_bias",
    "swiglu",
    "rope",
    "attention",
    "attention_part",
    "attention_merge",
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

/// Activations quantized to int8 by [`Batch::quantize`], for [`Batch::matmul_q`]: the values,
/// a scale per 32 and a scaled sum per 16.
pub struct GpuQ {
    q: CudaSlice<u32>,
    d: CudaSlice<f32>,
    s: CudaSlice<f32>,
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
    /// Heads per token in the second buffer.
    pub n_heads2: usize,
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
    mmq: Vec<(GgmlType, CudaFunction)>,
    /// The int8 matvecs (`mvq_*`) of the same types.
    mvq: Vec<(GgmlType, CudaFunction)>,
    /// `mmq` holds the tensor-core kernels (`mma_*`), which take larger blocks.
    mma: bool,
    other: Vec<(&'static str, CudaFunction)>,
    /// Zeros: the expert id of a plain matvec, one for each token of an int8 batch too short
    /// for the matmul kernels.
    zero_id: GpuIds,
    /// 0 and 1: the ids of [`Batch::matvec_q_pair`]'s two matrices.
    pair_ids: GpuIds,
    /// Stands in for the quantized output of a kernel that isn't asked for one.
    no_q: GpuQ,
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
        // A stream of our own, since the default one can't be captured into a graph. It is the
        // only stream, so cudarc's cross-stream events (several driver calls per launch) are off.
        // SAFETY: every buffer is used on this one stream.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().map_err(err("create stream"))?;
        // compute_61 (Pascal, 2016) for the int8 kernels' dp4a instruction.
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_61"),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(SOURCE, opts)
            .map_err(|e| format!("CUDA kernel compile failed: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(err("load CUDA module"))?;
        let func = |n: &str| module.load_function(n).map_err(err(n));
        let matvec = MATVEC_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, func(n)?, func(&n.replace("matvec", "matmul"))?)))
            .collect::<Result<_, String>>()?;
        // From compute capability 8.0 (Ampere) the tensor-core kernels replace the dp4a ones.
        // They are a second compile of the same source: fewer registers per thread let a
        // multiprocessor hold more of their warps, and the same cap slows the f32 matmuls.
        // RLI_CUDA_NO_MMA keeps the dp4a kernels, to check them on a newer GPU.
        let mma = std::env::var_os("RLI_CUDA_NO_MMA").is_none()
            && ctx
                .compute_capability()
                .map_err(err("compute capability"))?
                .0
                >= 8;
        let mmq = if mma {
            let opts = cudarc::nvrtc::CompileOptions {
                arch: Some("compute_80"),
                options: vec!["-DRLI_MMA".into()],
                maxrregcount: Some(MMA_REGS),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(SOURCE, opts)
                .map_err(|e| format!("CUDA tensor-core kernel compile failed: {e:?}"))?;
            let module = ctx.load_module(ptx).map_err(err("load CUDA module"))?;
            MMQ_KERNELS
                .iter()
                .map(|&(t, n)| {
                    let n = n.replace("mmq", "mma");
                    Ok((t, module.load_function(&n).map_err(err(&n))?))
                })
                .collect::<Result<_, String>>()?
        } else {
            MMQ_KERNELS
                .iter()
                .map(|&(t, n)| Ok((t, func(n)?)))
                .collect::<Result<_, String>>()?
        };
        let mvq = MMQ_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, func(&n.replace("mmq", "mvq"))?)))
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
            .alloc_zeros::<u8>(total + WEIGHT_SLACK)
            .map_err(err("allocate weight arena"))?;
        for p in &placed {
            let src = &mapped[p.file_off..p.file_off + p.len];
            stream
                .memcpy_htod(src, &mut weights.slice_mut(p.dev_off..p.dev_off + p.len))
                .map_err(err("upload weights"))?;
        }
        let zero_id = GpuIds(
            stream
                .alloc_zeros(MMQ_MIN_TOKENS)
                .map_err(err("allocate"))?,
        );
        let mut pair_ids = GpuIds(stream.alloc_zeros(2).map_err(err("allocate"))?);
        stream
            .memcpy_htod(&[0u32, 1], &mut pair_ids.0)
            .map_err(err("upload"))?;
        let no_q = GpuQ {
            q: stream.alloc_zeros(8).map_err(err("allocate"))?,
            d: stream.alloc_zeros(1).map_err(err("allocate"))?,
            s: stream.alloc_zeros(2).map_err(err("allocate"))?,
        };
        let name = format!("cuda ({})", ctx.name().map_err(err("device name"))?);
        Ok(Self {
            stream,
            weights,
            placed,
            matvec,
            mmq,
            mvq,
            mma,
            other,
            zero_id,
            pair_ids,
            no_q,
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

    /// Whether a `ty` matrix of `cols` columns can go through [`Batch::matmul_q`].
    pub fn has_matmul_q(ty: GgmlType, cols: usize) -> bool {
        MMQ_KERNELS.iter().any(|(t, _)| *t == ty) && cols.is_multiple_of(256)
    }

    /// Whether the int8 kernels can run the matrix of `ty` and `cols` columns at byte `offset`
    /// of the model file: [`Gpu::has_matmul_q`], and a start at a multiple of 16 bytes on the
    /// device, which the matvecs' 16-byte reads assume (GGUF aligns tensors to 32).
    pub fn has_int8(&self, ty: GgmlType, cols: usize, offset: usize) -> bool {
        Self::has_matmul_q(ty, cols) && self.device_offset(offset, 1).is_multiple_of(16)
    }

    /// Room for `n` quantized activations (a multiple of 32).
    pub fn alloc_q(&self, n: usize) -> GpuQ {
        let zeros = |n: usize| self.stream.alloc_zeros::<f32>(n.max(1));
        GpuQ {
            q: self
                .stream
                .alloc_zeros(n.div_ceil(4).max(1))
                .expect("CUDA buffer allocation failed"),
            d: zeros(n / 32).expect("CUDA buffer allocation failed"),
            s: zeros(n / 16).expect("CUDA buffer allocation failed"),
        }
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
        Batch {
            gpu: self,
            mode: Mode::Queue,
        }
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

/// Most arguments a kernel takes.
const MAX_ARGS: usize = 24;

/// A kernel argument as the bytes the driver copies: a device pointer or a scalar.
trait ArgBits {
    fn bits(&self, stream: &CudaStream) -> (u64, u8);
}
impl ArgBits for u32 {
    fn bits(&self, _: &CudaStream) -> (u64, u8) {
        (*self as u64, 4)
    }
}
impl ArgBits for f32 {
    fn bits(&self, _: &CudaStream) -> (u64, u8) {
        (self.to_bits() as u64, 4)
    }
}
impl ArgBits for u64 {
    fn bits(&self, _: &CudaStream) -> (u64, u8) {
        (*self, 8)
    }
}
impl<T> ArgBits for CudaSlice<T> {
    fn bits(&self, stream: &CudaStream) -> (u64, u8) {
        (self.device_ptr(stream).0, 8)
    }
}

/// One launch: its kernel, grid and arguments.
#[derive(Clone, Copy, PartialEq)]
struct Rec {
    func: *const CudaFunction,
    /// Grid x and y, and threads per block.
    dims: [u32; 3],
    bits: [u64; MAX_ARGS],
    sizes: [u8; MAX_ARGS],
    n: usize,
}

/// A launch being put together: the arguments as cudarc takes them, where it will be the one
/// to launch, and as bits to compare with a graph's.
struct Call<'a> {
    stream: &'a CudaStream,
    args: Option<LaunchArgs<'a>>,
    rec: Rec,
}

impl<'a> Call<'a> {
    fn arg<T: ArgBits>(&mut self, v: &'a T) -> &mut Self
    where
        LaunchArgs<'a>: PushKernelArg<&'a T>,
    {
        let (bits, size) = v.bits(self.stream);
        (self.rec.bits[self.rec.n], self.rec.sizes[self.rec.n]) = (bits, size);
        self.rec.n += 1;
        if let Some(a) = &mut self.args {
            a.arg(v);
        }
        self
    }
}

/// A token's launches as a CUDA graph ([`Gpu::run_graph`]): replaying it costs less per launch
/// than queueing each (2.8 against 4.0 microseconds on an A100) and takes one driver call.
pub struct Graph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    /// Per launch, in order: its node, the node's parameters and what was last launched.
    nodes: Vec<sys::CUgraphNode>,
    params: Vec<sys::CUDA_KERNEL_NODE_PARAMS>,
    recs: Vec<Rec>,
}

// SAFETY: the handles are only used through `Gpu::run_graph`, on the context's stream; the
// pointers inside `params` are rewritten before each use.
unsafe impl Send for Graph {}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: both handles came from `Gpu::run_graph` and are not used again.
        unsafe {
            sys::cuGraphExecDestroy(self.exec);
            sys::cuGraphDestroy(self.graph);
        }
    }
}

impl Graph {
    /// Makes launch `i` of the graph `rec` (same kernel, new grid or arguments).
    fn set(&mut self, i: usize, rec: Rec) {
        self.recs[i] = rec;
        let rec = &self.recs[i];
        let mut ptrs = [std::ptr::null_mut(); MAX_ARGS];
        for (p, b) in ptrs.iter_mut().zip(&rec.bits) {
            *p = b as *const u64 as *mut std::ffi::c_void;
        }
        let p = &mut self.params[i];
        (p.gridDimX, p.gridDimY, p.blockDimX) = (rec.dims[0], rec.dims[1], rec.dims[2]);
        p.kernelParams = ptrs.as_mut_ptr();
        // SAFETY: the node belongs to this graph; the driver copies the arguments (each 4 or
        // 8 bytes, the low bytes of its u64) before returning.
        unsafe { sys::cuGraphExecKernelNodeSetParams_v2(self.exec, self.nodes[i], p) }
            .result()
            .expect("CUDA graph update failed");
    }
}

impl Gpu {
    /// Queues the launches `f` makes, as a graph kept in `slot`: captured on the first call and
    /// replayed after that, with the launches whose grid or arguments changed updated first.
    /// When `f` makes other launches than the graph holds, it is captured again, so `f` may
    /// run twice and must do nothing but queue launches.
    pub fn run_graph(&self, slot: &mut Option<Graph>, f: impl Fn(&mut Batch<'_>)) {
        let stream = self.stream.cu_stream();
        if let Some(g) = slot {
            let mut b = Batch {
                gpu: self,
                mode: Mode::Replay(g, 0),
            };
            f(&mut b);
            let Mode::Replay(g, next) = b.mode else {
                unreachable!()
            };
            if next == g.recs.len() {
                // SAFETY: the graph is live and was captured on this stream.
                unsafe { sys::cuGraphLaunch(g.exec, stream) }
                    .result()
                    .expect("CUDA graph launch failed");
                return;
            }
            *slot = None;
        }
        let (mut graph, mut exec) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: the stream is live, and capture ends before the graph is used.
        unsafe {
            sys::cuStreamBeginCapture_v2(
                stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
        }
        .result()
        .expect("CUDA capture failed");
        let mut b = Batch {
            gpu: self,
            mode: Mode::Capture(Vec::new()),
        };
        f(&mut b);
        let Mode::Capture(recs) = b.mode else {
            unreachable!()
        };
        let mut n = recs.len();
        let mut nodes = vec![std::ptr::null_mut(); n];
        // SAFETY: `nodes` has room for the `n` it is told of; the handles outlive their uses.
        let params: Vec<sys::CUDA_KERNEL_NODE_PARAMS> = unsafe {
            sys::cuStreamEndCapture(stream, &mut graph)
                .result()
                .expect("CUDA capture failed");
            sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0)
                .result()
                .expect("CUDA graph failed");
            sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n)
                .result()
                .expect("CUDA graph nodes");
            nodes
                .iter()
                .map(|&node| {
                    let mut p = std::mem::zeroed();
                    sys::cuGraphKernelNodeGetParams_v2(node, &mut p)
                        .result()
                        .expect("CUDA graph node");
                    p
                })
                .collect()
        };
        // Updates address nodes by launch order, so the nodes must come back in that order.
        assert_eq!(
            n,
            recs.len(),
            "CUDA graph has other nodes than were launched"
        );
        for (p, r) in params.iter().zip(&recs) {
            assert_eq!(
                [p.gridDimX, p.gridDimY, p.blockDimX],
                r.dims,
                "CUDA graph node order"
            );
            for i in 0..r.n {
                // SAFETY: the node holds `r.n` arguments of these sizes.
                let have = unsafe {
                    let a = *p.kernelParams.add(i) as *const u8;
                    let mut v = [0u8; 8];
                    std::ptr::copy_nonoverlapping(a, v.as_mut_ptr(), r.sizes[i] as usize);
                    u64::from_le_bytes(v)
                };
                assert_eq!(have, r.bits[i], "CUDA graph node order");
            }
        }
        let g = slot.insert(Graph {
            graph,
            exec,
            nodes,
            params,
            recs,
        });
        // SAFETY: as above.
        unsafe { sys::cuGraphLaunch(g.exec, stream) }
            .result()
            .expect("CUDA graph launch failed");
    }

    /// Waits for all queued work.
    pub fn sync(&self) {
        self.stream.synchronize().expect("CUDA command failed");
    }
}

/// Where a [`Batch`]'s launches go.
enum Mode<'a> {
    /// To the stream.
    Queue,
    /// To the stream, which is capturing them, and into the list.
    Capture(Vec<Rec>),
    /// Nowhere: each is compared with the graph's launch of that index, which is updated if
    /// it differs. The index stops advancing at a launch the graph can't become.
    Replay(&'a mut Graph, usize),
}

impl<'a> Batch<'a> {
    fn call<'c>(&self, func: &'c CudaFunction) -> Call<'c>
    where
        'a: 'c,
    {
        let gpu: &'a Gpu = self.gpu;
        Call {
            stream: &gpu.stream,
            args: (!matches!(self.mode, Mode::Replay(..))).then(|| gpu.stream.launch_builder(func)),
            rec: Rec {
                func,
                dims: [0; 3],
                bits: [0; MAX_ARGS],
                sizes: [0; MAX_ARGS],
                n: 0,
            },
        }
    }

    fn launch(&mut self, mut call: Call<'_>, cfg: LaunchConfig) {
        call.rec.dims = [cfg.grid_dim.0, cfg.grid_dim.1, cfg.block_dim.0];
        if let Some(args) = &mut call.args {
            // SAFETY: each caller passes exactly the parameters of the kernel's signature in
            // cuda.cu, with buffer bounds asserted or sized by construction.
            unsafe { args.launch(cfg) }.expect("CUDA launch failed");
        }
        match &mut self.mode {
            Mode::Queue => {}
            Mode::Capture(recs) => recs.push(call.rec),
            Mode::Replay(g, next) => {
                let Some(have) = g.recs.get(*next) else {
                    *next = usize::MAX;
                    return;
                };
                if *have != call.rec {
                    if have.func != call.rec.func || have.sizes != call.rec.sizes {
                        *next = usize::MAX;
                        return;
                    }
                    g.set(*next, call.rec);
                }
                *next += 1;
            }
        }
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

/// Queued GPU work on the device stream; kernels run in order. [`Batch::finish`] waits.
pub struct Batch<'a> {
    gpu: &'a Gpu,
    mode: Mode<'a>,
}

impl<'a> Batch<'a> {
    /// [`Gpu::has_int8`] of the batch's device.
    pub fn has_int8(&self, ty: GgmlType, cols: usize, offset: usize) -> bool {
        self.gpu.has_int8(ty, cols, offset)
    }

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
            return self.matvec_id(ty, rows, cols, offset, 0, 1, zero, x, 0, 1, y, 0, 1);
        }
        let span = rows * ty.row_bytes(cols);
        let (f, w_off, row_bytes) = self.matrix(ty, rows, cols, offset, span, true);
        assert!(
            n_tok > 0 && x.1 + n_tok * cols <= x.0.len() && y.1 + n_tok * rows <= y.0.len(),
            "matmul buffer too small"
        );
        let (x_off, y_off) = (x.1 as u64, y.1 as u64);
        let (rows_u, cols_u, nt) = (rows as u32, cols as u32, n_tok as u32);
        let mut a = self.call(f);
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
        self.launch(
            a,
            grid2(
                rows.div_ceil(MATVEC_ROWS_PER_BLOCK),
                n_tok.div_ceil(MATMUL_TILE),
                (MATVEC_ROWS_PER_BLOCK * 32) as u32,
            ),
        );
    }

    /// Quantizes the `n` values at `x` (a multiple of 32) to int8 in `out`, a scale per 32.
    pub fn quantize(&mut self, x: At<'_>, n: usize, out: &GpuQ) {
        assert!(n.is_multiple_of(32) && x.1 + n <= x.0.len());
        assert!(out.q.len() * 4 >= n && out.d.len() * 32 >= n && out.s.len() * 16 >= n);
        let (x_off, blocks) = (x.1 as u64, (n / 32) as u32);
        let mut l = self.call(self.gpu.other("quant_x"));
        l.arg(&x.0.0)
            .arg(&x_off)
            .arg(&out.q)
            .arg(&out.d)
            .arg(&out.s)
            .arg(&blocks);
        self.launch(l, grid((n / 32).div_ceil(8), 256));
    }

    /// [`Batch::matmul`] over activations quantized by [`Batch::quantize`] (`n_tok` rows of
    /// `cols` in `xq`), for the matrices [`Gpu::has_int8`] accepts. The result differs from the
    /// f32 matmul's by the activations' rounding. Under [`MMQ_MIN_TOKENS`] tokens, `add` adds
    /// it to `y` instead of storing it.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        xq: &GpuQ,
        y: At<'_>,
        n_tok: usize,
        add: bool,
    ) {
        assert!(self.gpu.has_int8(ty, cols, offset) && rows > 0 && n_tok > 0);
        assert!(xq.q.len() * 4 >= n_tok * cols && y.1 + n_tok * rows <= y.0.len());
        if n_tok < MMQ_MIN_TOKENS {
            let zero = &self.gpu.zero_id;
            return self.matvec_q_id(
                ty, rows, cols, offset, 0, 1, zero, xq, cols, 1, y, rows, n_tok, add,
            );
        }
        assert!(!add, "the int8 matmul kernels don't accumulate");
        let row_bytes = ty.row_bytes(cols);
        let w_off = self.gpu.device_offset(offset, rows * row_bytes) as u64;
        let f = &self
            .gpu
            .mmq
            .iter()
            .find(|(t, _)| *t == ty)
            .expect("checked by has_matmul_q")
            .1;
        let (y_off, rb) = (y.1 as u64, row_bytes as u32);
        let (rows_u, cols_u, nt) = (rows as u32, cols as u32, n_tok as u32);
        let mut a = self.call(f);
        a.arg(&self.gpu.weights)
            .arg(&w_off)
            .arg(&xq.q)
            .arg(&xq.d)
            .arg(&xq.s)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&rb)
            .arg(&nt);
        let cfg = if self.gpu.mma {
            grid2(
                rows.div_ceil(MMA_TILE.0),
                n_tok.div_ceil(MMA_TILE.1),
                MMA_TILE.2,
            )
        } else {
            grid2(rows.div_ceil(32), n_tok.div_ceil(MMQ_TILE), 256)
        };
        self.launch(a, cfg);
    }

    /// [`Batch::matvec_id`] over activations quantized by [`Batch::quantize`], for the matrices
    /// [`Gpu::has_int8`] accepts; `x_stride` counts values, a multiple of 32.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_q_id(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        stride: usize,
        n_stack: usize,
        ids: &GpuIds,
        xq: &GpuQ,
        x_stride: usize,
        x_div: usize,
        y: At<'_>,
        y_stride: usize,
        n: usize,
        add: bool,
    ) {
        assert!(self.gpu.has_int8(ty, cols, offset) && rows > 0 && n > 0 && x_div > 0);
        assert!(
            ids.0.len() >= n
                && x_stride.is_multiple_of(32)
                && xq.q.len() * 4 >= (n - 1) / x_div * x_stride + cols
                && y.1 + (n - 1) * y_stride + rows <= y.0.len(),
            "matvec buffer too small"
        );
        let row_bytes = ty.row_bytes(cols);
        let span = (n_stack - 1) * stride + rows * row_bytes;
        let w_off = self.gpu.device_offset(offset, span) as u64;
        let (id_stride, xs, xd) = (stride as u64, x_stride as u32, x_div as u32);
        self.launch_mvq(
            ty, rows, cols, w_off, id_stride, ids, xq, xs, xd, y, y_stride, n, add,
        );
    }

    /// Two matrices of one type and shape at `offsets`, over the same quantized token in `xq`,
    /// as one launch: the second's result follows the first's at `y`.
    pub fn matvec_q_pair(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offsets: [usize; 2],
        xq: &GpuQ,
        y: At<'_>,
    ) {
        assert!(offsets.iter().all(|&o| self.gpu.has_int8(ty, cols, o)) && rows > 0);
        assert!(xq.q.len() * 4 >= cols && y.1 + 2 * rows <= y.0.len());
        let len = rows * ty.row_bytes(cols);
        let [a, b] = offsets.map(|o| self.gpu.device_offset(o, len) as u64);
        // The kernel adds id * stride to the first matrix's offset, modulo 2^64.
        let ids = &self.gpu.pair_ids;
        self.launch_mvq(
            ty,
            rows,
            cols,
            a,
            b.wrapping_sub(a),
            ids,
            xq,
            0,
            1,
            y,
            rows,
            2,
            false,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_mvq(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        w_off: u64,
        id_stride: u64,
        ids: &GpuIds,
        xq: &GpuQ,
        xs: u32,
        xd: u32,
        y: At<'_>,
        y_stride: usize,
        n: usize,
        add: bool,
    ) {
        let f = &self
            .gpu
            .mvq
            .iter()
            .find(|(t, _)| *t == ty)
            .expect("checked by has_matmul_q")
            .1;
        let (y_off, rb) = (y.1 as u64, ty.row_bytes(cols) as u32);
        let (rows_u, cols_u, ys, add) = (rows as u32, cols as u32, y_stride as u32, add as u32);
        let mut a = self.call(f);
        a.arg(&self.gpu.weights)
            .arg(&w_off)
            .arg(&xq.q)
            .arg(&xq.d)
            .arg(&xq.s)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&rb)
            .arg(&ids.0)
            .arg(&id_stride)
            .arg(&xs)
            .arg(&xd)
            .arg(&ys)
            .arg(&add);
        self.launch(
            a,
            grid2(
                rows.div_ceil(MATVEC_ROWS_PER_BLOCK),
                n,
                (MATVEC_ROWS_PER_BLOCK * 32) as u32,
            ),
        );
    }

    /// `n` matvecs in one launch, over matrices of a stack of `n_stack` that lie `stride` bytes
    /// apart from `offset`: matvec `i` multiplies matrix `ids[i]` into `x` at
    /// `i / x_div * x_stride` and writes `y` at `i * y_stride`.
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
        x_div: usize,
        y: At<'_>,
        y_stride: usize,
        n: usize,
    ) {
        let span = (n_stack - 1) * stride + rows * ty.row_bytes(cols);
        let (f, w_off, row_bytes) = self.matrix(ty, rows, cols, offset, span, false);
        assert!(
            n > 0
                && ids.0.len() >= n
                && x_div > 0
                && x.1 + (n - 1) / x_div * x_stride + cols <= x.0.len()
                && y.1 + (n - 1) * y_stride + rows <= y.0.len(),
            "matvec buffer too small"
        );
        let (x_off, y_off, id_stride) = (x.1 as u64, y.1 as u64, stride as u64);
        let (rows_u, cols_u) = (rows as u32, cols as u32);
        let (xs, xd, ys) = (x_stride as u32, x_div as u32, y_stride as u32);
        let mut a = self.call(f);
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
            .arg(&xd)
            .arg(&ys);
        self.launch(
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
    ) -> (&'a CudaFunction, u64, u32) {
        let gpu: &'a Gpu = self.gpu;
        assert!(
            rows > 0 && cols.is_multiple_of(32),
            "matvec cols must be a multiple of 32"
        );
        let row_bytes = ty.row_bytes(cols);
        let w_off = gpu.device_offset(offset, span) as u64;
        let (_, mv, mm) = gpu
            .matvec
            .iter()
            .find(|(t, ..)| *t == ty)
            .expect("unsupported type for CUDA");
        (if matmul { mm } else { mv }, w_off, row_bytes as u32)
    }

    /// For each of `n_tok` tokens: softmax over its `n_expert` values of `logits` (the router's
    /// output), then the `k` largest: their ids into `ids` and weights into `w` (renormalized
    /// to sum to 1 when `norm`). With `shared`, a token has `k + 1` weights, the last the
    /// sigmoid of its value of `shared_gate`.
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
        n_tok: usize,
    ) {
        assert!(n_expert <= MOE_MAX_EXPERTS && k <= n_expert);
        assert!(n_tok * n_expert <= logits.len() && n_tok <= shared_gate.len());
        assert!(ids.0.len() >= n_tok * k && w.len() >= n_tok * (k + usize::from(shared)));
        let (ne, k_u, norm_u, shared_u) = (n_expert as u32, k as u32, norm as u32, shared as u32);
        let mut l = self.call(self.gpu.other("moe_topk"));
        l.arg(&logits.0)
            .arg(&shared_gate.0)
            .arg(&ids.0)
            .arg(&w.0)
            .arg(&ne)
            .arg(&k_u)
            .arg(&norm_u)
            .arg(&shared_u);
        self.launch(l, grid(n_tok, 32));
    }

    /// For each of `n_tok` tokens `t`: row `t` of `x` `+= sum over r < m of
    /// w[t * w_stride + r] * (row t * m + r of src)`, rows of `n`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine(
        &mut self,
        x: At<'_>,
        src: At<'_>,
        w: At<'_>,
        w_stride: usize,
        m: usize,
        n: usize,
        n_tok: usize,
    ) {
        assert!(n_tok > 0 && x.1 + n_tok * n <= x.0.len());
        assert!(src.1 + n_tok * m * n <= src.0.len());
        assert!(w.1 + (n_tok - 1) * w_stride + m <= w.0.len());
        let (x_off, src_off, w_off) = (x.1 as u64, src.1 as u64, w.1 as u64);
        let (ws, n_u, m_u, nt) = (w_stride as u32, n as u32, m as u32, n_tok as u32);
        let mut l = self.call(self.gpu.other("moe_combine"));
        l.arg(&x.0.0)
            .arg(&x_off)
            .arg(&src.0.0)
            .arg(&src_off)
            .arg(&w.0.0)
            .arg(&w_off)
            .arg(&ws)
            .arg(&n_u)
            .arg(&m_u)
            .arg(&nt);
        let len = n * n_tok;
        self.launch(l, grid(len.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// RMS-normalizes `rows` consecutive rows of `n` from `x` into `out` (may alias).
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm(
        &mut self,
        x: At<'_>,
        out: At<'_>,
        w: &GpuBuf,
        rows: usize,
        n: usize,
        eps: f32,
        quant: Option<&GpuQ>,
    ) {
        assert!(x.1 + rows * n <= x.0.len() && out.1 + rows * n <= out.0.len());
        let (x_off, o_off, n_u) = (x.1 as u64, out.1 as u64, n as u32);
        let (q, quant) = self.quant_out(quant, rows * n, n);
        let mut a = self.call(self.gpu.other("rmsnorm"));
        a.arg(&x.0.0)
            .arg(&x_off)
            .arg(&out.0.0)
            .arg(&o_off)
            .arg(&w.0)
            .arg(&n_u)
            .arg(&eps)
            .arg(&q.q)
            .arg(&q.d)
            .arg(&q.s)
            .arg(&quant);
        self.launch(a, grid(rows, REDUCE_THREADS));
    }

    /// The buffers and flag a kernel takes for quantizing its `n` results (rows of `row`, a
    /// multiple of 32) on the way out, as [`Batch::quantize`] would.
    fn quant_out<'c>(&self, q: Option<&'c GpuQ>, n: usize, row: usize) -> (&'c GpuQ, u32)
    where
        'a: 'c,
    {
        let gpu: &'a Gpu = self.gpu;
        if let Some(q) = q {
            assert!(row.is_multiple_of(32));
            assert!(q.q.len() * 4 >= n && q.d.len() * 32 >= n && q.s.len() * 16 >= n);
        }
        (q.unwrap_or(&gpu.no_q), q.is_some() as u32)
    }

    /// `a += alpha * b`
    pub fn axpy(&mut self, a: At<'_>, b: At<'_>, alpha: f32, n: usize) {
        assert!(a.1 + n <= a.0.len() && b.1 + n <= b.0.len());
        let (a_off, b_off, n_u) = (a.1 as u64, b.1 as u64, n as u32);
        let mut l = self.call(self.gpu.other("axpy"));
        l.arg(&a.0.0)
            .arg(&a_off)
            .arg(&b.0.0)
            .arg(&b_off)
            .arg(&n_u)
            .arg(&alpha);
        self.launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
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
        let mut l = self.call(self.gpu.other("add_bias"));
        l.arg(&a.0.0).arg(&a_off).arg(&bias.0).arg(&n_u).arg(&m_u);
        self.launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// `gate = silu(gate) * up`, quantized into `quant` as well when given.
    pub fn swiglu(&mut self, gate: At<'_>, up: At<'_>, n: usize, quant: Option<&GpuQ>) {
        assert!(gate.1 + n <= gate.0.len() && up.1 + n <= up.0.len());
        let (g_off, u_off, n_u) = (gate.1 as u64, up.1 as u64, n as u32);
        let (q, quant) = self.quant_out(quant, n, n);
        let mut l = self.call(self.gpu.other("swiglu"));
        l.arg(&gate.0.0)
            .arg(&g_off)
            .arg(&up.0.0)
            .arg(&u_off)
            .arg(&n_u)
            .arg(&q.q)
            .arg(&q.d)
            .arg(&q.s)
            .arg(&quant);
        self.launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// Rotates the heads at `x`, and those at `x2` (`r.n_heads2` per token) in the same launch.
    pub fn rope(&mut self, x: At<'_>, x2: At<'_>, r: &RopeArgs<'_>) {
        let (heads, heads2) = (r.n_tok * r.n_heads, r.n_tok * r.n_heads2);
        assert!(x.1 + heads * r.head_dim <= x.0.len());
        assert!(x2.1 + heads2 * r.head_dim <= x2.0.len());
        let (x_off, x2_off) = (x.1 as u64, x2.1 as u64);
        let (nh2, per_tok2) = (heads2 as u32, r.n_heads2 as u32);
        let ff = r.freq_factors.unwrap_or(x.0);
        let has_ff = r.freq_factors.is_some() as u32;
        let (nh, hd, rd, pos) = (
            heads as u32,
            r.head_dim as u32,
            r.rope_dim as u32,
            r.pos as u32,
        );
        let (neox, per_tok) = (r.neox as u32, r.n_heads as u32);
        let mut l = self.call(self.gpu.other("rope"));
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
            .arg(&per_tok)
            .arg(&x2.0.0)
            .arg(&x2_off)
            .arg(&nh2)
            .arg(&per_tok2);
        let n = (heads + heads2) * r.rope_dim / 2;
        self.launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// f32s of `scores` scratch [`Batch::attention`] needs for one token at `max_ctx`.
    pub fn attention_scratch(n_head: usize, head_dim: usize, max_ctx: usize) -> usize {
        n_head * max_ctx.max(max_ctx.div_ceil(ATTN_PART) * (head_dim + 2))
    }

    /// One token's attention with the context split over blocks of [`ATTN_PART`] positions,
    /// then merged.
    #[allow(clippy::too_many_arguments)]
    fn attention_decode(
        &mut self,
        q: &GpuBuf,
        k: &GpuBuf,
        v: &GpuBuf,
        out: &GpuBuf,
        scores: &GpuBuf,
        p: &AttnParams,
        quant: Option<&GpuQ>,
    ) {
        let n_parts = (p.n_pos as usize).div_ceil(ATTN_PART);
        assert!(scores.len() >= (p.n_head * (p.head_dim + 2)) as usize * n_parts);
        let mut l = self.call(self.gpu.other("attention_part"));
        l.arg(&q.0)
            .arg(&k.0)
            .arg(&v.0)
            .arg(&scores.0)
            .arg(&p.n_head)
            .arg(&p.n_kv)
            .arg(&p.head_dim)
            .arg(&p.n_pos)
            .arg(&p.kv_dim)
            .arg(&p.scale);
        self.launch(l, grid2(p.n_kv as usize, n_parts, ATTN_PART_THREADS));
        let n_parts = n_parts as u32;
        let mut l = self.call(self.gpu.other("attention_merge"));
        let (xq, quant) =
            self.quant_out(quant, (p.n_head * p.head_dim) as usize, p.head_dim as usize);
        l.arg(&scores.0)
            .arg(&out.0)
            .arg(&p.head_dim)
            .arg(&n_parts)
            .arg(&xq.q)
            .arg(&xq.d)
            .arg(&xq.s)
            .arg(&quant);
        self.launch(l, grid(p.n_head as usize, ATTN_PART_THREADS));
    }

    /// Causal attention for `p.n_tok` query tokens over the cached positions. `q`, `k`, `v` and
    /// `out` start at element 0 of their buffers. `scores` is scratch of
    /// [`Batch::attention_scratch`] per token; when it holds fewer tokens than `p.n_tok`, the
    /// tokens run in several launches. The result is quantized into `quant` as well when given
    /// (head_dim a multiple of 32).
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        q: &GpuBuf,
        k: &GpuBuf,
        v: &GpuBuf,
        out: &GpuBuf,
        scores: &GpuBuf,
        p: &AttnParams,
        quant: Option<&GpuQ>,
    ) {
        let per_launch = (scores.len() / (p.n_head * p.max_ctx) as usize).min(p.n_tok as usize);
        assert!(per_launch > 0, "attention scores scratch too small");
        assert!(p.head_dim <= ATTN_THREADS, "head_dim too large for CUDA");
        let q_len = (p.n_tok * p.n_head * p.head_dim) as usize;
        assert!(q.len() >= q_len && out.len() >= q_len);
        assert!(p.n_pos + p.n_tok - 1 <= p.max_ctx);
        assert!(k.len() >= ((p.n_pos + p.n_tok - 1) * p.kv_dim) as usize && v.len() >= k.len());
        let (xq, quant_u) = self.quant_out(quant, q_len, p.head_dim as usize);
        if p.n_tok == 1
            && p.n_pos > ATTN_PART_MIN
            && p.n_head / p.n_kv <= ATTN_PART_HEADS
            && p.head_dim <= ATTN_PART_THREADS
            && p.head_dim.is_multiple_of(4)
            && p.n_pos as usize <= ATTN_PART * ATTN_PART_MAX
        {
            return self.attention_decode(q, k, v, out, scores, p, quant);
        }
        for tok0 in (0..p.n_tok).step_by(per_launch) {
            let mut l = self.call(self.gpu.other("attention"));
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
                .arg(&tok0)
                .arg(&xq.q)
                .arg(&xq.d)
                .arg(&xq.s)
                .arg(&quant_u);
            let n = per_launch.min((p.n_tok - tok0) as usize);
            self.launch(l, grid2(p.n_head as usize, n, ATTN_THREADS));
        }
    }

    /// Waits for all queued work.
    pub fn finish(self) {
        self.gpu.stream.synchronize().expect("CUDA command failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The int8 matmul, and the int8 matvec (under 8 tokens), against the f32 ones over the same
    /// random weights: they differ only by the rounding of activations to int8. The sizes
    /// leave a row tile, a token tile and a matvec step partly filled. Needs an NVIDIA GPU, so
    /// CI doesn't run it; `RLI_CUDA_NO_MMA=1` checks the dp4a matmuls on a GPU that would use
    /// the tensor-core ones.
    #[test]
    fn matmul_q_matches_f32_matmul() {
        let types = [(GgmlType::Q4_K, &[0, 2][..]), (GgmlType::Q6_K, &[208][..])];
        for ((rows, cols, n), (ty, scales_at)) in [(200, 512, 137), (203, 1280, 1), (64, 1024, 5)]
            .into_iter()
            .flat_map(|size| types.map(|t| (size, t)))
        {
            let mut seed = 1u32;
            let mut next = move || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed >> 8
            };
            let mut w: Vec<u8> = (0..rows * ty.row_bytes(cols))
                .map(|_| next() as u8)
                .collect();
            for block in w.chunks_exact_mut(ty.block_bytes()) {
                for &i in scales_at {
                    block[i + 1] = 0x24 | (block[i + 1] & 3); // an f16 in [2^-6, 2^-5)
                }
            }
            let x: Vec<f32> = (0..n * cols)
                .map(|_| next() as f32 / 8388608.0 - 1.0)
                .collect();

            let gpu = Gpu::new(&w, &[(0, w.len())]).expect("CUDA device");
            let (xb, xq) = (gpu.upload(&x), gpu.alloc_q(n * cols));
            let (y_f32, y_q) = (gpu.alloc(n * rows), gpu.alloc(n * rows));
            let mut b = gpu.batch();
            b.matmul(ty, rows, cols, 0, At(&xb, 0), At(&y_f32, 0), n);
            b.quantize(At(&xb, 0), n * cols, &xq);
            b.matmul_q(ty, rows, cols, 0, &xq, At(&y_q, 0), n, false);
            b.finish();
            let (mut want, mut got) = (vec![0f32; n * rows], vec![0f32; n * rows]);
            gpu.read(&y_f32, &mut want);
            gpu.read(&y_q, &mut got);

            let rms = |v: &mut dyn Iterator<Item = f32>| {
                (v.map(|a| (a * a) as f64).sum::<f64>() / (n * rows) as f64).sqrt()
            };
            let err = rms(&mut got.iter().zip(&want).map(|(a, b)| a - b));
            let size = rms(&mut want.iter().copied());
            assert!(
                err < 0.01 * size,
                "{ty:?}, {n} tokens: rms error {err} against rms {size}"
            );
        }
    }

    /// One token's attention past [`ATTN_PART_MIN`] positions (split over blocks and merged)
    /// against the batched kernel's first token, which sees the same positions. The shapes
    /// cover 1, 4 and 8 query heads per KV head and a last block partly filled. Needs an
    /// NVIDIA GPU, so CI doesn't run it.
    #[test]
    fn decode_attention_matches_batched_attention() {
        for (n_head, n_kv, head_dim, n_pos) in [(8, 2, 64, 300), (4, 4, 36, 129), (16, 2, 128, 517)]
        {
            let mut seed = 7u32;
            let mut rand = |n: usize| -> Vec<f32> {
                (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        (seed >> 8) as f32 / 8388608.0 - 1.0
                    })
                    .collect()
            };
            let gpu = Gpu::new(&[0; 256], &[(0, 256)]).expect("CUDA device");
            let (max_ctx, kv_dim, q_len) = (n_pos + 1, n_kv * head_dim, n_head * head_dim);
            let q = gpu.upload(&rand(2 * q_len));
            let (k, v) = (
                gpu.upload(&rand(max_ctx * kv_dim)),
                gpu.upload(&rand(max_ctx * kv_dim)),
            );
            let (one, two) = (gpu.alloc(q_len), gpu.alloc(2 * q_len));
            let scores = gpu.alloc(2 * Batch::attention_scratch(n_head, head_dim, max_ctx));
            let mut p = AttnParams {
                n_head: n_head as u32,
                n_kv: n_kv as u32,
                head_dim: head_dim as u32,
                n_tok: 1,
                n_pos: n_pos as u32,
                kv_dim: kv_dim as u32,
                max_ctx: max_ctx as u32,
                scale: 0.5,
            };
            let mut b = gpu.batch();
            b.attention(&q, &k, &v, &one, &scores, &p, None);
            p.n_tok = 2;
            b.attention(&q, &k, &v, &two, &scores, &p, None);
            b.finish();
            let (mut got, mut want) = (vec![0f32; q_len], vec![0f32; q_len]);
            gpu.read(&one, &mut got);
            gpu.read(&two, &mut want);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(want.iter().any(|w| w.abs() > 0.01), "degenerate reference");
            assert!(
                err < 1e-5,
                "{n_head}/{n_kv} heads of {head_dim}: max difference {err}"
            );
        }
    }
}
