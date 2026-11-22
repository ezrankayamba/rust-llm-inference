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
    LaunchConfig, PinnedHostSlice, PushKernelArg, sys,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

const SOURCE: &str = include_str!("cuda.cu");
const MATVEC_ROWS_PER_BLOCK: usize = 4;
/// Fewest tokens of a chunk whose attention runs as tiled products (three launches, and a
/// fourth to quantize the result); fewer take the one `attention` launch a decoded token
/// does (Mistral-7B: 2 tokens 239 -> 248 tok/s, 4 tokens 358 -> 366, 6 tokens 526 -> 535).
const ATTN_TILED_MIN: u32 = 8;
/// Grid rows (matrices or tokens of one launch) from which the Q4_K int8 matvec reads 4
/// super-blocks a step, not 8: see `mvq_q4_k_step8` in cuda.cu.
const MVQ_MANY: usize = 32;
/// f32/f16 matvecs of at most this many rows run a block per row (see `MATVEC` in cuda.cu).
const MATVEC_WIDE_ROWS: usize = 256;
const REDUCE_THREADS: u32 = 1024;
/// Downloads of at least this many f32s go through page-locked memory.
const PINNED_MIN: usize = 1 << 14;
/// Threads per attention block, which is also the largest head_dim the kernel takes.
pub const ATTN_THREADS: u32 = 1024;
/// The decode attention kernel's cached positions per block, threads per block, most query
/// heads per KV head and most blocks of context (`AP_T`, `AP_G` and `AP_PARTS` in the
/// source; 65536 positions).
const ATTN_PART: usize = 32;
const ATTN_PART_THREADS: u32 = 256;
const ATTN_PART_HEADS: u32 = 8;
const ATTN_PART_MAX: usize = 2048;
/// Decode attention's parts go in multiples of this many (see [`Batch::attention_decode`]).
const ATTN_PART_STEP: usize = 8;
/// Fewest cached positions it runs for; under that its second launch costs more than the
/// one-launch kernel loses (A100: 13-20 against 10-17 microseconds at 128 positions, 19-26
/// against 37-55 at 512).
const ATTN_PART_MIN: u32 = 128;
const ELT_THREADS: usize = 256;
/// Positions a part of one token's latent attention takes (`attention_mla` with `split`), and
/// the values a part keeps per head (`MLA_PART` in cuda.cu).
const MLA_PART_LEN: usize = 32;
const MLA_PART: usize = 514;
/// Weight placement alignment in the device arena (f32 weights are read as `float*`).
const WEIGHT_ALIGN: usize = 256;
/// Bytes past the last weight that kernels may read (and ignore): the Q6_K matvec takes its
/// unaligned blocks as whole 16-byte words.
const WEIGHT_SLACK: usize = 16;

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
/// Weight types with an int8 matmul kernel, which needs whole 256-weight super-blocks.
const MMQ_KERNELS: [(GgmlType, &str); 2] =
    [(GgmlType::Q4_K, "mmq_q4_k"), (GgmlType::Q6_K, "mmq_q6_k")];
/// Types of 32-weight blocks with an f16 matmul over stacked experts ([`Batch::gemm_id`]),
/// which takes them 64 columns at a time.
const GEMM_ID_BLOCK32: [(GgmlType, &str); 2] =
    [(GgmlType::Q8_0, "gemm_q8_0"), (GgmlType::Q5_0, "gemm_q5_0")];
/// Tokens per block of the int8 matmul kernels (`MQ_T` in cuda.cu).
const MMQ_TILE: usize = 64;
/// Fewest tokens the int8 matmul kernels take; fewer run as an int8 matvec each.
pub const MMQ_MIN_TOKENS: usize = 8;
/// Rows, tokens and threads of a block of the tensor-core matmuls (`MA_R`, `MA_T`, `MA_NW`
/// warps in cuda.cu), and their compile's cap on registers per thread.
const MMA_TILE: (usize, usize, u32) = (128, 64, 256);
/// Tokens a block of the kernels for chunks of no more than that: a block stages its whole
/// tile of tokens however many the chunk has.
const MMA_FEW: [usize; 2] = [16, 32];
/// Tokens a block of the lean kernels (`MATMUL_MMK` and `MATMUL_MMK6` in cuda.cu).
const MMK_TOKENS: [usize; 3] = [32, 64, 128];
/// The chunks [`Batch::gemm_dense`] takes, in tokens. Under them the int8 kernels' 16-token
/// blocks are ahead (Mistral-7B at 16 tokens: 8% slower this way), and over them writing the
/// matrix out once for [`Batch::gemm`] is: a block here decodes its weights again for every
/// 64 tokens (Mistral-7B, runs of 30: 6% faster at 224, 1.5% at 256, 2% slower at 320).
pub const GEMM_DENSE_TOKENS: std::ops::RangeInclusive<usize> = 24..=256;
/// The chunks whose Q4_K matrices take `gemm_q4_k_direct` instead (blocks of 128 rows by 32
/// tokens, each lane decoding its own weights): 14336x4096 by 16 tokens 74 us where the int8
/// tensor cores took 92, by 32 tokens 82 where the 64-token block took 99. A second tile of
/// tokens decodes the weights again, so not past 32. Its time hardly depends on the tokens,
/// so 2 to 4 take the int8 matvec that reads a row once for all of them (`mvq_q4_k_multi`;
/// Mistral-7B at 4 tokens 357 tok/s against 346 this way, at 5 tokens 321 against 434).
const GEMM_DIRECT_TOKENS: std::ops::RangeInclusive<usize> = 5..=32;
/// Most (token, expert) pairs an expert, on average, for which [`Batch::gemm_id`] takes
/// `gemm_*_direct_id` (128 rows by 32 pairs a block, no weights staged) and not `gemm_*_id`
/// (64 by 64, staged): Qwen3-30B-A3B's 128 experts, 8 a token, at 32 tokens 1167 -> 1280
/// tok/s, at 48 tokens 1512 -> 1581, at 64 level, at 96 tokens 2262 -> 2158, at 512 tokens
/// 4492 -> 3514.
const GEMM_DIRECT_ID_PAIRS: usize = 4;
/// Fewest tokens of a chunk whose Q6_K matrices take `gemm_q6_k_direct`: 14336x4096 by 16
/// tokens 120 us (int8 126), by 24 tokens 124 (141); by 8 tokens level, and a Mistral-7B
/// prompt 2% slower.
const GEMM_DIRECT_Q6_K_MIN: usize = 16;
/// Under [`MMQ_MIN_TOKENS`], chunks from this many tokens take `gemm_q6_k_direct` too; fewer
/// run `mvq_q6_k_multi*` (Mistral-7B with those against the direct kernel: 3 tokens 317 ->
/// 331 tok/s, 4 tokens 371 -> 390, 5 tokens level).
const GEMM_DIRECT_Q6_K_FEW: usize = 5;
/// Blocks of 128 tokens (`gemm_*_dense128`) decode a matrix once where 64-token blocks decode
/// it twice. Q6_K, whose decoding costs more, takes them at every length from 65 (4096x14336
/// by 96 tokens 278 us -> 221, by 192 tokens 437 -> 411). Q4_K takes them where the last one
/// has more than this many tokens, so 65 to 128 and 193 to 256: 14336x4096 by 72 / 128 / 224 /
/// 256 tokens 192 / 244 / 397 / 418 us in 64-token blocks, 180 / 219 / 378 / 404 in these; by
/// 160 / 192, where the last would hold 32 or 64, 301 / 327 against 329 / 353.
const GEMM_DENSE_WIDE_REST: usize = 64;
/// Most parts of a matrix's columns it takes.
const GEMM_DENSE_PARTS_MAX: usize = 8;
/// Most parts [`mma_parts`] splits a matrix's columns into.
const MMA_PARTS_MAX: usize = 16;
const MMA_REGS: usize = 80;
/// Rows, tokens and threads of a block of the f16 GEMM's kernels, and the columns they take
/// at a time (`GE_K` in cuda.cu).
const GEMM_TILES: [(usize, usize, u32); 3] = [(64, 64, 256), (64, 128, 128), (128, 128, 128)];
const GEMM_COLS: usize = 64;
/// Shared memory of a block of the last of them, in bytes (`GE_S` words a row and a token),
/// and the most parts [`gemm_parts`] splits a matrix's columns into.
const GEMM_ADD_SHARED: usize = (128 + 128) * 36 * 4;
const GEMM_PARTS_MAX: usize = 16;
/// Fewest blocks the GEMM's larger tile is used for.
const GEMM_BLOCKS_MIN: usize = 256;
/// Fewest tokens the f16 GEMM is used for.
pub const GEMM_MIN_TOKENS: usize = 129;
const OTHER_KERNELS: [&str; 23] = [
    "attention_mla_merge",
    "attention_mla",
    "rmsnorm_strided",
    "copy_heads",
    "quant_x",
    "to_f16_zero",
    "rmsnorm",
    "axpy",
    "zero_rows",
    "add_bias",
    "swiglu",
    "rope",
    "norm_rope",
    "attention",
    "attention_part",
    "attention_merge",
    "attn_scores",
    "attn_softmax",
    "attn_pv",
    "moe_topk",
    "moe_group",
    "moe_combine",
    "matmul_f32_tiled",
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

/// A device f16 buffer: what [`Batch::gemm`] multiplies.
pub struct GpuHalf(CudaSlice<u16>);

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

/// One side of [`Batch::copy_heads`]: a head's values start at `t * row + h * head + off`.
#[derive(Clone, Copy)]
pub struct Heads {
    pub row: usize,
    pub head: usize,
    pub off: usize,
}

pub struct AttnParams<'a> {
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
    /// Several tokens' scores and sums may run on f16 tensor cores (not under `precise`).
    pub half: bool,
    /// A device word and the value it holds, which the kernels read in place of that much of
    /// `n_pos`: launches that differ only in it are then the same (see [`RopeArgs::pos_dev`]).
    pub pos_dev: Option<(&'a GpuIds, u32)>,
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
    /// A device word and the value it holds, which the kernel reads in place of that much of
    /// `pos`: a replayed graph's launches then need no update as the position moves.
    pub pos_dev: Option<(&'a GpuIds, usize)>,
    /// The K cache, the V rows and the V cache: the rotated keys are stored into the cache at
    /// their positions instead of back in place, and V is copied there (`rope_dim ==
    /// head_dim`).
    pub store: Option<(&'a GpuBuf, &'a GpuBuf, &'a GpuBuf)>,
}

impl RopeArgs<'_> {
    /// The kernels' position argument, and the device word they add to it.
    fn dev_pos<'g>(&'g self, zero: &'g GpuIds) -> (u32, &'g GpuIds) {
        match self.pos_dev {
            Some((d, v)) => ((self.pos - v) as u32, d),
            None => (self.pos as u32, zero),
        }
    }

    /// The store arguments: K cache, V rows, V cache, the cache row length and the flag.
    fn store_args<'g>(&'g self, x: &'g GpuBuf, heads2: usize) -> [&'g GpuBuf; 3] {
        match self.store {
            Some((k, vs, vd)) => {
                assert_eq!(
                    self.rope_dim, self.head_dim,
                    "storing rope needs full rotation"
                );
                let at = (self.pos + self.n_tok) * self.n_heads2 * self.head_dim;
                assert!(k.len() >= at && vd.len() >= at);
                assert!(vs.len() >= heads2 * self.head_dim);
                [k, vs, vd]
            }
            None => [x, x, x],
        }
    }
}

/// An uploaded byte range of the model file: `[file_off, file_off + len)` lives at `dev_off`.
struct Placed {
    file_off: usize,
    len: usize,
    dev_off: usize,
}

pub struct Gpu {
    /// The tensor scale of each NVFP4 matrix, by file offset ([`Gpu::set_scale`]).
    scales: HashMap<usize, f32>,
    stream: Arc<CudaStream>,
    /// Page-locked host memory that large downloads land in, and its length.
    pinned: Mutex<Option<(PinnedHostSlice<f32>, usize)>>,
    weights: CudaSlice<u8>,
    placed: Vec<Placed>,
    /// Per weight type: the matvec kernel and the matmul one.
    matvec: Vec<(GgmlType, CudaFunction, CudaFunction)>,
    mmq: Vec<(GgmlType, CudaFunction)>,
    /// The tensor-core kernels of [`MMA_FEW`] tokens a block.
    mma_few: Vec<(GgmlType, usize, CudaFunction)>,
    /// The int8 matmuls over pairs grouped by expert (`mmq_*_id`).
    mmq_id: Vec<(GgmlType, CudaFunction)>,
    /// The int8 matvecs of the same types: `mvq_*`, and `mvq_*_pair` for two matrices.
    mvq: Vec<(GgmlType, CudaFunction, CudaFunction)>,
    /// `mvq_q4_k_many`, for launches of [`MVQ_MANY`] grid rows or more.
    mvq_many: CudaFunction,
    /// `mvq_q4_k_multi2`, `3` and `4`: for that many tokens a pass.
    mvq_multi: [CudaFunction; 3],
    /// The Q6_K ones, for 2, 3 and 4 tokens.
    mvq6_multi: [CudaFunction; 3],
    /// `mmq` holds the tensor-core kernels (`mma_*`), which take larger blocks.
    mma: bool,
    /// The lean tensor-core kernels (`mmk_*`) by type and tokens a block; empty where unused.
    mmk: Vec<(GgmlType, usize, CudaFunction)>,
    /// The same over stacked experts' pairs (`mmk_*_id*`), by type and pairs a block.
    mmk_id: Vec<(GgmlType, usize, CudaFunction)>,
    /// Per weight type, the kernel that writes a matrix out as f16; `to_f16` for activations;
    /// and the f16 tensor-core product of the two, where the GPU has one.
    dequant: Vec<(GgmlType, CudaFunction)>,
    /// `dequant`'s Q4_K kernel needs the matrix at a multiple of 16 bytes; this one doesn't.
    dequant_q4_k_any: CudaFunction,
    to_f16: CudaFunction,
    gemm: Option<Vec<CudaFunction>>,
    /// Multiprocessors, and how many blocks of the 128x128 GEMM tile they hold between them.
    gemm_room: (usize, usize),
    /// The f16 products for stacked experts, which decode as they go (`gemm_*_id`).
    gemm_id: Vec<(GgmlType, CudaFunction)>,
    /// The same over every token, a matrix's columns in parts.
    gemm_dense: Vec<(GgmlType, CudaFunction)>,
    gemm_dense128: Vec<(GgmlType, CudaFunction)>,
    /// For Q4_K over few tokens: blocks of 128 rows by 32 tokens.
    gemm_direct: Vec<(GgmlType, CudaFunction)>,
    gemm_direct_id: Vec<(GgmlType, CudaFunction)>,
    /// `attn_scores_h` and `attn_pv_h`.
    attn_half: Option<(CudaFunction, CudaFunction)>,
    /// Flash attention (`attn_flash_*`, the lean compile) by head size; empty where unused.
    attn_flash: Vec<(u32, CudaFunction)>,
    /// NVFP4 on FP4 tensor cores (the compute_120a compile): `quant_fp4` and `mmf_nvfp4`.
    fp4: Option<(CudaFunction, CudaFunction)>,
    other: Vec<(&'static str, CudaFunction)>,
    /// Zeros: the expert id of a plain matvec, one for each token of an int8 batch too short
    /// for the matmul kernels.
    zero_id: GpuIds,
    /// Stands in for the quantized output of a kernel that isn't asked for one.
    no_q: GpuQ,
    name: String,
}

impl Gpu {
    /// Opens device 0, compiles the kernels and copies the byte `ranges` (offset, len) of
    /// `mapped` (the model file) into VRAM. Matvecs may only read weights inside those ranges.
    pub fn new(mapped: &[u8], ranges: &[(usize, usize)]) -> Result<Self, String> {
        Self::with_owned(mapped, &[], ranges)
    }

    /// [`Gpu::new`] where offsets from `mapped.len()` on are in `owned`: matrices the loader
    /// made (DeepSeek's transposed `k_b`), addressed past the end of the file.
    pub fn with_owned(
        mapped: &[u8],
        owned: &[u8],
        ranges: &[(usize, usize)],
    ) -> Result<Self, String> {
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
        let (mut gemm, mut gemm_id, mut mma_few) = (None, Vec::new(), Vec::new());
        let (mut gemm_dense, mut gemm_direct) = (Vec::new(), Vec::new());
        let mut gemm_direct_id = Vec::new();
        let mut attn_half = None;
        let mut gemm_dense128 = Vec::new();
        // From compute capability 12.0 (Blackwell) the lean kernels take the int8 matmuls of
        // 8 tokens and more: a third compile, of them alone and with no register cap. They
        // were written for an RTX 5090 and measured only there, hence the capability;
        // RLI_CUDA_LEAN=1 or 0 turns them on or off anywhere tensor cores are in use.
        let lean = match std::env::var("RLI_CUDA_LEAN").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            _ => {
                ctx.compute_capability()
                    .map_err(err("compute capability"))?
                    .0
                    >= 12
            }
        };
        let (mut mmk, mut mmk_id) = (Vec::new(), Vec::new());
        let mut attn_flash = Vec::new();
        if mma && lean {
            let opts = cudarc::nvrtc::CompileOptions {
                arch: Some("compute_80"),
                options: vec!["-DRLI_MMA".into(), "-DRLI_MMK".into()],
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(SOURCE, opts)
                .map_err(|e| format!("CUDA lean kernel compile failed: {e:?}"))?;
            let module = ctx.load_module(ptx).map_err(err("load CUDA module"))?;
            for (t, name) in [(GgmlType::Q4_K, "q4_k"), (GgmlType::Q6_K, "q6_k")] {
                for tt in MMK_TOKENS {
                    let n = format!("mmk_{name}_t{tt}");
                    mmk.push((t, tt, module.load_function(&n).map_err(err(&n))?));
                }
                for tt in [32, 64] {
                    let n = format!("mmk_{name}_id{tt}");
                    mmk_id.push((t, tt, module.load_function(&n).map_err(err(&n))?));
                }
            }
            for tt in MMK_TOKENS {
                let n = format!("mmk_nvfp4_t{tt}");
                mmk.push((
                    GgmlType::NVFP4,
                    tt,
                    module.load_function(&n).map_err(err(&n))?,
                ));
            }
            // Fused attention for prompt chunks; RLI_CUDA_FLASH=0 keeps the three kernels.
            if std::env::var("RLI_CUDA_FLASH").as_deref() != Ok("0") {
                for hd in [64, 128] {
                    let n = format!("attn_flash_{hd}");
                    attn_flash.push((hd, module.load_function(&n).map_err(err(&n))?));
                }
            }
        }
        // On compute capability 12.0 (an RTX 5090) NVFP4 matrices can multiply on FP4 tensor
        // cores (`Batch::set_fp4_acts`), whose instructions belong to that capability alone
        // (compute_120a).
        let cc = ctx
            .compute_capability()
            .map_err(err("compute capability"))?;
        let fp4 = if cc == (12, 0) {
            let opts = cudarc::nvrtc::CompileOptions {
                arch: Some("compute_120a"),
                options: vec!["-DRLI_MMA".into(), "-DRLI_FP4".into()],
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(SOURCE, opts)
                .map_err(|e| format!("CUDA FP4 kernel compile failed: {e:?}"))?;
            let module = ctx.load_module(ptx).map_err(err("load CUDA module"))?;
            let f = |n| module.load_function(n).map_err(err(n));
            Some((f("quant_fp4")?, f("mmf_nvfp4")?))
        } else {
            None
        };
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
            gemm = Some(
                GEMM_TILES
                    .iter()
                    .map(|(r, t, _)| {
                        let n = format!("gemm_f16_{r}x{t}");
                        module.load_function(&n).map_err(err(&n))
                    })
                    .collect::<Result<_, String>>()?,
            );
            gemm_id = MMQ_KERNELS
                .iter()
                .map(|&(t, n)| (t, n.replace("mmq", "gemm")))
                .chain(GEMM_ID_BLOCK32.map(|(t, n)| (t, n.to_string())))
                .map(|(t, n)| {
                    let n = format!("{n}_id");
                    Ok((t, module.load_function(&n).map_err(err(&n))?))
                })
                .collect::<Result<_, String>>()?;
            for (t, n) in MMQ_KERNELS {
                let n = format!("{}_dense", n.replace("mmq", "gemm"));
                gemm_dense.push((t, module.load_function(&n).map_err(err(&n))?));
                let n = format!("{n}128");
                gemm_dense128.push((t, module.load_function(&n).map_err(err(&n))?));
            }
            for (t, n) in MMQ_KERNELS {
                let n = format!("{}_direct", n.replace("mmq", "gemm"));
                gemm_direct.push((t, module.load_function(&n).map_err(err(&n))?));
                let n = format!("{n}_id");
                gemm_direct_id.push((t, module.load_function(&n).map_err(err(&n))?));
            }
            let f = |n| module.load_function(n).map_err(err(n));
            attn_half = Some((f("attn_scores_h")?, f("attn_pv_h")?));
            for tt in MMA_FEW {
                for (t, n) in MMQ_KERNELS {
                    let n = format!("{}_t{tt}", n.replace("mmq", "mma"));
                    mma_few.push((t, tt, module.load_function(&n).map_err(err(&n))?));
                }
            }
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
        let mmq_id = MMQ_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, func(&format!("{n}_id"))?)))
            .collect::<Result<_, String>>()?;
        let mvq = MMQ_KERNELS
            .iter()
            .chain(&[(GgmlType::NVFP4, "mmq_nvfp4")])
            .map(|&(t, n)| {
                let n = n.replace("mmq", "mvq");
                Ok((t, func(&n)?, func(&format!("{n}_pair"))?))
            })
            .collect::<Result<_, String>>()?;
        let dequant = MATVEC_KERNELS
            .iter()
            .map(|&(t, n)| Ok((t, func(&n.replace("matvec", "dequant"))?)))
            .collect::<Result<_, String>>()?;
        let to_f16 = func("to_f16")?;
        let dequant_q4_k_any = func("dequant_q4_k_any")?;
        let other = OTHER_KERNELS
            .iter()
            .map(|&n| Ok((n, func(n)?)))
            .collect::<Result<_, String>>()?;

        let mut ranges = ranges.to_vec();
        ranges.sort_unstable();
        ranges.dedup();
        let mut placed = Vec::with_capacity(ranges.len());
        let mut total = 0;
        let (file, end) = (mapped.len(), mapped.len() + owned.len());
        for &(file_off, len) in &ranges {
            assert!(file_off + len <= end, "weight range out of file");
            assert!(
                file_off >= file || file_off + len <= file,
                "a range across the file's end"
            );
            // Ranges that touch or overlap in the file stay together, so a stack of experts
            // keeps its stride on the device.
            let last: Option<&mut Placed> = placed.last_mut();
            let p = match last {
                Some(p) if p.file_off + p.len >= file_off && file_off != file => {
                    p.len = p.len.max(file_off + len - p.file_off);
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
            let src = match p.file_off.checked_sub(file) {
                Some(at) => &owned[at..at + p.len],
                None => &mapped[p.file_off..p.file_off + p.len],
            };
            stream
                .memcpy_htod(src, &mut weights.slice_mut(p.dev_off..p.dev_off + p.len))
                .map_err(err("upload weights"))?;
        }
        let zero_id = GpuIds(
            stream
                .alloc_zeros(MMQ_MIN_TOKENS)
                .map_err(err("allocate"))?,
        );
        let no_q = GpuQ {
            q: stream.alloc_zeros(8).map_err(err("allocate"))?,
            d: stream.alloc_zeros(1).map_err(err("allocate"))?,
            s: stream.alloc_zeros(2).map_err(err("allocate"))?,
        };
        let name = format!("cuda ({})", ctx.name().map_err(err("device name"))?);
        let attr = |a| ctx.attribute(a).map_err(err("device attribute"));
        let sms = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize;
        let shared = attr(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        )? as usize;
        let gemm_room = (sms, sms * (shared / GEMM_ADD_SHARED));
        Ok(Self {
            stream,
            pinned: Mutex::new(None),
            weights,
            placed,
            matvec,
            mmq,
            mmq_id,
            mvq,
            mvq_many: func("mvq_q4_k_many")?,
            mvq_multi: [
                func("mvq_q4_k_multi2")?,
                func("mvq_q4_k_multi3")?,
                func("mvq_q4_k_multi4")?,
            ],
            mvq6_multi: [
                func("mvq_q6_k_multi2")?,
                // One that holds 3 tokens took 69 us a launch on Mistral-7B where this takes 56
                // for 3 or 4 (3-token chunks 345 -> 360 tok/s).
                func("mvq_q6_k_multi4")?,
                func("mvq_q6_k_multi4")?,
            ],
            mma,
            mmk,
            mmk_id,
            dequant,
            dequant_q4_k_any,
            to_f16,
            gemm,
            gemm_room,
            mma_few,
            gemm_dense,
            gemm_dense128,
            gemm_direct,
            gemm_direct_id,
            attn_half,
            attn_flash,
            fp4,
            gemm_id,
            other,
            zero_id,
            no_q,
            name,
            scales: HashMap::new(),
        })
    }

    /// Records the tensor scale of the NVFP4 matrix at file `offset`, which the kernels that
    /// read it multiply their results by. Every NVFP4 matrix must have one (1 when the file has
    /// no `.scale` for it).
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
    /// of the model file: [`Gpu::has_matmul_q`] (or NVFP4, which has only the matvecs: see
    /// [`Batch::has_int8`]), and a start at a multiple of 16 bytes on the device, which the
    /// matvecs' 16-byte reads assume (GGUF aligns tensors to 32).
    pub fn has_int8(&self, ty: GgmlType, cols: usize, offset: usize) -> bool {
        let nvfp4 = ty == GgmlType::NVFP4 && cols.is_multiple_of(256);
        (Self::has_matmul_q(ty, cols) || nvfp4) && self.device_offset(offset, 1).is_multiple_of(16)
    }

    pub fn alloc_half(&self, n: usize) -> GpuHalf {
        GpuHalf(
            self.stream
                .alloc_zeros(n.max(1))
                .expect("CUDA buffer allocation failed"),
        )
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
    pub fn write_ids(&self, dst: &mut GpuIds, data: &[u32]) {
        self.stream
            .memcpy_htod(data, &mut dst.0.slice_mut(..data.len()))
            .expect("CUDA upload failed");
    }

    pub fn write(&self, dst: &mut GpuBuf, data: &[f32]) {
        self.stream
            .memcpy_htod(data, &mut dst.0.slice_mut(..data.len()))
            .expect("CUDA upload failed");
    }

    /// Copies the start of `src` into `out`, waiting for queued work.
    pub fn read(&self, src: &GpuBuf, out: &mut [f32]) {
        let src = src.0.slice(..out.len());
        if out.len() < PINNED_MIN {
            self.stream
                .memcpy_dtoh(&src, out)
                .expect("CUDA download failed");
            self.stream.synchronize().expect("CUDA sync failed");
            return;
        }
        // A vocabulary of logits per token: into pageable memory the driver copies it in
        // stages (144 us for 128k floats on an A100), so it goes through a page-locked buffer.
        let mut pinned = self.pinned.lock().unwrap_or_else(|p| p.into_inner());
        if pinned.as_ref().is_none_or(|p| p.1 < out.len()) {
            // SAFETY: f32 has no invalid bit patterns, and the buffer is written before read.
            let host = unsafe { self.stream.context().alloc_pinned_with_flags(out.len(), 0) }
                .expect("allocate page-locked memory");
            *pinned = Some((host, out.len()));
        }
        let (host, _) = pinned.as_mut().expect("allocated above");
        // Exactly the source's length: the copy takes its size from the destination.
        let host = &mut host.as_mut_slice().expect("CUDA sync failed")[..out.len()];
        self.stream
            .memcpy_dtoh(&src, host)
            .expect("CUDA download failed");
        self.stream.synchronize().expect("CUDA sync failed");
        out.copy_from_slice(host);
    }

    pub fn batch(&self) -> Batch<'_> {
        Batch {
            gpu: self,
            mode: Mode::Queue,
            fp4_acts: false,
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
    /// The grid, and threads per block.
    dims: [u32; 4],
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
        [p.gridDimX, p.gridDimY, p.gridDimZ, p.blockDimX] = rec.dims;
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
                fp4_acts: false,
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
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }
        .result()
        .expect("CUDA capture failed");
        let mut b = Batch {
            gpu: self,
            mode: Mode::Capture(Vec::new()),
            fp4_acts: false,
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
                [p.gridDimX, p.gridDimY, p.gridDimZ, p.blockDimX],
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

static PROFILE: OnceLock<bool> = OnceLock::new();
static PROFILE_TIMES: Mutex<BTreeMap<String, (f64, u64)>> = Mutex::new(BTreeMap::new());

/// Whether `RLI_CUDA_PROF` is set: every queued launch is then run to completion and timed, and
/// the totals per kernel are printed when the device is dropped. A graph's launches can't be
/// timed, so callers queue instead of [`Gpu::run_graph`] while profiling.
pub fn profiling() -> bool {
    *PROFILE.get_or_init(|| std::env::var_os("RLI_CUDA_PROF").is_some())
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let times = std::mem::take(&mut *PROFILE_TIMES.lock().unwrap_or_else(|p| p.into_inner()));
        let mut rows: Vec<_> = times.into_iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let total: f64 = rows.iter().map(|r| r.1.0).sum();
        for (name, (t, n)) in rows {
            eprintln!(
                "{name:16} {:9.2} ms {:5.1}% {n:7} launches {:8.1} us each",
                t * 1e3,
                t / total * 100.0,
                t / n as f64 * 1e6
            );
        }
    }
}

impl Gpu {
    /// A kernel's name, for the profile.
    fn name_of(&self, f: *const CudaFunction) -> String {
        let eq = |g: &CudaFunction| std::ptr::eq(g, f);
        if let Some((n, _)) = self.other.iter().find(|(_, g)| eq(g)) {
            n.to_string()
        } else if let Some((t, ..)) = self.matvec.iter().find(|(_, a, b)| eq(a) || eq(b)) {
            format!("f32 {t:?}")
        } else if eq(&self.dequant_q4_k_any) {
            "dequant Q4_K (unaligned)".into()
        } else if let Some((t, _)) = self.dequant.iter().find(|(_, g)| eq(g)) {
            format!("dequant {t:?}")
        } else if eq(&self.to_f16) {
            "to_f16".into()
        } else if let Some(i) = self.gemm.iter().flatten().position(eq) {
            format!("gemm_f16 {}x{}", GEMM_TILES[i].0, GEMM_TILES[i].1)
        } else if self.fp4.as_ref().is_some_and(|f| eq(&f.0)) {
            "quant_fp4".to_string()
        } else if self.fp4.as_ref().is_some_and(|f| eq(&f.1)) {
            "fp4 matmul NVFP4".to_string()
        } else if let Some((hd, _)) = self.attn_flash.iter().find(|(_, g)| eq(g)) {
            format!("attn_flash {hd}")
        } else if self.attn_half.as_ref().is_some_and(|h| eq(&h.0)) {
            "attn_scores_h".to_string()
        } else if self.attn_half.as_ref().is_some_and(|h| eq(&h.1)) {
            "attn_pv_h".to_string()
        } else if let Some((t, _)) = self.gemm_direct_id.iter().find(|(_, g)| eq(g)) {
            format!("gemm direct by expert {t:?}")
        } else if let Some((t, _)) = self.gemm_direct.iter().find(|(_, g)| eq(g)) {
            format!("gemm direct {t:?}")
        } else if let Some((t, _)) = self.gemm_dense128.iter().find(|(_, g)| eq(g)) {
            format!("gemm in-block {t:?}, 128 tokens")
        } else if let Some((t, _)) = self.gemm_dense.iter().find(|(_, g)| eq(g)) {
            format!("gemm in-block {t:?}")
        } else if let Some((t, _)) = self.gemm_id.iter().find(|(_, g)| eq(g)) {
            format!("gemm by expert {t:?}")
        } else if let Some((t, _)) = self.mmq.iter().find(|(_, g)| eq(g)) {
            format!("int8 matmul {t:?}")
        } else if let Some((t, tt, _)) = self.mmk.iter().find(|(.., g)| eq(g)) {
            format!("lean matmul {t:?}, {tt} tokens")
        } else if let Some((t, tt, _)) = self.mmk_id.iter().find(|(.., g)| eq(g)) {
            format!("lean matmul by expert {t:?}, {tt} pairs")
        } else if let Some((t, tt, _)) = self.mma_few.iter().find(|(.., g)| eq(g)) {
            format!("int8 matmul {t:?}, {tt} tokens")
        } else if let Some((t, _)) = self.mmq_id.iter().find(|(_, g)| eq(g)) {
            format!("int8 matmul by expert {t:?}")
        } else if self.mvq6_multi.iter().any(eq) {
            "int8 matvec Q6_K, few tokens".to_string()
        } else if self.mvq_multi.iter().any(eq) {
            "int8 matvec Q4_K, few tokens".to_string()
        } else if eq(&self.mvq_many) {
            "int8 matvec Q4_K, many".to_string()
        } else if let Some((t, ..)) = self.mvq.iter().find(|(_, a, b)| eq(a) || eq(b)) {
            format!("int8 matvec {t:?}")
        } else {
            "?".into()
        }
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
                dims: [0; 4],
                bits: [0; MAX_ARGS],
                sizes: [0; MAX_ARGS],
                n: 0,
            },
        }
    }

    fn launch(&mut self, mut call: Call<'_>, cfg: LaunchConfig) {
        call.rec.dims = [
            cfg.grid_dim.0,
            cfg.grid_dim.1,
            cfg.grid_dim.2,
            cfg.block_dim.0,
        ];
        if let Some(args) = &mut call.args {
            let timed = profiling() && matches!(self.mode, Mode::Queue);
            if timed {
                self.gpu.sync();
            }
            let start = std::time::Instant::now();
            // SAFETY: each caller passes exactly the parameters of the kernel's signature in
            // cuda.cu, with buffer bounds asserted or sized by construction.
            unsafe { args.launch(cfg) }.expect("CUDA launch failed");
            if timed {
                self.gpu.sync();
                let dt = start.elapsed().as_secs_f64();
                let mut times = PROFILE_TIMES.lock().unwrap_or_else(|p| p.into_inner());
                let e = times.entry(self.gpu.name_of(call.rec.func)).or_default();
                (e.0, e.1) = (e.0 + dt, e.1 + 1);
            }
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

/// How many parts to split the columns of a `rows` x `cols` matrix into so that the blocks
/// of the 128x128 GEMM tile over `n_tok` tokens fill a GPU of `sms` multiprocessors that
/// holds `room` of them, if any number does. On an A100 a multiprocessor takes as long over
/// three of these blocks as over four, half as long over two, and blocks past `room` run
/// after the rest: so the count closest under an even number per multiprocessor, at least
/// three quarters of it (128 or 256 blocks on its 108 multiprocessors lost to the smaller
/// tile; 192, 336 and 384 beat it by 8-40%).
fn gemm_parts(rows: usize, cols: usize, n_tok: usize, sms: usize, room: usize) -> Option<usize> {
    let (tr, tt, _) = GEMM_TILES[2];
    let tiles = rows.div_ceil(tr) * n_tok.div_ceil(tt);
    (1..=(cols / GEMM_COLS).min(GEMM_PARTS_MAX))
        .map(|p| (p, tiles * p))
        .filter(|&(_, b)| b <= room && gemm_fills(b, sms))
        // The fullest, and of equals the fewest parts.
        .min_by(|&(p, b), &(q, c)| (c * gemm_full(b, sms), p).cmp(&(b * gemm_full(c, sms), q)))
        .map(|(p, _)| p)
}

/// How many parts the int8 tensor-core matmul splits the columns into: enough for four
/// times the blocks a GPU of `sms` multiprocessors runs at once (its registers allow three
/// each), of at least two 128-column steps. Measured on an A100: 4096x4096 by 64 tokens
/// 179 us in 1 part, 67 in 3, 56 in 16; 14336x4096 285, 188 (3), 165 (8 to 16), 183 (24).
fn mma_parts(rows: usize, cols: usize, token_tiles: usize, sms: usize) -> usize {
    let (tr, _, threads) = MMA_TILE;
    let tiles = rows.div_ceil(tr) * token_tiles;
    let held = sms * (65536 / (MMA_REGS * threads as usize));
    (4 * held)
        .div_ceil(tiles)
        .min(MMA_PARTS_MAX)
        .min(cols / 256)
        .max(1)
}

/// Tokens a block of the lean kernels for `n_tok` tokens: the largest of [`MMK_TOKENS`] whose
/// last tile is more than half full (the work is dealt out by tile and step, so a tile with
/// few tokens leaves its blocks idle: 192 tokens ran 8-10% faster in three tiles of 64 than
/// in two of 128) and that leaves two steps for each of `sms` blocks (Llama-3.2-1B, whose
/// matrices are 2048 wide: 128 tokens 7% faster in tiles of 64, 64 tokens 3.5% in 32).
fn lean_tile(n_tok: usize, rows: usize, cols: usize, sms: usize) -> usize {
    let fits = |&t: &usize| {
        let tiles = n_tok.div_ceil(t);
        tiles * t - n_tok < t / 2 && rows.div_ceil(128) * tiles * (cols / 128) >= 2 * sms
    };
    let all = MMK_TOKENS.iter().rev();
    *all.clone().find(|t| fits(t)).unwrap_or(&MMK_TOKENS[0])
}

/// How many parts [`Batch::gemm_dense`] splits the columns into, as [`mma_parts`]: its blocks
/// are 64 rows by 64 tokens, and a part is at least a super-block. Measured on an A100, us
/// for 1 / 2 / 4 / 8 / 16 parts: 4096x4096 by 64 tokens 98 / 64 / 44 / 40 / 45, 14336x4096
/// 156 / 136 / 121 / 122 / 135; by 256 tokens 147 / 129 / 129 / 129 and 437 / 412 / 414 / 434.
fn dense_parts(tiles: usize, cols: usize, sms: usize) -> usize {
    let held = sms * (65536 / (MMA_REGS * 256));
    (4 * held)
        .div_ceil(tiles)
        .min(GEMM_DENSE_PARTS_MAX)
        .min(cols / 256)
        .max(1)
}

/// The even number of blocks per multiprocessor that `blocks` is under, and whether it is
/// within a quarter of it.
fn gemm_full(blocks: usize, sms: usize) -> usize {
    blocks.div_ceil(2 * sms) * 2 * sms
}
fn gemm_fills(blocks: usize, sms: usize) -> bool {
    4 * blocks >= 3 * gemm_full(blocks, sms)
}

/// The launches of the 128x128 GEMM tile that make up a product, as (first row, rows, parts
/// of the columns), if they can fill the GPU. Where no number of parts does it for all the
/// rows, as many as fill it by themselves go first, four blocks a multiprocessor or two:
/// Mistral-7B's 14336 rows are 448 blocks at 512 tokens where an A100 holds 432, and in one
/// launch the last 16 cost 175 us of 590.
fn gemm_plan(
    rows: usize,
    cols: usize,
    n_tok: usize,
    sms: usize,
    room: usize,
) -> Option<Vec<(usize, usize, usize)>> {
    let (tr, tt, _) = GEMM_TILES[2];
    let ntt = n_tok.div_ceil(tt);
    let (mut plan, mut row) = (Vec::new(), 0);
    loop {
        if let Some(parts) = gemm_parts(rows - row, cols, n_tok, sms, room) {
            plan.push((row, rows - row, parts));
            return Some(plan);
        }
        let whole = [room, 2 * sms]
            .into_iter()
            .map(|blocks| blocks / ntt)
            .find(|&t| {
                t > 0 && t * ntt <= room && gemm_fills(t * ntt, sms) && t * tr < rows - row
            })?;
        plan.push((row, whole * tr, 1));
        row += whole * tr;
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
    /// NVFP4 prompts may take FP4 activations ([`Batch::set_fp4_acts`]).
    fp4_acts: bool,
}

impl<'a> Batch<'a> {
    /// Whether [`Batch::matmul_q`] takes the matrix over `n_tok` tokens: [`Gpu::has_int8`],
    /// and for NVFP4 from [`MMQ_MIN_TOKENS`] only on the lean kernels, its one int8 matmul.
    pub fn has_int8(&self, ty: GgmlType, cols: usize, offset: usize, n_tok: usize) -> bool {
        let lean = || self.gpu.mmk.iter().any(|(t, ..)| *t == ty);
        self.gpu.has_int8(ty, cols, offset)
            && (ty != GgmlType::NVFP4 || n_tok < MMQ_MIN_TOKENS || lean())
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
        let scale = self.gpu.scale(ty, offset);
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
            .arg(&nt)
            .arg(&scale);
        self.launch(
            a,
            grid2(
                rows.div_ceil(MATVEC_ROWS_PER_BLOCK),
                n_tok.div_ceil(MATMUL_TILE),
                (MATVEC_ROWS_PER_BLOCK * 32) as u32,
            ),
        );
    }

    /// [`Batch::matmul`] for an F32 matrix over `n_tok` tokens, added into y, which the
    /// caller zeroes (as [`Batch::to_f16_zero`] does): tiles of 32 rows by 32 tokens, the
    /// columns in as many parts as give every multiprocessor two blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f32_add(
        &mut self,
        rows: usize,
        cols: usize,
        offset: usize,
        x: At<'_>,
        y: At<'_>,
        n_tok: usize,
    ) {
        self.matmul_f32_add_heads(rows, cols, offset, (1, 0), (x, 0), (y, 0), n_tok);
    }

    /// [`Batch::matmul_f32_add`] for `n` matrices `w_stride` bytes apart, each over its own
    /// `n_tok` tokens: matrix `h` reads x from `h * x_stride` and adds into y from
    /// `h * y_stride` (values), all in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f32_add_heads(
        &mut self,
        rows: usize,
        cols: usize,
        offset: usize,
        (n, w_stride): (usize, usize),
        (x, x_stride): (At<'_>, usize),
        (y, y_stride): (At<'_>, usize),
        n_tok: usize,
    ) {
        let span = (n - 1) * w_stride + rows * cols * 4;
        let (_, w_off, _) = self.matrix(GgmlType::F32, rows, cols, offset, span, true);
        assert!(
            n > 0
                && n_tok > 0
                && x.1 + (n - 1) * x_stride + n_tok * cols <= x.0.len()
                && y.1 + (n - 1) * y_stride + n_tok * rows <= y.0.len(),
            "matmul buffer too small"
        );
        let tiles = rows.div_ceil(32) * n_tok.div_ceil(32) * n;
        let parts = (2 * self.gpu.gemm_room.0)
            .div_ceil(tiles)
            .clamp(1, cols / 32);
        let (x_off, y_off) = (x.1 as u64, y.1 as u64);
        let (rows_u, cols_u, nt, parts_u) = (rows as u32, cols as u32, n_tok as u32, parts as u32);
        let strides = [w_stride as u64, x_stride as u64, y_stride as u64];
        let mut a = self.call(self.gpu.other("matmul_f32_tiled"));
        a.arg(&self.gpu.weights)
            .arg(&w_off)
            .arg(&x.0.0)
            .arg(&x_off)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&nt)
            .arg(&parts_u);
        for v in &strides {
            a.arg(v);
        }
        let cfg = LaunchConfig {
            grid_dim: (
                rows.div_ceil(32) as u32,
                n_tok.div_ceil(32) as u32,
                (parts * n) as u32,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.launch(a, cfg);
    }

    /// Whether [`Batch::gemm`] takes `n_tok` tokens over a matrix of `ty` and `cols` columns:
    /// it pays for writing the matrix out as f16 from [`GEMM_MIN_TOKENS`]. Not for the types
    /// of the lean int8 kernels, where the GPU has them (see `MATMUL_MMK`); the others (Q5_0,
    /// Q8_0) ran in f32 there before, 6x slower over a 512-token chunk.
    pub fn has_gemm(&self, ty: GgmlType, cols: usize, n_tok: usize) -> bool {
        !self.gpu.mmk.iter().any(|(t, ..)| *t == ty) && self.gemm_fits(cols, n_tok)
    }

    fn gemm_fits(&self, cols: usize, n_tok: usize) -> bool {
        self.gpu.gemm.is_some() && cols.is_multiple_of(GEMM_COLS) && n_tok >= GEMM_MIN_TOKENS
    }

    /// Whether [`Batch::matmul_q`] can add into y for chunks of [`MMQ_MIN_TOKENS`] or more.
    pub fn has_matmul_q_add(&self) -> bool {
        self.gpu.mma
    }

    /// Whether [`Batch::gemm_add`] takes this product: as [`Batch::has_gemm`], and the larger
    /// tile's blocks can be made to fill the GPU (see [`gemm_plan`]).
    pub fn has_gemm_add(&self, ty: GgmlType, rows: usize, cols: usize, n_tok: usize) -> bool {
        self.has_gemm(ty, cols, n_tok) && self.gemm_plan(rows, cols, n_tok).is_some()
    }

    fn gemm_plan(
        &self,
        rows: usize,
        cols: usize,
        n_tok: usize,
    ) -> Option<Vec<(usize, usize, usize)>> {
        let (sms, room) = self.gpu.gemm_room;
        gemm_plan(rows, cols, n_tok, sms, room)
    }

    /// The `n` values at `x` as f16 in `out`.
    pub fn to_f16(&mut self, x: At<'_>, n: usize, out: &GpuHalf) {
        assert!(x.1 + n <= x.0.len() && n <= out.0.len());
        let (x_off, n_u) = (x.1 as u64, n as u32);
        let mut l = self.call(&self.gpu.to_f16);
        l.arg(&x.0.0).arg(&x_off).arg(&out.0).arg(&n_u);
        self.launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// [`Batch::to_f16`], and zeros in the `len` values at each of up to three `zs`.
    pub fn to_f16_zero(&mut self, x: At<'_>, n: usize, out: &GpuHalf, zs: &[(At<'_>, usize)]) {
        assert!(x.1 + n <= x.0.len() && n <= out.0.len() && zs.len() <= 3);
        assert!(zs.iter().all(|(z, len)| z.1 + len <= z.0.len()));
        let (x_off, n_u) = (x.1 as u64, n as u32);
        // A slot with no result zeroes nothing of x.
        let slot = |i: usize| {
            zs.get(i)
                .map_or((x.0, 0, 0), |&(z, len)| (z.0, z.1 as u64, len as u32))
        };
        let (a, b, c) = (slot(0), slot(1), slot(2));
        let mut l = self.call(self.gpu.other("to_f16_zero"));
        l.arg(&x.0.0).arg(&x_off).arg(&out.0).arg(&n_u);
        l.arg(&a.0.0).arg(&a.1).arg(&a.2);
        l.arg(&b.0.0).arg(&b.1).arg(&b.2);
        l.arg(&c.0.0).arg(&c.1).arg(&c.2);
        let most = zs.iter().map(|z| z.1).max().unwrap_or(0).max(n);
        self.launch(l, grid(most.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// Writes the `rows x cols` matrix of `ty` at byte `offset` of the model file out as f16
    /// into `wh`, for [`Batch::gemm`].
    pub fn dequant(&mut self, ty: GgmlType, rows: usize, cols: usize, offset: usize, wh: &GpuHalf) {
        assert!(cols.is_multiple_of(32) && wh.0.len() >= rows * cols);
        let gpu: &'a Gpu = self.gpu;
        let row_bytes = ty.row_bytes(cols);
        let w_off = gpu.device_offset(offset, rows * row_bytes) as u64;
        let (rows_u, cols_u, rb) = (rows as u32, cols as u32, row_bytes as u32);
        let (_, f) = gpu
            .dequant
            .iter()
            .find(|(t, _)| *t == ty)
            .expect("a kernel per supported type");
        // The wide Q4_K kernel takes a warp per 4 super-blocks and Q6_K's one per super-block,
        // the others one per 32 weights.
        let wide = ty == GgmlType::Q4_K && w_off.is_multiple_of(16);
        let (f, warps) = match (ty, wide) {
            (GgmlType::Q4_K, true) => (f, rows * (cols / 256).div_ceil(4)),
            (GgmlType::Q4_K, false) => (&gpu.dequant_q4_k_any, rows * cols / 32),
            (GgmlType::Q6_K, _) => (f, rows * cols / 256),
            _ => (f, rows * cols / 32),
        };
        // The kernels of the DEQUANT macro take the matrix's tensor scale; the wide Q4_K and
        // the Q6_K ones have none (only NVFP4 matrices do).
        let scaled = !matches!((ty, wide), (GgmlType::Q4_K, true) | (GgmlType::Q6_K, _));
        let scale = gpu.scale(ty, offset);
        let mut l = self.call(f);
        l.arg(&gpu.weights)
            .arg(&w_off)
            .arg(&wh.0)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&rb);
        if scaled {
            l.arg(&scale);
        }
        self.launch(l, grid(warps.div_ceil(8), 256));
    }

    /// [`Batch::matmul`] on f16 tensor cores, where [`Batch::has_gemm`]: the matrix as
    /// [`Batch::dequant`] left it in `wh`, by the `n_tok` rows of `cols` that
    /// [`Batch::to_f16`] left in `xh`. The result differs from the f32 matmul's by the
    /// rounding of weights and activations to f16.
    pub fn gemm(
        &mut self,
        rows: usize,
        cols: usize,
        wh: &GpuHalf,
        xh: &GpuHalf,
        y: At<'_>,
        n_tok: usize,
    ) {
        assert!(self.gemm_fits(cols, n_tok) && rows > 0);
        assert!(wh.0.len() >= rows * cols && xh.0.len() >= n_tok * cols);
        assert!(y.1 + n_tok * rows <= y.0.len());
        if let Some(plan) = self.gemm_plan(rows, cols, n_tok) {
            for (row, n_rows, parts) in plan {
                // Parts add their sums.
                if parts > 1 {
                    self.zero_rows(y, (row, n_rows, rows), n_tok);
                }
                self.gemm_tile(
                    2,
                    parts,
                    parts > 1,
                    (row, n_rows, rows),
                    cols,
                    wh,
                    xh,
                    y,
                    n_tok,
                );
            }
            return;
        }
        // The larger tile once it still leaves a few blocks per multiprocessor.
        let (lr, lt, _) = GEMM_TILES[1];
        let i = (rows.div_ceil(lr) * n_tok.div_ceil(lt) >= GEMM_BLOCKS_MIN) as usize;
        self.gemm_tile(i, 1, false, (0, rows, rows), cols, wh, xh, y, n_tok);
    }

    /// `y += W x` where [`Batch::has_gemm_add`], operands as in [`Batch::gemm`].
    pub fn gemm_add(
        &mut self,
        rows: usize,
        cols: usize,
        wh: &GpuHalf,
        xh: &GpuHalf,
        y: At<'_>,
        n_tok: usize,
    ) {
        assert!(wh.0.len() >= rows * cols && xh.0.len() >= n_tok * cols);
        assert!(y.1 + n_tok * rows <= y.0.len() && rows > 0);
        let plan = self.gemm_plan(rows, cols, n_tok);
        let plan = plan.filter(|_| self.gemm_fits(cols, n_tok));
        for (row, n_rows, parts) in plan.expect("checked by has_gemm_add") {
            self.gemm_tile(2, parts, true, (row, n_rows, rows), cols, wh, xh, y, n_tok);
        }
    }

    /// Tile `i` of [`GEMM_TILES`] over `parts` parts of the columns and `rows` of the matrix's
    /// rows (first, how many, of how many), added to what y holds with `acc`. Only the last
    /// tile takes parts (which need `acc`), `acc` or less than every row.
    #[allow(clippy::too_many_arguments)]
    fn gemm_tile(
        &mut self,
        i: usize,
        parts: usize,
        acc: bool,
        (base, rows, ldy): (usize, usize, usize),
        cols: usize,
        wh: &GpuHalf,
        xh: &GpuHalf,
        y: At<'_>,
        n_tok: usize,
    ) {
        assert!(i == 2 || (parts == 1 && !acc && base == 0 && rows == ldy));
        assert!(acc || parts == 1);
        let gpu: &'a Gpu = self.gpu;
        let (rows_u, cols_u) = (rows as u32, cols as u32);
        let (y_off, nt) = (y.1 as u64, n_tok as u32);
        let (parts_u, base_u, ldy_u, acc_u) = (parts as u32, base as u32, ldy as u32, acc as u32);
        let (tr, tt, threads) = GEMM_TILES[i];
        let mut l = self.call(&gpu.gemm.as_ref().expect("checked by has_gemm")[i]);
        l.arg(&wh.0)
            .arg(&xh.0)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&nt)
            .arg(&parts_u)
            .arg(&base_u)
            .arg(&ldy_u)
            .arg(&acc_u);
        let cfg = grid2(rows.div_ceil(tr), n_tok.div_ceil(tt) * parts, threads);
        self.launch(l, cfg);
    }

    /// Whether [`Batch::matmul_q_id`] runs this stack on the lean tensor-core kernels, which
    /// are then ahead of [`Batch::gemm_id`].
    pub fn has_lean_id(&self, ty: GgmlType, cols: usize, offset: usize, stride: usize) -> bool {
        self.gpu.mmk_id.iter().any(|(t, ..)| *t == ty)
            && cols.is_multiple_of(128)
            && stride.is_multiple_of(16)
            && self.gpu.has_int8(ty, cols, offset)
    }

    /// Whether [`Batch::gemm_id`] takes a stack of `ty` matrices of `cols` columns at byte
    /// `offset` of the model file, `stride` bytes apart: Q4_K or Q6_K on a GPU with f16 tensor
    /// cores, and for Q4_K every row at a multiple of 16 bytes on the device.
    pub fn has_gemm_id(&self, ty: GgmlType, cols: usize, offset: usize, stride: usize) -> bool {
        let aligned = stride.is_multiple_of(16) && self.gpu.has_int8(ty, cols, offset);
        let block32 = GEMM_ID_BLOCK32.iter().any(|(t, _)| *t == ty);
        self.gpu.gemm_id.iter().any(|(t, _)| *t == ty)
            && cols.is_multiple_of(if block32 { 64 } else { 256 })
            && (ty != GgmlType::Q4_K || aligned)
    }

    /// [`Batch::matmul_q_id`] on f16 tensor cores, over f16 activations (`xh`, rows of
    /// `cols`): the weights are decoded to f16 inside the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_id(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        stride: usize,
        n_stack: usize,
        (perm, start): (&GpuIds, &GpuIds),
        xh: &GpuHalf,
        x_div: usize,
        y: At<'_>,
        n: usize,
    ) {
        assert!(self.has_gemm_id(ty, cols, offset, stride) && rows > 0 && n > 0 && x_div > 0);
        assert!(
            perm.0.len() >= n
                && start.0.len() > n_stack
                && xh.0.len() >= ((n - 1) / x_div + 1) * cols
                && y.1 + n * rows <= y.0.len(),
            "matmul buffer too small"
        );
        let gpu: &'a Gpu = self.gpu;
        let row_bytes = ty.row_bytes(cols);
        let span = (n_stack - 1) * stride + rows * row_bytes;
        let w_off = gpu.device_offset(offset, span) as u64;
        let (stride, xd, y_off) = (stride as u64, x_div as u32, y.1 as u64);
        let (rows_u, cols_u, rb) = (rows as u32, cols as u32, row_bytes as u32);
        let direct =
            n <= GEMM_DIRECT_ID_PAIRS * n_stack && gpu.gemm_direct_id.iter().any(|(t, _)| *t == ty);
        let f = if direct {
            &gpu.gemm_direct_id
        } else {
            &gpu.gemm_id
        };
        let f = f.iter().find(|(t, _)| *t == ty);
        let mut a = self.call(&f.expect("checked by has_gemm_id").1);
        a.arg(&gpu.weights)
            .arg(&w_off)
            .arg(&stride)
            .arg(&perm.0)
            .arg(&start.0)
            .arg(&xh.0)
            .arg(&xd)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&rb);
        self.launch(
            a,
            grid2(rows.div_ceil(if direct { 128 } else { 64 }), n_stack, 256),
        );
    }

    /// Whether [`Batch::gemm_dense`] takes this matrix over a chunk of `n_tok` tokens.
    pub fn has_gemm_dense(&self, ty: GgmlType, cols: usize, offset: usize, n_tok: usize) -> bool {
        let min = match ty {
            GgmlType::Q4_K => *GEMM_DIRECT_TOKENS.start(),
            GgmlType::Q6_K if n_tok < MMQ_MIN_TOKENS => GEMM_DIRECT_Q6_K_FEW,
            GgmlType::Q6_K => GEMM_DIRECT_Q6_K_MIN,
            _ => *GEMM_DENSE_TOKENS.start(),
        };
        let lean = n_tok >= MMQ_MIN_TOKENS && self.gpu.mmk.iter().any(|(t, ..)| *t == ty);
        !lean
            && (min..=*GEMM_DENSE_TOKENS.end()).contains(&n_tok)
            && self.gpu.gemm_dense.iter().any(|(t, _)| *t == ty)
            && self.has_gemm_id(ty, cols, offset, 0)
    }

    /// `y = W x` (`y += W x` with `add`) on f16 tensor cores where [`Batch::has_gemm_dense`],
    /// over the `n_tok` rows of `cols` that [`Batch::to_f16`] left in `xh`. The weights are
    /// decoded to f16 inside the kernel, as for [`Batch::gemm_id`]: nothing is written out
    /// first, which costs as much as the product at these sizes. Blocks of 64 rows by 64
    /// tokens take the columns in parts and add their sums.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_dense(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        xh: &GpuHalf,
        y: At<'_>,
        n_tok: usize,
        add: bool,
    ) {
        assert!(self.has_gemm_dense(ty, cols, offset, n_tok) && rows > 0);
        assert!(xh.0.len() >= n_tok * cols && y.1 + n_tok * rows <= y.0.len());
        let direct = n_tok <= *GEMM_DIRECT_TOKENS.end();
        let wide =
            n_tok > 64 && (ty == GgmlType::Q6_K || (n_tok - 1) % 128 >= GEMM_DENSE_WIDE_REST);
        let (tr, tt) = if direct {
            (128, 32)
        } else if wide {
            (64, 128)
        } else {
            (64, 64)
        };
        let tiles = rows.div_ceil(tr) * n_tok.div_ceil(tt);
        let sms = self.gpu.gemm_room.0;
        let parts = if direct {
            // Measured on an A100: 14336x4096 by 16 tokens (112 tiles) 83 us in 4 parts,
            // 74 in 8, 78 in 12; 4096x14336 (32 tiles) 131 in 4, 86 in 8, 77 in 16.
            (8 * sms).div_ceil(tiles).min(16).min(cols / 256).max(1)
        } else {
            dense_parts(tiles, cols, sms)
        };
        if !add {
            self.zero_rows(y, (0, rows, rows), n_tok);
        }
        let gpu: &'a Gpu = self.gpu;
        let row_bytes = ty.row_bytes(cols);
        let w_off = gpu.device_offset(offset, rows * row_bytes) as u64;
        let (stride, xd, y_off) = (0u64, 1u32, y.1 as u64);
        let (rows_u, cols_u, rb) = (rows as u32, cols as u32, row_bytes as u32);
        let (nt, parts_u) = (n_tok as u32, parts as u32);
        let f = if direct {
            &gpu.gemm_direct
        } else if wide {
            &gpu.gemm_dense128
        } else {
            &gpu.gemm_dense
        };
        let f = f.iter().find(|(t, _)| *t == ty);
        let mut a = self.call(&f.expect("checked by has_gemm_id").1);
        a.arg(&gpu.weights)
            .arg(&w_off)
            .arg(&stride)
            .arg(&gpu.zero_id.0)
            .arg(&gpu.zero_id.0)
            .arg(&xh.0)
            .arg(&xd)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&rb)
            .arg(&nt)
            .arg(&parts_u);
        // Q4_K's blocks of 128 tokens have 16 warps (`NW` in cuda.cu).
        let threads = if wide && ty == GgmlType::Q4_K {
            512
        } else {
            256
        };
        self.launch(
            a,
            grid2(rows.div_ceil(tr), n_tok.div_ceil(tt) * parts, threads),
        );
    }

    /// Lets NVFP4 matrices over 8+ tokens multiply on FP4 tensor cores, their activations
    /// quantized to NVFP4 too ([`Batch::has_fp4`]): llama.cpp's default on Blackwell, and 1.8x
    /// the int8 kernel's speed over 512-token prompts, at about 75 times its KL divergence
    /// from f32 (`docs/benchmarks.md`, 2026-11-22). Off unless set.
    pub fn set_fp4_acts(&mut self, on: bool) {
        self.fp4_acts = on;
    }

    /// Whether [`Batch::matmul_fp4`] takes the `ty` matrix of `cols` columns at byte `offset`
    /// of the model file over `n_tok` tokens: with [`Batch::set_fp4_acts`], NVFP4 from
    /// [`MMQ_MIN_TOKENS`] tokens, on a GPU with FP4 tensor cores, rows at multiples of 16
    /// bytes (256 columns).
    pub fn has_fp4(&self, ty: GgmlType, cols: usize, offset: usize, n_tok: usize) -> bool {
        self.fp4_acts
            && self.gpu.fp4.is_some()
            && ty == GgmlType::NVFP4
            && n_tok >= MMQ_MIN_TOKENS
            && self.gpu.has_int8(ty, cols, offset)
    }

    /// Quantizes `n_tok` rows of `cols` at `x` to NVFP4 for [`Batch::matmul_fp4`]: the
    /// weights' block layout in `out`'s int8 room, a scale per token in its first scales.
    pub fn quantize_fp4(&mut self, x: At<'_>, n_tok: usize, cols: usize, out: &GpuQ) {
        let (quant, _) = self.gpu.fp4.as_ref().expect("checked by has_fp4");
        assert!(cols.is_multiple_of(256) && x.1 + n_tok * cols <= x.0.len());
        assert!(out.q.len() * 4 >= n_tok * cols && out.d.len() >= n_tok);
        let (x_off, cols_u) = (x.1 as u64, cols as u32);
        let mut l = self.call(quant);
        l.arg(&x.0.0)
            .arg(&x_off)
            .arg(&out.q)
            .arg(&out.d)
            .arg(&cols_u);
        self.launch(l, grid(n_tok, 256));
    }

    /// `y = W x` (`y += W x` with `add`) for the NVFP4 matrix at `offset`, over the `n_tok`
    /// tokens [`Batch::quantize_fp4`] left in `xf`, on FP4 tensor cores.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_fp4(
        &mut self,
        rows: usize,
        cols: usize,
        offset: usize,
        xf: &GpuQ,
        y: At<'_>,
        n_tok: usize,
        add: bool,
    ) {
        let ty = GgmlType::NVFP4;
        assert!(self.has_fp4(ty, cols, offset, n_tok) && rows > 0);
        assert!(xf.q.len() * 4 >= n_tok * cols && y.1 + n_tok * rows <= y.0.len());
        let gpu: &'a Gpu = self.gpu;
        let (_, f) = gpu.fp4.as_ref().expect("checked by has_fp4");
        let row_bytes = ty.row_bytes(cols);
        let w_off = gpu.device_offset(offset, rows * row_bytes) as u64;
        let (y_off, rows_u, cols_u, rb) = (y.1 as u64, rows as u32, cols as u32, row_bytes as u32);
        let (nt, add_u, scale) = (n_tok as u32, add as u32, gpu.scale(ty, offset));
        // Few tiles leave most multiprocessors idle: then each takes a part of the columns
        // and adds into y, two blocks' worth for every multiprocessor in all.
        let tiles = rows.div_ceil(128) * n_tok.div_ceil(64);
        let parts = (2 * gpu.gemm_room.0).div_ceil(tiles).min(cols / 256).max(1);
        if parts > 1 && !add {
            self.zero_rows(y, (0, rows, rows), n_tok);
        }
        let parts_u = parts as u32;
        let mut a = self.call(f);
        a.arg(&gpu.weights)
            .arg(&w_off)
            .arg(&xf.q)
            .arg(&xf.d)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&rb)
            .arg(&nt)
            .arg(&add_u)
            .arg(&scale)
            .arg(&parts_u);
        self.launch(
            a,
            grid2(rows.div_ceil(128), n_tok.div_ceil(64) * parts, 256),
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
    /// f32 matmul's by the activations' rounding. `add` adds it to `y` instead of storing it:
    /// under [`MMQ_MIN_TOKENS`] tokens, and above where [`Batch::has_matmul_q_add`].
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
        assert!(self.has_int8(ty, cols, offset, n_tok) && rows > 0 && n_tok > 0);
        assert!(xq.q.len() * 4 >= n_tok * cols && y.1 + n_tok * rows <= y.0.len());
        // A few tokens of a Q4_K or Q6_K matrix share one read of each row.
        let multi = match ty {
            GgmlType::Q4_K => Some(&self.gpu.mvq_multi),
            GgmlType::Q6_K => Some(&self.gpu.mvq6_multi),
            _ => None,
        };
        if let (Some(multi), true) = (multi, n_tok < MMQ_MIN_TOKENS && n_tok > 1) {
            let gpu: &'a Gpu = self.gpu;
            let row_bytes = ty.row_bytes(cols);
            let w_off = gpu.device_offset(offset, rows * row_bytes) as u64;
            let (y_off, rb, nt, add) = (y.1 as u64, row_bytes as u32, n_tok as u32, add as u32);
            let (rows_u, cols_u) = (rows as u32, cols as u32);
            let mut a = self.call(&multi[n_tok.min(4) - 2]);
            a.arg(&gpu.weights)
                .arg(&w_off)
                .arg(&xq.q)
                .arg(&xq.d)
                .arg(&xq.s)
                .arg(&y.0.0)
                .arg(&y_off)
                .arg(&rows_u)
                .arg(&cols_u)
                .arg(&rb)
                .arg(&nt)
                .arg(&add);
            let threads = (MATVEC_ROWS_PER_BLOCK * 32) as u32;
            return self.launch(a, grid(rows.div_ceil(MATVEC_ROWS_PER_BLOCK), threads));
        }
        if n_tok < MMQ_MIN_TOKENS {
            let zero = &self.gpu.zero_id;
            return self.matvec_q_id(
                ty, rows, cols, offset, 0, 1, zero, xq, cols, 1, y, rows, n_tok, add, None,
            );
        }
        assert!(
            !add || self.gpu.mma,
            "the dp4a matmul kernels don't accumulate"
        );
        let gpu: &'a Gpu = self.gpu;
        let mut lean = gpu.mmk.iter().filter(|(t, ..)| *t == ty);
        let tt = lean_tile(n_tok, rows, cols, gpu.gemm_room.0);
        // The kernels count rows' offsets and their units of work in 32 bits.
        let row_bytes = ty.row_bytes(cols);
        let units = rows.div_ceil(128) * n_tok.div_ceil(tt) * (cols / 128);
        let small = (units * gpu.gemm_room.0).max(rows * row_bytes) < u32::MAX as usize;
        if let Some((_, _, f)) = lean.find(|(_, t, _)| *t == tt && small) {
            if !add {
                self.zero_rows(y, (0, rows, rows), n_tok);
            }
            let w_off = gpu.device_offset(offset, rows * row_bytes) as u64;
            // The units of work (a tile's 128-column steps) are dealt evenly to a block per
            // multiprocessor, of at least two steps. Two blocks each was 3-6% slower over
            // Mistral-7B and Llama-3.2-1B prompts of 48 to 128 tokens.
            let blocks = gpu.gemm_room.0.min(units / 2).max(1);
            let (y_off, rb) = (y.1 as u64, row_bytes as u32);
            let (rows_u, cols_u, nt, one) = (rows as u32, cols as u32, n_tok as u32, 1u32);
            let scale = gpu.scale(ty, offset);
            let mut a = self.call(f);
            a.arg(&gpu.weights)
                .arg(&w_off)
                .arg(&xq.q)
                .arg(&xq.d)
                .arg(&xq.s)
                .arg(&y.0.0)
                .arg(&y_off)
                .arg(&rows_u)
                .arg(&cols_u)
                .arg(&rb)
                .arg(&nt)
                .arg(&one);
            if ty == GgmlType::NVFP4 {
                a.arg(&scale);
            }
            self.launch(a, grid2(blocks, 1, 256));
            return;
        }
        let few = gpu
            .mma_few
            .iter()
            .find(|(t, tt, _)| *t == ty && n_tok <= *tt);
        let tt = few.map_or(MMA_TILE.1, |f| f.1);
        // The tensor-core kernel in parts, which add: 0 for one part that stores.
        let parts = if gpu.mma {
            mma_parts(rows, cols, n_tok.div_ceil(tt), gpu.gemm_room.0)
        } else {
            1
        };
        if !add && parts > 1 {
            self.zero_rows(y, (0, rows, rows), n_tok);
        }
        let row_bytes = ty.row_bytes(cols);
        let w_off = gpu.device_offset(offset, rows * row_bytes) as u64;
        let f = match few {
            Some(f) => &f.2,
            None => {
                let f = gpu.mmq.iter().find(|(t, _)| *t == ty);
                &f.expect("checked by has_matmul_q").1
            }
        };
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
        let parts_u = if add || parts > 1 { parts as u32 } else { 0 };
        if self.gpu.mma {
            a.arg(&parts_u);
        }
        let cfg = if self.gpu.mma {
            grid2(
                rows.div_ceil(MMA_TILE.0),
                n_tok.div_ceil(tt) * parts,
                MMA_TILE.2,
            )
        } else {
            grid2(rows.div_ceil(32), n_tok.div_ceil(MMQ_TILE), 256)
        };
        self.launch(a, cfg);
    }

    /// Groups the `n` expert ids in `ids` (each under `n_expert`) for [`Batch::matmul_q_id`]:
    /// `perm` gets the positions in `ids` expert by expert, `start` (`n_expert + 1` long) where
    /// each expert's begin.
    pub fn moe_group(
        &mut self,
        ids: &GpuIds,
        perm: &GpuIds,
        start: &GpuIds,
        n: usize,
        n_expert: usize,
    ) {
        assert!(n_expert <= MOE_MAX_EXPERTS && start.0.len() > n_expert);
        assert!(ids.0.len() >= n && perm.0.len() >= n);
        let (n_u, ne) = (n as u32, n_expert as u32);
        let mut l = self.call(self.gpu.other("moe_group"));
        l.arg(&ids.0).arg(&perm.0).arg(&start.0).arg(&n_u).arg(&ne);
        self.launch(l, grid(1, 256));
    }

    /// [`Batch::matvec_q_id`] for many rows at once, over the grouping [`Batch::moe_group`]
    /// made of the `n` ids: an expert's weights are decoded once for all the rows that chose
    /// it. Row `i` of `y` (rows of `rows`) is its expert's matrix times row `i / x_div` of `xq`
    /// (rows of `cols`).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q_id(
        &mut self,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        offset: usize,
        stride: usize,
        n_stack: usize,
        (perm, start): (&GpuIds, &GpuIds),
        xq: &GpuQ,
        x_div: usize,
        y: At<'_>,
        n: usize,
    ) {
        assert!(self.gpu.has_int8(ty, cols, offset) && rows > 0 && n > 0 && x_div > 0);
        assert!(
            perm.0.len() >= n
                && start.0.len() > n_stack
                && xq.q.len() * 4 >= ((n - 1) / x_div + 1) * cols
                && y.1 + n * rows <= y.0.len(),
            "matmul buffer too small"
        );
        let row_bytes = ty.row_bytes(cols);
        let span = (n_stack - 1) * stride + rows * row_bytes;
        let w_off = self.gpu.device_offset(offset, span) as u64;
        // Blocks of 64 pairs where an expert has more than 4 on average: Qwen3-30B-A3B on an
        // RTX 5090, 64 against 32 a block, 32 tokens 3% slower, 64 level, 128 +4%, 256 +8%,
        // 512 +14%; 128 a block against 64 was 3-4% slower at 128 to 256 tokens, level at 384.
        let lean = self.has_lean_id(ty, cols, offset, stride);
        let tt = if n > 4 * n_stack { 64 } else { 32 };
        let f = if lean {
            let f = self
                .gpu
                .mmk_id
                .iter()
                .find(|(t, m, _)| *t == ty && *m == tt);
            &f.expect("checked by has_lean_id").2
        } else {
            let f = self.gpu.mmq_id.iter().find(|(t, _)| *t == ty);
            &f.expect("checked by has_matmul_q").1
        };
        let (stride, xd, y_off) = (stride as u64, x_div as u32, y.1 as u64);
        let (rows_u, cols_u, rb) = (rows as u32, cols as u32, row_bytes as u32);
        let mut a = self.call(f);
        a.arg(&self.gpu.weights)
            .arg(&w_off)
            .arg(&stride)
            .arg(&perm.0)
            .arg(&start.0)
            .arg(&xq.q)
            .arg(&xq.d)
            .arg(&xq.s)
            .arg(&xd)
            .arg(&y.0.0)
            .arg(&y_off)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&rb);
        self.launch(
            a,
            grid2(rows.div_ceil(if lean { 128 } else { 32 }), n_stack, 256),
        );
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
        second: Option<(usize, At<'_>)>,
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
        // The kernel adds the second stack's distance to the offset, modulo 2^64.
        let second_offset = second.map(|(offset2, _)| offset2);
        let second = second.map(|(offset2, y2)| {
            assert!(self.gpu.has_int8(ty, cols, offset2));
            assert!(y2.1 + (n - 1) * y_stride + rows <= y2.0.len());
            let w2 = self.gpu.device_offset(offset2, span) as u64;
            (w2.wrapping_sub(w_off), y2)
        });
        let (id_stride, xs, xd) = (stride as u64, x_stride as u32, x_div as u32);
        // NVFP4 stacks have a tensor scale per matrix, which the kernel doesn't take.
        assert!(ty != GgmlType::NVFP4 || n_stack == 1);
        let scales = [
            self.gpu.scale(ty, offset),
            second.map_or(1.0, |_| {
                self.gpu.scale(ty, second_offset.expect("set with second"))
            }),
        ];
        self.launch_mvq(
            ty, rows, cols, w_off, id_stride, ids, xq, xs, xd, y, y_stride, n, add, second, scales,
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
        let (zero, y2) = (&self.gpu.zero_id, At(y.0, y.1 + rows));
        let [a, b] = offsets;
        self.matvec_q_id(
            ty,
            rows,
            cols,
            a,
            0,
            1,
            zero,
            xq,
            cols,
            1,
            y,
            rows,
            1,
            false,
            Some((b, y2)),
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
        second: Option<(u64, At<'_>)>,
        scales: [f32; 2],
    ) {
        let (_, one, pair) = self
            .gpu
            .mvq
            .iter()
            .find(|(t, ..)| *t == ty)
            .expect("checked by has_matmul_q");
        let many = second.is_none() && ty == GgmlType::Q4_K && n >= MVQ_MANY;
        let f = if second.is_some() {
            pair
        } else if many {
            &self.gpu.mvq_many
        } else {
            one
        };
        let (y_off, rb) = (y.1 as u64, ty.row_bytes(cols) as u32);
        let (rows_u, cols_u, ys, add) = (rows as u32, cols as u32, y_stride as u32, add as u32);
        let (z_stride, y2) = second.unwrap_or((0, y));
        let y2_off = y2.1 as u64;
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
        if second.is_some() {
            a.arg(&y2.0.0).arg(&y2_off).arg(&z_stride);
        }
        if ty == GgmlType::NVFP4 {
            a.arg(&scales[0]);
            if second.is_some() {
                a.arg(&scales[1]);
            }
        }
        let mut cfg = grid2(
            rows.div_ceil(MATVEC_ROWS_PER_BLOCK),
            n,
            (MATVEC_ROWS_PER_BLOCK * 32) as u32,
        );
        cfg.grid_dim.2 = 1 + second.is_some() as u32;
        self.launch(a, cfg);
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
        // Only a plain matvec of an NVFP4 matrix: a stack's matrices would each need their
        // own scale (`models::cuda` routes NVFP4 experts one at a time).
        assert!(
            ty != GgmlType::NVFP4 || n_stack == 1,
            "NVFP4 stacks have no id kernels"
        );
        let scale = self.gpu.scale(ty, offset);
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
            .arg(&ys)
            .arg(&scale);
        // A small f32 matrix (a MoE router) gets a block of 8 warps per row: see MATVEC.
        let wide = matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
            && rows <= MATVEC_WIDE_ROWS;
        let cfg = if wide {
            grid2(rows, n, 256)
        } else {
            grid2(
                rows.div_ceil(MATVEC_ROWS_PER_BLOCK),
                n,
                (MATVEC_ROWS_PER_BLOCK * 32) as u32,
            )
        };
        self.launch(a, cfg);
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

    /// RMS-normalizes `rows` consecutive rows of `n` from `x` into `out` (may alias), and
    /// into `quant` quantized, or into `half` as f16 (as [`Batch::to_f16`] would), if given.
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
        half: Option<&GpuHalf>,
    ) {
        assert!(x.1 + rows * n <= x.0.len() && out.1 + rows * n <= out.0.len());
        let (quant, mode) = Self::half_out(quant, half, rows * n);
        let (x_off, o_off, n_u) = (x.1 as u64, out.1 as u64, n as u32);
        let (q, _) = self.quant_out(quant, rows * n, n);
        let mut a = self.call(self.gpu.other("rmsnorm"));
        a.arg(&x.0.0)
            .arg(&x_off)
            .arg(&out.0.0)
            .arg(&o_off)
            .arg(&w.0)
            .arg(&n_u)
            .arg(&eps);
        match half {
            Some(h) => a.arg(&h.0),
            None => a.arg(&q.q),
        };
        a.arg(&q.d).arg(&q.s).arg(&mode);
        // A thread per value where the block can hold them: with fewer, each thread reads
        // several values in turn and waits out the memory latency for every one.
        let threads = (n.next_multiple_of(32) as u32).min(REDUCE_THREADS);
        self.launch(a, grid(rows, threads));
    }

    /// The mode a kernel that writes its `n` results quantized (1) or as f16 (2) takes, and
    /// `quant` if that is the one it writes: `half` comes first.
    fn half_out<'c>(
        quant: Option<&'c GpuQ>,
        half: Option<&GpuHalf>,
        n: usize,
    ) -> (Option<&'c GpuQ>, u32) {
        match half {
            Some(h) => {
                assert!(h.0.len() >= n);
                (None, 2)
            }
            None => (quant, quant.is_some() as u32),
        }
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

    /// [`Batch::rmsnorm`] of the first `n` values of `rows` rows, `x_stride` apart at `x` and
    /// `out_stride` apart at `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_strided(
        &mut self,
        x: At<'_>,
        x_stride: usize,
        out: At<'_>,
        out_stride: usize,
        w: &GpuBuf,
        rows: usize,
        n: usize,
        eps: f32,
    ) {
        assert!(rows >= 1 && n <= x_stride.min(out_stride) && n <= w.len());
        assert!(x.1 + (rows - 1) * x_stride + n <= x.0.len());
        assert!(out.1 + (rows - 1) * out_stride + n <= out.0.len());
        let (x_off, o_off) = (x.1 as u64, out.1 as u64);
        let (xs, os, n_u) = (x_stride as u32, out_stride as u32, n as u32);
        let mut a = self.call(self.gpu.other("rmsnorm_strided"));
        a.arg(&x.0.0)
            .arg(&x_off)
            .arg(&xs)
            .arg(&out.0.0)
            .arg(&o_off)
            .arg(&os)
            .arg(&w.0)
            .arg(&n_u)
            .arg(&eps);
        let threads = (n.next_multiple_of(32) as u32).min(REDUCE_THREADS);
        self.launch(a, grid(rows, threads));
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
        assert!(m >= 1 && nh >= 1 && src.1 + last(from) <= src.0.len());
        assert!(dst.1 + last(to) <= dst.0.len());
        let u = |v: usize| u32::try_from(v).expect("fits u32");
        let (s_off, d_off) = (src.1 as u64, dst.1 as u64);
        let a = [
            u(m),
            u(nh),
            u(w),
            u(from.row),
            u(from.head),
            u(from.off),
            u(to.row),
            u(to.head),
            u(to.off),
        ];
        let mut l = self.call(self.gpu.other("copy_heads"));
        l.arg(&src.0.0).arg(&s_off).arg(&dst.0.0).arg(&d_off);
        for v in &a {
            l.arg(v);
        }
        let n = m * nh * w;
        self.launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
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

    /// Zeroes `rows` (first, how many, of how many) of each of the `n` tokens at `a`.
    fn zero_rows(&mut self, a: At<'_>, (base, rows, ldy): (usize, usize, usize), n: usize) {
        assert!(a.1 + n * ldy <= a.0.len() && base + rows <= ldy);
        let (a_off, n_u) = (a.1 as u64, n as u32);
        let (ldy_u, base_u, rows_u) = (ldy as u32, base as u32, rows as u32);
        let mut l = self.call(self.gpu.other("zero_rows"));
        l.arg(&a.0.0)
            .arg(&a_off)
            .arg(&ldy_u)
            .arg(&base_u)
            .arg(&rows_u)
            .arg(&n_u);
        self.launch(
            l,
            grid((rows * n).div_ceil(ELT_THREADS), ELT_THREADS as u32),
        );
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

    /// `gate = silu(gate) * up`; `quant` and `half` as for [`Batch::rmsnorm`].
    pub fn swiglu(
        &mut self,
        gate: At<'_>,
        up: At<'_>,
        n: usize,
        quant: Option<&GpuQ>,
        half: Option<&GpuHalf>,
    ) {
        assert!(gate.1 + n <= gate.0.len() && up.1 + n <= up.0.len());
        let (g_off, u_off, n_u) = (gate.1 as u64, up.1 as u64, n as u32);
        let (quant, mode) = Self::half_out(quant, half, n);
        let (q, _) = self.quant_out(quant, n, n);
        let mut l = self.call(self.gpu.other("swiglu"));
        l.arg(&gate.0.0)
            .arg(&g_off)
            .arg(&up.0.0)
            .arg(&u_off)
            .arg(&n_u);
        match half {
            Some(h) => l.arg(&h.0),
            None => l.arg(&q.q),
        };
        l.arg(&q.d).arg(&q.s).arg(&mode);
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
        let (pos, posd) = r.dev_pos(&self.gpu.zero_id);
        let (nh, hd, rd) = (heads as u32, r.head_dim as u32, r.rope_dim as u32);
        let (neox, per_tok) = (r.neox as u32, r.n_heads as u32);
        let [kd, vs, vd] = r.store_args(x.0, heads2);
        let (kv_dim, store) = ((r.n_heads2 * r.head_dim) as u32, r.store.is_some() as u32);
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
            .arg(&per_tok2)
            .arg(&posd.0)
            .arg(&kd.0)
            .arg(&vs.0)
            .arg(&vd.0)
            .arg(&kv_dim)
            .arg(&store);
        let copy = if r.store.is_some() {
            heads2 * r.head_dim
        } else {
            0
        };
        let n = (heads + heads2) * r.rope_dim / 2 + copy;
        self.launch(l, grid(n.div_ceil(ELT_THREADS), ELT_THREADS as u32));
    }

    /// Whether [`Batch::norm_rope`] takes heads of `head_dim`.
    pub fn has_norm_rope(head_dim: usize) -> bool {
        head_dim.is_multiple_of(32) && head_dim <= 1024
    }

    /// RMSNorm over each head (weights `wq` for the heads at `x`, `wk` for those at `x2`), then
    /// [`Batch::rope`], as one launch.
    pub fn norm_rope(
        &mut self,
        x: At<'_>,
        x2: At<'_>,
        wq: &GpuBuf,
        wk: &GpuBuf,
        eps: f32,
        r: &RopeArgs<'_>,
    ) {
        let (heads, heads2) = (r.n_tok * r.n_heads, r.n_tok * r.n_heads2);
        assert!(Self::has_norm_rope(r.head_dim) && r.rope_dim <= r.head_dim);
        assert!(wq.len() >= r.head_dim && wk.len() >= r.head_dim);
        assert!(x.1 + heads * r.head_dim <= x.0.len());
        assert!(x2.1 + heads2 * r.head_dim <= x2.0.len());
        let (x_off, x2_off) = (x.1 as u64, x2.1 as u64);
        let ff = r.freq_factors.unwrap_or(x.0);
        let has_ff = r.freq_factors.is_some() as u32;
        let (pos, posd) = r.dev_pos(&self.gpu.zero_id);
        let (nh, hd, rd) = (heads as u32, r.head_dim as u32, r.rope_dim as u32);
        let (neox, per_tok, per_tok2) = (r.neox as u32, r.n_heads as u32, r.n_heads2 as u32);
        let [kd, vs, vd] = r.store_args(x.0, heads2);
        let (kv_dim, nh2) = ((r.n_heads2 * r.head_dim) as u32, heads2 as u32);
        let store = r.store.is_some() as u32;
        let mut l = self.call(self.gpu.other("norm_rope"));
        l.arg(&x.0.0)
            .arg(&x_off)
            .arg(&wq.0)
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
            .arg(&wk.0)
            .arg(&per_tok2)
            .arg(&eps)
            .arg(&posd.0)
            .arg(&kd.0)
            .arg(&vs.0)
            .arg(&vd.0)
            .arg(&kv_dim)
            .arg(&nh2)
            .arg(&store);
        let copy = if r.store.is_some() { heads2 } else { 0 };
        self.launch(l, grid(heads + heads2 + copy, hd));
    }

    /// The attention kernels' `n_pos` argument, and the device word they add to it.
    fn attn_pos<'g>(p: &'g AttnParams<'_>, zero: &'g GpuIds) -> (u32, &'g GpuIds) {
        match p.pos_dev {
            Some((d, v)) => (p.n_pos - v, d),
            None => (p.n_pos, zero),
        }
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
        p: &AttnParams<'_>,
        quant: Option<&GpuQ>,
    ) {
        // Parts in multiples of ATTN_PART_STEP, those past the context left empty, so the
        // grid changes once in that many parts and not every ATTN_PART positions.
        let room = scores.len() / (p.n_head * (p.head_dim + 2)) as usize;
        let n_parts = (p.n_pos as usize).div_ceil(ATTN_PART);
        assert!(room >= n_parts);
        let n_parts = n_parts
            .next_multiple_of(ATTN_PART_STEP)
            .min(room)
            .min(ATTN_PART_MAX);
        let (n_pos, posd) = Self::attn_pos(p, &self.gpu.zero_id);
        let mut l = self.call(self.gpu.other("attention_part"));
        l.arg(&q.0)
            .arg(&k.0)
            .arg(&v.0)
            .arg(&scores.0)
            .arg(&p.n_head)
            .arg(&p.n_kv)
            .arg(&p.head_dim)
            .arg(&n_pos)
            .arg(&p.kv_dim)
            .arg(&p.scale)
            .arg(&posd.0);
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
        p: &AttnParams<'_>,
        quant: Option<&GpuQ>,
        half_out: Option<&GpuHalf>,
    ) {
        let per_launch = (scores.len() / (p.n_head * p.max_ctx) as usize).min(p.n_tok as usize);
        assert!(per_launch > 0, "attention scores scratch too small");
        assert!(p.head_dim <= ATTN_THREADS, "head_dim too large for CUDA");
        let q_len = (p.n_tok * p.n_head * p.head_dim) as usize;
        assert!(q.len() >= q_len && out.len() >= q_len);
        assert!(p.n_pos + p.n_tok - 1 <= p.max_ctx);
        assert!(k.len() >= ((p.n_pos + p.n_tok - 1) * p.kv_dim) as usize && v.len() >= k.len());
        // Only the tensor-core kernel writes f16 itself.
        let (quant, _) = Self::half_out(quant, half_out, q_len);
        let mut converted = half_out.is_none();
        let (xq, quant_u) = self.quant_out(quant, q_len, p.head_dim as usize);
        if p.n_tok == 1
            && p.n_pos > ATTN_PART_MIN
            && p.n_head / p.n_kv <= ATTN_PART_HEADS
            && p.head_dim <= ATTN_PART_THREADS
            && p.head_dim.is_multiple_of(4)
            && p.n_pos as usize <= ATTN_PART * ATTN_PART_MAX
        {
            self.attention_decode(q, k, v, out, scores, p, quant);
            if let Some(h) = half_out {
                self.to_f16(At(out, 0), q_len, h);
            }
            return;
        }
        // Several tokens run as tiled products where the kernels take the shape.
        let g_n = p.n_head / p.n_kv;
        if p.n_tok >= ATTN_TILED_MIN
            && 32 % g_n == 0
            && p.head_dim <= 128
            && p.head_dim.is_multiple_of(4)
        {
            assert!(
                p.pos_dev.is_none(),
                "tiled attention takes its positions from the host"
            );
            let gpu: &'a Gpu = self.gpu;
            let flash = gpu.attn_flash.iter().find(|(hd, _)| *hd == p.head_dim);
            if let Some((_, f)) = flash.filter(|_| p.half) {
                let mut l = self.call(f);
                l.arg(&q.0).arg(&k.0).arg(&v.0).arg(&out.0);
                match half_out {
                    Some(o) => l.arg(&o.0).arg(&1u32),
                    None => l.arg(&out.0).arg(&0u32),
                };
                l.arg(&p.n_head)
                    .arg(&p.n_kv)
                    .arg(&p.n_pos)
                    .arg(&p.kv_dim)
                    .arg(&p.scale)
                    .arg(&p.n_tok);
                let rows = (p.n_tok as usize).div_ceil((64 / g_n) as usize);
                self.launch(l, grid2(p.n_kv as usize, rows, 128));
                if let Some(quant) = quant {
                    self.quantize(At(out, 0), q_len, quant);
                }
                return;
            }
            for tok0 in (0..p.n_tok).step_by(per_launch) {
                let n = (per_launch as u32).min(p.n_tok - tok0);
                let seen = (p.n_pos + tok0 + n - 1) as usize;
                let (tiles, rows) = (
                    seen.div_ceil(32),
                    (n as usize).div_ceil((32 / g_n) as usize),
                );
                let gpu: &'a Gpu = self.gpu;
                // f16 tensor cores where there are any, unless the session is precise.
                let half = gpu
                    .attn_half
                    .as_ref()
                    .filter(|_| p.half && p.head_dim.is_multiple_of(32));
                let mut l = self.call(half.map_or(gpu.other("attn_scores"), |h| &h.0));
                l.arg(&q.0)
                    .arg(&k.0)
                    .arg(&scores.0)
                    .arg(&p.n_head)
                    .arg(&p.n_kv)
                    .arg(&p.head_dim)
                    .arg(&p.n_pos)
                    .arg(&p.kv_dim)
                    .arg(&p.max_ctx)
                    .arg(&p.scale)
                    .arg(&tok0)
                    .arg(&n);
                let cfg = LaunchConfig {
                    grid_dim: (p.n_kv, tiles as u32, rows as u32),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                self.launch(l, cfg);
                let mut l = self.call(self.gpu.other("attn_softmax"));
                l.arg(&scores.0)
                    .arg(&p.n_head)
                    .arg(&p.n_pos)
                    .arg(&p.max_ctx)
                    .arg(&tok0);
                self.launch(l, grid2(p.n_head as usize, n as usize, 256));
                let mut l = self.call(half.map_or(gpu.other("attn_pv"), |h| &h.1));
                l.arg(&scores.0)
                    .arg(&v.0)
                    .arg(&out.0)
                    .arg(&p.n_head)
                    .arg(&p.n_kv)
                    .arg(&p.head_dim)
                    .arg(&p.n_pos)
                    .arg(&p.kv_dim)
                    .arg(&p.max_ctx)
                    .arg(&tok0)
                    .arg(&n);
                if half.is_some() {
                    converted = true;
                    match half_out {
                        Some(o) => l.arg(&o.0).arg(&1u32),
                        None => l.arg(&out.0).arg(&0u32),
                    };
                }
                // The tensor-core kernel takes 64 query rows a block.
                let (rows, threads) = match half {
                    Some(_) => ((n as usize).div_ceil((64 / g_n) as usize), 256),
                    None => (rows, 2 * p.head_dim),
                };
                self.launch(l, grid2(p.n_kv as usize, rows, threads));
            }
            if let Some(quant) = quant {
                self.quantize(At(out, 0), q_len, quant);
            }
            if let (Some(h), false) = (half_out, converted) {
                self.to_f16(At(out, 0), q_len, h);
            }
            return;
        }
        let (n_pos, posd) = Self::attn_pos(p, &self.gpu.zero_id);
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
                .arg(&n_pos)
                .arg(&p.kv_dim)
                .arg(&p.max_ctx)
                .arg(&p.scale)
                .arg(&tok0)
                .arg(&xq.q)
                .arg(&xq.d)
                .arg(&xq.s)
                .arg(&quant_u)
                .arg(&posd.0);
            let n = per_launch.min((p.n_tok - tok0) as usize);
            self.launch(l, grid2(p.n_head as usize, n, ATTN_THREADS));
        }
        if let Some(h) = half_out {
            self.to_f16(At(out, 0), q_len, h);
        }
    }

    /// Whether [`Batch::attention_mla`] takes latent attention with these dims (`MLA_ROPE`,
    /// `MLA_LAT` and `MLA_H` in cuda.cu: DeepSeek V2's and V3's).
    pub fn has_attention_mla(rope: usize, latent: usize, n_head: usize) -> bool {
        (rope, latent) == (64, 512) && n_head.is_multiple_of(16)
    }

    /// DeepSeek's latent attention for `n_tok` query tokens, the first seeing `n_pos` cached
    /// positions and each later one one more: `q` holds a cache row's worth per head (the
    /// rotated dims, then the latent), `cache` is the keys and, past the rotated dims, the
    /// values. Each head's output goes past its rotated dims in `out`, laid out as `q`; the
    /// rest of `out` is left alone. One token's context is split into parts of
    /// [`MLA_PART_LEN`] positions, run side by side through `scores` (scratch of
    /// [`Batch::attention_scratch`] for a 576-dim head) and merged.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_mla(
        &mut self,
        q: &GpuBuf,
        cache: &GpuBuf,
        out: &GpuBuf,
        scores: &GpuBuf,
        n_head: usize,
        n_tok: usize,
        n_pos: usize,
        scale: f32,
    ) {
        let row = 576;
        assert!(Self::has_attention_mla(64, row - 64, n_head) && n_tok > 0 && n_pos > 0);
        assert!(q.len() >= n_tok * n_head * row && out.len() >= n_tok * n_head * row);
        assert!(cache.len() >= (n_pos + n_tok - 1) * row);
        // Even at short context: one block for every position took V2-Lite decode (128 tokens
        // from position 0) from 347 tok/s to 316.
        let split = n_tok == 1;
        // Parts in multiples of 8, those past the context left empty, so that the grid
        // changes once in 8 parts and not every part.
        let room = scores.len() / (n_head * MLA_PART);
        let n_parts = match split {
            true => n_pos.div_ceil(MLA_PART_LEN).next_multiple_of(8).min(room),
            false => 1,
        };
        assert!(
            !split || n_parts * MLA_PART_LEN >= n_pos,
            "attention scratch too small"
        );
        let (nh, np, part_len) = (n_head as u32, n_pos as u32, MLA_PART_LEN as u32);
        let split_u = u32::from(split);
        let mut l = self.call(self.gpu.other("attention_mla"));
        l.arg(&q.0)
            .arg(&cache.0)
            .arg(&out.0)
            .arg(&nh)
            .arg(&np)
            .arg(&scale)
            .arg(&split_u)
            .arg(&part_len)
            .arg(&scores.0);
        let cfg = LaunchConfig {
            grid_dim: ((n_head / 16) as u32, n_tok as u32, n_parts as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.launch(l, cfg);
        if split {
            let mut l = self.call(self.gpu.other("attention_mla_merge"));
            let n_parts = n_parts as u32;
            l.arg(&scores.0).arg(&out.0).arg(&n_parts);
            self.launch(l, grid(n_head, 256));
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

    /// The int8 matmul, and the int8 matvec (under 8 tokens; Q4_K's takes up to 4 tokens a pass,
    /// so 5 is a pass and a token and 3 a pass partly filled), against the f32 ones over the same
    /// random weights: they differ only by the rounding of activations to int8. The sizes
    /// leave a row tile, a token tile and a matvec step partly filled. Needs an NVIDIA GPU, so
    /// CI doesn't run it; `RLI_CUDA_NO_MMA=1` checks the dp4a matmuls on a GPU that would use
    /// the tensor-core ones.
    #[test]
    fn tiled_f32_matmul_matches_matmul() {
        for (rows, cols, n) in [
            (128, 2048, 512),
            (100, 256, 70),
            (60, 96, 9),
            (128, 2048, 2),
        ] {
            let mut seed = 7u32;
            let mut next = move || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / 8388608.0 - 1.0
            };
            let w: Vec<u8> = (0..rows * cols)
                .flat_map(|_| next().to_le_bytes())
                .collect();
            let x: Vec<f32> = (0..n * cols).map(|_| next()).collect();
            let gpu = Gpu::new(&w, &[(0, w.len())]).expect("CUDA device");
            let xb = gpu.upload(&x);
            let (want_b, got_b) = (gpu.alloc(n * rows), gpu.upload(&vec![0f32; n * rows]));
            let mut b = gpu.batch();
            b.matmul(GgmlType::F32, rows, cols, 0, At(&xb, 0), At(&want_b, 0), n);
            b.matmul_f32_add(rows, cols, 0, At(&xb, 0), At(&got_b, 0), n);
            b.finish();
            let (mut want, mut got) = (vec![0f32; n * rows], vec![0f32; n * rows]);
            gpu.read(&want_b, &mut want);
            gpu.read(&got_b, &mut got);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < 1e-3, "{rows}x{cols} by {n}: max difference {err}");
        }
    }

    #[test]
    fn matmul_q_matches_f32_matmul() {
        let types = [(GgmlType::Q4_K, &[0, 2][..]), (GgmlType::Q6_K, &[208][..])];
        for ((rows, cols, n), (ty, scales_at)) in [
            (200, 512, 137),
            (203, 1280, 1),
            (64, 1024, 5),
            (130, 768, 2),
            (130, 768, 3),
            (131, 512, 4),
            (130, 512, 13),
            (130, 768, 30),
            // Enough work for the lean kernels' 128-token blocks on up to 180 multiprocessors.
            (1100, 2560, 200),
        ]
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
            // Added to itself where the kernel can: twice the product.
            let added = n < MMQ_MIN_TOKENS || b.has_matmul_q_add();
            if added {
                b.matmul_q(ty, rows, cols, 0, &xq, At(&y_q, 0), n, true);
            }
            b.finish();
            let (mut want, mut got) = (vec![0f32; n * rows], vec![0f32; n * rows]);
            gpu.read(&y_f32, &mut want);
            gpu.read(&y_q, &mut got);
            if added {
                got.iter_mut().for_each(|v| *v /= 2.0);
            }

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

    /// The matmul over rows grouped by expert against one int8 matvec per row: the same
    /// arithmetic in another order. An expert with more rows than a block takes, experts with
    /// none, and a last row tile partly filled. Needs an NVIDIA GPU.
    #[test]
    fn matmul_q_id_matches_matvec_q_id() {
        let (rows, cols, ne, k) = (72, 512, 7, 3);
        // 90 tokens, and 9: few enough pairs an expert for the kernel that stages no weights.
        for (ty, scales_at, n_tok) in [
            (GgmlType::Q4_K, &[0, 2][..], 90),
            (GgmlType::Q6_K, &[208][..], 90),
            (GgmlType::Q4_K, &[0, 2][..], 9),
            (GgmlType::Q6_K, &[208][..], 9),
        ] {
            let mut seed = 7u32;
            let mut next = move || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed >> 8
            };
            let stride = rows * ty.row_bytes(cols);
            let mut w: Vec<u8> = (0..ne * stride).map(|_| next() as u8).collect();
            for block in w.chunks_exact_mut(ty.block_bytes()) {
                for &i in scales_at {
                    block[i + 1] = 0x24 | (block[i + 1] & 3); // an f16 in [2^-6, 2^-5)
                }
            }
            let x: Vec<f32> = (0..n_tok * cols)
                .map(|_| next() as f32 / 8388608.0 - 1.0)
                .collect();
            // Experts 0..5 only, and mostly expert 2.
            let n = n_tok * k;
            let ids: Vec<u32> = (0..n)
                .map(|_| if next() % 4 == 0 { next() % 5 } else { 2 })
                .collect();

            let gpu = Gpu::new(&w, &[(0, w.len())]).expect("CUDA device");
            let ids = GpuIds(gpu.stream.clone_htod(&ids).expect("upload"));
            let (perm, start) = (gpu.alloc_ids(n), gpu.alloc_ids(ne + 1));
            let (xb, xq) = (gpu.upload(&x), gpu.alloc_q(n_tok * cols));
            let (y_each, y_grouped) = (gpu.alloc(n * rows), gpu.alloc(n * rows));
            let mut b = gpu.batch();
            b.quantize(At(&xb, 0), n_tok * cols, &xq);
            b.matvec_q_id(
                ty,
                rows,
                cols,
                0,
                stride,
                ne,
                &ids,
                &xq,
                cols,
                k,
                At(&y_each, 0),
                rows,
                n,
                false,
                None,
            );
            b.moe_group(&ids, &perm, &start, n, ne);
            let groups = (&perm, &start);
            b.matmul_q_id(
                ty,
                rows,
                cols,
                0,
                stride,
                ne,
                groups,
                &xq,
                k,
                At(&y_grouped, 0),
                n,
            );
            b.finish();
            let (mut want, mut got) = (vec![0f32; n * rows], vec![0f32; n * rows]);
            gpu.read(&y_each, &mut want);
            gpu.read(&y_grouped, &mut got);
            let largest = |v: &[f32]| v.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff =
                |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let (size, err) = (largest(&want), diff(&got, &want));
            assert!(size > 0.0 && err < 1e-4 * size, "{ty:?}: {err} of {size}");

            // The f16 tensor-core kernel, where there is one, against f32 matvecs.
            let mut b = gpu.batch();
            if !b.has_gemm_id(ty, cols, 0, stride) {
                continue;
            }
            let xh = gpu.alloc_half(n_tok * cols);
            b.matvec_id(
                ty,
                rows,
                cols,
                0,
                stride,
                ne,
                &ids,
                At(&xb, 0),
                cols,
                k,
                At(&y_each, 0),
                rows,
                n,
            );
            b.to_f16(At(&xb, 0), n_tok * cols, &xh);
            let y = At(&y_grouped, 0);
            b.gemm_id(ty, rows, cols, 0, stride, ne, groups, &xh, k, y, n);
            b.finish();
            gpu.read(&y_each, &mut want);
            gpu.read(&y_grouped, &mut got);
            let (size, err) = (largest(&want), diff(&got, &want));
            assert!(
                size > 0.0 && err < 2e-3 * size,
                "f16 {ty:?}: {err} of {size}"
            );
        }
    }

    /// The latent attention kernel against the general one over the same random queries and
    /// cache (one KV head of 576 dims): they differ in the order of their sums. Needs an
    /// NVIDIA GPU.
    #[test]
    fn attention_mla_matches_attention() {
        let (nh, row, pos, n) = (16, 576, 300, 13);
        let max_ctx = pos + n;
        let mut seed = 5u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 8388608.0 - 1.0
        };
        let q: Vec<f32> = (0..n * nh * row).map(|_| 4.0 * next()).collect();
        let k: Vec<f32> = (0..max_ctx * row).map(|_| next()).collect();
        let w = vec![0u8; 64];
        let gpu = Gpu::new(&w, &[(0, w.len())]).expect("CUDA device");
        let (qb, kb) = (gpu.upload(&q), gpu.upload(&k));
        let (want_b, got_b) = (gpu.alloc(n * nh * row), gpu.alloc(n * nh * row));
        let scores = gpu.alloc(n * nh * max_ctx.max(Batch::attention_scratch(nh, row, max_ctx)));
        let p = AttnParams {
            n_head: nh as u32,
            n_kv: 1,
            head_dim: row as u32,
            n_tok: n as u32,
            n_pos: (pos + 1) as u32,
            kv_dim: row as u32,
            max_ctx: max_ctx as u32,
            scale: 0.07,
            half: false,
            pos_dev: None,
        };
        let mut b = gpu.batch();
        b.attention(&qb, &kb, &kb, &want_b, &scores, &p, None, None);
        b.attention_mla(&qb, &kb, &got_b, &scores, nh, n, pos + 1, 0.07);
        // The last token alone, its context split into parts.
        let last = (n - 1) * nh * row;
        let one = gpu.alloc(nh * row);
        let ql = gpu.upload(&q[last..]);
        b.attention_mla(&ql, &kb, &one, &scores, nh, 1, pos + n, 0.07);
        b.finish();
        let mut split = vec![0f32; nh * row];
        gpu.read(&one, &mut split);
        let (mut want, mut got) = (vec![0f32; n * nh * row], vec![0f32; n * nh * row]);
        gpu.read(&want_b, &mut want);
        gpu.read(&got_b, &mut got);
        let mut err = 0f32;
        for (w, g) in want.chunks_exact(row).zip(got.chunks_exact(row)) {
            for (a, b) in w[64..].iter().zip(&g[64..]) {
                err = err.max((a - b).abs());
            }
        }
        assert!(err < 1e-5, "max |diff| {err}");
        let mut err = 0f32;
        for (w, g) in want[last..].chunks_exact(row).zip(split.chunks_exact(row)) {
            for (a, b) in w[64..].iter().zip(&g[64..]) {
                err = err.max((a - b).abs());
            }
        }
        assert!(err < 1e-5, "one token split: max |diff| {err}");
    }

    /// The f16 matmul over stacked experts for the 32-weight block types, whose columns need
    /// only be a multiple of 64 (1408: DeepSeek-V2-Lite's experts' down), against f32
    /// matvecs. Needs an NVIDIA GPU with f16 tensor cores.
    #[test]
    fn gemm_id_takes_32_weight_blocks() {
        // And tiny_mixtral's gate and up: one 64-column step, rows of a block partly filled.
        let shapes = [(72, 1408), (96, 64)];
        let (ne, k, n_tok) = (7, 3, 90);
        for ((ty, _), (rows, cols)) in GEMM_ID_BLOCK32.iter().flat_map(|t| shapes.map(|s| (t, s))) {
            let ty = *ty;
            let mut seed = 11u32;
            let mut next = move || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed >> 8
            };
            let stride = rows * ty.row_bytes(cols);
            let mut w: Vec<u8> = (0..ne * stride).map(|_| next() as u8).collect();
            for block in w.chunks_exact_mut(ty.block_bytes()) {
                block[1] = 0x24 | (block[1] & 3); // an f16 in [2^-6, 2^-5)
            }
            let x: Vec<f32> = (0..n_tok * cols)
                .map(|_| next() as f32 / 8388608.0 - 1.0)
                .collect();
            let n = n_tok * k;
            let ids: Vec<u32> = (0..n)
                .map(|_| if next() % 4 == 0 { next() % 5 } else { 2 })
                .collect();
            let gpu = Gpu::new(&w, &[(0, w.len())]).expect("CUDA device");
            let mut b = gpu.batch();
            if !b.has_gemm_id(ty, cols, 0, stride) {
                return; // no f16 tensor cores
            }
            let ids = GpuIds(gpu.stream.clone_htod(&ids).expect("upload"));
            let (perm, start) = (gpu.alloc_ids(n), gpu.alloc_ids(ne + 1));
            let (xb, xh) = (gpu.upload(&x), gpu.alloc_half(n_tok * cols));
            let (y_each, y_grouped) = (gpu.alloc(n * rows), gpu.alloc(n * rows));
            let (each, grouped) = (At(&y_each, 0), At(&y_grouped, 0));
            let x0 = At(&xb, 0);
            b.matvec_id(
                ty, rows, cols, 0, stride, ne, &ids, x0, cols, k, each, rows, n,
            );
            b.moe_group(&ids, &perm, &start, n, ne);
            b.to_f16(x0, n_tok * cols, &xh);
            b.gemm_id(
                ty,
                rows,
                cols,
                0,
                stride,
                ne,
                (&perm, &start),
                &xh,
                k,
                grouped,
                n,
            );
            b.finish();
            let (mut want, mut got) = (vec![0f32; n * rows], vec![0f32; n * rows]);
            gpu.read(&y_each, &mut want);
            gpu.read(&y_grouped, &mut got);
            let size = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let err = got
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                size > 0.0 && err < 2e-3 * size,
                "{ty:?} {rows}x{cols}: {err} of {size}"
            );
        }
    }

    /// The f16 GEMM against the f32 matmul over the same random weights of every type: they
    /// differ by the rounding of weights and activations to f16. The sizes leave a row tile
    /// and a token tile partly filled. Needs an NVIDIA GPU with f16 tensor cores.
    #[test]
    fn gemm_matches_f32_matmul() {
        for (ty, _) in MATVEC_KERNELS {
            // The last two have rows and tokens enough for the larger tile: columns in parts,
            // and (on an A100) rows in two launches.
            for (rows, cols, n) in [
                (200, 512, 137),
                (64, 1280, GEMM_MIN_TOKENS),
                (17, 256, 300),
                (2048, 512, 512),
                (14400, 256, 500),
                // Chunks short enough for the kernel that decodes the weights itself.
                (200, 512, 30),
                (300, 768, 9),
                (300, 768, 17),
                (130, 768, 100),
                (130, 768, 230),
                (2048, 1024, 192),
            ] {
                let mut seed = 3u32;
                let mut next = move || {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    seed >> 8
                };
                let mut w: Vec<u8> = (0..rows * ty.row_bytes(cols))
                    .map(|_| next() as u8)
                    .collect();
                // Floats in the weights (f32, f16 and the blocks' scales) must be finite and
                // of one size: every other byte pair becomes an f16 in [2^-6, 2^-5), and f32
                // weights are written whole.
                match ty {
                    GgmlType::F32 => {
                        for c in w.as_chunks_mut::<4>().0 {
                            let v = next() as f32 / 8388608.0 - 1.0;
                            c.copy_from_slice(&v.to_le_bytes());
                        }
                    }
                    GgmlType::F16 => w
                        .as_chunks_mut::<2>()
                        .0
                        .iter_mut()
                        .for_each(|c| c[1] = 0x24 | (c[1] & 0x83)),
                    // bf16 in [2^-7, 2^-5).
                    GgmlType::BF16 => w
                        .as_chunks_mut::<2>()
                        .0
                        .iter_mut()
                        .for_each(|c| c[1] = 0x3c | (c[1] & 0x80)),
                    GgmlType::Q4_K | GgmlType::Q5_K => {
                        for b in w.chunks_exact_mut(ty.block_bytes()) {
                            (b[1], b[3]) = (0x24 | (b[1] & 3), 0x24 | (b[3] & 3));
                        }
                    }
                    GgmlType::Q6_K => w
                        .as_chunks_mut::<210>()
                        .0
                        .iter_mut()
                        .for_each(|b| b[209] = 0x24 | (b[209] & 3)),
                    GgmlType::NVFP4 => {
                        for b in w.chunks_exact_mut(ty.block_bytes()) {
                            b[..4].iter_mut().for_each(|d| *d = 0x20 | (*d & 15));
                        }
                    }
                    _ => {
                        for b in w.chunks_exact_mut(ty.block_bytes()) {
                            b[1] = 0x24 | (b[1] & 3);
                        }
                    }
                }
                let x: Vec<f32> = (0..n * cols)
                    .map(|_| next() as f32 / 8388608.0 - 1.0)
                    .collect();
                let mut gpu = Gpu::new(&w, &[(0, w.len())]).expect("CUDA device");
                gpu.set_scale(0, 0.37);
                let xb = gpu.upload(&x);
                let (wh, xh) = (gpu.alloc_half(rows * cols), gpu.alloc_half(n * cols));
                let (y_f32, y_h) = (gpu.alloc(n * rows), gpu.alloc(n * rows));
                let mut b = gpu.batch();
                let dense = b.has_gemm_dense(ty, cols, 0, n);
                if !dense && !b.has_gemm(ty, cols, n) {
                    continue; // no kernel for the type, no f16 tensor cores, or RLI_CUDA_NO_MMA
                }
                b.matmul(ty, rows, cols, 0, At(&xb, 0), At(&y_f32, 0), n);
                b.to_f16(At(&xb, 0), n * cols, &xh);
                // y_sum = y_f32 + W x, where there is a kernel for it.
                let y_sum = gpu.alloc(n * rows);
                let added = dense || b.has_gemm_add(ty, rows, cols, n);
                if added {
                    b.add(At(&y_sum, 0), At(&y_f32, 0), n * rows);
                }
                if dense {
                    b.gemm_dense(ty, rows, cols, 0, &xh, At(&y_h, 0), n, false);
                    b.gemm_dense(ty, rows, cols, 0, &xh, At(&y_sum, 0), n, true);
                } else {
                    b.dequant(ty, rows, cols, 0, &wh);
                    b.gemm(rows, cols, &wh, &xh, At(&y_h, 0), n);
                    if added {
                        b.gemm_add(rows, cols, &wh, &xh, At(&y_sum, 0), n);
                    }
                }
                b.finish();
                let (mut want, mut got) = (vec![0f32; n * rows], vec![0f32; n * rows]);
                gpu.read(&y_f32, &mut want);
                gpu.read(&y_h, &mut got);
                let rms = |v: &mut dyn Iterator<Item = f32>| {
                    (v.map(|a| (a * a) as f64).sum::<f64>() / (n * rows) as f64).sqrt()
                };
                let err = rms(&mut got.iter().zip(&want).map(|(a, b)| a - b));
                let size = rms(&mut want.iter().copied());
                assert!(
                    err < 1e-3 * size,
                    "{ty:?} {rows}x{cols}, {n} tokens: rms error {err} against rms {size}"
                );
                if added {
                    gpu.read(&y_sum, &mut got);
                    let err = rms(&mut got.iter().zip(&want).map(|(a, b)| a - 2.0 * b));
                    assert!(
                        err < 1e-3 * size,
                        "{ty:?} {rows}x{cols}, {n} tokens, added: rms error {err} against {size}"
                    );
                }
            }
        }
    }

    /// NVFP4's kernels against the CPU's f32 reference times the tensor scale they take from
    /// `Gpu::set_scale`: a matvec, the matmul (a token tile and a part-filled one) and the
    /// dequantized matrix the f16 GEMM reads. Needs an NVIDIA GPU.
    #[test]
    fn nvfp4_kernels_match_the_cpu_with_their_tensor_scale() {
        let (ty, rows, cols, scale) = (GgmlType::NVFP4, 37, 576, 0.37f32);
        let mut seed = 11u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        let mut w: Vec<u8> = (0..rows * ty.row_bytes(cols))
            .map(|_| next() as u8)
            .collect();
        for b in w.chunks_exact_mut(ty.block_bytes()) {
            b[..4].iter_mut().for_each(|d| *d = 0x20 | (*d & 15));
        }
        let m = crate::QMatrix::new(ty, rows, cols, &w);
        let mut gpu = Gpu::new(&w, &[(0, w.len())]).expect("CUDA device");
        gpu.set_scale(0, scale);
        for n in [1, 5, 40] {
            let x: Vec<f32> = (0..n * cols)
                .map(|_| next() as f32 / 8388608.0 - 1.0)
                .collect();
            let mut want = vec![0f32; n * rows];
            for (xt, yt) in x.chunks_exact(cols).zip(want.chunks_exact_mut(rows)) {
                m.matvec_with(xt, yt, true);
            }
            want.iter_mut().for_each(|v| *v *= scale);
            let (xb, y) = (gpu.upload(&x), gpu.alloc(n * rows));
            let mut b = gpu.batch();
            b.matmul(ty, rows, cols, 0, At(&xb, 0), At(&y, 0), n);
            b.finish();
            let mut got = vec![0f32; n * rows];
            gpu.read(&y, &mut got);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
                    "{n} tokens [{i}]: {g} vs {w}"
                );
            }
        }
        let wh = gpu.alloc_half(rows * cols);
        let mut b = gpu.batch();
        b.dequant(ty, rows, cols, 0, &wh);
        b.finish();
        let got: Vec<f32> = gpu
            .stream
            .clone_dtoh(&wh.0)
            .expect("download")
            .into_iter()
            .map(crate::half::f16_to_f32)
            .collect();
        let mut want = vec![0f32; rows * cols];
        crate::quant::dequantize_row(ty, &w, &mut want);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let w = w * scale;
            assert!(
                (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
                "dequant [{i}]: {g} vs {w}"
            );
        }
    }

    /// The NVFP4 int8 matvecs (one token, a few as a row each, added into y, and the pair
    /// kernel with a tensor scale per matrix) and, where the GPU has them, the lean int8
    /// matmuls (every block size, stored then added) against the CPU's f32 product. Each 32
    /// activations are whole numbers that include 127, which int8 holds exactly, so only float
    /// rounding separates them. Steps of 8 units of 256 columns and row tiles are left partly
    /// filled. Needs an NVIDIA GPU.
    #[test]
    fn nvfp4_int8_kernels_match_the_cpu() {
        let ty = GgmlType::NVFP4;
        for (rows, cols) in [(37, 768), (6, 2304)] {
            let mut seed = 5u32;
            let mut next = move || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed >> 8
            };
            let len = rows * ty.row_bytes(cols);
            let mut w: Vec<u8> = (0..2 * len).map(|_| next() as u8).collect();
            for b in w.chunks_exact_mut(ty.block_bytes()) {
                b[..4].iter_mut().for_each(|d| *d = 0x20 | (*d & 15));
            }
            let scales = [0.37f32, 1.9];
            let mut gpu = Gpu::new(&w, &[(0, w.len())]).expect("CUDA device");
            gpu.set_scale(0, scales[0]);
            gpu.set_scale(len, scales[1]);
            let n = 200;
            let x: Vec<f32> = (0..n * cols)
                .map(|i| match i % 32 {
                    7 => 127.0,
                    _ => (next() % 255) as f32 - 127.0,
                })
                .collect();
            let want: Vec<Vec<f32>> = (0..2)
                .map(|m| {
                    let q = crate::QMatrix::new(ty, rows, cols, &w[m * len..(m + 1) * len]);
                    let mut y = vec![0f32; n * rows];
                    for (xt, yt) in x.chunks_exact(cols).zip(y.chunks_exact_mut(rows)) {
                        q.matvec_with(xt, yt, true);
                    }
                    y.iter().map(|v| v * scales[m]).collect()
                })
                .collect();
            let (xb, xq) = (gpu.upload(&x), gpu.alloc_q(n * cols));
            let (one, many, pair) = (gpu.alloc(rows), gpu.alloc(3 * rows), gpu.alloc(2 * rows));
            let mut b = gpu.batch();
            b.quantize(At(&xb, 0), n * cols, &xq);
            b.matmul_q(ty, rows, cols, 0, &xq, At(&one, 0), 1, false);
            // Twice the product: stored, then added.
            b.matmul_q(ty, rows, cols, 0, &xq, At(&many, 0), 3, false);
            b.matmul_q(ty, rows, cols, 0, &xq, At(&many, 0), 3, true);
            b.matvec_q_pair(ty, rows, cols, [0, len], &xq, At(&pair, 0));
            b.finish();
            let read = |buf: &GpuBuf, len| {
                let mut v = vec![0f32; len];
                gpu.read(buf, &mut v);
                v
            };
            let check = |what: &str, got: &[f32], want: &[f32], k: f32| {
                for (i, (g, w)) in got.iter().zip(want).enumerate() {
                    let w = k * w;
                    assert!(
                        (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                        "{rows}x{cols} {what} [{i}]: {g} vs {w}"
                    );
                }
            };
            check("one token", &read(&one, rows), &want[0][..rows], 1.0);
            check("3 tokens, added", &read(&many, 3 * rows), &want[0], 2.0);
            let p = read(&pair, 2 * rows);
            check("pair, first", &p[..rows], &want[0][..rows], 1.0);
            check("pair, second", &p[rows..], &want[1][..rows], 1.0);
            if !gpu.mmk.iter().any(|(t, ..)| *t == ty) {
                eprintln!("no lean NVFP4 matmul: skipped");
                continue;
            }
            for nt in [8, 40, 70, 200] {
                let y = gpu.alloc(nt * rows);
                let mut b = gpu.batch();
                b.matmul_q(ty, rows, cols, 0, &xq, At(&y, 0), nt, false);
                b.matmul_q(ty, rows, cols, 0, &xq, At(&y, 0), nt, true);
                b.finish();
                check(&format!("{nt} tokens"), &read(&y, nt * rows), &want[0], 2.0);
            }
        }
    }

    /// The FP4 tensor-core matmul (stored, then added) against the CPU's f32 product of the
    /// weights with the activations as `quantize_fp4` left them, which the CPU decodes with the
    /// weights' own decoder; and those activations against the originals, each within twice
    /// its sub-block's scale (a value past 6 scales saturates, as the scale search allows).
    /// Row and token tiles are left partly filled. Needs an RTX 5090 (compute capability 12.0).
    #[test]
    fn fp4_matmul_matches_the_cpu_over_its_quantized_activations() {
        let ty = GgmlType::NVFP4;
        for (rows, cols, n) in [(200, 768, 70), (64, 256, 8)] {
            let mut seed = 3u32;
            let mut next = move || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed >> 8
            };
            let mut w: Vec<u8> = (0..rows * ty.row_bytes(cols))
                .map(|_| next() as u8)
                .collect();
            for b in w.chunks_exact_mut(ty.block_bytes()) {
                b[..4].iter_mut().for_each(|d| *d = 0x20 | (*d & 15));
            }
            let scale = 0.37f32;
            let mut gpu = Gpu::new(&w, &[(0, w.len())]).expect("CUDA device");
            if gpu.fp4.is_none() {
                eprintln!("no FP4 tensor cores: skipped");
                return;
            }
            gpu.set_scale(0, scale);
            // Values over a range of sizes, some sub-blocks all zero.
            let x: Vec<f32> = (0..n * cols)
                .map(|i| match (i / 16) % 11 {
                    3 => 0.0,
                    k => (next() as f32 / 8388608.0 - 1.0) * (1 << k) as f32,
                })
                .collect();
            let (xb, xf, y) = (gpu.upload(&x), gpu.alloc_q(n * cols), gpu.alloc(n * rows));
            let mut b = gpu.batch();
            b.set_fp4_acts(true);
            b.quantize_fp4(At(&xb, 0), n, cols, &xf);
            b.matmul_fp4(rows, cols, 0, &xf, At(&y, 0), n, false);
            b.finish();
            let mut got = vec![0f32; n * rows];
            gpu.read(&y, &mut got);
            let q: Vec<u8> = gpu.stream.clone_dtoh(&xf.q).expect("download")[..]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let g = gpu.stream.clone_dtoh(&xf.d).expect("download");
            let rb = ty.row_bytes(cols);
            let mut xd = vec![0f32; n * cols];
            for t in 0..n {
                let (row, out) = (&q[t * rb..(t + 1) * rb], &mut xd[t * cols..(t + 1) * cols]);
                crate::quant::dequantize_row(ty, row, out);
                out.iter_mut().for_each(|v| *v *= g[t]);
                for (i, (d, v)) in out.iter().zip(&x[t * cols..]).enumerate() {
                    let sb = row[(i / 64) * 36 + (i % 64) / 16];
                    // ue4m3 is the scale halved, for FP4_VALUES.
                    let step = 2.0 * crate::quant::ue4m3(sb) * g[t];
                    assert!(
                        (d - v).abs() <= 2.0 * step + 1e-6,
                        "token {t} value {i}: {d} for {v} (scale {step})"
                    );
                }
            }
            let mut wd = vec![0f32; rows * cols];
            crate::quant::dequantize_row(ty, &w, &mut wd);
            let want: Vec<f32> = (0..n * rows)
                .map(|i| {
                    let (t, r) = (i / rows, i % rows);
                    let dot: f64 = (0..cols)
                        .map(|k| wd[r * cols + k] as f64 * xd[t * cols + k] as f64)
                        .sum();
                    dot as f32 * scale
                })
                .collect();
            let check = |got: &[f32], k: f32| {
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let w = k * w;
                    assert!(
                        (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                        "{rows}x{cols} by {n} [{i}]: {g} vs {w}"
                    );
                }
            };
            check(&got, 1.0);
            let mut b = gpu.batch();
            b.set_fp4_acts(true);
            b.matmul_fp4(rows, cols, 0, &xf, At(&y, 0), n, true);
            b.finish();
            gpu.read(&y, &mut got);
            check(&got, 2.0);
        }
    }

    /// Large downloads go through a page-locked buffer that is kept: a shorter one after a
    /// longer one must copy only its own length. Needs an NVIDIA GPU, so CI doesn't run it.
    #[test]
    fn read_returns_the_buffer_at_any_length() {
        let gpu = Gpu::new(&[0; 256], &[(0, 256)]).expect("CUDA device");
        let data: Vec<f32> = (0..3 * PINNED_MIN).map(|i| i as f32).collect();
        let buf = gpu.upload(&data);
        for len in [
            3 * PINNED_MIN,
            PINNED_MIN,
            PINNED_MIN - 1,
            2 * PINNED_MIN + 5,
            7,
        ] {
            let mut out = vec![-1.0; len];
            gpu.read(&buf, &mut out);
            assert_eq!(out, data[..len], "len {len}");
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
                half: false,
                pos_dev: None,
            };
            let mut b = gpu.batch();
            b.attention(&q, &k, &v, &one, &scores, &p, None, None);
            p.n_tok = 2;
            b.attention(&q, &k, &v, &two, &scores, &p, None, None);
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

    /// The f16 tensor-core attention kernels against the f32 ones: several tiles of query
    /// rows (32 for the scores, 64 for the sums) and of positions, the last ones partly
    /// filled, and 1, 4 and 8 query heads per KV head. Needs an NVIDIA GPU.
    #[test]
    fn half_attention_matches_f32_attention() {
        for (n_head, n_kv, head_dim, n_pos, n_tok) in [
            (8, 2, 64, 40, 70),
            (16, 2, 128, 100, 37),
            (4, 4, 96, 10, 50),
            (32, 8, 128, 1, 300),
            (32, 4, 128, 517, 64),
            (32, 8, 64, 1, 9),
        ] {
            let mut seed = 11u32;
            let mut rand = |n: usize| -> Vec<f32> {
                (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        (seed >> 8) as f32 / 8388608.0 - 1.0
                    })
                    .collect()
            };
            let gpu = Gpu::new(&[0; 256], &[(0, 256)]).expect("CUDA device");
            if gpu.attn_half.is_none() {
                return;
            }
            let (max_ctx, kv_dim) = (n_pos + n_tok, n_kv * head_dim);
            let q_len = n_tok * n_head * head_dim;
            let q = gpu.upload(&rand(q_len));
            let (k, v) = (
                gpu.upload(&rand(max_ctx * kv_dim)),
                gpu.upload(&rand(max_ctx * kv_dim)),
            );
            let (full, half) = (gpu.alloc(q_len), gpu.alloc(q_len));
            let scores = gpu.alloc(n_tok * Batch::attention_scratch(n_head, head_dim, max_ctx));
            let mut p = AttnParams {
                n_head: n_head as u32,
                n_kv: n_kv as u32,
                head_dim: head_dim as u32,
                n_tok: n_tok as u32,
                n_pos: n_pos as u32,
                kv_dim: kv_dim as u32,
                max_ctx: max_ctx as u32,
                scale: 0.5,
                half: false,
                pos_dev: None,
            };
            let mut b = gpu.batch();
            b.attention(&q, &k, &v, &full, &scores, &p, None, None);
            p.half = true;
            b.attention(&q, &k, &v, &half, &scores, &p, None, None);
            b.finish();
            let (mut got, mut want) = (vec![0f32; q_len], vec![0f32; q_len]);
            gpu.read(&half, &mut got);
            gpu.read(&full, &mut want);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(want.iter().any(|w| w.abs() > 0.01), "degenerate reference");
            assert!(
                err < 5e-3,
                "{n_head}/{n_kv} heads of {head_dim}: max difference {err}"
            );
        }
    }
}
