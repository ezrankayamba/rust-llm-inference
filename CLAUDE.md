# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A from-scratch LLM inference engine in Rust for resource-constrained deployment: CPU-only hosts, Apple
unified memory, and low-VRAM GPUs. No ML framework dependency (no candle, burn, or ggml bindings):
tensors, kernels, the GGUF loader, and model graphs are all implemented here.

## Agreed scope

| Area | Decision |
|---|---|
| Foundation | From scratch: own tensor type, own CPU SIMD / Metal / CUDA kernels |
| Backends | CPU SIMD (baseline, always built) → Apple Metal → CUDA |
| Weight format | GGUF only, memory-mapped |
| Quantization | F16/BF16 (reference path), Q4_0/Q8_0, then K-quants (Q4_K, Q5_K, Q6_K); Q4_1/Q5_0/Q5_1 also load (Q4_K_M files contain Q5_0) |
| Models | Llama family (Llama 2/3, Mistral, Qwen dense), then MoE (Mixtral, Qwen-MoE) |
| Interfaces | CLI and an OpenAI-compatible HTTP server (`/v1/chat/completions`, SSE streaming) |
| Layout | Cargo workspace; GPU backends behind cargo features |
| Toolchain | Latest stable Rust, pinned in `rust-toolchain.toml`; SIMD via `std::arch` intrinsics |

## Architecture

Workspace crates (dependencies flow downward only):

- **rli-core** (`crates/core`): dtypes and GGUF block layouts (`quant.rs`, the f32 reference
  dequant/dot), `QMatrix` matvec (`matrix.rs`, rayon-parallel; every forward pass with CPU layers runs inside rli-models' own pool, `pool()` in `lib.rs`, so the calling thread is a worker rather than waiting on one per matvec), fast integer kernels over
  Q8-quantized activations (`qdot.rs`, NEON `dotprod` or AVX2 with runtime detection and a portable
  fallback; the NEON one-row Q4_K/Q5_K kernel (`k_rows`, decode) and the Q6_K arm of `dot_rows` keep a super-block in i32 with one float
  multiply-add at its end, as the AVX2 one does, since a float add per 32 weights was a chain every block
  waited on; on ARM, batches of 2+ tokens use `i8mm` `smmla` via inline asm when present, over activations
  laid out in token pairs once per matmul (`BlockActs`); on x86, batches of 8+ tokens (`BLOCK_MIN_TOKENS`)
  decode 8 weight rows at a time into an interleaved panel in L1, so one broadcast activation load serves
  all 8 rows (weights stay in the mapping; no repacked copy), or 16 rows with `vpdpbusd` where the CPU
  has AVX-512 VNNI, which must match the AVX2 kernel bit for bit. Either way batched int8 results differ from
  per-token ones by float rounding. The AVX2 one-row kernel, most of x86 decode time, keeps a K-quant
  super-block in i32 and spreads the 32-weight formats over four sums, so no sum waits on the last; on Intel CPUs
  only, its caller prefetches the bytes 1152 ahead of each row, which was worth 1.7x in decode there and
  cost up to 18% on AMD. `--repack` (`Model::repack`, off by default) copies the CPU layers' Q4_K
  weights into groups of 8 interleaved rows (`pack_rows`) that `matvec_packed` reads for one-token
  matvecs: +18% decode on Zen 4 for about the Q4_K weights' size in anonymous memory, nothing on
  Intel; prompts and `precise` still read the mapping), and ops (RMSNorm, RoPE, softmax, SwiGLU).
  GPU primitives live here too: `metal.rs` + `metal.metal` (feature `metal`) and `cuda.rs` +
  `cuda.cu` (feature `cuda`); both compile their kernel source at runtime.
- **rli-gguf** (`crates/gguf`): mmap GGUF parser; tensor offsets are absolute file offsets, and
  weights stay zero-copy. `writer.rs` is a minimal writer for tests.
- **rli-models** (`crates/models`): `config.rs` (`Arch`: Llama, Qwen2, Qwen3, Qwen2Moe, Qwen3Moe),
  `model.rs` (decoder forward, KV `Session`, MoE routing, the `Accelerator` trait), `tokenizer.rs`
  (SPM and GPT-2 byte-level BPE with per-model pre-tokenizers), `chat.rs` (format chosen by
  fingerprinting the GGUF Jinja template, default system text read from it), `sampler.rs`, and `engine.rs` (the generation API the
  binaries share, including KV prefix reuse and `stop.rs`'s stop strings). `metal.rs` / `cuda.rs` implement `Accelerator`, and
  `backend::attach` picks the one compiled in.
- **rli** (`crates/cli`) and **rli-server** (`crates/server`): thin binaries. The server is
  std-only HTTP (one thread per connection, `Connection: close`) with one shared session behind a
  mutex held only during generation. A client disconnect or a 30 s write stall cancels generation;
  `Engine::generate` polls between prompt slices (`Model::prompt_slice`: 256 tokens, 512 with
  every layer on a GPU) and before each token. Responses carry
  OpenAI's `usage` (streams need `stream_options.include_usage`) plus a non-standard `timings`.

Design points that span files:

- **Memory is the primary budget.** Weights stay in the mmap. Metal wraps the offloaded
  layers' weights as no-copy `MTLBuffer`s (unified memory), each at most `maxBufferLength`
  (14.3 GB on this 24 GB Mac), packed from per-layer file ranges so a tensor never straddles two;
  kernels bind the buffer holding their tensor. CUDA copies only the offloaded layers' weights into a VRAM
  arena (partial offload, `--gpu-layers N`); the remaining layers run on the CPU. The Metal KV
  cache is f16 (f32 when `Model.precise` is set at attach time), starts at 256 positions and
  doubles as needed up to the session's context; K/V are computed in f32 scratch, then stored.
  The CPU layers' KV cache is f16 too (`KvCache`; f32 when `precise` is set as the session is
  created), grown per token.
- **GPU offload is layer-level.** All offloaded layers of a token are queued as one batch with
  one sync (MoE included, see below). Per-matvec offload was measured slower than the CPU. `Model.accel` is declared first so it drops before the mapping.
- **Prompts are batched on Metal.** `Model::forward_batch` hands the whole prompt to
  `Accelerator::forward_batch` (the default implementation loops over `forward`). Metal runs it
  in chunks of up to 512 tokens. The matmul is tiled on `simdgroup_matrix`: 64 rows by 8, 16, 32
  or 64 tokens (a kernel per size, since a tile costs its weight decode however few tokens it
  holds; whole 64-token tiles, then the smallest that fits the rest), 64 columns a step, each
  thread decoding 32 weights into registers before the barrier. Tiles are half precision with
  f32 sums for quantized weights (logits within ~0.01 of f32 tiles), f32 for F32/F16 weights
  and under `precise`. A rest of up to 24 tokens (`MATMUL_FEW_TOKENS`; 16 for half Q4_K and
  Q6_K, whose 32-token tile is ahead on Mistral-7B from 17) takes `matmul_few_*` instead,
  llama.cpp's layout for few tokens: 16 rows by 8, 16 or 24 tokens, nothing staged in threadgroup
  memory (each lane puts the 16 weights it decodes into the matrix through its own two elements),
  the columns shared out among 8 simdgroups, f32. A tile's barrier per step is what few tokens
  can't pay for: 16-token prompts ran 1.12-1.2x faster, 4- and 8-token ones 1.35x; 32 tokens that
  way were 20% slower than the 32-token tile. A matmul's dispatches (whole tiles, then the rest)
  have no barrier between them: they write different rows, and the rest runs on weights the
  tiles have just read (72- and 88-token prompts 6-9% faster for it). A rest of 32 to 48 tokens
  is a 32-token tile and the few-token kernel, not a padded 64-token tile (40 tokens 1.3x
  faster; 56 as 32 and 24 was 4% slower on Mistral-7B). The half 32- and 64-token tiles of Q4_K
  and Q6_K are llama.cpp's kernel in its own terms (`MATMUL_K`; its tile is 32 tokens, ours
  also 64): 32-column steps, 8 KB of threadgroup memory where `MATMUL` stages 12 and 16,
  decoders that mask quants in place. **A tile's
  speed depends on the threadgroup memory it declares** (the same kernel: 811 us a dispatch
  with 8 KB, 844 with 14; ours with 12 KB more, +16%), so count it. A tile stages rows and tokens
  past the end as a repeat of row or token 0, not as zeros under a test (their results are
  never written): the test per staged value was 3.5-5% of the 32-token tile. Causal `attention_batch` also runs on the matrix units with an online
  softmax: over an f16 cache, `attention_batch_staged_<head size>` (16, 32, 64, 96, 128) stages
  32 keys of K and V once for up to four query heads of a KV head; other head sizes and f32
  caches take the general kernel. Loop bounds in these kernels are compile-time constants:
  with run-time bounds the compiler keeps the matrices in memory (Llama-3.2-1B pp512 was 4.5%
  slower for it). Decode uses matvec, and attention split over
  chunks of 256 cached positions that run in parallel and are merged (`attention_part_*` +
  `attention_merge`), so decode speed holds up at long context.
  head_dim not a multiple of 8 or above 256 falls back to one token at a time.
  The output and down projections add into the residual stream (`matmul_add`: a whole tile adds
  y after its last step; seeding the sums with y under a run-time test ran 3x slower).
  Against llama.cpp's Metal backend on the M4 (`docs/benchmarks.md`, 2026-11-17): pp512 level
  on Mistral-7B and Llama-3.2-1B, (2026-11-18) pp512 3-4% ahead, 40 to 200 tokens 1-42% ahead, 32 tokens level, 4 to 24
  between 1% behind (4 tokens on Mistral-7B) and 12% ahead; decode
  (2026-11-16) ahead by 2-11% on three of four models and level on Llama-3.2-1B. llama.cpp's
  matmul kernel, ported into this harness, ran exactly as fast as ours: don't look there.
- **Metal decode is weight streaming plus 5-9 us for every dispatch that must wait** for the
  one before it. The encoder is concurrent; `Batch` puts a barrier before each dispatch unless
  `Batch.free` says it reads nothing the ones just before it write (Q, K and V; RoPE and the
  cache stores). So a token's kernels are fused, not just fast: `matvec_add` adds into the
  residual stream, `matvec_bias` adds its bias, `matvec_swiglu` is gate, up and the activation
  in one (two matrices of one type), `rope_store` rotates K into the cache, and decode
  attention over a single 256-position chunk writes its result itself. Quantized matvecs take
  two rows a simdgroup (`MATVEC2`; four from 16384 rows, `MATVEC4`), read x once for them and
  build four weights per 32-bit word (`pair_*`, `two_*` in metal.metal); Q5_K, F16 and F32
  still take a row and a chunk per lane. Kernels avoid arrays and
  vectors indexed by a loop: the compiler keeps them in memory. `Batch::start` submits a
  token's first layer while the rest is encoded.
- **`RLI_METAL_PROF=1` times every Metal dispatch** alone in its own command buffer and prints
  the totals per kernel when the device is dropped. Alone is not how they run: a dispatch after
  an idle gap is 10-40% slower than in sequence, by an amount that differs per kernel
  (llama.cpp's tile: 1060 us alone, 772 repeated; ours 961 and 871). To compare two kernels,
  set `RLI_METAL_PROF_REPEAT=16` as well: each dispatch then runs that many times in its
  command buffer and its time is divided. A prompt is 90-97% the half-precision matmul
  tiles (`matmul_h_*_t64`), so that kernel is the whole of the prompt gap to llama.cpp. A
  128-token tile on 256 threads was 5% slower than the 64-token one, which is 11% ahead of 32.
- **Benchmark the two engines alternately with rests** (30-45 s): the second of two back-to-back
  runs is slower by several percent from heat alone, and anything else using the GPU (an
  animated wallpaper) shows up as noise.
- **CPU prompts are batched too**, in chunks of 128 tokens (`CPU_CHUNK`): `QMatrix::matmul_with`
  runs each block of weight rows against every token of the chunk while it is in cache. One
  row is the decode case, so CPU decode and prompts share `Model::cpu_layers`. CPU MoE layers
  route each row, then run each used expert once over the rows that picked it.
- **The last layer of a prompt computes one row.** Nothing reads the other rows after it, so
  they get their keys and values only, and a batch that returns no logits stops there
  (`cpu_layers`' `tail`, `m` and `lo` in the Metal and CUDA layer loops; on CUDA those rows keep
  their queries and attention, which its kernels take with the keys). llama.cpp does the same
  for the FFN; it was 5% of a 16-layer prompt.
- **MoE on Metal.** Decode routes on the GPU, with no sync: `moe_topk` (one simdgroup) writes
  the k expert ids and weights, then two dispatches (expert id read from the buffer, experts
  at a fixed stride in the stacked tensor): `matvec_swiglu_id` is the k experts' gate, up and
  activation, and `matvec_id_add` their down matvecs added into the residual stream by weight,
  a simdgroup's lanes dealt the column units of all k experts (one expert's 768 columns leave
  20 of 32 lanes idle). Types without pair kernels take `matvec_id` per matrix, `swiglu` and
  `moe_combine`. A per-layer sync cost ~0.35 ms; removing it took Qwen3-30B-A3B decode from
  19.6 to 29.1 tok/s, and the fused kernels to 43.5 (llama.cpp 42). Experts stored as separate tensors (old
  Mixtral files) and prompts take the CPU-routed path: per layer and chunk, one router matmul
  and one sync; the CPU picks each token's top-k and sorts the (token, expert) pairs by expert.
  Then one `gather_rows` of every pair's input, each used expert's gate and up as matmuls over
  its own rows with no barrier between experts (a wait per expert halved the speed), one
  SwiGLU, every down, and `moe_combine_rows` adds each token's k outputs, weighted. A shared
  expert runs over all rows with a per-token sigmoid gate (`scatter_axpy`). That scratch is a
  row per token and chosen expert (92 MB for 512 tokens of Qwen3-30B-A3B), so the batch
  scratch is sized for the longest chunk seen, not for 512: on a model that barely fits,
  Metal fails a command buffer with "Insufficient Memory" when its buffers can't all be
  wired, and a few tens of MB decide it. Check `/usr/bin/time -l` (peak memory footprint)
  after changing what is allocated.
- **Accuracy.** The fast CPU path quantizes activations to int8 (Q8 blocks): logits differ from
  f32 by ~0.1–0.6 but top-1 matches. `Model.precise` (the `--precise` flag in `dump_logits`)
  keeps f32 activations and matches the NumPy reference to ≤4e-5 on the CPU and ≤1.1e-4 with Metal
  batched prompts (the matrix units accumulate in a different order); tests that compare logits use it.
  The f16 KV caches (Metal and CPU) move logits by up to ~0.015 against f32 on real models
  (top-1 unchanged). Int8 activations also amplify float rounding: a tiny difference flips a value
  across a `round()` boundary and cascades through the layers, so NEON, AVX2 and portable kernels
  each give logits about 0.4 apart and can differ in temperature-0 text on near-ties. Compare a
  kernel with `--precise`, not with another kernel's text. These differences are enough to change sampled output: at high temperature, the
  same seed gives one reply when a prompt is evaluated in full and another when its prefix is cached.
- **Model files are untrusted.** `rli-gguf` rejects malformed headers, and `Model::load` and
  `Tokenizer::from_gguf` check tensor shapes and metadata against each other, because GPU kernels
  have no bounds checks and a wrong size can allocate without limit. `crates/models/tests/malformed.rs`
  corrupts header fields and requires an error or a clean CPU run, never a panic. New tensors need
  a shape check in `Model::load`.
- **CPU SIMD uses runtime feature detection**, and every kernel keeps a scalar/portable fallback.
- **NEON row kernels are short of vector operations, and of sums.** The M4 runs four a cycle, so
  a decode kernel is counted in operations per block: values that reach a vector register through
  a general one, a multiplier broadcast again per use (LLVM does that to a lane of a 16-bit
  product; `smull` avoids it), a multiply and an add where one fused instruction would do. A float
  sum per row is a chain: blocks go into four sums in turn, indexed by a loop of constant length
  (indexed by `block % 4`, or through a closure, the sums end up in memory and it is slower). Read
  the disassembly (`objdump -d` on `bench_matvec`) before believing a change did what it says.
  `smmla` costs as much per multiply as `sdot` there; the block kernel is ahead of the row kernel
  only because its scale arithmetic is shared by more multiplies. Against llama.cpp on the M4 CPU
  (`docs/benchmarks.md`, 2026-11-16): Qwen2.5-0.5B (Q5_0/Q8_0) prompts 2.2x ahead and decode level
  on four threads; Llama-3.2-1B (K-quants) decode level on four threads and 12-14% behind on one,
  prompts 7% behind on one thread and 16% on four (2026-11-17; llama.cpp scales activations per 256, we per 32: three float operations
  per 32 weights, row pair and token pair that it does not have). `--coarse-acts` (`Model.coarse_acts`, off by
  default) quantizes prompt activations per 256 too and runs K-quants through `rows2_coarse`: ahead of
  llama.cpp on one thread, 9% behind on four, at about twice the logit error. It has no x86 kernel: an AVX-512 one gained nothing on a Xeon 6975P. An Intel AMX block kernel (`run_amx`, tiles through inline asm, bit-equal to the AVX-512 one) runs where the CPU has it (`RLI_AMX=0` turns it off): it stores 8 chunks' integer sums, then `settle` does their float arithmetic with 16 tokens' sums in registers (done per chunk with the sums in memory it was 12% slower than AVX-512; now 5% faster on a Xeon 6975P-C). llama.cpp's AMX path, on weights it repacks at load and one activation scale per 256, keeps its K-quant prompts 1.13-1.19x ahead on Sapphire Rapids and later Xeons (2026-11-17); a prompt chunk here is 13% panel fill, the rest token loop, so repacking alone would not close that. GitHub runners are mostly AMD: the `Benchmark` workflow's `cpu_flag=amx_int8` stops a run at once on anything else.
- **CUDA kernels keep a warp's reads adjacent.** Each of a warp's 32 lanes takes the next weight
  and the next value of x: `wt_*` decodes one weight per lane (the matmuls), `sb_*` 8 weights per
  lane, 256 per step, so a block's scales are decoded once (the quantized matvecs). Lanes that
  each took a separate 32-weight chunk ran prompts 6-7x slower on an A100. Prompts run in chunks
  of 512 tokens (`BATCH_CHUNK`; a matrix is decoded once per chunk, so 512 beat 256 by 14-30%): a warp multiplies its weight row into a tile of 32 tokens, and
  causal attention runs as tiled products (`attn_scores`, `attn_softmax`, `attn_pv`: 32 positions
  of K or V against 32 query rows per block, so a cached row is read once per tile, not once
  per head and token; a thread reads its share of a tile as 4 `float4` before storing any, and
  `attn_pv` reads the next 32 positions while it multiplies these: 1.3-1.7x on those kernels). Its scores need a row per head and token,
  capped at 128 MB per session (`SCORE_BUDGET`); past that a chunk's attention takes several
  launches. Decode past 128 cached positions splits the context instead: `attention_part` takes
  a KV head and 32 positions per block, scoring the query heads that share the KV head
  together, and `attention_merge` combines the blocks (decode at 2048 positions: Mistral-7B
  71 -> 107 tok/s). A thread of either issues all its reads of K, V or the parts before its
  first barrier, since K, softmax, then V was three waits on memory in a row (142 -> 153 tok/s); under 128 the second launch costs more than it saves. MoE layers are routed on the GPU with no sync, in decode and in prompt chunks alike:
  `moe_topk` takes a warp per token, then every token's k experts run as grid rows of one launch
  per matrix (each matvec kernel takes an expert id per row, `Batch::matvec_id`), so a (token,
  expert) pair decodes its expert's weights again. Chunks of 64+ tokens (`MOE_GROUP_TOKENS`)
  don't: `moe_group` sorts the chunk's pairs by expert on the GPU, and a block of
  `gemm_q4_k_id` / `gemm_q6_k_id` takes 64 rows of one expert and the pairs that chose it,
  decoding the weights to f16 in shared memory on the way into f16 `mma.sync` (no scratch
  copy; Qwen3-30B-A3B prompts 830 -> 3115 tok/s, 3900 with 512-token chunks). Without f16
  tensor cores the same grouping runs `dp4a` (`mmq_*_id`, 1810 tok/s). File ranges
  that touch are uploaded as one, so stacked experts keep their stride in VRAM. Experts stored
  as separate tensors are routed on the CPU, one token at a time with a sync per layer.
  Prompt chunks of 8+ tokens (`MMQ_MIN_TOKENS`) use int8 activations unless `precise`:
  `quant_x` quantizes a matmul's input once (one scale per 32 values), and `mmq_q4_k` /
  `mmq_q6_k` decode a 32-row, 256-column weight tile to int8 in shared memory and run `dp4a`
  over 64 tokens per block; 3.4-4.4x on dense prompts, logits within ~0.16 of f32 (0.6 on
  Qwen3-30B-A3B). Decode and shorter chunks run the same matrices as int8 matvecs (`mvq_q4_k`,
  `mvq_q6_k`): a Q4_K lane takes one sub-block in four aligned 16-byte reads, since the number
  of separate loads per step, not the arithmetic, was what the f32 matvec spent its time on
  (Mistral-7B decode 72 -> 129 tok/s). Those reads need the tensor at a multiple of 16 bytes in
  VRAM (`Gpu::has_int8`), else f32 runs. Q6_K blocks are 210 bytes, so its lanes read the
  aligned 16-byte words around each span and shift (`ld24`; the arena ends in 16 bytes of slack
  for that). Launches cost about 4 microseconds each, so a token's are kept few: kernels whose
  result feeds int8 matrices quantize it on the way out (`quant_block` in `rmsnorm`, `swiglu`,
  `attention`), the int8 matvec can add into the residual stream, `rope` takes queries and keys
  together and a token's gate and up are one launch (`matvec_q_pair`; the `mvq_*_pair` kernels
  take a grid two deep, and are separate because their three extra parameters made the plain
  ones 2-5% slower): 12 launches a dense layer, was 20. A Qwen3 MoE layer takes 14: its chosen
  experts' gate and up stacks are one launch too (decode only; 1.5% slower over a prompt chunk),
  `norm_rope` does the per-head query and key norms with the rotation, and an f32/f16 matvec of
  up to 256 rows (the router) gets a block of 8 warps per row. That took Qwen3-30B-A3B decode
  from 155 to 178 tok/s, past llama.cpp's 171.5. One token's launches are captured as a CUDA graph and replayed (`Gpu::run_graph`):
  `Batch` compares each launch with the graph's and updates the nodes whose arguments changed,
  so the queueing code is the same either way; a pass whose launches differ is captured anew,
  and one that reads back mid-pass (CPU-routed MoE) can't be replayed. Stacked MoE experts run
  the int8 matvec too (`matvec_q_id`); other weight types stay f32. `dp4a` has no header
  under NVRTC, so it is inline PTX, and the source is compiled for compute capability 6.1.
  On capability 8.0+ (Ampere) `mma_q4_k` / `mma_q6_k` take their place (`RLI_CUDA_NO_MMA=1`
  keeps `dp4a`): int8 tensor cores through inline `mma.sync`, blocks of 128 rows x 64 tokens
  (every block stages its tokens again, so tall), 128 columns at a time, a warp's 16 rows of
  weights held in registers. The columns go in parts whose blocks add their sums into y
  (`mma_parts`: four times the blocks the GPU runs at once; a 4096x4096 matrix by 64 tokens
  took 179 us as 32 blocks and 56 as 512), which also lets them add into the residual
  stream. They are a second
  compile of `cuda.cu` (`-DRLI_MMA`, only those kernels) capped at 80 registers per thread,
  which lets a multiprocessor hold more warps but slows the f32 matmuls. About 2x `dp4a` on
  Mistral-7B prompts; not bit-equal to it. Their tile loads issue all of a thread's
  global reads before its first store: a read followed by a store waits out the memory
  latency, which was most of the loading time with so few warps per multiprocessor
  (65536 / registers / 32 fit). Don't carry that over: the same reordering made the decode
  matvecs, the f32 matmul and the `dp4a` matmul 2-28% slower (`docs/benchmarks.md`).
  Prompt chunks of 129+ tokens (`GEMM_MIN_TOKENS`) skip all of that for dense matrices
  (llama.cpp 0.9 runs its int8 tensor-core kernel at every length on this GPU, not cuBLAS;
  writing the matrices out costs as much as multiplying them at 64 tokens, and int8 in parts
  is ahead up to 128, level at 192, 24% behind at 256 on Mistral-7B): `to_f16` converts a matmul's input, `dequant_*` writes the matrix
  out as f16 into a scratch buffer the size of the largest one (`Scratch::wh`), and
  `gemm_f16_64x64` / `_64x128` multiply them with f16 `mma.sync` and f32 sums (the second
  compile again). Closer to f32 than int8 (~0.07 on dense models) and 1.25-1.3x faster. Not
  under `precise`, not for F32/F16 weights (they keep f32 activations); MoE experts have their
  own kernels (above).
  Q4_K and Q6_K have wide dequantizers (16-byte stores; Q6_K reads through `ld24`); the int8
  matmuls remain for chunks of 8-128 tokens. The 64x128 tile runs four warps of four row groups
  (a read of the token tile feeds four products); it does not wait on global memory.
  What it is short of is the copy into shared memory: with the tiles staged once it runs
  1.5-1.9x faster, and every tile shape tried moves 2.2-2.7 TB/s. cuBLAS on the same A100 is
  level on 4096x4096 x 512 tokens (156 us against 169) and far ahead on Mistral-7B's FFN
  (422 against 513, 314 against 603), which is most of the prompt gap. A 128x128 tile on four
  warps with `cp.async` into two buffers reaches 135-157 TFLOPS, but only where the block
  count is a multiple of what the GPU runs at once: a few blocks more run as a second wave
  that costs a whole block's time, and real shapes don't divide. Tried and dropped
  (`docs/benchmarks.md`, 2026-11-18): `ldmatrix`, eight warps a block, 32-column steps,
  blocks that take part of a tile, a swizzled 32-word layout, a higher register cap.
  So `gemm_f16_128x128` (four warps of 64 rows by 64 tokens) runs where its blocks can be
  made to fill the GPU, and the 64-row tiles elsewhere (`gemm_plan`, `gemm_parts` in
  cuda.rs): a matrix of few rows has its columns split into parts, each a block that adds
  its sums into y with `atomicAdd` (free here; y is zeroed first, or is the residual
  stream: `Batch::gemm_add`, which saves the `axpy` too), and one of more rows than the
  GPU holds blocks goes as much as fills it, then the rest in parts. The counts that work
  are an A100's (108 multiprocessors, 4 such blocks each: as long over 3 blocks as over 4,
  half as long over 2), found by measuring; no other GPU has been tried. **The order
  the parts add in varies, so a prompt's logits differ from run to run** in the last bits,
  which the layers amplify (0.17 and 0.19 from `precise` on two runs of one prompt).
  Graph capture is thread-local: in global mode another thread's allocation fails while it
  is open.
  Against llama.cpp's CUDA backend on the same A100 (`docs/benchmarks.md`, 2026-11-15): decode
  is 10-16% ahead on all three models; decode at 2048 positions is level on Llama-3.2-1B and 2%
  ahead on Mistral-7B (2026-11-17), and prompts
  level on Llama-3.2-1B (within its noise) and Qwen3-30B-A3B, 1.16x behind on Mistral-7B;
  (2026-11-18, with the 128x128 tile) pp512 3% behind on Mistral-7B and 5-6% ahead on the
  other two; short prompts (int8 in parts) behind on Mistral-7B, 1.65x at 16 tokens, 1.35x
  at 32, 1.15x at 64, 1.24x at 128; Llama-3.2-1B 1.3x behind at 16, 1.14x at 32, level from
  64; Qwen3-30B-A3B 2x behind at 16 and 32 (its expert matmuls, not these), level at 64,
  1.3x ahead at 128.
- **A decoded token's CPU time is GPU idle time.** On CUDA it was a quarter of a Llama-3.2-1B
  token. A pass that runs wholly on an accelerator skips `pool().install` (a thread wake-up per
  token), `argmax` is written to vectorize, and `Gpu::read` downloads the logits through
  page-locked memory. Time the phases before the kernels.
- **CUDA kernels wait on memory per read.** A thread that reads values in a loop and uses each
  at once waits out the latency every time round, so a small kernel wants a thread per value
  (`rmsnorm` with 256 threads for a 2048-wide row cost twice a trivial launch). `RLI_CUDA_PROF=1`
  times every launch and prints the totals per kernel when the device is dropped; it is how
  that was found.
- **CUDA is checked by hand.** CI only compiles it, so after changing it run on an NVIDIA GPU:
  the tests with `--no-default-features --features rli-models/cuda`, and `dump_logits --precise`
  against `ref_forward.py` both batched and with `--per-token` (the decode kernels).
  `scripts/cuda_cpu_check` checks the weight decoders on the CPU; CI doesn't run it. The tensor-core
  kernels decode inline, so only the GPU test `matmul_q_matches_f32_matmul` (rli-core, run it
  with and without `RLI_CUDA_NO_MMA=1`) covers them. It first ran
  on 2026-11-15, on a rented A100; `libnvrtc` had to be put on the library path there
  (`LD_LIBRARY_PATH=/opt/conda/lib` on a PyTorch image).

## Commands

`CARGO_TARGET_DIR` is set globally on this machine, so binaries are at
`$CARGO_TARGET_DIR/release/{rli,rli-server}`, not `./target`.

`rli` and `rli-server` enable `metal` by default (a no-op off macOS), and offload every layer
unless `--gpu-layers N` says otherwise (`0` = CPU only). Without the flag, a model larger than
Metal's recommended working set (19.1 GB here, less a 1 GB margin) offloads as many leading
layers as fit, and a GPU that can't take the model at
all (no device or driver, unsupported quant, not enough VRAM) falls back to the CPU; both warn.
With the flag, either is an error (`rli_models::backend::attach_default`).

```sh
cargo build --release                                  # macOS: CPU + Metal
cargo build --release --no-default-features            # CPU only
cargo build --release --no-default-features --features rli/cuda,rli-server/cuda   # CUDA (runtime-loads driver + NVRTC; no toolkit needed to build)
cargo test --workspace --release                       # all tests, including Metal on macOS
cargo test --workspace --release --no-default-features # CPU-only tests
cargo test -p <crate> <test_name>                      # single test
cargo clippy --workspace --all-targets -- -D warnings  # also with --no-default-features and --features rli-models/cuda
cargo fmt --all
```

CI (`.github/workflows/ci.yml`) runs on pushes to `main` and on PRs: fmt and clippy, plus the tests
on Linux x86-64 (the only run of the AVX2 kernels; this Mac can't run them, even under Rosetta). Each job has a 4-minute timeout, so a change that
makes it slower fails rather than creeping up. There is no macOS job, to avoid its billing: Metal
and NEON are checked only locally, so run the tests and clippy above before pushing changes to them.

Run: `rli -m model.gguf -p "prompt"` (without `-p` it starts an interactive chat), or
`rli-server -m model.gguf --port 8080`, which also serves a chat page for manual testing at `/`
(`crates/server/src/chat.html`, embedded in the binary). Real models go in `models/`, which is
gitignored.

## Validation workflow

The Python tools need `pip install gguf numpy` (plus `tokenizers huggingface_hub` for the tokenizer check):

- Logits: `cargo run --release -p rli-models --example dump_logits -- <model> "<prompt>" out.f32 --precise [--gpu-layers=N] [--per-token]`,
  then `python scripts/ref_forward.py <model> <ids> out.f32` (an independent NumPy forward pass).
  `--per-token` runs the prompt through the decode path instead of the batched one.
- Tokenizer: `python scripts/check_tokenizer.py <model.gguf> <hf-repo>`.
- Chat format: `python scripts/check_chat_template.py <model.gguf>` renders the file's real Jinja
  template and compares it with `ChatTemplate::render` (needs `pip install gguf jinja2`).
- Int8 accuracy: `cargo run --release -p rli-models --example eval_quant -- <model> <text file> [--tokens 512]`
  runs a text through the decode path with f32 (`precise`) and int8 activations and prints both
  perplexities, the KL divergence of int8 from f32, and top-1 agreement. Measured 2026-11-09 (prose and
  code, 256-512 tokens): dense models KL mean 0.0003-0.002, top-1 agreement 96-99.6%, perplexity
  within ±0.3%; Qwen3-30B-A3B (MoE) KL mean 0.004-0.015, max ~1.0, as int8 noise flips expert choices.
  One activation scale per 256 values for K-quants (llama.cpp's Q8_K) raised KL 1.4-6x and cut top-1
  agreement by 1-3 points, so we keep one per 32.
  Each block's scale is its 256-value group's largest block scale times r/256 (`Q8Act`), which lets the
  AVX2 K-quant row kernels multiply by r as an integer and convert to float once per super-block; against
  free per-32 scales that moved KL mean by x0.8-1.25 and top-1 agreement by under 1 point (2026-11-11).
- Fixtures: `scripts/gen_quant_fixture.py` (dequant reference) and `scripts/make_tiny_moe.py`
  (tiny MoE GGUFs plus their reference logits, used by `crates/models/tests/moe_reference.rs`).
- Throughput: `cargo run --release -p rli-models --features metal --example bench -- <model> [--pp 512] [--tg 128] [--depth D] [--reps 3] [--gpu-layers N]`
  (fixed-length runs that ignore EOS; `--depth` prefills D tokens before timing decode; defaults to CPU). Kernel microbenchmarks: `-p rli-core --example bench_matvec` / `bench_metal`.
  Background load (e.g. a Docker VM) makes Metal numbers swing by ~2x on this machine; measure on a quiet system.

- Against llama.cpp: `python3 scripts/bench_compare.py --bin-dir <dir> --llama-dir <dir> --model <model> [--threads 1,4]`
  prints throughput, first-token time and peak memory on the CPU. The manual `Benchmark` workflow
  (`.github/workflows/bench.yml`) runs it on x86; GitHub runners vary in CPU and have 2 vCPUs on one core, so
  compare only within one run and don't read thread scaling from it. On macOS build llama.cpp with
  `-DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF` (and `-DGGML_METAL=OFF` for CPU runs; with Accelerate on, Q5_0
  prompts still ran at one speed whatever the thread count): its default BLAS backend dequantizes Q5_0 and similar weights and runs prompts on
  Apple's matrix hardware (Qwen2.5 0.5B on the M4: 517 tok/s against 110.6 without), so that's not a
  comparison of int8 kernels, and it doesn't carry over to x86.

Performance-sensitive changes to kernels or the hot path need before/after benchmark numbers. Debug
builds are not representative, so always measure with `--release`. Record results worth keeping in
`docs/benchmarks.md`, with the run or command that produced them.
