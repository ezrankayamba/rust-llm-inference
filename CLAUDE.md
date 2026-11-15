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
  dequant/dot), `QMatrix` matvec (`matrix.rs`, rayon-parallel; every forward pass runs inside rli-models' own pool, `pool()` in `lib.rs`, so the calling thread is a worker rather than waiting on one per matvec), fast integer kernels over
  Q8-quantized activations (`qdot.rs`, NEON `dotprod` or AVX2 with runtime detection and a portable
  fallback; on ARM, batches of 2+ tokens use `i8mm` `smmla` via inline asm when present, over activations
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
  `Engine::generate` polls between 256-token prompt slices and before each token. Responses carry
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
  in chunks of up to 256 tokens: a tiled `simdgroup_matrix` matmul (f32; under 5 tokens, one
  matvec per token, since a tile costs about 5-6 matvecs however few tokens it holds), and causal
  `attention_batch` with an online softmax. Decode uses matvec, and attention split over
  chunks of 256 cached positions that run in parallel and are merged (`attention_part_*` +
  `attention_merge`), so decode speed holds up at long context.
  head_dim not a multiple of 32 or above 256 falls back to one token at a time.
- **CPU prompts are batched too**, in chunks of 128 tokens (`CPU_CHUNK`): `QMatrix::matmul_with`
  runs each block of weight rows against every token of the chunk while it is in cache. One
  row is the decode case, so CPU decode and prompts share `Model::cpu_layers`. CPU MoE layers
  route each row, then run each used expert once over the rows that picked it.
- **MoE on Metal.** Decode routes on the GPU, with no sync: `moe_topk` (one simdgroup) writes
  the k expert ids and weights, `matvec_id` runs all k experts' gate/up/down as one dispatch
  each (expert id read from the buffer, experts at a fixed stride in the stacked tensor), and
  `moe_combine` adds the weighted outputs. A per-layer sync cost ~0.35 ms; removing it took
  Qwen3-30B-A3B decode from 19.6 to 29.1 tok/s. Experts stored as separate tensors (old
  Mixtral files) and prompts take the CPU-routed path: per layer and chunk, one router matmul
  and one sync; the CPU picks each token's top-k, then each used expert runs as a matmul over
  the rows gathered for it (`gather_rows`), added back weighted (`scatter_axpy`). A shared
  expert runs over all rows with a per-token sigmoid gate.
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
- **CUDA kernels keep a warp's reads adjacent.** Each of a warp's 32 lanes takes the next weight
  and the next value of x: `wt_*` decodes one weight per lane (the matmuls), `sb_*` 8 weights per
  lane, 256 per step, so a block's scales are decoded once (the quantized matvecs). Lanes that
  each took a separate 32-weight chunk ran prompts 6-7x slower on an A100. Prompts run in chunks
  of 256 tokens (`BATCH_CHUNK`): a warp multiplies its weight row into a tile of 32 tokens, and
  causal attention runs one block per head and token. Its scores need a row per head and token,
  capped at 128 MB per session (`SCORE_BUDGET`); past that a chunk's attention takes several
  launches. MoE layers are routed on the GPU with no sync, in decode and in prompt chunks alike:
  `moe_topk` takes a warp per token, then every token's k experts run as grid rows of one launch
  per matrix (each matvec kernel takes an expert id per row, `Batch::matvec_id`), so a (token,
  expert) pair decodes its expert's weights again; Metal's gathered matmuls don't. File ranges
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
  VRAM (`Gpu::has_int8`), else f32 runs. Other weight types and MoE experts stay f32. `dp4a` has no header
  under NVRTC, so it is inline PTX, and the source is compiled for compute capability 6.1.
  On capability 8.0+ (Ampere) `mma_q4_k` / `mma_q6_k` take their place (`RLI_CUDA_NO_MMA=1`
  keeps `dp4a`): int8 tensor cores through inline `mma.sync`, blocks of 64 rows x 128 tokens,
  128 columns at a time, a warp's 16 rows of weights held in registers. They are a second
  compile of `cuda.cu` (`-DRLI_MMA`, only those kernels) capped at 96 registers per thread,
  which lets a multiprocessor hold more warps but slows the f32 matmuls. About 2x `dp4a` on
  Mistral-7B prompts; not bit-equal to it. Their tile loads issue all of a thread's
  global reads before its first store: a read followed by a store waits out the memory
  latency, which was most of the loading time with so few warps per multiprocessor
  (65536 / registers / 32 fit). Don't carry that over: the same reordering made the decode
  matvecs, the f32 matmul and the `dp4a` matmul 2-28% slower (`docs/benchmarks.md`).
  llama.cpp's CUDA backend was 1.9-2.6x faster on dense prompts, 12x on Qwen3-30B-A3B's and
  2-3x on decode on the same A100 (`docs/benchmarks.md`, 2026-11-15).
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
  `-DGGML_BLAS=OFF`: its default BLAS backend dequantizes Q5_0 and similar weights and runs prompts on
  Apple's matrix hardware (Qwen2.5 0.5B on the M4: 517 tok/s against 110.6 without), so that's not a
  comparison of int8 kernels, and it doesn't carry over to x86.

Performance-sensitive changes to kernels or the hot path need before/after benchmark numbers. Debug
builds are not representative, so always measure with `--release`. Record results worth keeping in
`docs/benchmarks.md`, with the run or command that produced them.
