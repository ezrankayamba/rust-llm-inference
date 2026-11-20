# Architecture

This is a map of the engine: what each part does, how a request flows through it, and the
design rules that shape the code. The README covers using it; `CLAUDE.md` holds the detailed
design notes and the measurements behind each decision; `docs/benchmarks.md` holds every
number. Where this document says a choice "was faster", the run is recorded there.

## Goals and constraints

- **Small machines first.** CPU-only hosts, Apple unified memory and GPUs with little VRAM.
  Memory, not arithmetic, is the budget that decides most designs.
- **No ML framework.** Tensors, kernels, the GGUF loader, tokenizers and model graphs are all
  written here. The only GPU dependencies are the platform APIs (Metal; the CUDA driver and
  NVRTC, loaded at run time).
- **GGUF only**, memory-mapped. Quantized weights are read where they lie in the file.
- **One code path per decision, measured.** A kernel or layout is kept because a benchmark
  showed it faster on the hardware it targets; ones that lost are documented and removed.

## Crates

Dependencies flow downward only.

```
 rli (crates/cli)            rli-server (crates/server)
 one-shot + chat CLI         OpenAI-compatible HTTP + chat page
          \                      /
           \                    /
            rli-models (crates/models)
            config, model graph, KV sessions, tokenizers, chat templates,
            sampler, stop strings, Engine; the Metal and CUDA accelerators
                         |
            rli-core (crates/core)
            dtypes and quant blocks, CPU kernels (SIMD), ops,
            GPU primitives: metal.rs + metal.metal, cuda.rs + cuda.cu
                         |
            rli-gguf (crates/gguf)
            mmap GGUF parser (and a small writer for tests)
```

| Crate | Main files | Role |
|---|---|---|
| `rli-gguf` | `lib.rs`, `writer.rs` | Parses the header, metadata and tensor table of a memory-mapped file; tensors are absolute file offsets. Rejects malformed headers. |
| `rli-core` | `quant.rs`, `matrix.rs`, `qdot.rs`, `simd.rs`, `ops.rs`, `half.rs` | Block layouts and f32 reference dequantization; `QMatrix` matvec/matmul over quantized rows; int8 dot-product kernels (NEON, AVX2, AVX-512 VNNI, AMX, portable); RMSNorm, RoPE, softmax, SwiGLU. |
| | `metal.rs`, `metal.metal` | Metal device, buffers and kernels (feature `metal`). |
| | `cuda.rs`, `cuda.cu` | CUDA device, buffers, launch batching and graphs, and kernels (feature `cuda`). |
| `rli-models` | `config.rs`, `model.rs` | Architecture config (`Arch`: Llama, Qwen2, Qwen3, Qwen2Moe, Qwen3Moe); weight loading and shape checks; the decoder forward pass on the CPU; KV `Session`; the `Accelerator` trait. |
| | `metal.rs`, `cuda.rs`, `backend.rs` | The two `Accelerator` implementations, and `backend::attach` which picks the one compiled in. |
| | `tokenizer.rs`, `chat.rs`, `sampler.rs`, `stop.rs`, `engine.rs` | SPM and byte-level BPE tokenizers; chat formatting chosen by fingerprinting the GGUF's Jinja template; sampling; stop strings; `Engine`, the generation loop both binaries share. |
| `rli`, `rli-server` | `main.rs`, `http.rs`, `api.rs`, `chat.html` | Thin binaries: argument parsing, I/O, and (server) std-only HTTP with SSE streaming. |

## Life of a request

1. **Load.** `Engine::load` maps the GGUF file, builds `Config` from its metadata, and
   `Model::load` resolves every tensor to a `Weight` (type, shape, file offset) after checking
   its shape against the config. Nothing is copied: a `Weight` is a view into the mapping. The
   tokenizer and chat template are read from metadata.
2. **Offload.** `backend::attach_default` gives the first *N* layers to the GPU backend built in
   (all of them by default, or as many as fit; `--gpu-layers N` makes it exact). The backend
   becomes the model's `Accelerator`.
3. **Prompt.** The server or CLI renders the chat messages with the model's template and
   tokenizes them. `Engine::generate` compares the tokens with what the session already holds
   and keeps the common prefix's KV cache, then evaluates the rest in slices
   (`Model::prompt_slice`, 256 tokens or 512 when the GPU runs every layer), checking for
   cancellation between slices.
4. **Forward pass.** `Model::forward_batch` embeds the tokens, hands them to the accelerator
   (`Accelerator::forward_batch`) for its layers, then runs any remaining layers on the CPU
   (`cpu_layers`, in chunks of 128 tokens), the final norm and the output head. Only the last
   token's logits are computed.
5. **Decode.** Each step samples a token (`Sampler`), checks end-of-turn tokens and stop
   strings, streams the text out as it becomes valid UTF-8, and runs a one-token forward pass.
   A client disconnect or a stalled write cancels generation; the session stays consistent.

## The model graph

A decoder layer is the usual pre-norm block: RMSNorm, Q/K/V projections (with biases or
per-head Q/K norms where the architecture has them), RoPE, causal attention over the KV cache,
output projection added to the residual stream, RMSNorm, then the feed-forward network: a
SwiGLU MLP (`Ffn::Dense`) or a routed mixture of experts (`Ffn::Moe`), whose router picks the
top-k experts per token and whose outputs are summed by the router's weights (plus a shared
expert for Qwen2-MoE).

The same graph runs on every backend; backends differ only in how they schedule and fuse it.

## Memory model

- **Weights stay in the mapping.** The CPU reads them in place. Metal wraps the offloaded
  layers' file ranges as no-copy buffers (unified memory). CUDA copies only the offloaded
  layers into one VRAM arena. `--repack` (CPU, off by default) is the one exception: it copies
  Q4_K weights into an interleaved layout that is faster on some x86 CPUs.
- **KV caches are f16** on every backend (f32 in `precise` mode), grown as tokens arrive: the
  CPU cache per token, Metal's by doubling, CUDA's preallocated per session.
- **Scratch is sized to the work seen**, not the worst case: on a model that barely fits, a
  few tens of MB decide whether the GPU can run it.
- **Model files are untrusted.** Shapes and metadata are checked against each other at load,
  because GPU kernels do no bounds checks. A test corrupts headers and requires an error, never
  a panic.

## Backends

### CPU (always built)

- Activations are quantized to int8 per 32 values (`Q8Act`), and weights are multiplied in
  their own block format by integer dot-product kernels in `qdot.rs`, chosen at run time from
  the CPU's features. Every kernel has a portable fallback.
- Decode is one row at a time (`k_rows`, `dot_rows`); prompts multiply blocks of weight rows
  against a chunk of tokens while they are in cache (`QMatrix::matmul_with`), using `smmla` on
  ARM and interleaved 8- or 16-row panels with AVX2 / AVX-512 VNNI / AMX on x86.
- Work is spread over rayon's pool (`pool()`), with the calling thread as one of its workers.

### Metal (macOS)

- Each token's offloaded layers are encoded into one command buffer with one sync; the encoder
  is concurrent and barriers are placed only where a dispatch reads what the one before it
  wrote.
- Decode is weight streaming, so kernels are fused to cut dependent dispatches (matvec plus
  residual add, gate/up/activation in one, RoPE plus cache store).
- Prompts run in 512-token chunks on `simdgroup_matrix` tiles; attention over the f16 cache has
  staged and split-context kernels.
- MoE decode routes on the GPU with no sync; prompts route on the CPU once per layer and chunk.

### CUDA

- Kernels are compiled at run time with NVRTC from `cuda.cu`, in up to three compiles: the
  base one; the tensor-core kernels (`RLI_MMA`, capability 8.0+); and the "lean" kernels for
  Blackwell (`RLI_MMK`, capability 12.0+, e.g. the RTX 5090).
- `Batch` queues a pass's launches. A decoded token, and each prompt-chunk length, is captured
  as a CUDA graph and replayed; launches whose arguments change update their graph node. A
  decoded token's position is read from a device word, so a replay usually updates nothing.
- Kernels that produce a matmul's input write it in the form the matmul reads (int8 or f16,
  the `Fed` enum), so long prompts have no separate conversion launches.
- The matmul kernel is picked per matrix and chunk length (`mm_each`): int8 matvecs for decode,
  direct and dense f16 tensor-core kernels for short and medium chunks, dequantize-plus-GEMM or
  the lean int8 kernels for long ones.
- Prompt attention is one fused flash kernel on the lean GPUs (otherwise three tiled launches);
  decode splits long contexts over blocks and merges them.
- MoE layers route on the GPU in decode and prompts alike. Long chunks sort (token, expert)
  pairs by expert (`moe_group`) so each expert's weights are decoded once for all its tokens.

### Choosing and mixing

Offload is by layer, never by individual matmul (per-matmul offload measured slower than the
CPU). A model that doesn't fit runs its first *N* layers on the GPU and the rest on the CPU;
the two halves hand over the hidden state once per pass.

## Accuracy

There are two numeric modes:

- **Default (fast).** Int8 activations on the CPU and in CUDA's int8 kernels, f16 tiles and
  caches on the GPUs. Logits differ from f32 by roughly 0.1–0.6, and the top-1 token is
  preserved.
- **`precise`.** f32 activations everywhere. Logits match an independent NumPy forward pass
  (`scripts/ref_forward.py`) to about 1e-5 on the CPU and 1e-4 on Metal.

Tests that compare logits use `precise`. Different SIMD kernels, and some GPU kernels whose
parts add in varying order, can legitimately differ in the last bits. So a kernel is validated
against `precise`, not against another kernel's output text.

## Validation and tooling

| What | How |
|---|---|
| Unit and integration tests | `cargo test --workspace --release` (CPU, plus Metal on macOS); CI runs fmt, clippy and the Linux x86 tests, which are the only run of the AVX2 kernels |
| CUDA | checked by hand on an NVIDIA GPU: tests with and without the tensor-core and lean kernels, and logits against the NumPy reference |
| Logits | `dump_logits --precise` against `scripts/ref_forward.py`, batched and `--per-token` |
| Tokenizer, chat format | `scripts/check_tokenizer.py`, `scripts/check_chat_template.py` |
| Int8 accuracy | `eval_quant`: perplexity, KL divergence and top-1 agreement against f32 |
| Speed | `bench` (fixed-length prompt and decode runs), kernel microbenchmarks, `scripts/bench_compare.py` against llama.cpp; `RLI_METAL_PROF=1` / `RLI_CUDA_PROF=1` time each kernel |

## Where to start reading

- One token end to end on the CPU: `Model::forward_batch` and `cpu_layers` in
  `crates/models/src/model.rs`.
- How a backend plugs in: the `Accelerator` trait in the same file, then `Cuda::pass` in
  `crates/models/src/cuda.rs` or `Metal::pass` in `crates/models/src/metal.rs`.
- A kernel and its host side: for example `attn_flash` in `crates/core/src/cuda.cu` and
  `Batch::attention` in `crates/core/src/cuda.rs`.
- The generation loop: `Engine::generate` in `crates/models/src/engine.rs`.
