# rust-llm-inference

An LLM inference engine written from scratch in Rust, for machines with little to spare: CPU-only
hosts, Apple unified memory and low-VRAM GPUs.

It has no ML framework underneath (no candle, burn or ggml bindings). The tensors, the SIMD and GPU
kernels, the GGUF loader, the tokenizers and the model graphs are all implemented here.

- **`rli`**: a command-line tool for one-shot prompts and interactive chat.
- **`rli-server`**: an OpenAI-compatible HTTP server (`/v1/chat/completions`, with streaming) that
  also serves a small chat page at `/`.

## What it supports

| Area | Support |
|---|---|
| Weights | GGUF, memory-mapped and read in place |
| Quantization | F16/BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K; Q4_1, Q5_0 and Q5_1 also load |
| Models | Llama family (Llama 2/3, Mistral, dense Qwen) and mixture-of-experts (Mixtral, Qwen MoE) |
| CPU | AVX2 and AVX-512 VNNI on x86, NEON on ARM, chosen at runtime, with a portable fallback |
| GPU | Apple Metal. A CUDA backend compiles in CI but has never been run on an NVIDIA GPU |

## Quick start

You need Rust (the version is pinned in `rust-toolchain.toml`) and a GGUF model file.

```sh
cargo build --release                         # macOS: CPU + Metal
cargo build --release --no-default-features   # CPU only

target/release/rli -m model.gguf -p "Explain memory mapping in one paragraph."
target/release/rli -m model.gguf              # interactive chat
target/release/rli-server -m model.gguf --port 8080
```

Useful flags: `--gpu-layers N` (`0` keeps everything on the CPU), `--threads N`, `--ctx N`,
`--temp F`, and `--repack` (see below). `rli --help` lists the rest.

## Performance

One thread, Llama-3.2-1B Q4_K_M, against llama.cpp built natively on the same machine (measured
2026-11-13). These are
GitHub-hosted runners, whose CPUs vary from run to run, so compare within a row only.

| CPU | rli decode tok/s | rli `--repack` | llama.cpp | rli peak memory | llama.cpp peak memory |
|---|---:|---:|---:|---:|---:|
| EPYC 9V74 (Zen 4) | 25.4 | 30.1 | 29.9 | 864 MiB | 1331 MiB |
| EPYC 9V45 (Zen 5) | 33.7 | 36.1 | 36.6 | 864 MiB | 1331 MiB |
| EPYC 7763 (Zen 3) | 17.4 | 18.9 | 17.1 | 866 MiB | 1330 MiB |
| Xeon 8573C | 12.6 | 12.6 | 10.3 | 864 MiB | 1331 MiB |

By default the weights stay in the file mapping, which is why `rli` needs less memory. `--repack`
copies the Q4_K weights into an interleaved layout at load, as llama.cpp does: decode speeds up on
AMD CPUs, and memory use rises to about llama.cpp's (1322 MiB in these runs).

The largest remaining gap is prompt processing for Llama-3.2-1B on Intel Xeons with AMX, where
llama.cpp has a dedicated kernel and `rli` has none.

Every number, the run that produced it, and the experiments that did not work are in
[docs/benchmarks.md](docs/benchmarks.md).

## Accuracy

The fast CPU path quantizes activations to 8 bits. Against 32-bit float activations, on dense models
the next-token choice agrees 96-99.6% of the time and perplexity stays within 0.3%. An exact mode
matches an independent NumPy forward pass (`scripts/ref_forward.py`) to within 4e-5 on the CPU.

## Layout

A Cargo workspace; dependencies flow downward only.

| Crate | Contents |
|---|---|
| `crates/core` | Quantized block formats, matrix kernels (CPU SIMD, Metal, CUDA) and basic ops |
| `crates/gguf` | Memory-mapped GGUF parser |
| `crates/models` | Model graphs, KV cache, tokenizers, chat templates, sampling and the generation API |
| `crates/cli` | The `rli` binary |
| `crates/server` | The `rli-server` binary |

`CLAUDE.md` holds the detailed design notes and the validation workflow.

## Tests

```sh
cargo test --workspace --release
cargo clippy --workspace --all-targets -- -D warnings
```

## Licence

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT),
at your option.
