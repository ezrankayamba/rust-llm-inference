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
| GPU | Apple Metal, and CUDA (checked by hand on an NVIDIA A100; CI only compiles it) |

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

## Tool calls

On models whose chat template has Qwen's tool format (Qwen2.5, Qwen3), `/v1/chat/completions`
takes OpenAI `tools` and returns `tool_calls`. Function parameters are limited to an object of
required strings, each optionally an `enum`; anything else and `tool_choice` other than `"auto"`
or `"none"` are rejected with a 400. With `stream`, calls arrive as OpenAI's `delta.tool_calls`
once each block closes, and Qwen3 reasoning as `delta.reasoning_content`. While the model writes a call, its
sampling is constrained so the call can only name a listed function, give each of its keys once
and pick enum values from their lists. Qwen3 answers tools requests without thinking first.

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d @- <<'JSON'
{
  "messages": [
    {
      "role": "system",
      "content": "You operate a loan app by calling tools. The loan_application form has the fields amount, full_name and phone. The user may speak English, Swahili or both. Write amounts as digits (elfu = 1000, laki = 100000, milioni = 1000000)."
    },
    {
      "role": "user",
      "content": "Nifungulie fomu ya mkopo, jina ni Asha Juma, kiasi laki tano"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "fill_field",
        "description": "Fill one field of a form.",
        "parameters": {
          "type": "object",
          "properties": {
            "form_id": {
              "type": "string",
              "enum": [
                "loan_application"
              ]
            },
            "field_id": {
              "type": "string",
              "enum": [
                "amount",
                "full_name",
                "phone"
              ]
            },
            "value": {
              "type": "string"
            }
          },
          "required": [
            "form_id",
            "field_id",
            "value"
          ],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0
}
JSON
```

With Qwen3-8B Q4_K_M this returns two calls, `fill_field` with `full_name` = `Asha Juma` and with
`amount` = `500000` (`finish_reason: "tool_calls"`). On an M4 Mac the request takes 2.3 s on
Metal and 3.0 s on the CPU once the system prompt and tools are in the KV cache, nearly all of
it the 78-token reply. Inside a call, the text the schemas leave no choice over (the JSON around
the values, here the function name too) is evaluated in one batch instead of a token at a time,
which halved that; the constraint itself adds about 0.1 ms a token (`docs/benchmarks.md`,
2026-11-21).

`--draft models/Qwen3-0.6B-Q8_0.gguf --draft-tokens 6` (rli-server and rli) speculates with the
smaller model: it guesses the next tokens and the main model checks them in one pass. At
temperature 0 it keeps those it would have chosen itself, so replies stay the same; above it,
standard speculative sampling keeps the main model's distribution of replies (they did on every request of the
tool-call eval, `scripts/eval_tool_calls.py`). The request above then takes about 1.2 s warm on
Metal, and tool-call requests about half their time on Qwen3-8B and Qwen3-14B. The draft stops
guessing while too few of its guesses stand (Qwen3-0.6B at Swahili prose). It needs its 0.64 GB
of weights in memory, plus 0.3 GB measured for its cache and scratch on two requests (the cache
grows with use, to 0.47 GB at 4096 positions).

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
