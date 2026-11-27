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
| Quantization | F16/BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, NVFP4; Q4_1, Q5_0 and Q5_1 also load |
| Models | Llama family (Llama 2/3, Mistral, dense Qwen) and mixture-of-experts (Mixtral, Qwen MoE, DeepSeek V2 and V3 on the CPU, Metal and CUDA) |
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

## Docker

`ezrankayamba/rli-server` is a CPU-only image of the server (linux/amd64 and linux/arm64) that
serves one model. The model is not in the image: mount the folder that holds it and name the
file with `RLI_MODEL`.

```sh
docker run -d --name rli --restart unless-stopped -p 8080:8080 \
  -v /srv/models:/models:ro \
  -e RLI_MODEL=/models/Qwen3-4B-Q4_K_M.gguf \
  ezrankayamba/rli-server:latest

curl localhost:8080/healthz   # ok
```

Or with `compose.yaml` from this repo:
`MODEL_DIR=/srv/models RLI_MODEL=/models/Qwen3-4B-Q4_K_M.gguf docker compose up -d`.

| Variable | Flag | Default |
|---|---|---|
| `RLI_MODEL` | `--model` | `/models/model.gguf` |
| `RLI_CTX` | `--ctx` | 4096 |
| `RLI_THREADS` | `--threads` | one per core the container may use |
| `RLI_DRAFT`, `RLI_DRAFT_TOKENS` | `--draft`, `--draft-tokens` | none |
| `RLI_EXTRA_ARGS` | switches such as `--repack` | none |
| `RLI_PORT` | `--port` | 8080 |

Arguments after the image name go to `rli-server` as they are. The container runs as uid 10001,
which must be able to read the model, and reports healthy once `/healthz` answers. The weights
are memory-mapped, so give it RAM for the model file plus about 1 GB. The API has no
authentication: keep the port on a trusted network or behind a reverse proxy.

Build it yourself with `docker build -t rli-server .`, or for both architectures with
`docker buildx build --platform linux/amd64,linux/arm64 -t <name> --push .`.

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

On 60 written English and Swahili requests (`scripts/tool_cases.json`), Qwen3-14B with thinking
off and the draft below gets 59 right at a median of 1.8 s on an M4 Mac, Qwen3-8B 56 at 0.9 s,
when the client first writes spoken amounts as digits (`scripts/number_words.py`); without that
step the models get amounts wrong (elfu ishirini as 2000). Thinking did not help (58 of 60, about
21 s). See `docs/benchmarks.md`, 2026-11-21.

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

Against llama.cpp v0.9.0 on the same machines (`docs/verdict.md`, 2026-11-21), rli runs at
1.13x its speed on a weighted average, 1.07x without one outlier, ahead in 37 of 64 cases and
behind in 13. Speed ratios, rli / llama.cpp:

| Backend | Prompts | Decode |
|---|---:|---:|
| CUDA (RTX 5090) | 1.12 | 0.97 |
| Metal (Apple M4) | 0.97 | 1.04 |
| Apple M4 CPU | 1.45 | 0.99 |
| x86 CPU (two Xeons with AMX, an EPYC 7763) | 1.37 | 1.26 |

One thread, Llama-3.2-1B Q4_K_M, on GitHub-hosted runners (CPUs vary from run to run, so
compare within a row only):

| CPU | rli prompt tok/s | llama.cpp | rli decode tok/s | llama.cpp | rli peak memory | llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| Xeon 6975P-C | 126.4 | 140.9 | 16.3 | 11.8 | 862 MiB | 1661 MiB |
| Xeon 8573C | 100.8 | 71.3 | 12.7 | 10.1 | 862 MiB | 1331 MiB |
| EPYC 7763 (Zen 3) | 52.0 | 48.3 | 17.9 | 17.3 | 862 MiB | 1331 MiB |

By default the weights stay in the file mapping, which is why `rli` needs less memory. `--repack`
copies the Q4_K weights into an interleaved layout at load, as llama.cpp does: one-thread decode
on Zen 4 and Zen 5 gains up to 18%, and memory use rises to about llama.cpp's.

Where rli is behind: Metal prompts on very small models (SmolLM2-135M 15-21%), Llama-3.2-1B on
the M4 CPU (prompts 7-12%, one-thread decode 12%), long Mistral-7B prompts on the RTX 5090 (6-8%
from 512 tokens), and CUDA decode (1-5%).

Every number, the run that produced it, and the experiments that did not work are in
[docs/benchmarks.md](docs/benchmarks.md).

## Accuracy

The fast CPU path quantizes activations to 8 bits. Against 32-bit float activations, on dense models
the next-token choice agrees 96-99.6% of the time and perplexity stays within 0.3%. An exact mode
matches an independent NumPy forward pass (`scripts/ref_forward.py`) to within 4e-5 on the CPU.

NVFP4 models (NVIDIA's ModelOpt checkpoints) keep 8-bit activations too, on GPUs as well: on an
RTX 5090, Qwen3-8B-NVFP4 agrees with 32-bit activations on 99.2% of next tokens. `--fp4-acts`
runs their prompts with 4-bit activations on Blackwell's FP4 tensor cores instead, as llama.cpp
does by default: up to 1.8x faster over long prompts, but 90% agreement.

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
