# Verdict: rli against llama.cpp (2026-11-19)

**Score: 1.07** — on a weighted average, rli runs 7% faster than llama.cpp v0.9.0.
1.00 would be level. Without one outlier (below) the score is **1.01**.

Of 136 measured cases, rli is ahead in 81, level (within 3%) in 36 and behind in 19.

## Score by backend

Each number is rli's speed divided by llama.cpp's (above 1 = ahead).

| Backend | Prompts | Decode | Score |
|---|---:|---:|---:|
| CUDA (A100) | 1.10 | 1.07 | 1.09 |
| Metal (Apple M4) | 0.95 | 1.04 | 1.00 |
| Apple M4 CPU | 1.36 | 0.96 | 1.14 |
| x86 CPU | 1.08 | 1.07 | 1.08 |
| **Overall** | | | **1.07** |

## How the score is weighted

- The four backends count equally.
- Within a backend, prompts and decode count equally.
- Within prompts or decode, each model counts equally (on x86, each CPU type), however many
  prompt lengths or thread counts were measured for it.
- Every average is a geometric mean of the ratios, so 2x ahead and 2x behind cancel.

## The outlier

Qwen2.5-0.5B prompts on the M4 CPU are 2.2x ahead, because that file is mostly Q5_0 and Q8_0
weights, where llama.cpp has no fast kernel without BLAS. It is one of two models in that
cell and lifts the M4 CPU prompt score from 0.84 to 1.36. Leaving it out, the M4 CPU scores
0.89 and the overall score is 1.01.

## Ahead

- CUDA prompts from 32 tokens on Llama-3.2-1B (2-22%) and Qwen3-30B-A3B (15-58%), and
  512-token prompts on all three models (11-18%).
- Metal prompts on Llama-3.2-1B and Mistral-7B at most lengths (up to 42%), and decode on
  four of five models.
- x86 prompts with AVX-512 (3-16%), and decode on Xeons with AMX (20-35%).

## Behind

- **Metal prompts on small models:** Qwen2.5-0.5B 14%, SmolLM2-135M 22% (last measured
  2026-11-16, before later Metal work).
- **M4 CPU, Llama-3.2-1B:** prompts 12-21%, one-thread decode 16%.
- **x86 prompts on Xeons with AMX:** 14-22% on Llama-3.2-1B.
- **x86 one-thread decode on Zen 4 and Zen 5:** up to 12% unless `--repack` is on.
- **CUDA, Mistral-7B prompts:** 2-9% at 72, 160, 192 and 224 tokens, 2-5% at 2 to 4 tokens.
- **CUDA, Qwen3-30B-A3B prompts:** 5% at 3 and 16 tokens.

## Caveats

- This is a roll-up of figures already in `docs/benchmarks.md` (its last section has the
  tables), not a fresh run. Each ratio is rli and llama.cpp measured side by side, on dates
  from 2026-11-16 to 2026-11-19.
- CUDA decode has not been compared with llama.cpp since 2026-11-15, when it was 10-16%
  ahead; the score uses 10%.
- Two Qwen2.5-0.5B figures on the M4 CPU ("2.2x ahead" on prompts, "level" on decode) are
  recorded only as ratios.
- Run-to-run noise is about 5%, so a case within 3% is counted as level.
- Different weights give a different number. Counting only K-quant models on the hosts this
  project targets first (x86 and M4 CPU) would score lower; counting long GPU prompts more
  would score higher.
