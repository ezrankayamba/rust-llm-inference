# Verdict: rli against llama.cpp (2026-11-21)

**Score: 1.13**: on a weighted average, rli v0.3.0 runs 13% faster than llama.cpp v0.9.0.
1.00 would be level. Without one outlier (below) the score is **1.07**.

Of 64 measured cases, rli is ahead in 37, level (within 3%) in 14 and behind in 13.

The previous verdict (2026-11-19) scored 1.07, or 1.01 without the outlier, with CUDA measured on
an A100. That GPU is no longer rented, so CUDA is now the RTX 5090.

## Score by backend

Each number is rli's speed divided by llama.cpp's (above 1 = ahead).

| Backend | Prompts | Decode | Score |
|---|---:|---:|---:|
| CUDA (RTX 5090) | 1.12 | 0.97 | 1.04 |
| Metal (Apple M4) | 0.97 | 1.04 | 1.00 |
| Apple M4 CPU | 1.45 | 0.99 | 1.20 |
| x86 CPU | 1.37 | 1.26 | 1.32 |
| **Overall** | | | **1.13** |

## How the score is weighted

- The four backends count equally.
- Within a backend, prompts and decode count equally.
- Within prompts or decode, each model counts equally (on x86, each CPU type), however many
  prompt lengths or thread counts were measured for it.
- Every average is a geometric mean of the ratios, so 2x ahead and 2x behind cancel.

## The outlier

Qwen2.5-0.5B prompts on the M4 CPU are 2.3x ahead, because that file is mostly Q5_0 and Q8_0
weights, where llama.cpp has no fast kernel without BLAS. It is one of two models in that cell
and lifts the M4 CPU score from 0.95 to 1.20. Leaving it out, the overall score is 1.07.

## Ahead

- **x86, every CPU:** prompts 1.08-1.84x and decode 1.03-1.55x, except Llama-3.2-1B prompts on
  the Xeon 6975P-C. Peak memory about half of llama.cpp's (weights stay in the file mapping).
- **CUDA prompts from 8 to 256 tokens:** Mistral-7B 2-19%, Llama-3.2-1B 19-38%, Qwen3-30B-A3B
  3-55% (except 16 tokens), and Llama-3.2-1B at every length.
- **Metal:** Llama-3.2-1B at 16 tokens (38%), Mistral-7B prompts (1-4%), decode on small models
  (4-15%).

## Behind

- **Metal prompts on very small models:** SmolLM2-135M 15-21%, Qwen2.5-0.5B 13% at 64 tokens.
- **M4 CPU, Llama-3.2-1B:** prompts 7-12%, one-thread decode 12%.
- **CUDA, Mistral-7B prompts from 512 tokens:** 6-8%; both engines run at the 575 W limit.
- **CUDA decode:** 1-5% (Mistral-7B 4%, Qwen3-30B-A3B 5%, Llama-3.2-1B level).
- **x86, Llama-3.2-1B prompts on the Xeon 6975P-C:** 10% (llama.cpp's AMX path).

## Caveats

- One run per backend on 2026-11-21 (`docs/benchmarks.md`, first section), not a best of many.
  Run-to-run noise is about 5%, so a case within 3% is counted as level.
- The Mac was shared with a Docker VM. Qwen3-30B-A3B could not be measured on Metal (it paged
  from disk), so Metal's score has no MoE model; on 2026-11-16 its decode was 4% ahead.
- Someone was using the RTX 5090's PC during its run; the two rounds of each cell still agreed
  within 1% but one.
- GitHub runners are picked at random: this time two Xeons with AMX and an EPYC 7763, no Zen 4
  or Zen 5, where one-thread decode was up to 12% behind without `--repack`.
- Different weights give a different number. Counting only K-quant models on the CPU hosts
  would score lower; counting short GPU prompts more would score higher.
