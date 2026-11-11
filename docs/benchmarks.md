# Benchmark results

Measured numbers, newest first. Each section says how to reproduce it. GitHub runners vary in CPU
from run to run, so compare numbers only within one table.

## 2026-11-11: x86 CPU against llama.cpp, one thread

`Benchmark` workflow on `main` at 679b213 (run 42957871229), AMD EPYC 9V45 (AVX-512), llama.cpp
v0.9.0 built natively and restricted to AVX2. Prompt 256 tokens, decode 64, 3 reps.

| Model | Engine | Prompt tok/s | Decode tok/s | First token s | Peak anon MiB |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B Q4_K_M | rli | 172.0 | 61.2 | 1.34 | 64 |
| | llama.cpp native | 199.3 | 61.3 | 1.44 | 101 |
| | llama.cpp AVX2 only | 126.5 | 55.9 | 2.02 | 101 |
| Llama-3.2-1B Q4_K_M | rli | 77.2 | 35.4 | 2.86 | 87 |
| | llama.cpp native | 148.2 | 39.7 | 2.18 | 553 |
| | llama.cpp AVX2 only | 74.9 | 37.5 | 3.54 | 553 |

Decode is level on Qwen2.5-0.5B and 5-11% behind on Llama-3.2-1B. Before the row-kernel work below it
was 2-2.4x behind on Llama-3.2-1B. The open gap is prompt speed on AVX-512 hosts: rli has no AVX-512
kernel.

Reproduce: run the `Benchmark` workflow with no `base`.

## 2026-11-10/11: AVX2 row kernel (decode), one thread

`Benchmark` workflow with `base=main`, AMD EPYC 9V74, decode tok/s. Each step is one run comparing
the commit with its parent; prompt speed was unchanged in all of them.

| Step | Llama-3.2-1B Q4_K_M | Qwen2.5-0.5B Q4_K_M |
|---|---:|---:|
| Before (e6fcc25) | 16.5 | 33.8 |
| Independent running sums (c278b7b, run 42861186821) | 19.2 | 37.1 |
| Integer K-quant super-blocks (9ff501d, run 42955285665) | 24.1 | 39.5 |

Single-matrix decode time (`bench_matvec`, 5632 x 2048), before the first step and after the last:

| Format | Before ms | After ms |
|---|---:|---:|
| Q8_0 | 0.462 | 0.399 |
| Q4_0 | 0.527 | 0.404 |
| Q5_0 | 0.714 | 0.574 |
| Q4_K | 0.523 | 0.323 |
| Q5_K | 0.617 | 0.396 |
| Q6_K | 0.669 | 0.445 |

The two columns come from different runs on the same CPU model, so read them as approximate.

What changed:

- One running sum per row held the kernel to a multiply-add's latency per 32 weights. Blocks now go
  to four sums (32-weight formats) or stay in i32 within a super-block (K-quants).
- Each activation block's scale is its 256-value group's largest block scale times r/256, so the
  K-quant kernels multiply by r as an integer and convert to float once per super-block.

`llvm-mca -mcpu=znver3` predicted larger gains than were measured each time (Q4_K first step:
predicted 1.45x, measured 1.18x). Use it for direction only.

### Accuracy of the r/256 activation scales

`eval_quant`, 384 tokens (Mistral-7B 192), NEON kernel on an Apple M4, int8 against f32 activations.
"Before" has a free scale per 32 values.

| Model, text | KL mean before | KL mean after | Top-1 before | Top-1 after |
|---|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M, `CLAUDE.md` | 0.000723 | 0.000753 | 97.92% | 98.70% |
| Llama-3.2-1B Q4_K_M, `sampler.rs` | 0.000512 | 0.000631 | 99.22% | 98.96% |
| Qwen2.5-0.5B Q4_K_M, `CLAUDE.md` | 0.002118 | 0.002141 | 97.14% | 96.35% |
| Qwen2.5-0.5B Q4_K_M, `sampler.rs` | 0.001731 | 0.001353 | 97.92% | 97.92% |
| TinyLlama-1.1B Q4_0, `CLAUDE.md` | 0.000568 | 0.000582 | 98.70% | 99.22% |
| TinyLlama-1.1B Q4_0, `sampler.rs` | 0.000460 | 0.000480 | 99.48% | 99.22% |
| Qwen3-0.6B Q8_0, `CLAUDE.md` | 0.002200 | 0.002202 | 98.18% | 98.44% |
| Qwen3-0.6B Q8_0, `sampler.rs` | 0.001825 | 0.001683 | 97.92% | 97.66% |
| Mistral-7B Q4_K_M, `CLAUDE.md` | 0.000306 | 0.000345 | 98.44% | 98.44% |

The texts are the files at e6fcc25 (`CLAUDE.md`, `crates/models/src/sampler.rs`). Perplexity stayed
within 0.35% of f32 in every row.

## 2026-11-11: thread scaling on Apple M4, CPU only

`bench --pp 0 --tg 32 --reps 1 --threads N`, built with `--no-default-features`, at 679b213. Fastest
of five runs. The M4 has 4 performance and 6 efficiency cores.

| Threads | Llama-3.2-1B Q4_K_M tok/s | Mistral-7B Q4_K_M tok/s |
|---:|---:|---:|
| 1 | 23.5 | 4.2 |
| 2 | 44.8 | 7.6 |
| 3 | 56.6 | 8.8 |
| 4 | 70.3 | 10.1 |
| 6 | 73.7 | 11.3 |
| 10 | 77.7 | 12.9 |

Llama-3.2-1B reaches 3.0x on the 4 performance cores. Mistral-7B reaches 2.4x; each token reads
about 4.4 GB of weights, so memory speed is the likely limit (not measured). Thread scaling on x86
has not been measured: CI runners have 2 vCPUs on one core.

## 2026-11-10: sharing KV rows across query heads (`attn-group`, not merged)

Branch `attn-group` (f4a890a) converts each KV head's f16 rows to f32 once per 256 positions for all
of its query heads, and splits long contexts into spans for threads. It was slower than `main`:

- x86, one thread, decode at 2048 tokens of context (run 42707591562, EPYC 9V74): Llama-3.2-1B
  12.4 -> 11.7 tok/s, Qwen2.5-0.5B 26.5 -> 25.2.
- Apple M4, attention step alone, fastest of 40-400 runs: decode at 2048 positions 60-75% slower
  with 4 threads; a 64-token prompt chunk at 2048 positions 33-60% slower with 4 threads. Only
  one-thread decode at 8192 positions was faster (about 28%, Llama-3.2-1B shape).
