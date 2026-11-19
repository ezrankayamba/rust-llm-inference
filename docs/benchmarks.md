# Benchmark results

Measured numbers, newest first. Each section says how to reproduce it. GitHub runners vary in CPU
from run to run, so compare numbers only within one table.

## 2026-11-17: NEON Q6_K row kernel with a super-block in integers

One-thread decode on the M4 is 92% `dot_rows_q8` (`sample`), and the Q6_K output head is about
30% of a Llama-3.2-1B token: its row kernel ran 34 weights/ns against Q4_K's 60. It did a float
multiply-add per 32 weights; now each 16-weight half's i8 scale times its chunk's activation
ratio multiplies the half's dot product in i32, as `k_rows` does for Q4_K, with four float
multiply-adds per super-block (a sum holds four halves before it could overflow).

`RAYON_NUM_THREADS=1 bench_matvec`, Q6_K matvec (5632 x 2048): 0.336 ms (28.1 GB/s) before,
0.302, 0.302, 0.305 ms (31.0-31.3 GB/s) after. `bench --pp 0 --tg 96 --reps 3 --threads T`,
Llama-3.2-1B Q4_K_M, 20 s rests, two rounds: one thread 38.3 tok/s before, 40.1 and 39.3 after
(llama.cpp 45.5); four threads 84.6-84.9 before, 84.7 and 84.1 after (llama.cpp 84.9).

`eval_quant models/Llama-3.2-1B-Instruct-Q4_K_M.gguf README.md --tokens 256`: KL mean 0.000732,
max 0.0035, top-1 agreement 99.22% before; 0.000769, 0.0035, 98.44% after.

## 2026-11-17: the last layer of a prompt computes one row

llama.cpp's Metal matmul kernel (`kernel_mul_mm`, Q4_K, 64 rows x 32 tokens), ported into our
harness in place of our 32-token tile, ran a 512-token Llama-3.2-1B prompt at 1207 tok/s against
our own tile's 1206, with identical logits: its lead was not in the kernel. It was in the graph:
with one output, llama.cpp takes the output row after the last layer's attention, so that
layer's FFN runs on one token. We ran every layer in full.

Now, in the last layer, rows whose output nothing reads get their keys and values only (CPU and
Metal; a batch that returns no logits stops there), and the row the logits come from goes through
the decode kernels. On CUDA those rows also keep their queries and attention, which the kernels
take together with the keys, and skip the output projection and the FFN.

Metal, M4, `bench --pp N --tg 0 --reps 5 --gpu-layers 999` against `llama-bench -p N -n 0 -r 5`
(v0.9.0), alternating with 35 s rests, two rounds, tok/s (before: the table of the next section):

| Model | Tokens | before | after | llama.cpp |
|---|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 512 | 1336-1372 | 1414.0, 1421.7 | 1423.3, 1420.5 |
| Llama-3.2-1B Q4_K_M | 64 | 1146-1169 | 1206.0, 1200.3 | 1277.9, 1283.6 |
| Llama-3.2-1B Q4_K_M | 16 | 696-720 | 723.9, 728.0 | 786.4, 785.7 |
| Mistral-7B Q4_K_M | 512 | 196.0 | 196.0, 194.3 | 194.5, 192.2 |
| Mistral-7B Q4_K_M | 64 | 186.7-190.7 | 190.0, 191.4 | 198.8, 200.0 |
| Mistral-7B Q4_K_M | 16 | 128.0 | 129.2, 129.1 | 142.0, 142.0 |

Logits of a 500-token prompt moved by 0.003 (the last row is now computed by the f32 decode
kernels, not half tiles). Level at 512 tokens; 4-6% behind at 64 and 8-9% at 16, where the output
head (`matvec4_q6_k`, 2.1 ms on Llama-3.2-1B) is 4% and 9% of a pass.

M4 CPU, Llama-3.2-1B Q4_K_M, `bench --pp 256 --tg 0 --reps 3 --threads T` against
`llama-bench -p 256 -n 0 -r 3 -t T` (built without BLAS, Accelerate or Metal), 30 s rests, two
rounds, tok/s:

| Threads | before | after | after, `--coarse-acts` | llama.cpp |
|---:|---:|---:|---:|---:|
| 1 | 100.9 | 107.3, 107.5 | 123.8, 123.7 | 115.6, 115.7 |
| 4 | 324.6 | 345.1, 345.2 | 392.4, 392.0 | 409.3, 407.9 |

CUDA, A100, `bench --pp 512 --tg 128 --reps 5 --gpu-layers 999`, the two builds alternating,
three rounds, pp512 tok/s: Mistral-7B 4191, 4190, 4161 before and 4108 (± 260), 4243, 4239
after; Qwen3-30B-A3B 4172, 4151, 4212 and 4277, 4136 (± 314), 4168; Llama-3.2-1B too noisy to
tell (12.9k-16.9k and 14.5k-18.1k). Decode unchanged (Mistral-7B 170.5-171.4 and 170.4-170.8).
The CUDA tests pass there (152).

x86 at 98206b5, the `Benchmark` workflow (`threads=1,2`, llama.cpp v0.9.0 native build; runs
44143935896, 44143926384, 44143879584, 44143900412), Llama-3.2-1B Q4_K_M, prompt and decode tok/s:

| CPU | Threads | rli | rli `--repack` | llama.cpp |
|---|---:|---:|---:|---:|
| EPYC 9V74 (Zen 4) | 1 | 52.0 / 21.8 | 52.4 / 27.0 | 45.2 / 23.1 |
| EPYC 9V74 (Zen 4) | 2 | 99.3 / 38.2 | 98.4 / 36.8 | 88.8 / 36.3 |
| Xeon 6975P-C (AMX) | 1 | 120.2 / 16.0, 124.9 / 16.0 | 122.2 / 14.8, 127.2 / 14.9 | 138.1 / 11.8, 141.4 / 11.8 |
| Xeon 6975P-C (AMX) | 2 | 226.4 / 29.3, 233.9 / 30.3 | 223.5 / 27.9, 238.2 / 28.6 | 267.5 / 21.8, 276.5 / 22.0 |
| Xeon 8573C (AMX) | 1 | 91.1 / 11.6 | 90.3 / 11.9 | 107.1 / 9.6 |
| Xeon 8573C (AMX) | 2 | 176.9 / 22.9 | 174.3 / 23.3 | 210.2 / 18.9 |

On the AMX Xeons prompts are now 1.13-1.19x behind (1.20-1.27x at 495d66f on the 6975P-C) and
decode 1.2-1.4x ahead; on Zen 4 prompts are 12-15% ahead, and one-thread decode 6% behind
without `--repack`. Qwen2.5-0.5B is 1.5-1.8x ahead in prompts and decode on all three.

Tried on the way, on Llama-3.2-1B pp512 with Metal, none kept: the dispatch grid turned so that
neighbouring threadgroups share weight rows, as llama.cpp's is (1346-1349 tok/s against
1335-1338); weights copied into a Metal-allocated buffer instead of the wrapped mapping (1336-1341
against 1336-1337); the weight tile stored transposed and loaded plainly (1316-1339); a session
whose KV buffers had been used before (64 tokens: 1212-1215 against 1204-1209). The port of
llama.cpp's kernel gained 3% from declaring 6 KB of threadgroup memory instead of 14 KB (1246
against 1207), but our tile staging 32 columns a step to halve its 16 KB was far slower (1161-1171):
the extra barriers cost more.

## 2026-11-17: CUDA against llama.cpp again, and a 128-thread f16 GEMM tile

A100-PCIE-40GB (EPYC 7F72 host), llama.cpp v0.9.0 `-ngl 99 -p 512 -n 128 -r 3`, ours
`bench --pp 512 --tg 128 --reps 3 --gpu-layers 999`, alternating with 8 s rests, two rounds each.
At 3f55afd, tok/s:

| Model | pp512 rli | llama.cpp | tg128 rli | llama.cpp |
|---|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 10872, 12077 | 16262, 14499 | 592.1, 586.5 | 526.4, 536.2 |
| Mistral-7B Q4_K_M | 3537, 3507 | 4908, 4759 | 172.1, 171.8 | 155.9, 155.5 |
| Qwen3-30B-A3B Q4_K_M | 3922, 3911 | 4242, 4073 | 193.6, 194.4 | 171.0, 171.5 |

Decode at 2048 positions (`--pp 0 --tg 128 --depth 2048`): Llama-3.2-1B 463.8 against 513.8,
Mistral-7B 141.9 against 150.7.

`gemm_f16_64x128` now runs four warps of two row groups each (128 threads a block) where it ran
eight warps of one. `bench_cuda`, f16 GEMM of 512 tokens, us: 4096x4096 203.1 -> 181.9, 14336x4096
596.6 -> 552.4, 4096x14336 717.5 -> 631.8. Blocks of 128 rows (208.8 / 553.5 / 728.0), 256 tokens
(260.2 / 770.5 / 909.1) and 128 rows by 64 tokens (264.8 / 671.7 / 911.6) were no better, a
register cap of 96 or more was 40% slower (285.7 for 4096x4096), and a GEMM that decodes Q4_K on
the way into shared memory, with no dequantized copy, lost to dequantizing first (296.7 us against
45.4 + 203.1). Logits of a 541-token prompt are bit-identical before and after on Llama-3.2-1B and
Mistral-7B, and the CUDA tests pass. Same protocol after the change:

| Model | pp512 rli | llama.cpp | tg128 rli | llama.cpp |
|---|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 14244, 12399 | 16443, 14962 | 610.7, 607.4 | 522.8, 531.0 |
| Mistral-7B Q4_K_M | 3780, 3620 | 4561, 4657 | 172.8, 172.0 | 155.5, 155.4 |
| Qwen3-30B-A3B Q4_K_M | 3762, 3936 | 4239, 4241 | 193.7, 197.0 | 171.3, 170.8 |

Decode 11-15% ahead; prompts 1.1-1.2x behind on Llama-3.2-1B (its numbers swing by 15% from run to
run on both engines), 1.25x on Mistral-7B, 1.1x on Qwen3-30B-A3B.

Where a Mistral-7B prompt of 512 tokens goes then (`RLI_CUDA_PROF=1`, 3 reps): `gemm_f16` 57%
(504 us a launch for the 64x128 tile), `dequant` Q4_K 12.5% and Q6_K 5.8% (100 and 280 us a
matrix), `attn_scores` + `attn_pv` + `attn_softmax` 16%. Q6_K had no wide dequantizer: with a warp
per super-block, 8 weights a lane through `ld24` and one 16-byte store, a matrix takes 97 us, the
logits are bit-identical and Mistral-7B pp512 is 3860-3917 tok/s (llama.cpp 4561-4908).

## 2026-11-17: CUDA decode at long context: every read of a block issued first

Decode at 2048 positions cost 39 us a layer more than at depth 0 on Mistral-7B (llama.cpp: 6).
`RLI_CUDA_PROF=1 bench --pp 0 --tg 64 --depth D --reps 1` on the A100, us a launch:

| Model, depth | `attention_part` before | after | `attention_merge` before | after |
|---|---:|---:|---:|---:|
| Llama-3.2-1B, 192 | 21.2 | 16.5 | 13.9 | 13.9 |
| Llama-3.2-1B, 1024 | 24.4 | 20.9 | 14.8 | 12.7 |
| Llama-3.2-1B, 2048 | 35.0 | 26.0 | 16.0 | 12.3 |
| Mistral-7B, 192 | 23.3 | 15.5 | 13.0 | 11.7 |
| Mistral-7B, 1024 | 29.4 | 25.3 | 15.6 | 12.2 |
| Mistral-7B, 2048 | 45.7 | 33.4 | 19.3 | 13.1 |

(A trivial launch is about 11 us under the profiler.) `attention_part` read K, computed the
softmax, then read V in one or two rounds: three or four waits on memory in a row. Now a thread
issues its reads of K and V before the first barrier, and `attention_merge` its parts likewise;
reads that were clamped to a valid address and thrown away are skipped (6 of a thread's 8 reads of
K with heads of 64). Halving the kernel's shared memory changed nothing. Per-token logits of a
541-token prompt are bit-identical on both models and the CUDA tests pass.
`bench --pp 0 --tg 128 --depth 2048 --reps 3` against `llama-bench -p 0 -n 128 -d 2048 -r 3`,
alternating, two rounds, tok/s:

| Model | before | after | llama.cpp |
|---|---:|---:|---:|
| Llama-3.2-1B | 463.8 | 513.0, 514.0 | 522.8, 510.7 |
| Mistral-7B | 141.9 | 153.2, 153.1 | 150.4, 149.8 |

## 2026-11-17: CUDA prompt attention: tiles read in 16-byte words, a step ahead

`attn_scores` and `attn_pv` filled their shared-memory tiles a float at a time, each read followed
by its store (16 + 16 a thread in `attn_scores`; 16 + 4 per 32 positions in `attn_pv`). Now a thread
reads 4 `float4` per tile into registers before storing any, and `attn_pv` reads the next 32
positions' tiles while the current ones are multiplied. `RLI_CUDA_PROF=1 bench --pp 512 --tg 0
--reps 3` on the A100, us a launch (512 tokens):

| Model | `attn_scores` before | after | `attn_pv` before | after |
|---|---:|---:|---:|---:|
| Llama-3.2-1B | 255.3 | 202.8 | 304.4 | 181.4 |
| Mistral-7B | 354.9 | 227.9 | 365.0 | 245.4 |
| Qwen3-30B-A3B | 333.7 | 208.7 | 339.6 | 224.9 |

Logits of a 541-token prompt are bit-identical on all three and the CUDA tests pass. pp512 against
llama.cpp, alternating with 8 s rests, two rounds, tok/s: Llama-3.2-1B 12502, 15333 against 14464,
15784; Mistral-7B 3932, 3981 against 4638, 4919; Qwen3-30B-A3B 4043, 4142 against 4195, 4165.

## 2026-11-17: the 64x128 f16 GEMM tile on four warps of four row groups

A warp of `gemm_f16_64x128` now holds all 64 rows' weights for a step (four groups of 16) and
takes a quarter of the tokens, so each read of the token tile from shared memory feeds four
`mma.sync`, where it fed two. `bench_cuda` on the A100, f16 GEMM of 512 tokens, us:

| Tile (rows, tokens, warps, row groups) | 4096x4096 | 14336x4096 | 4096x14336 |
|---|---:|---:|---:|
| 64, 128, 8, 1 (to 9aac7c3) | 203.1 | 596.6 | 717.5 |
| 64, 128, 4, 2 (9aac7c3) | 181.7 | 552.4 | 631.5 |
| 64, 128, 4, 4 (now) | 168.9 | 512.4 | 597.1 |
| 64, 128, 2, 4 | 181.2 | 546.9 | 613.1 |
| 128, 128, 8, 4 | 300.6 | 770.3 | 1020.7 |
| 64, 64, 2, 4 | 195.5 | 686.6 | 682.2 |
| 64, 64, 4, 4 | 222.3 | 773.6 | 763.7 |

The register cap (80, 128, 168) made no difference to any of them, and issuing a step's reads of
both tiles before the first store did nothing either (120.3 us both ways at 256 tokens): the
kernel does not wait on global memory. Logits of a 541-token prompt are bit-identical on all three
models and the CUDA tests pass. pp512 against llama.cpp (`--reps 5`, alternating with 8 s rests,
two rounds), tok/s:

| Model | rli | llama.cpp |
|---|---:|---:|
| Llama-3.2-1B | 13923, 16784 | 15380, 16617 |
| Mistral-7B | 4230, 4138 | 4853, 4957 |
| Qwen3-30B-A3B | 4258, 4270 | 4232, 4237 |

Mistral-7B was 3520 this morning. What is left of its prompt: `gemm_f16` about 60%, `dequant`
Q4_K 14%, attention 12%.

## 2026-11-17: M4 CPU against llama.cpp at 401aeeb

Llama-3.2-1B Q4_K_M, `bench --pp 256 --tg 64 --reps 3 --threads T` against `llama-bench -ngl 0 -t T
-p 256 -n 64 -r 3` (v0.9.0, no BLAS, no Accelerate), alternating with 30 s rests, tok/s:

| Threads | Prompt rli | llama.cpp | Decode rli | llama.cpp |
|---:|---:|---:|---:|---:|
| 1 | 100.9 | 114.7 | 38.3 | 45.5 |
| 4 | 324.6 | 408.6 | 84.6 | 84.9 |

With `--coarse-acts` the one-thread prompt is 116.4 tok/s.

## 2026-11-17: Metal prompts against llama.cpp by prompt length

M4, `bench --pp N --tg 0 --reps 5 --gpu-layers 999` against `llama-bench -ngl 99 -p N -n 0 -r 5`
(v0.9.0), alternating with 35 s rests, tok/s:

| Model | Tokens | rli | llama.cpp |
|---|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 16 | 720.0 | 654.4 (± 94) |
| Llama-3.2-1B Q4_K_M | 64 | 1169.1 | 1302.7 |
| Llama-3.2-1B Q4_K_M | 512 | 1371.6 | 1464.2 |
| Mistral-7B Q4_K_M | 16 | 128.0 | 146.0 |
| Mistral-7B Q4_K_M | 64 | 190.7 | 204.7 |
| Mistral-7B Q4_K_M | 512 | 196.0 | 200.4 |

`RLI_METAL_PROF=1` (each dispatch alone in a command buffer, timed on the GPU's clock) says
where a prompt goes: `matmul_h_q4_k_t64` 74-80% and `matmul_h_q6_k_t64` 15-18% at both lengths on
both models, attention 1.2-1.4% at 64 tokens and 4.4-5% at 512, everything else under 4%.

Tile size, Llama-3.2-1B pp512 (`--reps 5`, 30 s rests, two rounds): 32-token tiles only 1203.8,
1201.5 tok/s; 64-token tiles 1338.1, 1334.8; a 128-token half tile on 256 threads (the first 128
decoding the weights, eight simdgroups multiplying) 1271.2. So the 64-token tile stays the largest.

Per token we run 64 tokens as fast as 512 (one 64-token tile either way) and llama.cpp runs 512
slower than 64, so the gap at 512 (2-7%) understates the matmul's: at 64 tokens, where attention
is small, it is 7-11%.

## 2026-11-17: the AMX kernel's float arithmetic after its tile loop: on by default

`time_parts` (branch `parts-1117`, run 44112265695, an AMX Xeon), ns per chunk of 16 rows of Q4_K:
decode 54.2, panel fill 51.9, the AVX-512 loop 37.7 per 16 tokens, the AMX loop 52.4, of which
its arithmetic alone (`settle` on sums that stay put) 27.8. On an EPYC (run 44111663158): fill
72.2, AVX-512 loop 59.9. So the fill is 13-15% of a 128-token chunk, and a repacked copy of the
weights could save no more than that.

`settle` ran after every tile step, each token's float sums read from and written to memory.
Now `run_amx` stores the integer sums of 8 chunks (16 KB for two groups of tokens, so they stay
in L1) and `settle` then runs them through with 16 tokens' sums in registers, as `run16` holds
its four. Bit-equal to AVX-512 still (`amx_block_kernel_equals_avx512_block_kernel_exactly`
passes on the runner). Xeon 6975P-C, `Benchmark` with `base`, pp256 tok/s, two rounds:

| Run | Model, threads | before | after | after, `RLI_AMX=0` |
|---|---|---:|---:|---:|
| 44113113405 (settle after the whole panel) | Llama-3.2-1B, 1 | 92.5, 94.5 | 103.1, 102.6 | |
| 44114046932 (every 8 chunks, against that) | Llama-3.2-1B, 1 | 98.1, 98.3 | 110.7, 110.9 | |
| 44114997185 (against main) | Llama-3.2-1B, 1 | 94.7, 93.1 | 118.6, 117.7 | 112.4, 111.2 |
| 44114997185 | Llama-3.2-1B, 2 | 177.9, 167.8 | 223.8, 219.2 | 212.3, 209.0 |
| 44114997185 | Qwen2.5-0.5B, 1 | 203.3, 206.5 | 232.2, 230.2 | 218.5, 222.4 |
| 44114997185 | Qwen2.5-0.5B, 2 | 379.2, 373.5 | 422.5, 412.5 | 400.9, 403.8 |

("before" ran the old tile kernel, `RLI_AMX=1`.) The tile kernel is 4-6% ahead of AVX-512 now,
so it runs wherever the CPU has AMX; `RLI_AMX=0` turns it off.

## 2026-11-17: x86 against llama.cpp at e92dfcd, four CPUs without AMX

`Benchmark` workflow, `threads=1,2`, Llama-3.2-1B Q4_K_M, llama.cpp's native build. Prompt /
decode tok/s:

| CPU (run) | Threads | rli | rli `--repack` | llama.cpp |
|---|---:|---:|---:|---:|
| EPYC 7763, AVX2 (44117045976) | 1 | 49.3 / 17.9 | 49.2 / 19.5 | 49.3 / 17.3 |
| EPYC 7763 | 2 | 96.5 / 30.9 | 96.7 / 33.5 | 96.6 / 31.9 |
| EPYC 7763 (44117084833) | 1 | 49.3 / 18.5 | 48.9 / 19.7 | 49.0 / 17.9 |
| EPYC 7763 | 2 | 96.8 / 33.6 | 96.7 / 35.0 | 96.6 / 32.3 |
| Xeon 8370C, AVX-512 (44117026280) | 1 | 67.8 / 12.5 | 68.0 / 11.9 | 61.3 / 12.4 |
| Xeon 8370C | 2 | 129.9 / 23.8 | 129.6 / 21.0 | 120.9 / 23.2 |
| EPYC 9V74, AVX-512 (44117007792) | 1 | 97.0 / 26.3 | 96.3 / 31.8 | 83.5 / 29.9 |
| EPYC 9V74 | 2 | 188.0 / 41.9 | 186.0 / 41.6 | 164.3 / 40.5 |
| EPYC 9V45, AVX-512 (44117063465) | 1 | 170.1 / 34.5 | 167.4 / 36.6 | 156.0 / 37.4 |
| EPYC 9V45 | 2 | 312.9 / 57.4 | 326.3 / 56.3 | 312.1 / 52.4 |
| EPYC 9V45 (44117103784) | 1 | 169.8 / 34.5 | 168.4 / 36.3 | 149.8 / 34.9 |
| EPYC 9V45 | 2 | 310.3 / 53.2 | 312.6 / 53.4 | 303.3 / 50.5 |

Prompts are level on AVX2 and 3-16% ahead with AVX-512. One-thread decode on Zen 4 and Zen 5 is
0-12% behind without `--repack` (26.3 against 29.9 on the 9V74) and ahead or level with it.

## 2026-11-17: Xeons with AMX against llama.cpp, the tile kernel on (33a0b92)

`Benchmark` with `cpu_flag=amx_int8`, `threads=1,2`, llama.cpp's native build (its AMX path).
Prompt / decode tok/s:

| CPU (run) | Model | Threads | rli | llama.cpp |
|---|---|---:|---:|---:|
| Xeon 6973P-C (44122568841) | Llama-3.2-1B | 1 | 111.4 / 15.5 | 129.0 / 12.0 |
| Xeon 6973P-C | Llama-3.2-1B | 2 | 202.5 / 27.0 | 256.3 / 22.5 |
| Xeon 6973P-C | Qwen2.5-0.5B | 1 | 218.3 / 30.5 | 139.7 / 19.6 |
| Xeon 6973P-C | Qwen2.5-0.5B | 2 | 395.8 / 53.4 | 276.7 / 36.9 |
| Xeon 6975P-C (44122670519) | Llama-3.2-1B | 1 | 119.1 / 16.3 | 142.6 / 12.3 |
| Xeon 6975P-C | Llama-3.2-1B | 2 | 218.6 / 29.5 | 278.5 / 21.9 |

K-quant prompts are 1.16-1.2x behind on one thread (1.27-1.35x this morning, 1.8x before that)
and 1.27x on two; decode is 1.2-1.35x ahead, and peak memory 862 MiB against 1661.

## 2026-11-17: x86 against llama.cpp after the panel fixes (3f55afd)

`Benchmark` workflow, `threads=1,2`. The AVX2 panel got the AVX-512 one's changes (3f55afd).
Prompt / decode tok/s, rli against llama.cpp's native build:

| CPU (run) | Model | Threads | rli | llama.cpp |
|---|---|---:|---:|---:|
| EPYC 7763, AVX2 (44080708394, at c521434) | Llama-3.2-1B | 1 | 45.8 / 18.2 | 48.9 / 17.9 |
| EPYC 7763 (44080708394) | Llama-3.2-1B | 2 | 89.0 / 33.1 | 96.0 / 31.8 |
| EPYC 7763 (44085078793, at 3f55afd) | Llama-3.2-1B | 1 | 49.0 / 18.3 | 48.9 / 17.5 |
| EPYC 7763 (44085078793) | Llama-3.2-1B | 2 | 94.4 / 32.7 | 95.5 / 31.9 |
| EPYC 7763 (44085078793) | Qwen2.5-0.5B | 1 | 114.1 / 32.6 | 77.9 / 28.7 |
| EPYC 7763 (44085078793) | Qwen2.5-0.5B | 2 | 200.3 / 58.6 | 153.6 / 50.7 |
| Xeon with AMX (44085060758, at 3f55afd) | Llama-3.2-1B | 1 | 109.0 / 16.0 | 138.6 / 11.8 |
| Xeon with AMX (44085060758) | Llama-3.2-1B | 2 | 209.7 / 30.6 | 282.7 / 22.8 |
| Xeon with AMX (44085060758) | Qwen2.5-0.5B | 1 | 221.0 / 32.2 | 136.4 / 20.8 |
| Xeon with AMX (44085060758) | Qwen2.5-0.5B | 2 | 403.1 / 57.7 | 264.9 / 35.3 |

On the Xeon (rli on AVX-512; `RLI_AMX` unset) Llama-3.2-1B prompts are 1.27-1.35x behind, where
they were 1.8x this morning (78.6 against 144.2). Against llama.cpp's AVX2 build there: 111.5
against 61.1.

## 2026-11-17: the AVX-512 panel: what filling it cost (c521434)

`bench_amx` on a Xeon 6975P-C (run 44077130823): a whole AMX step (load 16 rows' weights for a
chunk, load 16 tokens' activations, `tdpbsud`, store) is 17.5 ns, 4.6 of it the multiply, and the
activations' stride (32 to 8192 bytes) changes it by 1-2 ns. The block kernel spends about 107 ns
on the same work, so neither kernel was waiting on multiplies. A diagnostic on branch `amx-parts`
timed the pieces on a Xeon 8573C, ns per chunk of 16 rows:

| Type | Decode 16 rows | Fill the panel (with decode) | AVX-512 loop, 16 tokens | AMX loop, 16 tokens |
|---|---:|---:|---:|---:|
| Q4_K | 70.2 | 150.6 | 41.5 | 59.2 |
| Q5_K | 69.8 | 157.9 | 40.9 | 59.1 |
| Q6_K | 53.9 | 152.3 | 53.2 | 77.1 |
| Q5_0 | 47.7 | 141.9 | 40.0 | 59.1 |
| Q8_0 | 43.4 | 144.4 | 45.7 | 51.8 |

A fill costs what 3-4 groups of 16 tokens do, and the AMX loop's float step (`settle`) costs more
than all of the AVX-512 loop. Filling with a transpose in registers (e11e230) measured nothing by
itself (Q4_K matmul n=64 9.5-9.8 ms either way on two Xeons). Finding a Q4_K/Q5_K super-block's
scales once, not per chunk (c521434; `base=e11e230`, one thread, AMD EPYC 9V74, runs 44080003922,
44079986104 and 44079948936):

| | Before | After |
|---|---:|---:|
| Q4_K matmul n=64, ms | 9.11-9.47 | 7.01-7.86 |
| Llama-3.2-1B pp256, tok/s | 86.6-88.1 | 97.1-98.1 |
| Qwen2.5-0.5B pp256, tok/s | 185.1-194.7 | 192.1-195.1 |

## 2026-11-17: x86 prompts: results written in place, rounding without a call (7559c77)

A profile of a Qwen2.5-0.5B prompt on a Xeon 6975P-C (`Benchmark` workflow, `profile=true`,
`profile_run="--pp 256 --tg 0 --reps 3"`, run 44072280829) had the block kernel at 42% and its
weight decode at 10%; transposing each matmul's [row][token] result took 16%, and quantizing
activations 14% (`roundf` is a call per value on x86-64's baseline). Tasks now write their rows into
the output, and x86 rounds from a truncation. Run 44072998317 (`base=6f3e480`, AMD EPYC 9V74), pp256
tok/s, both alternations:

| Model | Threads | Before | After |
|---|---:|---:|---:|
| Qwen2.5-0.5B Q4_K_M | 1 | 178.1, 175.7 | 188.8, 185.4 |
| Qwen2.5-0.5B Q4_K_M | 2 | 335.2, 336.9 | 364.3, 351.5 |
| Llama-3.2-1B Q4_K_M | 1 | 84.3, 84.7 | 86.4, 86.9 |
| Llama-3.2-1B Q4_K_M | 2 | 161.7, 164.6 | 170.2, 167.1 |

So 4-8% on Qwen and 2-5% on Llama-3.2-1B there, less than the Intel profile's shares. M4,
Llama-3.2-1B pp128 on 4 threads: 331 tok/s, was 330.

The quantizer was still 12% of that prompt on the Xeon (run 44073808388): a block's largest
magnitude was one chain of 32 maxima. With eight running maxima (1df8566; `base=f84d3d3`, one
thread, `RLI_AMX=1` on both sides), pp256 tok/s:

| CPU (run) | Model | Before | After |
|---|---|---:|---:|
| Xeon 6975P-C (44074406644) | Qwen2.5-0.5B | 184.8, 184.7 | 191.0, 195.7 |
| Xeon 6975P-C | Llama-3.2-1B | 79.5, 81.0 | 80.8, 83.3 |
| Xeon Platinum 8573C (44074439768) | Qwen2.5-0.5B | 177.6, 177.9 | 183.1, 182.8 |
| Xeon Platinum 8573C | Llama-3.2-1B | 75.5, 75.0 | 76.2, 76.9 |

## 2026-11-17: an Intel AMX block kernel, exact but slower (opt-in, `RLI_AMX=1`)

llama.cpp's lead on Xeon prompts (below) is its AMX kernels. `run_amx` multiplies a chunk of 16
weight rows into 16 tokens with one `tdpbsud`, on the AVX-512 kernel's panel and float steps. It
agrees with that kernel bit for bit on a Xeon Platinum 8573C and a 6975P-C, and is slower
(`base=4e08472`, `cpu_flag=amx_int8`, one thread, 6975P-C unless noted):

| | AVX-512 | AMX, c6f7bdd (run 44069875416, 8573C) | AMX, e80d25a (run 44071369144) |
|---|---:|---:|---:|
| Q4_K matmul n=64, ms | 8.58-8.85 | 9.96 (base 8.65) | 9.70 |
| Q6_K matmul n=64, ms | 9.59-9.73 | 12.32 (base 9.72) | 11.88-12.49 |
| Llama-3.2-1B pp256, tok/s | 79.4-80.6 | 70.6 (base 83.1) | 68.6-70.7 |

e80d25a shares a weight tile between two groups of tokens and reads a step's sums after the next
step's tiles are issued, as llama.cpp does; it changed little, so the cost is elsewhere and not yet
found. Off unless `RLI_AMX=1`.

## 2026-11-17: x86 on an Intel Xeon 6975P-C, and an AVX-512 coarse path that did nothing (run 44063716638)

Same workflow at ebd2672, which gave `run16` a variant for `--coarse-acts` (a chunk's sum times the
rows' scale, a group's times the activations' once). The runner was an Intel Xeon 6975P-C (AVX-512
VNNI, AMX), 4 logical CPUs. Prompt / decode tok/s:

| Model | Threads | rli | rli `--coarse-acts` | llama.cpp native | llama.cpp AVX2 build |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B Q4_K_M | 1 | 163.0 / 32.1 | 164.5 / 31.5 | 137.5 / 20.7 | 92.1 / 15.3 |
| Qwen2.5-0.5B Q4_K_M | 2 | 301.9 / 56.4 | 304.6 / 56.6 | 272.2 / 40.1 | 178.9 / 26.5 |
| Llama-3.2-1B Q4_K_M | 1 | 78.6 / 15.8 | 77.7 / 16.1 | 144.2 / 12.4 | 61.0 / 14.9 |
| Llama-3.2-1B Q4_K_M | 2 | 156.0 / 29.5 | 152.7 / 29.7 | 285.9 / 22.8 | 120.7 / 24.5 |

The coarse path changed nothing (77.7 against 78.6), so it was removed again; it was never measured
on AMD. On this CPU Llama-3.2-1B prompts are 1.8x behind llama.cpp's native build, against 1.2x on
the EPYC 9V45 below: llama.cpp is as fast on both (144 and 148), rli is not (79 and 124). Decode is
ahead on both models, and Qwen prompts are.

## 2026-11-17: x86 against llama.cpp at 58ff301 (run 44060706489)

`Benchmark` workflow, `threads=1,2`, llama.cpp v0.9.0 built both ways. AMD EPYC 9V45 (AVX-512
VNNI), 4 logical CPUs; prompt 256 tokens, decode 64, 3 reps. Prompt / decode tok/s:

| Model | Threads | rli | rli `--repack` | llama.cpp native | llama.cpp AVX2 build |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B Q4_K_M | 1 | 262.9 / 56.2 | 263.3 / 59.0 | 187.9 / 55.4 | 118.7 / 51.2 |
| Qwen2.5-0.5B Q4_K_M | 2 | 502.6 / 88.7 | 474.6 / 83.4 | 368.0 / 71.2 | 230.3 / 67.0 |
| Llama-3.2-1B Q4_K_M | 1 | 124.1 / 33.4 | 124.4 / 35.0 | 147.9 / 35.2 | 69.6 / 34.5 |
| Llama-3.2-1B Q4_K_M | 2 | 238.9 / 45.3 | 244.2 / 45.9 | 288.3 / 45.9 | 137.3 / 45.7 |

(rli's numbers are from the tables beside the native build; beside the AVX2 build they were within
2%.) Ahead or level everywhere except Llama-3.2-1B prompts against the native build, 16-17% behind,
and its one-thread decode without `--repack`, 5% behind. First token 1.01 s against 1.63 s (Qwen)
and 2.03 against 2.16 (Llama); peak RSS 454 against 580 MiB and 864 against 1331.

## 2026-11-17: `--coarse-acts`: one activation scale per 256 values for CPU prompts (opt-in)

Off by default. `Q8Act::coarse` gives every block of a 256-value group the group's scale, as
llama.cpp's Q8_K does, and the NEON block kernel then keeps a K-quant super-block in integers
(`rows2_coarse`: a chunk's results times its 6-bit scale, and its min times the chunk's sum of
activations, one `mla` each; the float scales once per super-block). Decode keeps a scale per 32.
`RAYON_NUM_THREADS=1 bench_matvec`, 5632 x 2048, 64 tokens, ms, per-32 -> coarse: Q4_K 6.26 ->
5.43, Q5_K 7.52 -> 6.78, Q6_K 7.82 -> 6.76; the other types are unchanged (same kernel).

`bench <model> --pp 128 --tg 0 --reps 3 --threads N [--coarse-acts]`, M4 CPU, tok/s, and the
largest logit difference from the `--precise` reference over a 578-token prompt (`dump_logits`,
batched; top-1 the same in every case):

| Model | Threads | per-32 | coarse | llama.cpp | logits per-32 | coarse |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 1 | 103.1 | 119.4 | 116.4 | 0.16 | 0.32 |
| Llama-3.2-1B Q4_K_M | 4 | 328.9 | 376.4 | 411.9 | | |
| Qwen2.5-0.5B Q4_K_M | 1 | 274.5 | 283.3 | 117.2 | 0.37 | 0.78 |
| Qwen2.5-0.5B Q4_K_M | 4 | 864.8 | 882.3 | 372.7 | | |

On x86 the flag only changes the quantization: the kernels there have no coarse path (one for
AVX-512 was tried and removed, see the x86 entry of this date), so it costs accuracy for nothing.

More threads, Llama-3.2-1B Q4_K_M (`--pp 128 --tg 32 --reps 3`; llama.cpp `-t N -p 128 -n 32 -r 3`).
The M4 has 4 performance and 6 efficiency cores; rli defaults to one thread per logical core (10),
llama.cpp to 4 here:

| Threads | pp128 rli | rli `--coarse-acts` | llama.cpp | tg32 rli | llama.cpp |
|---:|---:|---:|---:|---:|---:|
| 4 | 329.5 | 376.4 | 403.5 | 84.7 | 85.0 |
| 6 | 385.3 | | 442.1 | 101.2 | 106.0 |
| 10 | 458.7 | 523.5 | 529.1 | 105.1 | 96.1 |

So with each engine's defaults rli is ahead (458.7 against 403.5, 105.1 against 84.9); at equal
thread counts its prompts are 13-18% behind without the flag and 1-7% with it.

Mistral-7B-Instruct-v0.3 Q4_K_M (`--pp 64 --tg 16 --reps 2`; llama.cpp `-p 64 -n 16 -r 2`):

| Threads | pp64 rli | rli `--coarse-acts` | llama.cpp | tg16 rli | llama.cpp |
|---:|---:|---:|---:|---:|---:|
| 4 | 45.9 | 53.9 | 58.4 | 16.2 | 16.2 |
| 10 | 67.8 | | 78.1 | 22.0 | 19.9 |

## 2026-11-16: CPU prompts: the steps between matmuls, off the one thread

`sample` of a four-thread prompt showed the workers waiting 6-8% of the time while the main thread
ran the per-token steps of a layer. `Rope::apply` found each angle (`powf`, `sin_cos`) again for
every head; it now finds them once per token. The norms and the rotation of a chunk's tokens run on
the pool. Llama-3.2-1B Q4_K_M, `bench --pp 128 --tg 32 --reps 3 --threads 4`, alternated, tok/s:

| | pp128 | tg32 |
|---|---:|---:|
| before | 317.7 | 83.7 |
| angles once per token | 325.6 | 84.8 |
| norms and rotation on the pool | 333.4 | 85.0 |

One thread: 102.8 -> 103.8 and 38.4 -> 38.4. llama.cpp: 411.9 and 85.2 on four threads.

## 2026-11-16: NEON block kernel: K-quant scale decode inlined

`k_scales` stayed a call from `rows2` (a `#[target_feature]` function is only inlined at the
compiler's discretion), returning 16 vectors through the stack once per super-block and row pair.
Marked `#[inline(always)]`. `RAYON_NUM_THREADS=1 bench_matvec`, 64 tokens, ms: Q4_K 6.55 -> 6.35,
Q5_K 7.83 -> 7.65. Llama-3.2-1B Q4_K_M, `bench --pp 128 --tg 0 --reps 3 --threads N`, alternated:
one thread 100.1 -> 102.9 tok/s (llama.cpp 116.4), four 309.3 -> 318.5 (411.9).

## 2026-11-16: NEON Q4_K/Q5_K row kernel: multipliers taken by lane

`k_rows` multiplied each sub-block's dot product by a multiplier it had stored as i32 and loaded
into every lane. The compiler saw through the store and, since the multipliers came from a 16-bit
product, broadcast and widened one again for every sub-block (`dup.4h` + `sshll`, 16 operations a
super-block; asking for the lane directly compiled to the same). Computed with widening multiplies
(`smull`), they are 32-bit from the start and `mla` takes its lane. The mins are also negated once
per super-block as a scalar. `RAYON_NUM_THREADS=1 bench_matvec`, 5632 x 2048, ms: Q4_K 0.226 ->
0.198, Q5_K 0.355 -> 0.315. Four integer sums a super-block instead of two changed nothing.

Llama-3.2-1B Q4_K_M, `bench --pp 0 --tg 64 --reps 3 --threads N`, alternated, tok/s: one thread
35.1 -> 38.2 (llama.cpp 45.6), two 66.3 -> 71.6, four 83.2 -> 83.5 (85.2). Logits and `eval_quant`
as before (the arithmetic is the same).

## 2026-11-16: NEON one-row kernels for Q8_0, Q4_0, Q5_0: four sums

These kernels were waiting on their one float sum after all: the same matrix laid out with longer
rows ran slower (`bench_matvec` with 2048 x 5632 turned to 32768 columns: Q8_0 0.296 -> 0.383 ms),
because shorter rows let consecutive rows overlap. Blocks now go into four sums in turn, in an inner
loop of constant length. Two earlier attempts changed nothing or lost 5-12% because the sums ended
up in memory: one indexed them by `block % 4`, the other passed the index to a closure.
`RAYON_NUM_THREADS=1 bench_matvec`, 5632 x 2048, ms: Q8_0 0.292 -> 0.231, Q4_0 0.293 -> 0.241,
Q5_0 0.324 -> 0.309.

Qwen2.5-0.5B Q4_K_M, `bench --pp 0 --tg 64 --reps 3 --threads N`, alternated: one thread 74.8 ->
78.8 tok/s (llama.cpp 80.5), four threads 155.7 -> 157.3 (156.4). Logits per token against the
`--precise` reference 0.35, top-1 the same; `eval_quant` KL mean 0.0017, top-1 agreement 96.0%.

## 2026-11-16: NEON one-row kernels for Q8_0, Q4_0, Q5_0: fewer vector operations per block

A block's f16 scale is loaded straight into a vector register (`ld16`; it went through a general
one), the multiply and add of its sum are one fused operation, and the Q5_0 decoder builds its
weights with `bsl` and `sri` from the inverted high bits. `RAYON_NUM_THREADS=1 bench_matvec`,
5632 x 2048, ms per matvec: Q8_0 0.298 -> 0.292, Q4_0 0.322 -> 0.293, Q5_0 0.354 -> 0.324.

Qwen2.5-0.5B Q4_K_M, `bench --pp 0 --tg 64 --reps 3 --threads N`, alternated: one thread 67.1 ->
74.9 tok/s, four threads 150.9 -> 155.9. Logits against the `--precise` reference 0.37 batched,
0.35 per token, top-1 the same; `eval_quant` KL mean 0.0017, top-1 agreement 96.0%.

Against llama.cpp built with `-DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF -DGGML_METAL=OFF`
(`llama-bench -t N -p 128 -n 64 -r 3`; the build used for the entries above had Accelerate on,
and its Qwen prompts did not change with the thread count: 511, 480, 547 tok/s at 1, 2, 4):

| Model | Threads | pp128 rli | llama.cpp | tg64 rli | llama.cpp |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1 | 270.7 | 117.2 | 74.9 | 80.5 |
| Qwen2.5-0.5B | 4 | 814.8 | 372.7 | 155.9 | 156.4 |
| Llama-3.2-1B | 1 | 99.6 | 116.4 | 35.2 | 45.6 |
| Llama-3.2-1B | 4 | 307.5 | 411.9 | 83.5 | 85.2 |

## 2026-11-16: NEON Q5_0 decoder kept in vector registers

A Q5_0 block's 32 high bits were moved byte by byte from general registers into vector ones. `q5_0`
loads the word into every lane, spreads its bytes with a table, and turns a clear bit into 0xf0
(the nibble less 16), so there is no subtraction either. `RAYON_NUM_THREADS=1 bench_matvec`, Q5_0
5632 x 2048: matvec 0.553 -> 0.354 ms, 64-token matmul 6.54 -> 5.80 ms.

Qwen2.5-0.5B Q4_K_M (two thirds of what a token reads is Q5_0), `bench --pp 128 --tg 64 --reps 3
--threads N`, alternated, tok/s:

| Threads | pp128 before | after | tg64 before | after | llama.cpp tg64 |
|---:|---:|---:|---:|---:|---:|
| 1 | 249.9 | 268.6 | 51.7 | 67.1 | 79.7 |
| 4 | 757.1 | 805.2 | 132.8 | 151.9 | 155.7 |

Logits against the `--precise` reference unchanged (0.37, top-1 the same). Unrolling the Q8_0,
Q4_0 and Q5_0 loops over two sums changed nothing (0.298 / 0.322 / 0.551 ms), so they are not
waiting on their sums.

## 2026-11-16: NEON block kernel (prompts): token groups of 16

`dot_block` decoded each row pair's weights once per group of 8 tokens; with 16 the decode is shared
by twice as many `smmla`. `RAYON_NUM_THREADS=1 bench_matvec`, 5632 x 2048, 64 tokens, ms:

| | Q8_0 | Q4_0 | Q5_0 | Q4_K | Q5_K | Q6_K |
|---|---:|---:|---:|---:|---:|---:|
| groups of 8 | 5.54 | 6.03 | 8.14 | 6.77 | 9.70 | 10.18 |
| groups of 16 | 5.25 | 5.46 | 6.54 | 6.55 | 7.83 | 8.00 |

`bench --pp 128 --tg 0 --reps 3 --threads N`, alternated: Qwen2.5-0.5B Q4_K_M (Q5_0 and Q8_0),
4 threads, 663.1 -> 760.8 tok/s (llama.cpp 552); Llama-3.2-1B Q4_K_M (its layers are mostly Q4_K),
one thread 96.6 -> 99.8 (llama.cpp 116.3), 4 threads 311.2 -> 309.5 (411.4).

What is left on Llama-3.2-1B is the kernel: with 4 threads each spends 82% of its time in it and
6% waiting, and the kernel alone scales 3.5x (6.27 -> 1.80 ms). `smmla` costs as much per multiply
as `sdot` on the M4 (0.127 ns for 32 against 0.062 ns for 16, 16 independent chains), so a pair of
rows and tokens over 32 weights is 2.25 cycles of all four vector pipes before the three float
operations that apply the scales per 32 activations (llama.cpp has one activation scale per 256).
Writing each task's results straight into the token outputs instead of transposing afterwards
gained 1%, within noise, so it was dropped.

## 2026-11-16: NEON Q6_K decode kernel: fewer instructions per chunk

The one-row Q6_K kernel is bound by instruction count, not by a chain (about 21 vector operations
per 32 weights on four pipes). A chunk's weights now take two or three operations per 16 where they
took five (`sli` inserts the high bits), and the 16 scales are widened to i32 once per super-block
and loaded into lanes, where each was moved from a general register. `RAYON_NUM_THREADS=1
bench_matvec`, Q6_K 5632 x 2048: 0.434 -> 0.380 (decode) -> 0.354 ms (scales). Precomputing a
super-block's eight float scales per token as well was slower (0.404 ms), so that stays scalar.

Llama-3.2-1B Q4_K_M, `bench --pp 0 --tg 64 --reps 3 --threads N`, alternated: one thread 31.7 ->
35.2 tok/s (llama.cpp 45.5), four threads 82.7 -> 83.1 (85.3). Logits per token against the
`--precise` reference 0.17, top-1 unchanged; `eval_quant` KL mean 0.0005, top-1 agreement 97.6%.

## 2026-11-16: NEON decode kernel for Q4_K/Q5_K in integers, on the Apple M4 CPU

The one-row NEON kernel added each 32-weight chunk's dot product to one float sum, and subtracted
its min term from one scalar, so every chunk waited on the one before it. `k_rows` now does what the
AVX2 kernel does: sub-block scale times activation ratio as an integer multiplier, i32 sums over a
super-block (two, taken in turn), one float multiply-add per super-block, and the mins as two
vectors. The block kernel's `k_scales` (14% of a one-thread prompt in `sample`) builds its scales
with vectors instead of byte by byte, and the dense SwiGLU runs on the pool (4 threads, pp128:
299.8 -> 309.6, alternated twice).

`RAYON_NUM_THREADS=1 bench_matvec`, 5632 x 2048, ms per matvec: Q4_K 0.389 -> 0.227, Q5_K 0.474 ->
0.355; Q4_0, Q5_0, Q8_0 and Q6_K unchanged (dealing their chunks over four sums indexed by block ran
5-12% slower, so they keep one).

`bench <model> --pp 128 --tg 64 --reps 2 --threads N` (CPU), before and after alternated with 15 s
rests; llama.cpp (`-DGGML_BLAS=OFF`) `llama-bench -ngl 0 -t N -p 128 -n 64 -r 2`. Llama-3.2-1B
Q4_K_M, tok/s:

| Threads | pp128 before | after | llama.cpp | tg64 before | after | llama.cpp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 85.0 | 96.4 | 116.3 | 23.4 | 31.8 | 45.5 |
| 2 | 157.9 | 178.4 | 229.2 | 45.1 | 60.5 | 81.1 |
| 4 | 273.4 | 310.7 | 411.4 | 71.2 | 82.9 | 85.3 |

Qwen2.5-0.5B Q4_K_M (mostly Q5_0 and Q8_0), 4 threads: pp128 648.7 -> 665.9, tg64 130.6 -> 130.9.

Still behind llama.cpp per core: 1.2x on prompts and 1.4x on decode with one thread. The model's
other 40% of bytes are Q6_K, whose kernel is unchanged. Logits against the `--precise` reference
(Llama-3.2-1B, 578-token prompt): 0.16 batched, 0.16 per token, top-1 unchanged; `eval_quant`
KL mean 0.0005, top-1 agreement 99.2%.

## 2026-11-16: Metal MoE decode: two-row expert matvecs, fused with what follows them

Qwen3-30B-A3B Q4_K_M, same commands and protocol as the entry below. tg48, tok/s:

| | tok/s |
|---|---|
| before | 37.9 |
| two-row `matvec_id` kernels; gate, up and SwiGLU of the chosen experts as one dispatch | 39.7 |
| the experts' down matvecs and their weighted sum as one dispatch (`matvec_id_add`) | 43.5 |
| llama.cpp | 42.0-42.3 |

The last step gained more than a saved dispatch: an expert's down matrix has 768 columns, 12
units of 64, so 20 of a simdgroup's 32 lanes had nothing to do; dealing all 8 experts' units
over the lanes gives each three.

## 2026-11-16: Metal MoE prompts: experts side by side, and 600 MB of scratch that wasn't needed

Qwen3-30B-A3B Q4_K_M (18.6 GB) on the 24 GB M4, every layer offloaded; alternating, rested.

    bench models/Qwen3-30B-A3B-Q4_K_M.gguf --pp <n> --tg 48 --reps 3 --gpu-layers 999
    llama-bench -m models/Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -p <n> -n 48 -r 3

| | before | after | llama.cpp |
|---|---|---|---|
| pp128 | 121-125 | 254 | 284-288 |
| pp512 | | 290 | fails: Insufficient Memory |
| tg48 | 37.9 | 37.9 | 42.0 |
| peak memory footprint, pp128 | 784 MB | 185 MB | 190 MB |

- **The gate and up scratch was a chunk of vocabulary-sized rows.** `max_ff` took the most rows
  of any matrix with n_embd columns, which includes the output head: 2 x 512 x 151936 floats,
  622 MB of GPU memory (525 MB on Llama-3.2-1B, whose footprint went 682 -> 218 MB). On this
  model the machine has about 15 MB to spare, so it also decided what else could run: Metal
  fails a command buffer with "Insufficient Memory" when its buffers can't all be wired, and a
  larger scratch that still fits slows everything (the same pp128 with scratch for 512 tokens:
  165 tok/s against 253). The batch scratch is now sized for the longest chunk seen.
- **A layer's experts run side by side.** Per expert there was a gather, three matmuls, the
  activation and a scatter, each waiting for the one before: 640 waits a layer for a few rows
  of work each. Now one gather of every (token, expert) pair sorted by expert, each expert's
  gate and up over its own rows with no barrier between them, one activation, every down, and
  one combine per token (`moe_combine_rows`). With the barriers put back: 85 tok/s against 153
  (measured at 64 tokens a group).
- What remains is the small tile: a chunk of 128 tokens gives an expert 8 rows on average, so
  all 17 GB of experts are decoded for 8-token tiles, at the 37 GB/s that dense 16-token
  prompts get too (llama.cpp: 41).

## 2026-11-16: Metal prompts: direct 16-weight decoders, a matmul that adds into its output

Same protocol as the decode entry below (alternating, rested). pp512, tok/s:

| model | before | after | llama.cpp |
|---|---|---|---|
| SmolLM2-135M | 7170 | 7710 | 8500-9200 |
| Qwen2.5-0.5B | 2910 | 3146 | 3375 |
| Llama-3.2-1B | 1306 | 1327 | 1418 |
| Mistral-7B | 183 | 187.5 | 196.6 |

Decode is unchanged (363, 184.6, 106.9, 21.7).

- **Q4_0, Q5_0, Q8_0, F16 and F32 tiles decode their 16 weights directly**, a word at a time,
  where they went through the 32-weight scalar decoders and an array. Q4_K_M files of small
  models hold Q5_0 and Q8_0 matrices: Qwen2.5-0.5B 2910 -> 3086, SmolLM2 7170 -> 7607.
- **`matmul_add`**: the output and down projections add into the residual stream, as decode's
  matvecs do, saving a pass over n x n_embd per projection. A whole tile adds what is in y
  after its last step (`acc * I + y`); seeding the sums with y under `if (p.add)` put the
  matrices in memory and ran prompts 3x slower (Qwen2.5-0.5B: 828 tok/s).
- The tile step is written out twice, with and without its second 32-column chunk, in place
  of a test per step, and `add`, `axpy`, `swiglu` and the cache stores take four values a
  thread. Together with the above: Llama-3.2-1B 1306 -> 1327.

- Later the same day: a Q4_K tile step reads each word once for both of its chunks (they are
  the low and high nibbles of the same bytes; `deq16_q4_k_x2`). Llama-3.2-1B pp16 672 -> 700,
  pp512 1327 -> 1340; Mistral-7B pp512 187.5 -> 189. Short prompts against llama.cpp after
  it (pp16 / pp32 / pp64): Llama-3.2-1B 700 / 950 / 1156 against 791 / 1097 / 1280, Mistral-7B
  126.5 / 160 / 187 against 142 / 186 / 196, Qwen2.5-0.5B 1208 / 1787 / 2244 against
  1422 / 2067 / 2793. A tile matmul over 8192x2048 Q4_K costs 0.13 ms plus 9.9 us a token
  (`bench_metal`-style run over distinct matrices: 0.207, 0.277, 0.445 and 0.76 ms for 8, 16,
  32 and 64 tokens); storing a thread's weights eight halves at a time changed nothing.

Where a 512-token prompt goes (skipping one kernel class at a time): the matmuls are 90% on
Llama-3.2-1B and 80% on Qwen2.5-0.5B; on SmolLM2 they are 66%, attention 25%.

## 2026-11-16: Metal decode against llama.cpp: two-row matvecs, fewer stages per layer

Same machine and builds as the entry below. The two engines alternate with 30-45 s of rest
before each run: back to back, whichever runs second inherits the other's heat (consecutive
Mistral-7B runs fell from 20.5 to 19.0 tok/s), and an animated wallpaper was enough to add
several percent of noise.

    bench <model> --pp 512 --tg 128 --reps 3 --gpu-layers 999
    llama-bench -m <model> -ngl 99 -p 512 -n 128 -r 3

| model | tg128 before | tg128 after | llama.cpp | pp512 | llama.cpp pp512 |
|---|---|---|---|---|---|
| SmolLM2-135M | 222 | 363 | 326 | 7170 | 9180 |
| Qwen2.5-0.5B | 153.5 | 185.2 | 177.9 | 2910 | 3382 |
| Llama-3.2-1B | 92.8 | 106.8 | 107.2 | 1306 | 1422 |
| Mistral-7B | 18.4-19.1 | 21.7 | 21.3 | 183 | 196 |

Decode is ahead on three models and level on Llama-3.2-1B. Prompts are 6.5-8% behind on the
two larger models, 14% on Qwen2.5-0.5B and 22% on SmolLM2 (these rested figures replace the
"level on Mistral-7B" of the entry below, which was taken back to back).

A decoded token is weight streaming plus what each dispatch adds. Llama-3.2-1B reads 799 MB a
token; at the ~100 GB/s the kernels reach that is 8.0 ms, and the token took 10.8.

- **Two rows a simdgroup, weights a word at a time.** A lane takes 64 columns (K-quants: the
  two 32-weight chunks sharing their bytes) or one block (Q4_0, Q5_0, Q8_0) of two rows, reads
  x once for both, and assembles four weights per 32-bit word (`bench_metal`, GB/s, matvecs
  back to back): Q4_K 4096x4096 71.7 -> 85-92; Q6_K 78.5 -> 89-98; Q4_0 59 -> 78, Q5_0 66 -> 86.
  One row a lane was 87 where two reach 100: x is half the loads. Values indexed by a loop (an
  array of x, a `uint4` of weights) are kept in memory and ran 10-25% slower, so the kernels
  are written without them. Four rows a simdgroup is 8% faster on the output head (Q6_K
  128256x2048: 2347 -> 2176 us) and slower on a layer's matrices, so it is used from 16384 rows.
- **Dispatches that must wait cost 5-9 us each.** `bench_metal`, Q4_K 2048x2048: 33 us after a
  barrier, 24 when it may overlap its neighbour; a small kernel (`add`) costs 1.1-2 us and an
  8-row matvec 4.1, so it is the matvec's last threads finishing and the next one's starting,
  not the hand-over. The encoder is concurrent now with a barrier before each dispatch that
  isn't marked `free`: Q, K and V run together, as do RoPE and the cache stores. A dense layer
  went from 13 stages that wait to 8: the output and down matvecs add into the residual
  stream, gate, up and the SwiGLU are one kernel (the pair kernel's two rows taken from two
  matrices), K is rotated straight into the cache, a bias is added by its matvec, and decode
  attention over one chunk of positions writes its result without the merge pass.
- **Decode attention** (144 positions, Llama-3.2-1B): 0.41 -> 0.12 ms a token with four values
  per load and the positions dealt out over the threads sharing a dim group.
- **rmsnorm** with a thread per two values: 3.7 -> 2.7 us (256 -> 1024 threads; 32 threads took
  17 us).
- **The first layer is submitted before the rest is encoded** (`Batch::start`): +0.5%.

Not kept: one row a lane, 16-byte weight loads (87 GB/s against 100), threadgroups of 1, 2, 8
or 16 simdgroups (no change), thread QoS (no change), `sleep` then poll (oversleeps by 1 ms+).

Polling for completion instead of sleeping in `waitUntilCompleted` gave Llama-3.2-1B another
3.5% on a cool machine (106.5 -> 110.2 tok/s): with a CPU core awake the GPU itself streams
faster (Q4_K 8192x2048: 84 -> 91 GB/s). Not adopted: it holds a core at 100%. llama.cpp gets
part of this from its thread pool's polling (`llama-bench --poll 0`: 105.5 tok/s, default 107,
`--poll 100` 108.3).

## 2026-11-16: Metal prompts against llama.cpp: half tiles, a 64-row matmul, matrix-unit attention

Apple M4 (24 GB), Q4_K_M files, every layer on Metal, llama.cpp v0.9.0 (cd959ba) built with
Metal and `-DGGML_BLAS=OFF`. tok/s, one run after the other on a quiet machine:

    bench <model> --pp N --tg 128 --reps 3 --gpu-layers 999
    llama-bench -m <model> -ngl 99 -p N -n 128 -r 3

| model | test | before | after | llama.cpp |
|---|---|---|---|---|
| Llama-3.2-1B | pp512 | 757.5 | 1314.6 | 1395.0 |
| Llama-3.2-1B | pp16 | 279.1 | 611.1 | 796.1 |
| Llama-3.2-1B | tg128 | 92.5 | 92.8 | 110.6 |
| Qwen2.5-0.5B | pp512 | 1689.7 | 2896.4 | 3358.3 |
| Qwen2.5-0.5B | pp16 | | 996.2 | 1447.3 |
| Qwen2.5-0.5B | tg128 | 153.3 | 153.5 | 182.3 |
| Mistral-7B | pp512 | 117.9 | 181.6 | 181.9 |
| Mistral-7B | pp16 | | 81.4 | 116.5 |
| Mistral-7B | tg128 | 19.1 | 18.4 | 21.2 |

Prompts went from 1.6-2x behind to level on Mistral-7B, 6% behind on Llama-3.2-1B and 14% on
Qwen2.5-0.5B; short prompts are 1.3-1.45x behind and decode (untouched here) 16-19%.

What moved Llama-3.2-1B pp512, in order (tok/s): half-precision tiles with f32 sums 757 -> 791;
decoding weights into registers before the barrier 839; every thread loading the same amount
(32 weights and 32 activations) and full tiles stored straight to `y` 940; 64 columns per step
973; a 64x64 tile on four simdgroups 1044; batched attention on the matrix units with an online
softmax 1192; K and V staged per 32 keys for the query heads of a KV head 1254; 512-token chunks
1284; a kernel per head size 1344 (with run-time loop bounds the compiler keeps the matrices in
memory). A 64x128 tile on eight simdgroups was slower. The later odd-chunk fix (a matrix whose
columns are an odd multiple of 32) and the tile sizes below cost about 2%.

Short prompts: one 64-token tile cost the same however few tokens it held, so a tile is now
compiled for 8, 16, 32 and 64 tokens and the smallest that fits the rest is used. Llama-3.2-1B:

| tokens | 2 | 4 | 5 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| before (matvec under 5, else a 64-token tile) | | 126.4 | 88.2 | 140.3 | 279.1 | 549.9 | 1062.5 |
| after | 95.9 | 188.5 | 235.5 | 375.1 | 622.0 | 859.1 | 1060.2 |
| llama.cpp | | 313.9 | 388.8 | 621.4 | 802.6 | 1109.2 | 1290.2 |

A pass costs about 17 ms plus 0.68 ms a token there (a decoded token is 10.8 ms).

Where the rest of pp512 goes (Llama-3.2-1B, about 390 ms): the multiplies alone are ~275 ms
(the GPU's matrix units reach about 3.6 TFLOPS), weight reads and decode ~32 ms, activation
reads ~27 ms, tile loads ~15 ms, attention ~30 ms, other kernels ~20 ms.

Accuracy, last-token logits of a 97-token prompt against `--precise` (which keeps f32 tiles and
an f32 cache): 0.0072 on Llama-3.2-1B, 0.0111 on Qwen2.5-0.5B, 0.0035 on Mistral-7B, top-1
unchanged. `--precise` batched against `ref_forward.py`: 2.98e-5.

## 2026-11-15: CUDA prompts: experts grouped per chunk, 512-token chunks

Same A100, Q4_K_M, all layers offloaded, `bench --pp 512 --tg 128 --reps 5`, each build twice;
`--pp 2048 --tg 0 --reps 2` once. tok/s. The baseline is ab28911, so the rows include the
section below.

| Model | Measure | ab28911 | Now | llama.cpp |
|---|---|---:|---:|---:|
| Llama-3.2-1B | pp512 | 9048-9459 | 12042-15164 | 14500.7 |
| Llama-3.2-1B | pp2048 | 7238 | 10012 | |
| Mistral-7B | pp512 | 2169-2174 | 3424-3479 | 4911.9 |
| Mistral-7B | pp2048 | 1726 | 2336 | |
| Qwen3-30B-A3B | pp512 | 802-803 | 3859-3879 | 4241.7 |
| Qwen3-30B-A3B | pp2048 | 712 | 2214 | |

Llama's pp512 takes 35-45 ms and varies by ±2500 tok/s from run to run. Decode (tg128) did
not change: 595-611 / 171 / 195. Still behind llama.cpp on prompts: Mistral 1.4x, Qwen3 1.1x.

Qwen3-30B-A3B pp512, step by step (each measured twice, within 0.3%):

| Step | tok/s |
|---|---:|
| One int8 matvec per (token, expert) pair, after the section below | 828-831 |
| Pairs grouped by expert (`moe_group`), `dp4a` matmul per expert (`mmq_*_id`) | 1554-1555 |
| An expert's pairs dealt out to the 8 warps in turn, not 8 to a warp | 1812-1814 |
| Q4_K stacks decoded to f16 in the block, f16 `mma.sync` (`gemm_q4_k_id`) | 2862-2865 |
| The same for Q6_K stacks (`gemm_q6_k_id`) | 3115 |
| 512-token chunks | 3932-3947 |

- **Grouping.** A 256-token chunk has 2048 pairs over 128 experts, so each expert's three
  matrices were decoded 16 times a chunk on average. `moe_group` (one block: count, prefix
  sum, scatter) lists the pairs expert by expert, and block (row tile, expert) runs that
  expert's rows against all of its pairs.
- **Token tiles of the `dp4a` kernel.** 16 / 32 / 64 pairs per pass: 1590 / 1700 / 1810
  tok/s. Smaller tiles let more blocks share a multiprocessor but decode a busy expert again;
  the profile had the kernel at 73% of the prompt either way, bound by the multiply.
- **f16 in the block.** Writing the expert stacks out as f16 first, as the dense GEMM does,
  would move 600M weights per layer and chunk through device memory (about 160 ms a prompt
  at the dequantizer's rate). Decoding into shared memory has no such traffic.
- **512-token chunks** halve the decodes per token for every matrix, dense ones included:
  pp512 Llama 12.4k -> 14.1k, Mistral 2720 -> 3536, Qwen3 3115 -> 3940. The engine's
  cancellation poll follows (`Model::prompt_slice`), only when every layer is on CUDA.

Profile of a Qwen3 prompt before the Q6_K kernel and the larger chunk (`RLI_CUDA_PROF=1`,
ms per 512 tokens): experts 78 (Q4_K, f16) + 37 (Q6_K, `dp4a`), dense GEMM 28, attention
(`attn_scores`, `attn_pv`, `attn_softmax`) 39, dequant 10, router 8.

Logits against `--precise` (largest difference; top-1 always unchanged), on a prompt of
about 660 (Llama), 820 (Mistral) and 760 (Qwen3) tokens, so two chunks: Llama 0.063 (was
0.184), Mistral 0.075 (was 0.691). Qwen3 over eight prompt lengths from about 150 to 790
tokens: 0.37-1.04, mean 0.59 (was 0.50-0.92, mean 0.66), KL lower or equal each time; its
largest differences come with expert choices that flip. `--precise` batched against
per-token on the two-chunk prompt: 1.1e-5 / 6.3e-5 / 8.1e-6. Token counts are estimated from
the 14-sentence version of the prompt (357 / 444 / 408).

## 2026-11-15: long CUDA prompts through an f16 tensor-core GEMM

llama.cpp's CUDA backend writes a quantized matrix out as f16 and multiplies it with cuBLAS for
batches of 64+ tokens. Prompt chunks of 64+ tokens (`GEMM_MIN_TOKENS`) now do the same with our
own kernels: `to_f16` converts the activations once per matmul input, `dequant_*` writes the
matrix to an f16 scratch buffer, and `gemm_f16_*` multiplies them with `mma.sync` m16n8k16
(f16 inputs, f32 sums). Same A100, Q4_K_M, all layers offloaded, `bench --pp 512 --tg 128
--reps 3`, each build twice (Llama: `--reps 10`, three times, as its 50 ms prompt is noisy);
tok/s. The baseline is ab28911.

| Model | pp512 before | pp512 now | llama.cpp |
|---|---:|---:|---:|
| Llama-3.2-1B | 9426-9807 | 12069-12598 | 14500.7 |
| Mistral-7B | 2160-2166 | 2716-2722 | 4911.9 |
| Qwen3-30B-A3B | 811-812 | 828-831 | 4241.7 |

So prompts are still behind: 1.2x, 1.8x and 5.1x. Qwen3 barely moved because 81% of its prompt
time is the experts, which still run as one int8 matvec per (token, expert) pair.

Logits of a 357-444 token prompt against `--precise` (max difference; top-1 unchanged): Llama
0.164 -> 0.064, Mistral 0.393 -> 0.070, Qwen3 1.017 -> 1.055 (its experts are still int8).
F32 and F16 weights keep the f32 matmul: on them the GEMM's f16 activations are the only
rounding there is (0.0026 on the tiny test model, where batched and per-token otherwise agree
to 1e-3).

`bench_cuda`, one matrix against 256 tokens:

| Matrix | dequant us | GEMM us | GEMM TFLOPS |
|---|---:|---:|---:|
| Q4_K 4096x4096 | 45.8 | 120.5 | 71 |
| Q4_K 14336x4096 | 146.0 | 352.1 | 85 |
| Q4_K 4096x14336 | 148.2 | 436.0 | 69 |
| Q6_K 4096x4096 | 125.9 | 121.8 | 71 |

What was tried on the way:

- **The int8 tensor-core matmul is bound by instruction count.** On Mistral pp512 (234 ms),
  removing parts in turn: the float scaling after each `mma` is about 49% of the time, tile
  loads 25%, the `mma` itself 5%. Register cap, tile shape and software pipelining each moved
  it by 6% or less. An f16 GEMM has no per-block scaling, which is why it wins.
- **Tile shapes.** Blocks of 64 rows x 64 tokens and 64 x 128 are kept; the larger one is used
  when the matmul still fills 256 blocks (`GEMM_BLOCKS_MIN`). 128x128, 64x256, 32x64, 64x32,
  32x128 and 32x32 were slower; 128x256 needs more than 48 KB of shared memory.
- **Tile copies** were half of the GEMM's time one value at a time; they now move 16 bytes
  per thread and step.
- **Q4_K dequantizer**: a lane takes a sub-block in 16-byte reads, as the int8 matvec does.
  The other types use a one-weight-per-lane decoder, which is why Q6_K above takes 2.7x as
  long as Q4_K; a wide Q6_K one is not written yet.
- **Register cap** of the second compile: 96 -> 80 (it now holds the GEMM too).
- **Kernel profile mode misleads here**: ablations timed with `RLI_CUDA_PROF=1` gave
  contradictory results for the GEMM; the numbers above are from `bench_cuda`.

Running the tests several at once also turned up a race: graph capture was in global mode, so
an allocation on another thread's `Gpu` failed while a capture was open
(`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). Capture is thread-local now.

## 2026-11-15: the CPU's share of a decoded token on CUDA

Timing the phases of a token showed the GPU idle for about 480 us of Llama-3.2-1B's 2.0 ms and
700 us of Qwen3-30B-A3B's 5.9 ms: 66 / 220 us re-queueing the graph, 100-190 us downloading the
logits, and 300-380 us outside the backend. Same A100, Q4_K_M, all layers offloaded,
`bench --pp 512 --tg 128 --reps 3` and `--pp 0 --tg 128 --depth 2048`, each build twice; tok/s.
The baseline is ab28911, so the Qwen3 rows include the section below.

| Model | Measure | ab28911 | Now | llama.cpp |
|---|---|---:|---:|---:|
| Llama-3.2-1B | tg128 | 503.6-521.6 | 607.7-608.3 | 524.4 |
| Llama-3.2-1B | tg128 at depth 2048 | 404.5-409.6 | 476.1-476.4 | 522.4 |
| Mistral-7B | tg128 | 164.5-165.9 | 171.5-172.4 | 155.4 |
| Mistral-7B | tg128 at depth 2048 | 137.7-139.9 | 141.8-141.9 | 150.9 |
| Qwen3-30B-A3B | tg128 | 154.2-158.6 | 194.3-197.2 | 171.5 |
| Qwen3-30B-A3B | tg128 at depth 2048 | 124.3-125.5 | 147.6-147.7 | 166.4 |

Prompts did not change (7754-9358 / 2175-2191 / 809-813 against 7657-7725 / 2161-2174 /
804-812).

- **No pool hand-off for an all-GPU pass** (`Model::forward_batch`). `pool().install` wakes a
  worker and blocks the caller once per token, for a pass that has no CPU work.
- **`argmax`** carried the index through a fold, which is not vectorized: about 100 us over
  128k logits. It now takes 16 running maxima, then finds the first position of the largest.
  Those two took the time outside the backend from 300 to 60 us (Llama 503 -> 557 tok/s,
  Qwen3 171 -> 180, 256-token runs).
- **Logits through page-locked memory** (`Gpu::read`, from 16k floats up), and the hidden
  state is not downloaded when the logits are: Llama 557 -> 571, Qwen3 180 -> 184.5.

What is left of the CPU's share is the graph replay: 49 us a token on Llama and 178 us on
Qwen3, of which updating the 4 nodes a layer whose arguments follow the position is 22 and
67 us (0.35 us each) and the launch call 6-12 us. Not changed.

Two things measured on the way and not acted on: capping registers for the main compile
(`maxrregcount` 32-56) moved the int8 matvecs by less than their run-to-run spread, and that
spread is large for Q6_K in `bench_cuda` (128256 x 2048, the output matrix: 196-305 us from
identical runs with identical device addresses), so its numbers there are not to be read
singly.

Checks: tests locally (150) and on the A100 (144, plus `read_returns_the_buffer_at_any_length`,
added because the first version of the page-locked download copied the buffer's length rather
than the request's and failed after a prompt); `dump_logits` int8 against f32 and `--precise`
batched against `--per-token` unchanged on all three models, from fresh output files.

## 2026-11-15: Qwen3-30B-A3B decode on CUDA: 17 launches a layer to 14

The profile (`RLI_CUDA_PROF=1`) put a decoded Qwen3 token at 17 launches a layer, 816 a token,
which at 2.9 us a replayed launch is 2.3 ms of its 6.4 ms. Same A100, Q4_K_M, 48 layers
offloaded, `bench --pp 512 --tg 128 --reps 3` and `--pp 0 --tg 128 --depth 2048`, the build at
ab28911 and the new one run alternately, twice each; tok/s.

| Model | Measure | ab28911 | Now | llama.cpp |
|---|---|---:|---:|---:|
| Qwen3-30B-A3B | tg128 | 155.1-156.2 | 178.1-178.6 | 171.5 |
| Qwen3-30B-A3B | tg128 at depth 2048 | 121.4-124.1 | 134.8-137.6 | 166.4 |
| Qwen3-30B-A3B | pp512 | 802.2-813.3 | 802.5-804.2 | 4241.7 |
| Mistral-7B | tg128 | 165.8-167.4 | 164.1-165.1 | 155.4 |
| Llama-3.2-1B | tg128 | 500.4-505.4 | 501.1-506.4 | 524.4 |

What changed, with decode measured after each step against 153-157 for ab28911:

- **Router matvec** (f32, 128 x 2048), 160-165 tok/s. A warp per row read one value per lane
  and used it, 64 times over. Issuing 8 steps' reads before using any took the launch from 19.7
  to 19.1 us in the profile (floor about 11.5); a block of 8 warps per row, each taking every
  8th step, took it to 12.3 (`bench_cuda`: 11.6 -> 5.2 us). f32/f16 matrices of up to 256 rows
  take the wide shape. `moe_topk` now copies its logits to shared memory once: 17.3 -> 16.1 us.
- **`norm_rope`**, 167-174 tok/s: Qwen3's per-head RMSNorm of queries and keys and the rotation
  were three launches and are one. No change to prompts.
- **Gate and up of the chosen experts as one launch**, 175-178 tok/s: `mvq_*_pair` takes a
  grid two deep. Prompt chunks keep two launches: one was 1.5% slower there (794-801 against
  807-813 tok/s).

One thing that cost more than it looked: giving the plain int8 matvec kernels the pair's three
extra parameters and a branch on `blockIdx.z` made them 2-5% slower per launch (Q4_K over a
prompt chunk 1457-1520 -> 1556-1562 us), which was 3.5% of Qwen3 prompts. The pair is a second
instantiation of the macro instead.

Checks: the GPU tests, with and without `RLI_CUDA_NO_MMA=1`; Qwen3 `dump_logits --precise` on
CUDA against the CPU (`--gpu-layers=0`), batched and `--per-token`: 1.0e-5 and 1.3e-5, so
`norm_rope` and the wide matvec agree with an independent path; int8 against f32 unchanged
(0.30 batched, 0.26 per token on Qwen3).

## 2026-11-15: reading before storing in the other CUDA kernels (not merged)

The section below found that the tensor-core kernels' tile loads were waiting out one memory
latency per row, and guessed the other CUDA kernels might be held back the same way. Three
tries on the same A100 say they are not; none was kept. `bench --reps 3`, Q4_K_M, tok/s.

| Kernel, change | Measure | Before | After |
|---|---|---:|---:|
| Q4_K decode matvec: all reads of 1 / 2 / 4 / 8 super-blocks, then the arithmetic | Llama-3.2-1B tg128 | 276.4 | 259.8 / 268.2 / 269.7 / 248.0 |
| | Mistral-7B tg128 | 73.6 | 67.6 / 68.7 / 68.2 / 60.1 |
| | Qwen3-30B-A3B tg128 | 96.2 | 91.3 / 92.0 / 89.8 / 83.5 |
| f32 matmul: a tile's 32 values of x read before the multiplies | Mistral-7B pp512 `--precise` | 227.6 | 163.4 |
| | Llama-3.2-1B pp512 `--precise` | not rerun (1537.1 at 95d536b) | 1270.3 |
| `dp4a` matmul (`RLI_CUDA_NO_MMA=1`): a block's tokens read before they are stored | Mistral-7B pp512 | 968.9 | 818.4 |
| | Llama-3.2-1B pp512 | 5119.4 ± 559.7 | 4804.9 ± 131.4 |

The decode variant's logits still matched `ref_forward.py` (6.7e-6, `--per-token`), and the
`dp4a` one passed `matmul_q_matches_f32_matmul`. Why these lose was not investigated. The
decode matvecs run 4 warps a block at 32-40 registers a thread, so a multiprocessor holds many
warps and has others to run while one waits; the tensor-core kernels at 128 x 128 held 8.
For decode the number to look at is arithmetic instead: about 13.6 ms a token on Mistral-7B
for 7.1e9 f32 multiply-adds, where llama.cpp's decode runs on int8 activations.

## 2026-11-15: a per-kernel profile on CUDA, rmsnorm with a thread per value, tiled prompt attention

Same A100 instance. `RLI_CUDA_PROF=1` waits for every launch and prints the time per kernel
when the device is dropped (a launch and its wait cost about 11-12 microseconds by themselves,
so read small kernels against `rope` or `swiglu`). Two things it showed:

**`rmsnorm` cost twice a trivial kernel** (23.2 microseconds a launch in the profile against
12.7 for `rope`), at 2 launches a layer: its 256 threads each read 8 values of a 2048-wide row
in turn, waiting out the memory latency for every one. With up to 1024 threads a block it is
15.4. Decode, before is b99a488 (`bench --reps 3 --pp 0 --tg 128 [--depth 2048]`, Q4_K_M, tok/s):

| Model | tg128 before | after | tg128 at 2048 before | after | llama.cpp tg128 / at 2048 |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 467.2 | 513.8 | 380.5 | 416.4 | 524.4 / 522.4 |
| Mistral-7B | 149.3 | 166.1 | 126.8 | 138.6 | 155.4 / 150.9 |
| Qwen3-30B-A3B (MoE) | 144.5 | 162.9 | 116.7 | 124.7 | 171.5 / 166.4 |

**Prompt attention read every cached row again for each query head and token** (a block per
head and token). It now runs as three launches of tiled products (`attn_scores`,
`attn_softmax`, `attn_pv`): a block holds 32 positions of K or V and 32 query rows (the query
heads sharing the KV head, for 32 / heads-per-KV-head tokens) in shared memory. pp512, 5 reps,
before is b99a488: Llama-3.2-1B 7663.5 ± 1409.9 -> 8801.2 ± 1442.4, Mistral-7B 1885.3 -> 2167.7,
Qwen3-30B-A3B 743.5 -> 814.8 (llama.cpp 14500.7, 4911.9, 4241.7). pp4: 672.4, 199.3, 193.0.

Where a 512-token prompt's time goes now (profile, share of the kernels' total):

| Kernel | Llama-3.2-1B | Mistral-7B | Qwen3-30B-A3B |
|---|---:|---:|---:|
| int8 matmuls (tensor cores) | 75.8% | 85.6% | 8.4% |
| int8 matvecs (a token's experts, one by one) | 0.4% | 0.0% | 81.4% |
| attention (3 kernels) | 16.5% | 9.6% | 5.9% |
| everything else | 7.3% | 4.8% | 4.3% |

So the dense models' prompts are bound by the tensor-core matmul, and Qwen3's by running each
(token, expert) pair as its own matvec, which decodes an expert's weights again for every token
that picked it.

`--precise`: Mistral-7B batched against `ref_forward.py` 4.8e-6 (36-word prompt); decode against
batched on the long prompts 9.8e-6 (Llama-3.2-1B), 8.2e-5 (Mistral-7B), 9.5e-6 (Qwen3-30B-A3B);
Llama-3.2-1B decode against the reference 7.6e-6. Int8 against f32, max logit difference:
0.154 / 0.144, 0.132 / 0.122, 0.302 / 0.519 (batched / decode), top-1 the same.

## 2026-11-15: decode replayed as a CUDA graph, and MoE experts in int8

Same A100 instance. One token's launches are captured into a CUDA graph the first time and
replayed after that (`Gpu::run_graph`): every launch still goes through `Batch`, which compares
its kernel, grid and arguments with the graph's launch of the same index and updates the node if
they differ (the KV write offsets, the RoPE position and the attention length: 4 nodes a
layer), then one `cuGraphLaunch` runs them. Launches that no longer match (the attention kernel
changes at 128 positions, a prompt may not want logits) capture a new graph. Replaying costs 2.8
microseconds a launch against 4.0 queued (`bench_cuda`, 256x256 matvecs), and the CPU thread no
longer paces the GPU. Layers routed on the CPU read back mid-pass and are not replayed.

MoE experts stored as a stack now run the int8 matvec (`matvec_q_id`) for their gate, up and
down matrices, in decode and in prompt chunks; they were the last f32 matvecs on the hot path.

Before is caf0996, `bench --gpu-layers N --reps 5` (3 at depth), Q4_K_M, tok/s:

| Model | tg128 before | graph | and int8 experts | tg128 at 2048 before | after | pp512 before | after | llama.cpp tg128 / at 2048 / pp512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 437.8 | 453.2 | 467.2 | 353.8 | 380.5 | 7295.6 ± 896.3 | 7663.5 ± 1409.9 | 524.4 / 522.4 / 14500.7 |
| Mistral-7B | 143.5 | 147.8 | 149.3 | 121.9 | 126.8 | 1881.4 | 1885.3 | 155.4 / 150.9 / 4911.9 |
| Qwen3-30B-A3B (MoE) | 114.9 | 122.3 | 144.5 | 96.9 | 116.7 | 357.8 | 743.5 | 171.5 / 166.4 / 4241.7 |

The dense models' last column of tg128 is a second run of the graph build (nothing in the expert
change touches them); the host was quieter for it. pp4: 743.7, 189.2, 182.2. Int8 logits against
f32 on Qwen3-30B-A3B, max difference: 0.483 batched, 0.451 through decode (0.461 / 0.267 with
f32 experts), top-1 the same; the dense models' are unchanged. `--precise` is unchanged.

## 2026-11-15: fewer launches per decoded token on CUDA, and a Q6_K matvec of wide reads

Same A100 instance. A launch costs about 4 microseconds however little it does (a 256x256
matvec, back to back), and a dense layer took 20 of them, so launches were 1.3 of
Llama-3.2-1B's 2.9 ms a token. Now 12:

- `rmsnorm`, `swiglu`, `attention` and `attention_merge` quantize their result on the way out
  when int8 matrices read it (`quant_block`), which was a `quant_x` launch each: 4 a layer.
- The int8 matvec adds into `y` on request, so the attention output and FFN down projections
  write the residual stream themselves: 2 `axpy` launches a layer.
- `rope` rotates the queries and the keys in one launch.
- One token's gate and up projections are one launch (`matvec_q_pair`: the int8 matvec's
  per-row matrix id, with the two tensors' distance as the stride).

The output projection now runs in int8 too (it was f32: 215 MB of Q6_K a token on
Llama-3.2-1B), and the Q6_K matvec reads 16 bytes at a time as the Q4_K one does: a lane takes
32 weights of one of 4 super-blocks in 10 loads where 16 weights took 24. Q6_K blocks are 210
bytes, so it reads the two aligned 16-byte words around each unaligned span and shifts
(`ld24`). `bench_cuda`, microseconds a matvec: Q6_K 4096x4096 35.9 -> 27.4 (f32 36.8), 4096x14336
76.8 (f32 113.1).

Before is d4622a4, `bench --gpu-layers N --reps 3 --pp 0 --tg 128 [--depth 2048]`, Q4_K_M, tok/s:

| Model | tg128 before | fused launches | and Q6_K | tg128 at 2048 before | after | llama.cpp tg128 / at 2048 |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 349.7 | 391.4 | 437.8 | 282.8 | 353.8 | 524.4 / 522.4 |
| Mistral-7B | 129.4 | 135.6 | 143.5 | 107.2 | 121.9 | 155.4 / 150.9 |
| Qwen3-30B-A3B (MoE) | 104.2 | 100.4 ± 7.1 | 114.9 | 89.5 | 96.9 | 171.5 / 166.4 |

The host's load average was 65-100 during these runs (other tenants), and decode here waits on
the CPU thread that queues launches, so a run can come out 10-20% low: one Llama-3.2-1B run of
the last build gave 354.2 ± 36.0. Prompts are unchanged (pp512 7295.6 ± 896.3, 1881.4, 357.8).
Int8 logits against f32, max difference on the 36-word prompt, batched / through decode:
Llama-3.2-1B 0.148 / 0.145, Mistral-7B 0.107 / 0.121, Qwen3-30B-A3B 0.461 / 0.267; top-1 the
same. `--precise` is unchanged (decode against batched: 7.2e-6, 1.7e-5, 6.7e-6).

The device stream is now one of our own, since the default stream cannot be captured into a
graph (the section above).

## 2026-11-15: decode attention split over the context on CUDA

Same A100 instance. Past 128 cached positions, one token's attention now runs as
`attention_part` (a block per KV head and 32 positions, scoring the query heads that share the
KV head together, so K and V are read once) and `attention_merge`. The one-launch kernel had a
block per query head walk the whole context. One layer's attention, microseconds
(`bench_cuda`, 16 layers' caches in turn), old -> new by cached positions:

| Heads / KV heads x head_dim | 128 | 512 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| 32 / 8 x 64 (Llama-3.2-1B) | 9.8 -> 13.2 | 37.2 -> 19.0 | 132.2 -> 36.0 | 259.8 -> 61.3 |
| 32 / 8 x 128 (Mistral-7B) | 16.8 -> 16.1 | 54.6 -> 24.9 | 201.9 -> 49.2 | 400.6 -> 89.1 |
| 32 / 4 x 128 (Qwen3-30B-A3B) | 12.0 -> 19.6 | 51.5 -> 25.9 | 191.4 -> 42.2 | 380.3 -> 75.2 |

At 16 and 64 positions the old kernel takes 5.6-8.9 and the new pair 7.4-19.1 (the merge is a
second launch, 5-7 microseconds), so the old one still runs up to 128. Whole models, before is
0a05f6a, `bench --gpu-layers N --reps 3 --pp 0 --tg 128 [--depth 2048]`, Q4_K_M, tok/s:

| Model | tg128 before | after | tg128 at 2048 before | after | llama.cpp tg128 / at 2048 |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 339.1 | 349.7 | 198.1 | 282.8 ± 17.9 | 524.4 / 522.4 |
| Mistral-7B | 129.4 | 129.4 | 70.8 | 107.2 | 155.4 / 150.9 |
| Qwen3-30B-A3B (MoE) | 106.9 | 104.2 | 54.4 | 89.5 | 171.5 / 166.4 |

With `--precise`, a 357-token prompt through decode matches `ref_forward.py` to 6.7e-6
(Llama-3.2-1B), and decode matches the batched prompt to 1.7e-5 on Mistral-7B (444 tokens) and
6.7e-6 on Qwen3-30B-A3B (408).

Tried on the way: a block per 256 positions with a thread per position (2048 positions: 34-55
microseconds, but 8-15 at 16 positions and whole-model tg128 down 4-20%), and 64 positions with
4 threads each (the same at depth, 10-26 at 64 positions). In the kept kernel a query head's 32
scores are one warp, so the softmax needs no block-wide reduction. Issuing 8 reads before using
the first, as the tensor-core loaders do, changed nothing measurable here. At 2048 positions the
part kernel is 28-38 of the microseconds and the merge the rest; the part kernel is bound by its
arithmetic (every product reads a query value from shared memory), not by memory.

## 2026-11-15: int8 decode matvec on CUDA

Same A100 instance. Decode, and prompts of under 8 tokens, now run Q4_K and Q6_K matrices over
int8 activations too (`mvq_q4_k`, `mvq_q6_k`; `--precise` keeps f32): `quant_x` as for the prompt
matmuls, then `dp4a`. A Q4_K lane takes one 32-weight sub-block, 4 super-blocks per warp step, in
four 16-byte reads: the block's 16-byte header (both f16 scales and all 12 scale bytes, so no lane
waits on another's), its 16 bytes of weights, and two 16-byte halves of activations. Q6_K lanes
take 16 weights of one of 2 super-blocks. Before is 0948384, `bench --gpu-layers N --reps 5`
(3 at depth), Q4_K_M, tok/s:

| Model | tg128 before | after | tg128 at 2048 before | after | pp4 before | after | pp512 before | after | llama.cpp tg128 / at 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 262.3 | 339.1 | 142.9 | 198.1 | 139.8 | 596.4 | 7214.8 ± 1320.0 | 6991.5 ± 1351.3 | 524.4 / 522.4 |
| Mistral-7B | 72.4 | 129.4 | 50.4 | 70.8 | 32.1 | 173.1 | 1876.4 | 1863.2 | 155.4 / 150.9 |
| Qwen3-30B-A3B (MoE) | 93.6 | 106.9 | 51.1 | 54.4 | 86.6 | 144.6 | 353.5 | 353.5 | 171.5 / 166.4 |

llama.cpp is still 1.2-1.6x ahead on decode and 2.1-3.1x at 2048 tokens of context. Qwen3's
experts are still f32 (`matvec_id`). Short prompts gain most because the f32 matmul charged a
whole 32-token tile for 4 tokens.

One matvec (`-p rli-core --example bench_cuda`, back-to-back launches over 1 GB of weights,
microseconds, f32 -> int8): Q4_K 4096x4096 53.3 -> 18.0 (524 GB/s of weights), 14336x4096
143.7 -> 41.8, 4096x14336 166.3 -> 44.8; Q6_K 4096x4096 44.8 -> 35.9. A 256x256 matvec costs
3.7, which is the launch itself. The CPU spends about 3 microseconds queueing each launch,
2.1 ms a token on Mistral-7B, overlapping the GPU's 5.3 ms.

Logits against the GPU's f32 run, a 36-word prompt, max difference (top-1 the same in all):
Llama-3.2-1B 0.147 batched, 0.151 through decode; Mistral-7B 0.104, 0.105; Qwen3-30B-A3B 0.418,
0.201. With `--precise`, against `ref_forward.py`: Llama-3.2-1B 4.3e-6 batched and 4.0e-6 per
token, Mistral-7B 3.8e-6 and 3.3e-6.

What it took, Mistral-7B tg128 (73 with f32):

- The f32 kernel's layout (`sb_*`: 8 weights a lane, 256 a step) with `dp4a` in place of the
  float products: 77. Llama-3.2-1B lost 14% and
  Qwen3 7%, so integer arithmetic was not the cost.
- One lane decoding a sub-block's scale and sharing it by shuffle: 80.5. Issuing the next step's
  reads before using this step's: 83.
- A lane per sub-block, so each lane reads its own scale bytes: 96-101.
- The same with every read 16 bytes wide and aligned, 4 reads a lane instead of about 12 of 1-4
  bytes: 129. The count of separate loads per step was the cost. This needs the tensor at a
  multiple of 16 bytes in VRAM (`Gpu::has_int8`; GGUF aligns to 32), else f32 runs.
- Not kept: 2 or 8 rows per block instead of 4, and capping registers at 40 for all kernels,
  were within noise or slower on whole models, though the cap alone took the Q6_K matvec from
  36 to 23 microseconds.

## 2026-11-15: int8 tensor-core prompt matmul on CUDA

Same A100 instance as the next three sections. On compute capability 8.0 and up, `mma_q4_k` and
`mma_q6_k` replace the `dp4a` kernels of the next section: one `mma.sync` (inline PTX)
multiplies 16 weight rows by 8 tokens over 32 columns (Q4_K) or 16 (Q6_K) for a whole warp. A
block takes 64 rows and 128 tokens, 128 columns at a time; they are a second compile of
`cuda.cu`, capped at 96 registers per thread. `bench --gpu-layers N --reps 5`, Q4_K_M, tok/s;
"dp4a" is the same build with `RLI_CUDA_NO_MMA=1`.

| Model | pp512 dp4a | pp512 tensor cores | tg128 dp4a | tg128 tensor cores | llama.cpp pp512 |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 5119.4 ± 559.7 | 7511.1 ± 1637.1 | 276.9 | 280.9 | 14500.7 |
| Mistral-7B | 968.9 | 1895.2 | 72.7 | 73.4 | 4911.9 |
| Qwen3-30B-A3B (MoE) | 347.5 | 356.8 | 97.0 | 95.0 | 4241.7 |

llama.cpp is now 1.9x, 2.6x and 11.9x ahead on prompts. Short prompts gain too (5 reps, tensor
cores / dp4a): Mistral-7B pp8 160.0 / 109.6, pp32 543.1 / 402.5, pp128 1491.7 / 901.7;
Llama-3.2-1B pp8 463.7 / 290.9, pp32 1467.0 / 1077.6, pp128 3663.6 / 3169.9. The second compile
adds 0.5-0.8 s to attaching (2.07 s -> 2.6-2.9 s for a one-token run of Llama-3.2-1B).

Last-token logits of a 99-token prompt against the GPU's f32 (`--precise`) run: Llama-3.2-1B
max 0.173, mean 0.030, KL 3.6e-4; Mistral-7B 0.126, 0.022, 4.0e-4; Qwen3-30B-A3B 0.600, 0.105,
4.2e-3; top-1 the same in each. The integer sums are exact, but the result is not bit-equal to
the dp4a kernels' (float sums in another order), and with int8 activations that moves logits by
as much as the int8 rounding itself (0.126 between the two on Mistral-7B). Other checks: the
workspace tests with `--features rli-models/cuda`, among them a new one comparing `matmul_q`
with the f32 matmul on random Q4_K and Q6_K weights (run with and without `RLI_CUDA_NO_MMA`);
Llama-3.2-1B `--precise` against `ref_forward.py`, 421 tokens, 3.8e-5 batched and 3.3e-5 per
token.

How it got there, Mistral-7B pp512 (3 reps), each step measured by compiling parts of the
kernel out. With the kernel returning at once the prompt ran at 6200-6800 tok/s, so everything
outside these matmuls is about 0.16 ms a token.

| Step | tok/s |
|---|---:|
| `dp4a` kernel (next section) | 961 |
| `mma.sync` in the same loop, 16 columns a step, 32 rows x 64 tokens | 1282 |
| 32 columns a step for Q4_K (one scale per 32 weights) | 1349 |
| 128 x 128 tiles, weights of a warp's 16 rows held in registers, 8 warps | 784 |
| the same with the loops that load tiles unrolled | 937 |
| tile loads read everything into registers, then store | 1598 |
| 64 x 128 tiles, at most 96 registers per thread | 1885 |

- The multiplies were never the limit: with the `mma.sync` replaced by an xor the time did not
  change, and 32 columns a step in place of 16 gained 5%.
- Loading tiles was. A warp that reads global memory and stores the value before the next read
  waits out one memory latency per row: loading alone took 0.69 ms a token at 128 x 128, 0.18
  once all of a thread's reads were issued before its first store. The same change made the
  other CUDA kernels slower (section above).
- Registers set how many warps a multiprocessor holds (65536 / registers / 32). Left alone the
  compiler used 128-255 a thread for these kernels; a 96 cap gave 1611 -> 1885 at 64 x 128
  (72-104 all within 1%), and the same cap applied to the whole file took f32 (`--precise`)
  prompts from 227.6 to 177.6, hence the separate compile. 16 warps a block would not launch.
- No gain: converting the sums to float with an integer add and a float add in place of
  `(float)` (identical speed and results).

## 2026-11-15: int8 prompt matmul on CUDA (`dp4a`)

Same A100 instance as the next two sections. Prompt chunks of 8+ tokens now quantize activations
to int8 once per input (`quant_x`, one scale per 32 values) and multiply Q4_K and Q6_K matrices
with `mmq_*`: a block takes 32 weight rows and 64 tokens, decodes 256 columns of weights to int8
in shared memory, and each thread runs `dp4a` over its row and its warp's 8 tokens. `--precise`,
decode, the other weight types and MoE experts keep the f32 kernels. `bench --gpu-layers N
--reps 5`, Q4_K_M, tok/s; "before" is 95d536b rerun in the same session.

| Model | pp512 before | pp512 after | tg128 before | tg128 after | llama.cpp pp512 |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 1537.1 | 5201.6 ± 250.2 | 263.9 | 260.9 | 14500.7 |
| Mistral-7B | 219.1 | 965.5 | 70.9 | 72.0 | 4911.9 |
| Qwen3-30B-A3B (MoE) | 299.4 | 347.9 | 95.5 | 93.9 | 4241.7 |

llama.cpp is now 2.8x, 5.1x and 12.2x ahead on prompts (was 9-22x). Qwen3-30B-A3B gains only
its attention matrices: the experts are still f32, one (token, expert) pair at a time.

Last-token logits of a 99-token prompt against the GPU's own f32 (`--precise`) run, with the
CPU's int8 path (AVX2) for scale. Top-1 matched in every row.

| Model | GPU int8 max / mean abs diff | KL | CPU int8 max / mean abs diff | KL |
|---|---:|---:|---:|---:|
| Llama-3.2-1B | 0.157 / 0.030 | 2.7e-4 | 0.161 / 0.027 | 6.2e-4 |
| Mistral-7B | 0.121 / 0.021 | 4.0e-4 | 0.168 / 0.030 | 9.2e-4 |
| Qwen3-30B-A3B | 0.615 / 0.110 | 3.3e-3 | 0.418 / 0.076 | 1.2e-2 |

One prompt per model, so this shows the size of the difference, not a perplexity result;
`eval_quant` runs the decode path and does not cover this. Other checks: the workspace tests with
`--features rli-models/cuda`; Llama-3.2-1B `--precise` against `ref_forward.py`, 421 tokens,
3.8e-5 batched and 3.3e-5 per token; `scripts/cuda_cpu_check` (its new `mq` mode rebuilds
weights from the tile loaders: Q4_K 1.8e-7, Q6_K 4.1e-7 relative).

The kernel source is now compiled for compute capability 6.1 (`dp4a`), so a GPU older than
Pascal (2016) fails to attach and falls back to the CPU. Not tested on one.

What is left of the gap: llama.cpp's kernels on this GPU are not `dp4a` ones. On Turing and
newer its `ggml_cuda_should_use_mmq` always picks MMQ, and MMQ there uses int8 tensor-core
instructions (read from its source, v0.9.0; `GGML_CUDA_FORCE_MMQ` changes nothing on an A100,
which is why that build measured the same). Mistral-7B at 965 tok/s is about 7e12 int8
multiply-adds a second; the A100's 6912 cores at about 1.4 GHz give `dp4a` a ceiling near 4e13
with nothing else in the loop, and this loop spends about one instruction per multiply-add
(estimated from the source, not profiled). llama.cpp's 4912 tok/s is 3.5e13.

Tried first and dropped: giving each warp several weight rows in the f32 matmul (rows x tokens
per warp, exact f32). Best was 4 rows x 32 tokens at 1518 tok/s against 1563 for one row
(Llama-3.2-1B pp512), so it was not kept.

## 2026-11-15: CUDA against llama.cpp on the A100, and batched MoE prompts

Same instance as the next section. llama.cpp v0.9.0 (cd959ba) built with `-DGGML_CUDA=ON` against
the CUDA 12.4 toolkit; `llama-bench -ngl 99 -p 512 -n 128 -r 3 -t 8`, and `-p 0 -n 128 -d 2048`.
rli: `bench --gpu-layers N` and `--pp 0 --depth 2048`, 3 reps. Q4_K_M files, tok/s, peak VRAM
from `nvidia-smi` sampled during the prompt and decode run.

| Model | Engine | Prompt | Decode | Decode at 2048 | Peak VRAM MiB |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B | rli | 1563.5 | 264.9 | 168.3 | 1291 |
| | llama.cpp | 14500.7 | 524.4 | 522.4 | 1597 |
| Mistral-7B | rli | 221.1 | 71.4 | 49.9 | 4747 |
| | llama.cpp | 4911.9 | 155.4 | 150.9 | 5005 |
| Qwen3-30B-A3B (MoE) | rli | 299.0 | 91.8 | 50.6 | 18155 |
| | llama.cpp | 4241.7 | 171.5 | 166.4 | 18603 |

llama.cpp is 9-22x ahead on prompts, about 2x on decode and 3x on decode at 2048 tokens of
context, for 3-24% more VRAM. Its prompt numbers varied by up to 7% between repetitions and
rli's by under 1%. Why it is ahead was not profiled. Things to look at first: its prompt path
(rli's matmul is a hand-written kernel that decodes each weight once per 32 tokens and has no
tiling of x in shared memory), and rli's attention at long context, where decode loses 30-45%
against llama.cpp's 0-3%.

Batched MoE prompts, the change measured in the rli rows above: a chunk's tokens are routed on
the GPU (`moe_topk` with a warp per token) and every token's 8 experts run as rows of one launch
per matrix. Before it (previous section) MoE prompts ran one token at a time.

| Qwen3-30B-A3B | Before | After |
|---|---:|---:|
| pp512 | 94.8 | 299.0 |
| pp24 (5 reps) | not measured | 242.5 ± 20.7 |
| tg128 | 93.5 | 91.8 |
| Peak VRAM MiB, decode only / with a 512-token prompt | 17995 | 18155 |

Dense models did not move (Llama-3.2-1B 1550.6 -> 1563.5 prompt, 262.2 -> 264.9 decode;
Mistral-7B 220.5 -> 221.1, 70.7 -> 71.4). Checks on this code: the workspace tests with
`--features rli-models/cuda`; Qwen3-30B-A3B `--precise` logits for a 99-token prompt within
1.5e-5 of the CPU's with 48 layers offloaded and 9.1e-6 with 24; a 309-token prompt (two chunks)
within 7.2e-6 of the same prompt run one token at a time on the GPU; Llama-3.2-1B against
`ref_forward.py` 1.0e-5 batched and 8.3e-6 per token.

Each (token, expert) pair decodes its expert's weights again. Gathering the tokens that chose an
expert into one matmul, as Metal does, would decode them once per chunk; not tried.

The same change stopped sizing the feed-forward scratch from the vocabulary matrix, which had
made a batched prompt on a dense model allocate about 2 x 150 MB it never used (computed from
the sizes, not measured before the fix).

About the host: the GPU numbers repeat within 1%, but the machine is shared (load average 8-15
from other tenants while idle, 46 of its 96 threads ours), so its CPU numbers swing by 2-10% and
the small model's decode, which is limited by the CPU thread that launches kernels, swung by
20% under load. The CPU is a Zen 2 EPYC 7F72: AVX2 only, so it says nothing about the AVX-512
kernels.

## 2026-11-15: first CUDA runs, on an A100

The CUDA backend ran on an NVIDIA GPU for the first time: a rented A100-PCIE-40GB (driver
580.178.04, NVRTC 12.1 from the image's conda), host CPU EPYC 7F72 with 46 threads available.
`bench <model> --gpu-layers N` (pp512, tg128, 3 reps) and `--pp 0 --depth 2048`, all layers
offloaded, Q4_K_M files. tok/s:

| Model | | Prompt | Decode | Decode at 2048 |
|---|---|---:|---:|---:|
| Llama-3.2-1B | v0.2.0 | 107.4 | 88.6 | 59.6 |
| | after | 1550.6 | 262.2 | 175.8 |
| | CPU only | 260.3 | 36.7 | |
| Mistral-7B | v0.2.0 | 17.2 | 17.2 | |
| | after | 220.5 | 70.7 | 49.9 |
| | CPU only | 60.0 | 12.5 | |
| Qwen3-30B-A3B (MoE) | v0.2.0 | 27.8 | 27.0 | |
| | after | 94.8 | 93.5 | 50.6 |
| | CPU only (pp256, tg64) | 21.3 | 7.1 | |

The v0.2.0 code was correct as written: `dump_logits --precise` against `ref_forward.py` gave
max |diff| 1.0e-5 with all 16 layers of Llama-3.2-1B offloaded, 2.2e-5 with 8, and 5.7e-6 on
Mistral-7B. After the changes: Llama-3.2-1B 421 tokens (two chunks) 3.8e-5 batched and 3.3e-5 with
`--per-token`; 99 tokens 1.0e-5 batched, 8.3e-6 per token, 1.3e-5 and 1.0e-5 with 8 layers;
Mistral-7B 16 tokens 4.3e-6 batched and 5.7e-6 per token; Qwen3-30B-A3B against the CPU 2.4e-5 (48 layers) and 2.1e-5
(24). The workspace tests pass with `--features rli-models/cuda`.

What changed, in order, with the run each number came from (one instance, so comparable):

| Step | Llama-3.2-1B | Mistral-7B |
|---|---:|---:|
| Prompts one token at a time (v0.2.0), pp512 | 107.4 | 17.2 |
| Batched, a lane per 32-weight chunk, 8-token tile | 151.6 | 21.3 |
| A lane per weight, 8-token tile | 958.9 | 142.5 |
| 16-token tile | 1287.1 | 180.5 |
| 32-token tile | 1491.6 | 216.3 |
| Matvec a lane per 32-weight chunk (v0.2.0), tg128 | 88.6 | 17.2 |
| A lane per weight | 89.4 | 20.2 |
| 8 weights per lane, 256 per step (`sb_*`) | 230.4 ± 52 | 65.9 |
| Decode at 2048 with those | 71.5 | 26.4 |
| Attention's value sum split over `1024 / head_dim` threads per value | 133.5 | 41.1 |
| Attention's scores a warp per position | 172.0 | 50.0 |

- With a lane per chunk, the GPU showed 100% utilization at 105 W of 250: a warp's 32 loads of x
  were 128 bytes apart and its weight loads 18. Every workload moved weight bytes at 73-88 GB/s
  whatever the model. That was not memory (the card streams about 1500 GB/s) but instructions
  per weight, which is why decoding 8 weights per lane and a block's scales once gave 2.6-3.3x.
- A 24-token prompt ran at 860 tok/s with a 16-token tile and 864 with 32.
- Qwen3-30B-A3B decode: 27.0 -> 49.4 from the matvec kernels, -> 93.5 with the experts picked on
  the GPU (`moe_topk`, one launch per matrix for the 8 experts instead of 4 per expert, no sync
  per layer). Its prompts still ran one token at a time here.

Not measured: any other NVIDIA GPU, partial offload speed, Q4_0/Q5_0/Q8_0/Q5_K/F16 speed (their
kernels are checked by the tests and `cuda_cpu_check`, not timed), and thread count's effect on
the launch path (decode of the 1B model swung by 20% while another process used the CPU).

## 2026-11-13: `--repack`, an 8-row copy of Q4_K weights for x86 decode

`Benchmark` on the PR #11 branch (llama.cpp v0.9.0, native build), one thread, Llama-3.2-1B Q4_K_M.
The flag copies the Q4_K weights (457 MiB here) at load into groups of eight interleaved rows that
a dedicated AVX2 kernel reads for one-token matvecs; prompts still read the mapping.

| CPU (run) | Engine | Prompt tok/s | Decode tok/s | First token s | Peak RSS MiB | Peak anon MiB |
|---|---|---:|---:|---:|---:|---:|
| EPYC 9V74 (43485095711) | rli | 81.1 | 25.4 | 2.97 | 864 | 91 |
| | rli --repack | 80.8 | 30.1 | 3.26 | 1322 | 549 |
| | llama.cpp | 78.4 | 29.9 | 3.67 | 1331 | 553 |
| EPYC 9V45 (43485123307) | rli | 125.8 | 33.7 | 1.97 | 864 | 91 |
| | rli --repack | 127.5 | 36.1 | 2.21 | 1322 | 549 |
| | llama.cpp | 138.9 | 36.6 | 2.31 | 1331 | 553 |
| EPYC 7763 (43485102243) | rli | 42.0 | 17.4 | 5.21 | 866 | 93 |
| | rli --repack | 42.3 | 18.9 | 5.68 | 1319 | 550 |
| | llama.cpp | 47.9 | 17.1 | 5.61 | 1330 | 553 |
| Xeon 8573C (43485115112) | rli | 68.6 | 12.6 | 3.35 | 864 | 91 |
| | rli --repack | 68.2 | 12.6 | 3.62 | 1322 | 549 |
| | llama.cpp | 70.3 | 10.3 | 4.23 | 1331 | 553 |

- Decode gains 18% on Zen 4, 7% on Zen 5, 3-9% on Zen 3 (two more EPYC 7763 runs: 15.6 -> 16.7 and
  17.5 -> 18.0) and nothing on the Xeon, for 458 MiB of memory and 0.25-0.5 s at load.
- Qwen2.5-0.5B Q4_K_M has only 28 MiB of Q4_K weights (the rest is Q5_0 and Q6_K), so it moved by
  under 2% either way.
- Greedy text was the same with and without the flag on both models (48 tokens, EPYC 9V74 and the
  Xeon), though results differ by float rounding and can diverge on a near-tie.
- Tried just before it and dropped: Q6_K's offset term precomputed per token instead of a second
  multiply per 32 weights. `llvm-mca` predicted 6% for the loop; Llama-3.2-1B decode on the EPYC
  9V74 went 25.5 -> 24.4 in both runs (43482959603, 43482996129) and was level elsewhere.

## 2026-11-13: decode on Zen 4 against llama.cpp's

Two changes to the AVX2 row kernel, `Benchmark` with `base=main`, one thread, decode tok/s (prompts
unchanged in both):

| Change (run) | CPU | Llama-3.2-1B | Qwen2.5-0.5B |
|---|---|---:|---:|
| Kernel set up once per 16 rows, PR #9 (43367028654) | EPYC 9V74 | 24.7 -> 24.8 | 40.3 -> 42.2 |
| (43367061182) | EPYC 9V45 | 32.7 -> 34.1 | 57.8 -> 61.4 |
| (43367038062) | EPYC 7763 | 17.0 -> 17.5 | 29.5 -> 31.3 |
| (43367042601) | Xeon 8573C | 12.6 -> 12.9 | 24.9 -> 25.3 |
| f16 scale loaded with `vmovd`, PR #10 (43472418497) | EPYC 9V74 | 24.9 -> 26.0 | 43.2 -> 43.5 |
| (43472451322) | EPYC 9V45 | 32.4 -> 33.0 | 56.6 -> 57.6 |
| (43472427574) | EPYC 7763 | 17.8 -> 18.1 | 32.3 -> 32.6 |
| (43472435357) | Xeon 6975P-C | 15.7 -> 16.1 | 31.4 -> 31.4 |

What the measurements behind them showed (scratch branches, deleted):

- A row cost a fixed 11-14 ns on the EPYC 9V74 on top of about 5.2 ns per 256 Q4_K weights, 7.8
  for Q6_K and 6.9 for Q8_0 (run 43363411780). In RAM instead of cache the per-weight cost rose 8%.
- `perf annotate` of decode on the EPYC 9V74 (run 43471675407): the Q6_K loop took 42% of the
  kernel's samples for 33% of Llama-3.2-1B's weights, and a third of that sat on a `vpinsrw` the
  compiler made of the scale's load and the three instructions after it. Hence PR #10.
- llama.cpp's own decode profile on the same machine (run 43470015824): 53% in
  `ggml_gemv_q4_K_8x8_q8_K` and 31% in `ggml_vec_dot_q6_K_q8_K`. The first runs on a copy of the
  Q4_K weights interleaved eight rows at a time, made at load (`repack_q4_K_to_q4_K_8_bl`), which
  is most of its 553 MiB against rli's 91. Per weight it spent about 20% less than rli on Q4_K and
  about 27% less on Q6_K, by these shares and the tok/s of the comparison below.
- `llvm-mca -mcpu=znver4` put the Q4_K loop at 18 cycles per 256 weights, which is what the EPYC
  measured, with the four vector pipes as the limit.

Tried and dropped, no gain:

- VNNI `vpdpwssd` in the Q4_K loop (256-bit): 3% in the loop on the EPYC 9V74, 0-4% slower on the
  EPYC 9V45, 7-13% faster in cache on the Xeon 6975P-C (runs 43367018119, 43366525353, 43366520289).
- Decoding the Q4_K scales in 8 fewer instructions per super-block: no change on four CPUs (runs
  43468821016 to 43468864401).
- Working the multipliers out one super-block ahead: more instructions and slower in `llvm-mca`;
  not run.

Left: Llama-3.2-1B decode on the EPYC 9V74 is at about 26 tok/s against llama.cpp's 30. An idea
not tried: Q6_K's offset of 32 costs a second `maddubs` per 32 weights, which could come from the
activations once per matvec.

## 2026-11-12: x86 CPU against llama.cpp after the 128-token chunk, one thread

`Benchmark` workflow on `main` at f8057bd, llama.cpp v0.9.0 built natively. Prompt 256 tokens,
decode 64, 3 reps. One run per CPU, so compare within a row only.

| CPU (run) | Model | rli prompt | llama.cpp prompt | rli decode | llama.cpp decode |
|---|---|---:|---:|---:|---:|
| EPYC 7763, AVX2 only (43277552926) | Qwen2.5-0.5B | 99.4 | 77.7 | 28.8 | 28.3 |
| | Llama-3.2-1B | 42.7 | 48.1 | 17.1 | 16.5 |
| EPYC 9V74 (43277586143) | Qwen2.5-0.5B | 171.8 | 140.8 | 40.3 | 36.3 |
| | Llama-3.2-1B | 82.1 | 79.4 | 24.5 | 30.1 |
| EPYC 9V45 (43277578138) | Qwen2.5-0.5B | 259.0 | 186.4 | 55.6 | 55.9 |
| | Llama-3.2-1B | 132.0 | 137.7 | 33.2 | 36.4 |
| Xeon 6975P-C (43277570337) | Qwen2.5-0.5B | 158.4 | 130.6 | 29.8 | 19.7 |
| | Llama-3.2-1B | 77.9 | 143.0 | 15.2 | 11.4 |

All tok/s. Peak anonymous memory: rli 67-93 MiB, llama.cpp 101-882 MiB.

Where rli is still behind: Llama-3.2-1B prompts on the Xeon (1.8x), on the EPYC 7763 (11%) and on
the EPYC 9V45 (4%), and Llama-3.2-1B decode on the two Zen 4 CPUs (9-19%). On the Xeon,
llama.cpp's native build reports `AMX_INT8 = 1` and runs matmuls on AMX tiles
(`ggml/src/ggml-cpu/amx/mmq.cpp`: Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, weights repacked at
load, hence its 882 MiB); its AVX2 build manages 60.4 tok/s there. rli has no AMX kernel.

## 2026-11-12: CPU prompts in chunks of 128 tokens

`CPU_CHUNK` 64 -> 128 (PR #8). `Benchmark` with `base=main` on scratch branches, one thread, prompt
of 256 tokens; decode did not change.

| CPU (run) | Chunk | Llama-3.2-1B prompt tok/s | Qwen2.5-0.5B prompt tok/s |
|---|---|---:|---:|
| EPYC 9V74 (43274603182) | 64 -> 128 | 65.8 -> 81.5 | 140.5 -> 171.3 |
| EPYC 9V45 (43274641028) | 64 -> 256 | 100.5 -> 127.4 | 206.7 -> 237.0 |
| Xeon 8573C (43274633974) | 64 -> 256 | 68.0 -> 88.4 | 141.2 -> 163.9 |
| Xeon 6975P-C (43274648611) | 64 -> 256 | 66.5 -> 81.4 | 139.1 -> 145.8 |
| EPYC 9V74 (43275713291) | 128 -> 256 | 82.2 -> 84.7 | 173.0 -> 166.3 |
| Xeon 6975P-C (43275721192) | 128 -> 256 | 76.8 -> 81.7 | 154.2 -> 151.2 |
| EPYC 7763 (43275729603) | 128 -> 256 | 41.9 -> 43.7 | 96.7 -> 96.2 |
| EPYC 9V45 (43275744600) | 128 -> 256 | 135.7 -> 140.0 | 279.2 -> 263.3 |

The x86 block kernels decode each group of weight rows once per chunk, which the profile below put
at 37% of prompt time, so more tokens per chunk spread it thinner. 256 helped one model and hurt
the other, so 128. On the Apple M4 (NEON), prompts were the same at 64, 128 and 256 (Llama-3.2-1B
45.6, 46.4, 46.0 tok/s on one thread), and peak memory for a 256-token prompt rose by 1.6 MB.

Ruled out on the way: matmul speed does not depend on the matrix shape (`bench_matvec` at 8192 x
2048, 2048 x 8192 and 2048 x 2048 stayed within 10% of 5632 x 2048 in GOP/s on four CPUs), apart
from a fixed cost per call that shows on small matrices (512 x 2048: about 30% lower).

## 2026-11-11: prefetching rows on Intel (decode), one thread

`Benchmark` workflow with `base=main` on the PR #7 branch (run 43075955195), Intel Xeon 6975P-C.

| Model | Decode before tok/s | Decode after tok/s | Prompt tok/s |
|---|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 9.2 | 15.8 | 69 (unchanged) |
| Qwen2.5-0.5B Q4_K_M | 17.6 | 30.8 | 145 (unchanged) |

This answers the open question in the next section. Decode reads every weight once per token, and
on Intel runners the one-row kernel slowed down once the weights no longer fit in cache. Scratch
branch `intel-probe` (deleted) ran `bench_matvec` with 400000 rows instead of 5632 (run 43064994879,
Xeon 6975P-C against run 43064986231, EPYC 9V74):

| Format | Xeon, 5632 rows GB/s | Xeon, 400000 rows GB/s | EPYC, 5632 rows GB/s | EPYC, 400000 rows GB/s |
|---|---:|---:|---:|---:|
| Q8_0 | 23.2 | 11.8 | 30.7 | 30.6 |
| Q4_K | 17.1 | 5.5 | 20.3 | 18.9 |
| Q6_K | 18.4 | 11.9 | 21.4 | 21.7 |

llama.cpp (native build) was slow on the same Xeon: Llama-3.2-1B decode 11.5 tok/s against rli's
8.7, where on the EPYC 9V74 it was 30.3 against 25.0. Why the Xeon's own prefetching does not keep
up was not investigated.

The fix asks for the bytes 1152 ahead of each row before the row kernel runs, on Intel CPUs only.
Distances tried on Xeon 6973P-C and 6975P-C (runs 43070276411, 43070287058, 43073436141,
43073570890), Llama-3.2-1B decode tok/s: off 8.5, 1152 bytes 14.8, 2304 14.7, 4608 10.2, 9216 11.9,
18432 12.1; starting at the next row instead 14.1. With two threads on the runner's one core,
Llama-3.2-1B went 15.8 -> 16.4 and Qwen2.5-0.5B 27.6 -> 32.4.

The same prefetch made AMD slower, so it is off there: EPYC 9V74 decode 25.0 -> 20.3 at 1152 bytes
(run 43068339866), EPYC 9V45 33.9 -> 32.3, EPYC 7763 17.1 -> 16.3. With the vendor check, six
`base=main` runs on those three CPUs showed no change.

Not measured: Intel generations before Granite Rapids (the Xeon 8370C and 8573C runners showed the
same slow decode, 10-11 tok/s, but no run with the prefetch landed on them), and the NEON kernel.

## 2026-11-11: AVX-512 VNNI block kernel (prompts), one thread

`Benchmark` workflow with `base=main` on eebd36d (run 42961812565), AMD EPYC 9V74.

| Model | Prompt before tok/s | Prompt after tok/s | Decode tok/s |
|---|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 51.9 | 67.1 | 25.2 (unchanged) |
| Qwen2.5-0.5B Q4_K_M | 119.4 | 142.6 | 40.6 -> 41.1 |

64-token matmul (`bench_matvec`, 5632 x 2048): Q8_0 15.2 -> 8.9 ms, Q4_0 12.3 -> 9.2, Q5_0 12.3 ->
9.0, Q4_K 12.6 -> 9.2, Q5_K 12.3 -> 9.3, Q6_K 16.1 -> 9.8.

The kernel takes 16 rows per panel and uses `vpdpbusd`; CPUs without AVX-512 VNNI keep the AVX2
kernel. `llvm-mca -mcpu=znver4` put its inner loop at 2.2-2.5x the AVX2 one per row; the measured
matmul gain was 1.3-1.7x.

A perf profile of a 256-token prompt after the change (Llama-3.2-1B, run 43050339226, EPYC 9V74,
`profile=true`, `profile_run="--pp 256 --tg 0 --reps 6"`, with the fill and inner loop kept out of
line): inner loop 39%, panel fill 24%, block decode 13%, `Q8Act::new` and `roundf` 7%.

Transposing the panel's rows in registers instead of through memory (PR #6, not merged) did not
recover the fill's share: Xeon Platinum 8573C prompt +1% and +4% (run 43050974549); a slower EPYC
9V74 instance -4% and 0%, with the Q4_K matmul 5% slower (run 43052040901). The fill is where a
matmul first reads its weights, so its share is probably memory, not the interleaving. Not
measured.

Open question from run 43050974549: on the Xeon Platinum 8573C, Llama-3.2-1B decoded at 10.1 tok/s
against about 25 on the EPYC 9V74, while prompts ran at 60 tok/s against 52 (AVX2 kernel on both).
The Q4_K matvec was only 1.4x slower (0.445 ms against 0.322), so the row kernel alone does not
explain it. Answered above: the weights do not fit in cache during decode.

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

## 2026-11-17: a few-token Metal matmul (`matmul_few_*`)

llama.cpp does not run its tile kernel on 3 to 16 tokens: `kernel_mul_mv_mma_gen` stages nothing
in threadgroup memory, so a step has no barrier. Ours padded those prompts into 8- and 16-token
tiles. `matmul_few_*` takes that layout (16 rows by 8, 16 or 24 tokens a threadgroup, 8
simdgroups sharing the columns, f32) for a rest of up to 24 tokens. M4, Metal, all layers,
`bench <model> --pp N --tg 0 --reps 5 --gpu-layers 999` against `llama-bench -p N -n 0 -r 5`,
before, after and llama.cpp in turn with 20 s rests; prompt tok/s:

| Tokens | Llama-3.2-1B before | after | llama.cpp | Mistral-7B before | after | llama.cpp |
|---|---|---|---|---|---|---|
| 4 | 220.5 | 310.4 | 317.5 | 42.5 | 58.2 | 61.8 |
| 8 | 443.8 | 606.4 | 603.6 | 84.5 | 112.9 | 115.8 |
| 16 | 727.0 | 866.0 | 787.8 | 129.2 | 144.5 | 142.6 |
| 20 | 624.4 | 721.3 | 684.7 | 104.6 | 115.2 | 117.6 |
| 24 | 749.6 | 863.1 | 820.1 | 124.6 | 137.5 | 141.5 |
| 64 | 1206.1 | 1206.2 | 1282.9 | 192.2 | 192.1 | 200.1 |
| 88 | 1065.3 | 1106.4 | 1245.6 | 167.3 | 171.9 | 184.4 |
| 512 | 1429.5 | 1429.8 | 1433.9 | 195.2 | 194.7 | 193.4 |

Sizes tried: 8 simdgroups and 16 are level, 4 is 13-30% slower at 16 tokens; 8 rows a
threadgroup instead of 16 is 14-18% slower at 16 tokens, 32 rows the same as 16; four token
tiles (32 tokens) ran 800 and 127 tok/s where the 32-token tile kernel runs 992 and 164.
Logits of a 12-token prompt are within 0.009 of the half-precision tiles', and under `--precise`
within 3e-5 of the CPU's.

## 2026-11-17: a Metal matmul's rest runs beside its whole tiles

The rest of a matmul (after whole 64-token tiles) writes other rows of the output, so its
dispatch no longer waits for the tiles'. It then reads weights the tiles have just read. Same
runs as above (before, after, llama.cpp in turn, 20 s rests, 5 reps);
"before" is a962da6, so the 72- to 200-token rows include the few-token kernel on their rest
(88 tokens: 1066 at a962da6, 1106 with the few-token kernel, 1204 with this).

| Tokens | Llama-3.2-1B a962da6 | now | llama.cpp | Mistral-7B a962da6 | now | llama.cpp |
|---|---|---|---|---|---|---|
| 40 | 768.4 | 767.4 | 815.8 | 122.5 | 122.6 | 123.2 |
| 72 | 1054.0 | 1192.4 | 1004.3 | 168.9 | 187.6 | 152.3 |
| 88 | 1066.4 | 1204.1 | 1236.0 | 167.2 | 183.2 | 184.2 |
| 100 | 981.0 | 1063.0 | 1099.7 | 151.5 | 160.1 | 159.0 |
| 200 | 1330.8 | 1403.6 | 1278.7 | 191.9 | 198.8 | 179.9 |

llama.cpp's own switches, on its 64-token Llama-3.2-1B prompt (1276-1286 tok/s): no residency
sets 1281-1284, no fusion 1280, no graph reordering 1250, no concurrency 1241, private buffers
1235. None of them is the 6% it has on us there.

## 2026-11-17: a Metal matmul's rest of 32 to 48 tokens as a 32-token tile and the few-token kernel

Such a rest was padded into a 64-token tile, which costs its weight decode and 64 tokens of
products whatever it holds. Before (fcfd8a1), after, llama.cpp in turn, 20 s rests, 5 reps:

| Tokens | Llama-3.2-1B before | after | llama.cpp | Mistral-7B before | after | llama.cpp |
|---|---|---|---|---|---|---|
| 32 | 983.1 | 990.3 | 1095.2 | 164.8 | 164.6 | 187.6 |
| 40 | 769.0 | 1012.7 | 806.4 | 122.7 | 162.0 | 127.0 |
| 48 | 919.8 | 1092.4 | 972.8 | 146.8 | 163.3 | 151.3 |
| 56 | 1052.2 | 1052.9 | 1126.4 | 168.3 | 168.4 | 175.2 |
| 100 | 1062.3 | 1215.4 | 1101.5 | 159.5 | 181.3 | 159.1 |
| 112 | 1185.2 | 1287.1 | 1230.1 | 178.4 | 188.4 | 177.5 |

A rest of 56 as 32 and 24 ran 1087 against 1051 on Llama-3.2-1B and 161.8 against 168.1 on
Mistral-7B, so 49 to 63 stay in one 64-token tile.

### Looked for and not found: the one-tile gap (32, 56 and 64 tokens)

At 32 tokens both engines run one 32-token tile per 64 rows and llama.cpp is 10-14% ahead
(1095 against 990 tok/s on Llama-3.2-1B, 187.6 against 164.6 on Mistral-7B). None of these
accounts for it:

- llama.cpp's `kernel_mul_mm` for Q4_K ported into this harness in place of
  `matmul_h_q4_k_t32`: 1000 against 980 and 166.7 against 163.3 (2%).
- Its graph for that prompt is the same work: 225 matmuls a pass on Mistral-7B, ours 219 and
  the last layer's single-row matvecs.
- Its thread count (the number of command buffers a pass is split into): 186.6-187.1 from 1 to 8.
- A session reused across repetitions here instead of a fresh one: under 1.3%.
- 28 or 32 tokens as a 16- or 8-token tile beside the few-token kernel: 5-22% slower than the
  32-token tile.
- A per-kernel profile (`RLI_METAL_PROF=1`) of the 32-token Mistral-7B prompt is 96% the two
  tile kernels, Q6_K 17% slower per weight than Q4_K.

## 2026-11-18: the one-tile Metal gap: llama.cpp's 32-token tile, and threadgroup memory

Found by timing every op inside llama.cpp (a local patch to `ggml_metal_graph_compute` that
encodes each node in its own command buffer, as `RLI_METAL_PROF` does here, optionally 16 times
over). Mistral-7B, 32 tokens, us per matmul dispatch:

| | Q4_K alone | Q4_K repeated 16x | Q6_K repeated 16x |
|---|---|---|---|
| llama.cpp `kernel_mul_mm` in llama.cpp | 1060 | 772 | 873 |
| our `matmul_h_*_t32` (12 KB staged) | 961 | 871 | 1139 |
| ... declaring 4 KB more / 12 KB more | | 876 / 1010 | |
| ... staged 32 columns at a time (8 KB, twice the barriers) | | 931 | |
| llama.cpp's kernel ported here, 14 KB declared | 954 | 844 | |
| ... 8 KB (edge output shares the staging) | | 811 | 942 |
| ... 8 KB, our `deq16_*` decoders | | 851 | 1060 |

So alone ours was the faster kernel and in sequence the slower one, which is why the earlier
port "ran as fast as ours". `MATMUL_K32` is the 8 KB port, for Q4_K and Q6_K half tiles of 32
tokens; 49 to 64 tokens use it too. Prompt tok/s (before = 815c673; before, after, llama.cpp
in turn, 20 s rests, 5 reps):

| Tokens | Llama-3.2-1B before | after | llama.cpp | Mistral-7B before | after | llama.cpp |
|---|---|---|---|---|---|---|
| 32 | 985.8 | 1055.9 | 1091.6 | 163.8 | 180.0 | 187.0 |
| 40 | 1012.0 | 1066.2 | 814.9 | 160.9 | 170.5 | 126.3 |
| 48 | 1086.7 | 1126.5 | 960.2 | 166.9 | 175.6 | 150.5 |
| 56 | 1051.1 | 1128.5 | 1127.3 | 168.2 | 169.3 | 175.1 |
| 64 | 1208.1 | 1248.8 | 1282.6 | 192.1 | 192.3 | 196.3 |
| 88 | 1205.8 | 1204.7 | 1243.0 | 183.0 | 183.0 | 183.7 |
| 512 | 1426.7 | 1426.7 | 1426.2 | 194.8 | 193.1 | 192.2 |

That tile as the whole tile at every length: 64 tokens 1247 against 1200 (Llama-3.2-1B) and
191.5 against 190.9 (Mistral-7B); 100 tokens level and 2% slower; 128 tokens level and 4%
slower; 192 and 512 tokens 3-6% slower. Hence 49 to 64 only.

## 2026-11-18: Metal tiles stage out-of-range rows without a test

Both tile kernels staged `row_ok ? weight : 0` and `tok_ok ? x : 0`. Rows and tokens past
the end already read row or token 0, and an edge tile writes only what is in range, so the
tests are gone. `RLI_METAL_PROF=1 RLI_METAL_PROF_REPEAT=16`, Mistral-7B, us per dispatch:

| | before | after | llama.cpp |
|---|---|---|---|
| 32 tokens, Q4_K (`MATMUL_K32`) | 815 | 788 | 786 |
| 32 tokens, Q6_K | 937 | 890 | 880 |
| 128 tokens, Q4_K (`matmul_h_q4_k_t64`) | 2875 | 2818-2855 | |
| 128 tokens, Q6_K | 3222 | 3139-3165 | |

Tried on the way: the grid turned as llama.cpp's is (tokens on x): 821-823 against 815;
the add-into-y path skipped: no change. Prompt tok/s (before = 451b0da; before, after,
llama.cpp in turn, 20 s rests, 5 reps; the 32-token Llama-3.2-1B "after" was one disturbed
run of 904 +- 179, run again twice):

| Tokens | Llama-3.2-1B before | after | llama.cpp | Mistral-7B before | after | llama.cpp |
|---|---|---|---|---|---|---|
| 32 | 1061.9 | 1092.6, 1094.5 | 1092.7 | 180.2 | 186.4 | 187.1 |
| 56 | 1124.6 | 1144.4 | 1126.2 | 169.0 | 171.2 | 174.8 |
| 64 | 1249.1 | 1274.8 | 1283.5 | 189.1 | 198.4 | 199.4 |
| 128 | 1351.2 | 1366.3 | 1397.0 | 202.4 | 204.6 | 200.7 |
| 512 | 1425.8 | 1440.8 | 1421.9 | 194.2 | 192.6 | 191.7 |

## 2026-11-18: llama.cpp's Q4_K and Q6_K decoders in the few-token Metal kernel

`matmul_few_q4_k_*` and `_q6_k_*` decode through `ref_deq_*` (as `MATMUL_K32` does, here in
f32) instead of `deq16_*`. `RLI_METAL_PROF_REPEAT=16`, Mistral-7B, 4 tokens, us per
dispatch: Q4_K 299 -> 279 (llama.cpp's few-row kernel 263), Q6_K 381 -> 307 (291). `--precise`
logits of a 12-token prompt stay within 3e-5 of the CPU's. Prompt tok/s (before = d0f0890;
before, after, llama.cpp in turn, 20 s rests, 5 reps):

| Tokens | Llama-3.2-1B before | after | llama.cpp | Mistral-7B before | after | llama.cpp |
|---|---|---|---|---|---|---|
| 4 | 312.2 | 327.6 | 319.4 | 58.0 | 62.3 | 62.6 |
| 8 | 600.3 | 662.9 | 593.9 | 112.6 | 121.1 | 115.9 |
| 16 | 866.2 | 869.4 | 788.2 | 144.4 | 144.2 | 142.2 |
| 24 | 860.9 | 848.8 | 824.2 | 137.3 | 136.5 | 141.5 |
| 88 | 1218.6 | 1221.3 | 1237.3 | 185.3 | 185.2 | 183.8 |

## 2026-11-18: the 8 KB tile at 64 tokens (`MATMUL_K` with MT = 64)

The 32-token tile's layout with four token blocks a simdgroup, not two: still 32-column
steps and 8 KB (an edge tile's output goes through it 32 tokens at a time), llama.cpp-style
decoders. It replaces `MATMUL`'s 64-token half tile (16 KB) for Q4_K and Q6_K.
`RLI_METAL_PROF_REPEAT=16`, Mistral-7B, 128 tokens, us per dispatch: Q4_K 2818-2855 -> 2684,
Q6_K 3139-3165 -> 2803. With it, 49 to 64 tokens as 32-token tiles (the rule of 451b0da) lost:
64 tokens 198.6 against 204.4 and 56 tokens 170.5 against 179.4 on Mistral-7B, level on
Llama-3.2-1B; the rule is gone. Prompt tok/s (before = fc8b3fd; before, after, llama.cpp in
turn, 20 s rests, 5 reps):

| Tokens | Llama-3.2-1B before | after | llama.cpp | Mistral-7B before | after | llama.cpp |
|---|---|---|---|---|---|---|
| 56 | 1144.3 | 1131.5 | 1092.5 | 171.6 | 179.7 | 175.2 |
| 64 | 1285.2 | 1289.6 | 1282.2 | 198.8 | 204.4 | 199.5 |
| 88 | 1221.4 | 1255.2 | 1230.6 | 185.1 | 187.1 | 183.8 |
| 128 | 1370.5 | 1426.9 | 1399.8 | 202.4 | 212.8 | 201.3 |
| 200 | 1427.0 | 1457.5 | 1278.9 | 201.2 | 206.8 | 178.7 |
| 512 | 1441.9 | 1468.3 | 1425.9 | 195.1 | 198.8 | 191.8 |

Logits of a 193-token prompt (three 64-token tiles and a one-token rest) are within 0.008 of
the CPU's `--precise` ones.

## 2026-11-18: 17 to 24 tokens of Q4_K and Q6_K on the 32-token Metal tile

A rest of 17 to 24 tokens took the few-token kernel (3 token blocks); for half Q4_K and Q6_K
it now takes the 8 KB 32-token tile (`few_max` in `matmul_impl`). Faster on Mistral-7B, 2%
slower on Llama-3.2-1B. The other lengths are here because they had not been run since
3bb15f6; they do not change. Prompt tok/s (before = 3bb15f6; before, after, llama.cpp in
turn, 20 s rests, 5 reps):

| Tokens | Llama-3.2-1B before | after | llama.cpp | Mistral-7B before | after | llama.cpp |
|---|---|---|---|---|---|---|
| 20 | 711.6 | 695.6 | 683.2 | 114.8 | 117.7 | 117.8 |
| 24 | 848.2 | 832.3 | 824.4 | 136.5 | 141.6 | 141.6 |
| 40 | 1121.6 | 1119.2 | 818.4 | 179.8 | 179.8 | 127.0 |
| 48 | 1177.8 | 1178.4 | 973.6 | 181.4 | 181.3 | 151.3 |
| 72 | 1265.4 | 1262.4 | 1026.3 | 194.9 | 196.9 | 152.3 |
| 100 | 1309.7 | 1303.3 | 1101.4 | 193.4 | 194.1 | 158.8 |
| 112 | 1363.7 | 1365.7 | 1232.4 | 199.4 | 199.2 | 177.2 |

## 2026-11-18: what the CUDA f16 GEMM is short of (A100; nothing changed in the kernel)

Mistral-7B pp512 is 4131 tok/s here against llama.cpp's 4850-4960; `RLI_CUDA_PROF=1` puts 61%
of it in `gemm_f16` and 16% in `dequant`. cuBLAS's f16 GEMM on the same GPU (PyTorch
`half @ half`, 200 runs, us) against ours (`bench_cuda`, which now runs these shapes at 512
and 128 tokens):

| Matrix | 512 tokens: ours | cuBLAS | 128 tokens: ours | cuBLAS |
|---|---:|---:|---:|---:|
| 4096x4096 | 168.8 | 155.9 | 91.4 | 69.6 |
| 14336x4096 | 512.7 | 422.3 | 225.3 | 158.5 |
| 4096x14336 | 603.5 | 313.7 | 304.0 | 120.7 |
| 2048x2048 | 61.5 | 36.0 | 34.3 | 18.1 |
| 8192x2048 | 171.8 | 94.1 | 65.4 | 40.6 |
| 2048x8192 | 236.5 | 99.8 | 169.6 | 46.4 |

A Mistral-7B layer is 2297 us of GEMM here and 1782 with cuBLAS: 16 ms of a 124 ms prompt,
most of the gap. Diagnostics on the 64x128 tile (512 tokens; 4096x4096 / 14336x4096 /
4096x14336, us):

| Variant | | | |
|---|---:|---:|---:|
| as it is | 168.9 | 512.7 | 603.5 |
| every block reads the same 64 weight rows (weights in L2) | 164.8 | 475.8 | 558.6 |
| tiles staged at the first step only, barriers kept | 110.1 | 304.6 | 360.6 |
| ... and no barriers | 102.6 | 279.2 | 333.8 |
| tokens staged once, weights every step | 123.1 | 411.1 | 410.0 |
| weights staged once, tokens every step | 148.8 | 416.6 | 500.7 |
| column steps outermost (16 sums between revisits, not 4) | 199.9 | 603.9 | 696.4 |
| operands through `ldmatrix.x4` (weights only: 179.5 / 529.9 / 628.5) | 184.6 | 542.1 | 634.7 |
| rows 32 words apart, 16-byte chunks swizzled | 187.7 | 563.1 | 669.1 |

So the products alone run at 156-215 TFLOPS and the copy into shared memory is 35-40% of the
time, the token tile (two thirds of the bytes) more than the weights; not memory latency,
not the shared-memory loads, not the register cap (80, 168 and 224 give the same times).
Every shape and tile moves 2.2-2.7 TB/s from global to shared memory.

Other tiles, with rows chosen so the block count is a multiple of the 108 multiprocessors
(x 4096 columns, 512 tokens, us):

| Tile (rows x tokens, warps) | 13824 | 6912 | 3456 | 14336 | 4096 | 4096x14336 |
|---|---:|---:|---:|---:|---:|---:|
| 64x128, 4 (as it is) | 507.9 | 294.5 | 159.7 | 512.7 | 169.0 | 606.1 |
| 128x128, 4 | 415.6 | 222.1 | 168.5 | 590.8 | 194.6 | 672.9 |
| 128x128, 8 | | | | 770.1 | 300.4 | 1020.4 |
| 64x256, 8 | | | | 830.7 | 317.3 | 1097.9 |
| 128x128, 4, `cp.async` into two buffers | 403.7 | 208.0 | 129.3 | 507.7 | 183.1 | 605.4 |
| 64x128, 4, `cp.async` (2 blocks a multiprocessor) | 611.2 | 294.5 | 149.8 | 687.0 | 265.9 | 894.5 |
| 128x256, 8, `cp.async` | 369.7 | 193.0 | 187.7 | 545.9 | 184.6 | 619.1 |
| 256x128, 8, `cp.async` | 376.2 | 193.2 | 190.6 | 557.7 | 187.7 | 630.9 |

The larger tiles reach 135-157 TFLOPS where the blocks divide evenly (13824 and 6912 rows)
and lose it otherwise: 14336 rows are 16 blocks more than 13824 and cost 175 us more, a
second wave that takes a block's whole time with most of the GPU idle. Letting the caller
give a block fewer rows and tokens than its tile, so the count fits (4096x512 as 215 blocks
of 96x104), did not help: 180.3 / 557.1 / 622.0, because a block that multiplies 59% of a
tile still copies 78% of it. An earlier run of the `cp.async` kernels that looked 15-25%
better staged half of each step's columns by mistake; the numbers above are the corrected
ones. Not tried: splitting the columns over blocks (more, shorter blocks, so a last wave
costs less; needs the partial sums added).

## 2026-11-18: the CUDA f16 GEMM on a 128x128 tile, columns in parts

`gemm_f16_128x128` where `gemm_plan` can fill the GPU with its blocks (A100: 432 at once,
and an even number per multiprocessor): the columns of a matrix of few rows are split into
parts whose blocks add their sums into y (`atomicAdd`: the same times as separate buffers),
a matrix of more rows than 432 blocks take goes 432 blocks first and the rest in parts, and
the output and down projections add into the residual stream directly. Probes on the way,
128x128 tile, 512 tokens, us: 4096x4096 in 1 / 2 / 3 / 4 / 6 parts 194.7 / 207.2 / 155.7 /
174.2 / 166.1; 4096x14336 674.3 / 654.9 / 479.2 / 534.3 / 487.0; 2048x8192 316.7 / 195.7 /
141.7 / 187.3 / 151.5. Passing the choices at run time to the 64-row kernels made them 4-8%
slower, so the tile that adds is a kernel of its own. `bench_cuda`, us (before = c5fd822;
"adding" is `gemm_add`):

| Matrix | 512 tokens before | after | adding | cuBLAS | 128 tokens before | after | adding | cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096x4096 | 168.8 | 165.2 | 156.1 | 155.9 | 91.4 | 63.1 | 58.6 | 69.6 |
| 14336x4096 | 512.7 | 450.0 | 454.4 | 422.3 | 225.3 | 170.2 | 161.8 | 158.5 |
| 4096x14336 | 603.5 | 491.2 | 479.9 | 313.7 | 304.0 | 147.6 | 143.1 | 120.7 |
| 2048x2048 | 61.5 | 49.7 | 44.1 | 36.0 | 34.3 | 25.7 | 22.1 | 18.1 |
| 8192x2048 | 171.8 | 162.4 | 164.0 | 94.1 | 65.4 | 57.6 | 51.6 | 40.6 |
| 2048x8192 | 236.5 | 147.6 | 142.0 | 99.8 | 169.6 | 52.8 | 49.3 | 46.4 |

Prompt tok/s on the A100 (`bench --pp N --tg 0 --reps 5 --gpu-layers 999` before and after,
`llama-bench -ngl 99 -p N -n 0 -r 5`, in turn with 6 s rests, two rounds; this host's runs
swing by up to 10%):

| Model, tokens | before | after | llama.cpp |
|---|---:|---:|---:|
| Mistral-7B 512 | 4310, 4275 | 4658, 4680 | 4799, 4823 |
| Mistral-7B 256 | 3243, 3160 | 3712, 3901 | 3927, 4073 |
| Mistral-7B 128 | 2012, 1957 | 2430, 2323 | 3088, 3155 |
| Mistral-7B 64 | 1105, 1143 | 1324, 1353 | 2240, 2318 |
| Llama-3.2-1B 512 | 17363, 13706 | 16778, 16384 | 15917, 15588 |
| Llama-3.2-1B 256 | 10745, 12144 | 11329, 11923 | 11767, 11390 |
| Llama-3.2-1B 128 | 6974, 6448 | 7773, 8463 | 8871, 9443 |
| Llama-3.2-1B 64 | 3443, 3480 | 4750, 4673 | 6920, 6651 |
| Qwen3-30B-A3B 512 | 4271, 4318 | 4444, 4435 | 4148, 4245 |
| Qwen3-30B-A3B 256 | 3522, 3557 | 3565, 3660 | 2861, 2924 |
| Qwen3-30B-A3B 128 | 2118, 2052 | 2124, 2139 | 1765, 1766 |
| Qwen3-30B-A3B 64 | 1300, 1319 | 1318, 1376 | 1501, 1504 |

Short prompts had not been compared on CUDA before: the dense models are 1.4-1.7x behind at
64 tokens and 1.1-1.3x at 128. The CUDA tests pass with and without `RLI_CUDA_NO_MMA=1`
(152). Logits of a 601-token prompt (Mistral-7B) and a 540-token one (Llama-3.2-1B) against
`--precise`: 0.076 and 0.159 before, 0.085 and 0.174-0.194 after, top-1 the same. The order
in which parts add their sums is not fixed, so the logits now differ between runs.

## 2026-11-18: short CUDA prompts on the int8 tensor-core matmul, columns in parts

llama.cpp (0.9.0, cd959ba) no longer takes cuBLAS on this GPU: `ggml_cuda_should_use_mmq`
returns true for any NVIDIA GPU from Turing on, so every prompt length runs its int8 kernel
on the quantized weights. Our f16 path writes each matrix out first, which `bench_cuda`
puts at 46 / 146 us for 4096x4096 / 14336x4096 against 44 / 136 us for the product at 64
tokens (cuBLAS: 43 / 127), and `mma_q4_k` was slower still: 64 blocks for a 4096-row matrix
on 108 multiprocessors, each staging 128 tokens. Now a block is 128 rows by 64 tokens and a
matrix's columns go in parts that add their sums (`mma_parts`). `bench_cuda`, us, Q4_K by
64 tokens, by number of parts (0 = the 64x128 tile as it was):

| Matrix | 0 | 1 | 2 | 3 | 6 | 8 | 12 | 16 | 24 | f16: dequant + gemm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096x4096 | 137.2 | 179.2 | 93.7 | 67.2 | 61.2 | 62.6 | 60.0 | 55.8 | 62.1 | 89.9 |
| 14336x4096 | 317.0 | 284.8 | 224.1 | 187.9 | 175.9 | 165.6 | 165.0 | 165.9 | 183.4 | 282.0 |
| 4096x14336 | 499.1 | 650.0 | 330.2 | 227.5 | 177.7 | 201.5 | 176.9 | 165.9 | 169.0 | 270.6 |

The f16 GEMM stays from 129 tokens (`GEMM_MIN_TOKENS`, was 64). Int8 against f16 by prompt
length, tok/s: Mistral-7B 64: 1996 / 1401, 96: 2300 / 2000, 128: 2715 / 2551, 192: 2955 /
3027, 256: 2904 / 3802; Llama-3.2-1B 64: 6276 / 4907, 128: 8750 / 8683, 256: 11893 / 11393.

Prompt tok/s (as the table above; "before" is the build before the 128x128 f16 tile, two
commits back):

| Model, tokens | before | after | llama.cpp |
|---|---:|---:|---:|
| Mistral-7B 16 | 299, 314 | 633, 619 | 1088, 972 |
| Mistral-7B 32 | 562, 600 | 1181, 1209 | 1556, 1686 |
| Mistral-7B 64 | 1094, 1202 | 2001, 1858 | 2222, 2225 |
| Mistral-7B 128 | 2119, 1979 | 2508, 2493 | 2952, 3236 |
| Llama-3.2-1B 16 | 866, 882 | 2149, 2126 | 2641, 3031 |
| Llama-3.2-1B 32 | 1603, 1593 | 3872, 3853 | 4404, 4387 |
| Llama-3.2-1B 64 | 3389, 3401 | 6308, 6256 | 6438, 6430 |
| Llama-3.2-1B 128 | 6102, 6146 | 8150, 8574 | 8100, 8149 |
| Qwen3-30B-A3B 16 | 371, 384 | 456, 396 | 833, 834 |
| Qwen3-30B-A3B 32 | 577, 580 | 649, 623 | 1182, 1176 |
| Qwen3-30B-A3B 64 | 1386, 1324 | 1401, 1386 | 1501, 1202 |
| Qwen3-30B-A3B 128 | 2109, 2069 | 2320, 2280 | 1663, 1762 |

The CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (152). Logits of a ~110-token
prompt against `--precise`: Mistral-7B 0.059 before (f16), 0.146-0.190 now (int8, two
runs); Llama-3.2-1B 0.129 and 0.207-0.217; top-1 the same.

## 2026-11-18: 16- and 32-token blocks for the CUDA int8 tensor-core matmul

Kernels of 16 and 32 tokens a block for chunks that fit (`MMA_FEW`), and in the 16-token
Q4_K one each lane reads its operands from the matrix instead of through shared memory.
`bench_cuda`, Q4_K by 16 tokens, us: 64-token blocks 39.3 / 109.1 / 112.2 (4096x4096 /
14336x4096 / 4096x14336) -> direct 34.1 / 91.3 / 94.8; direct reads at 32 tokens 44.8 /
128.9 / 131.5 against 45.5 / 129.2 / 130.9 staged, and at 64 tokens 68.6 / 197.5 / 204.6
against 59.2 / 170.2 / 169.3, so only the 16-token kernel has them. More parts than
`mma_parts` gives change nothing at these sizes. Prompt tok/s, A100, 8 reps:

| Model, tokens | 64-token blocks | 16/32-token blocks | and direct reads | llama.cpp |
|---|---:|---:|---:|---:|
| Llama-3.2-1B 8 | 1130 | 1248, 1245, 1241 | 1325, 1352 | 1427, 1365 |
| Llama-3.2-1B 16 | 2162 | 2348, 2344, 2364 | 2520, 2559 | 2743, 2726 |
| Llama-3.2-1B 24 | 3071 | 3263, 3293, 3317 | 3265, 3289 | 3803, 3674 |
| Llama-3.2-1B 32 | 3874 | 4104, 4255, 4102 | 4165, 4342 | 4675, 4650 |
| Mistral-7B 8 | 325 | 339, 339, 439 | 381, 425 | 475, 486 |
| Mistral-7B 16 | 732 | 804, 657, 706 | 654, 701 | 1122, 1168 |
| Mistral-7B 24 | 1048 | 1144, 928, 1095 | 1068, 1060 | 1365, 1385 |
| Mistral-7B 32 | 1168 | 1367, 1305, 1165 | 1168, 1165 | 1801, 1788 |

Mistral-7B's runs on this host spread by 10-30% within a run; Llama-3.2-1B's by under 1%.
The CUDA tests pass (152).

Tried and dropped: reading the next step's weights while the 16-token kernel multiplies the
current ones (16 more registers a lane): 41.2 / 113.3 / 105.3 us against 34.1 / 91.3 / 94.8.
Both engines' short-prompt figures on this host rise with the number of repetitions (the
GPU's clocks): Mistral-7B at 16 tokens is 642 / 828 / 924 / 1047 tok/s here and 979 / 1083 /
1175 / 1289 for llama.cpp over 3 / 8 / 20 / 60 repetitions, so the table's Mistral-7B rows
overstate the gap (19% at 60 repetitions, not 40%).

## 2026-11-18: CUDA prompts against llama.cpp with the GPU warmed up (A100, 847ca5a)

`bench --pp N --tg 0 --reps R --gpu-layers 999` and `llama-bench -ngl 99 -p N -n 0 -r R` in
turn with 3 s rests, R = 2400 / N between 8 and 60, two rounds; tok/s. These replace the
8-repetition short-prompt figures above, which caught both engines before the GPU's clocks
came up.

| Tokens | Mistral-7B | llama.cpp | Llama-3.2-1B | llama.cpp | Qwen3-30B-A3B | llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 574, 572 | 526, 528 | 1847, 1807 | 1919, 1761 | 442, 445 | 523, 514 |
| 16 | 1046, 1049 | 1293, 1279 | 3216, 3508 | 3789, 3918 | 611, 617 | 836, 861 |
| 32 | 1620, 1620 | 2106, 2074 | 5834, 5905 | 6604, 6428 | 735, 739 | 1207, 1208 |
| 64 | 2365, 2379 | 2942, 2871 | 8608, 8375 | 8422, 9043 | 1669, 1675 | 1501, 1507 |
| 128 | 2723, 2749 | 3518, 3333 | 9457, 10535 | 10958, 11462 | 2574, 2544 | 1780, 1780 |
| 256 | 3817, 3738 | 4267, 4158 | 12643, 12121 | 12469, 12070 | 3658, 3529 | 2941, 2941 |
| 512 | 4812, 4757 | 4817, 4859 | 20317, 20980 | 19818, 19099 | 4402, 4408 | 4259, 4178 |

Ahead or level: 512 tokens on all three, Mistral-7B at 8, Llama-3.2-1B at 8, 64 and 256,
Qwen3-30B-A3B from 64. Behind: Mistral-7B 19-23% at 16 to 128 and 10% at 256;
Llama-3.2-1B 9-13% at 16, 32 and 128; Qwen3-30B-A3B 14% at 8, 28% at 16, 39% at 32.

## 2026-11-18: CUDA MoE expert matmuls grouped by expert from 20 tokens (was 64)

`MOE_GROUP_TOKENS` 64 -> 20. Qwen3-30B-A3B on the A100, `bench --pp N --tg 0 --reps 60
--gpu-layers 999`, tok/s, a pair at a time / grouped: 8 tokens 435 / 344, 16: 621 / 594,
20: 650 / 686, 24: 691 / 808, 28: 718 / 919, 32: 740 / 997, 48: 791 / 1337. Against
llama.cpp (`-r 60`, in turn, two rounds):

| Tokens | before | after | llama.cpp |
|---|---:|---:|---:|
| 8 | 442, 445 | 461, 447 | 548, 548 |
| 16 | 611, 617 | 616, 621 | 861, 861 |
| 24 | 691, 691 | 810, 810 | 1030, 1019 |
| 32 | 735, 739 | 996, 1001 | 1202, 1185 |
| 48 | 791 | 1337, 1352 | 1370, 1399 |

The CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (152). Logits of a 32-token prompt
against `--precise`: 0.83 before, 0.89 after, top-1 the same. Also tried and dropped for the
dense int8 matmul: a 128-token block for chunks of 65 to 128 (64 sums a lane under the
80-register cap): 121.9 / 382.4 / 345.0 us at 128 tokens against 103.1 / 314.3 / 311.4 as
two 64-token blocks.

## 2026-11-18: the MoE router of a short CUDA prompt chunk as matvecs

The router (F32, 128 x 2048 on Qwen3-30B-A3B) ran a prompt chunk through `matmul_f32`, one
warp walking each row's 64 column steps for a tile of 32 tokens: 75 us a launch at 16 tokens
(`RLI_CUDA_PROF=1`), 9% of the prompt. Chunks of up to 64 tokens (`ROUTER_MATVEC_TOKENS`) now
take the decode kernel, a block of 8 warps per row and token: 26 us. `bench --pp N --tg 0
--gpu-layers 999`, 60 / 60 / 60 / 18 / 8 reps, matmul against matvecs in turn, two rounds,
tok/s: 8 tokens 451, 454 -> 495, 500; 16: 615, 626 -> 660, 653; 32: 997, 999 -> 1072, 1072;
128: 2555, 2509 against 2575, 2517; 512: 4375, 4373 against 4176, 4184 (so not there).
`--precise` logits of a 32-token prompt are the old build's to 7e-6. CUDA tests pass with and
without `RLI_CUDA_NO_MMA=1` (152). Also tried on the dense int8 matmul and dropped: one
16-byte load for a token pair's scale and sum and one 8-byte load for its two operand words
(170.0 us against 170.2 for 14336x4096 by 64 tokens); a register cap of 64, 96, 112 or 128+
(the same to 96, 3% slower at 112, 1.4x slower from 128, where a multiprocessor holds two
blocks). With the float arithmetic after each `mma` reduced to one add that matmul takes
103 us, and with the tiles staged once 123: those are where its time is.

## 2026-11-18: CUDA prompt chunks of 24 to 192 tokens on f16 tensor cores, weights decoded in the block

`gemm_q4_k_dense` / `gemm_q6_k_dense` are the MoE experts' kernel (`GEMM_ID`: 64 rows by 64
tokens, a step's weights decoded to f16 on the way into shared memory) over every token of a
chunk, the columns in parts that `atomicAdd` into y (`dense_parts`, up to 8). Nothing is
written out first, which is what `dequant_*` + `gemm_f16_*` could not afford under 129
tokens. `bench_cuda` (`RLI_DENSE=1` for this kernel), us a matmul, int8 tensor cores against
this at its best part count: 4096x4096 by 64 tokens 59.3 -> 40.4, 14336x4096 170.4 -> 121.4,
4096x14336 169.5 -> 121.4; by 128 tokens 103 -> 70, 315 -> 216, 312 -> 215; by 16 tokens
34.3 against 32.4, 91.6 against 93.0 (so not there). By 256 tokens 129 / 414 / 398 against
dequant + GEMM's 133 / 453 / 396, by 512 it loses (234 / 796 / 784 against 210 / 595 / 644):
a block decodes its weights again for every 64 tokens.

`bench --pp N --tg 0 --gpu-layers 999`, R = 2400 / N repetitions (at most 60), before and
after in turn (two rounds each, tok/s). Mistral-7B: 16 tokens 1040, 1041 against 953, 965
(not used); 24: 1283, 1283 -> 1366, 1365; 32: 1609, 1643 -> 1735, 1755; 48: 2015, 2006 ->
2389, 2353; 64: 2365, 2428 -> 2841, 2861; 96: 2343, 2324 -> 2829, 2813; 128: 2742, 2754 ->
3236, 3324; 192: 3120, 3150 -> 3422, 3471; 256: 3702, 3812 against 3552, 3575 (not used).
Llama-3.2-1B was noisy that hour (±20%): 48 tokens 7535, 7840 -> 8831, 8663; 64: 9328, 7990
-> 10436, 8992; the rest within noise. Logits of a 400-character prompt against `--precise`:
Mistral-7B 0.27 -> 0.085, Llama-3.2-1B 0.20 -> 0.10 (f16 activations, not int8).

Against llama.cpp after it (ours / llama.cpp, two rounds in turn, means): Mistral-7B 24:
1382 / 1717; 32: 1763 / 2074; 48: 2387 / 2600; 64: 2817 / 2904; 96: 2776 / 3322; 128: 3361 /
3379; 192: 3442 / 4036. Llama-3.2-1B 24: 4762 / 5080; 32: 6176 / 6340; 48: 8197 / 8263; 64:
9187 / 9026; 96: 11188 / 10929; 128: 11194 / 11142; 192: 11835 / 12812. Qwen3-30B-A3B 24:
877 / 1037; 32: 1098 / 1183; 48: 1441 / 1391; 64: 1778 / 1518; 96: 2195 / 1693; 128: 2595 /
1774; 192: 3213 / 2783. CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (152 each);
`gemm_matches_f32_matmul` now covers the new kernel, stored and added.

## 2026-11-18: zeroing folded into the f16 conversion ahead of the in-block CUDA matmuls

`RLI_CUDA_PROF=1` on a 32-token Mistral-7B prompt: `zero_rows` 6.1% and `to_f16` 4.9% of the
time, nine launches a layer between them. `to_f16_zero` converts a matmul's input and zeroes
up to three results in one launch (Q, K and V; gate and up), five launches fewer a layer.
`bench --pp N --tg 0 --gpu-layers 999`, R = 2400 / N repetitions (at most 60), before and
after in turn, tok/s on Mistral-7B: 24 tokens 1350, 1359 -> 1401, 1420; 32: 1746, 1762 ->
1813, 1783; 64: 2834, 2865 -> 2898, 2923; 128: 3282, 3269 -> 3315, 3419. Against llama.cpp
after it (ours / llama.cpp, means of two rounds): Mistral-7B 24: 1393 / 1734; 32: 1809 / 2105;
48: 2423 / 2563. Qwen3-30B-A3B 24: 888 / 1028; 32: 1110 / 1201; 48: 1480 / 1375. CUDA tests
pass with and without `RLI_CUDA_NO_MMA=1` (152 each).

Tried and dropped there: a block of 128 rows by 32 tokens for chunks of up to 32 (at 24 and
32 tokens half of the 64x64 block's warps have no tokens): 14336x4096 by 32 tokens 102.7 ->
94.7 us, 4096x14336 104.1 -> 108.0, 4096x4096 34.5 -> 35.6. And a diagnostic: with the
per-weight decode (int to float, multiply-add, f16) replaced by a mask and an add, 14336x4096
by 32 tokens takes 91.4 us, not 102.8, so the decode arithmetic is a tenth of that matmul.

## 2026-11-18: the in-block CUDA f16 matmul's Q4_K scales, and where its time goes

Diagnostics on `gemm_q4_k_dense`, 14336x4096 by 32 tokens (102.7 us; `RLI_DENSE=1
bench_cuda`): weights and tokens staged on the first step only 46.1; one `mma` a step in
place of up to 16: 88.2; weights staged once 90.4; tokens staged once 100.9; the per-weight
arithmetic replaced by a mask 91.4. The parts don't add up to the whole, and neither waiting
less (step ks + 1's global reads issued before step ks is multiplied: 101.7, and 126.9
against 122.0 at 64 tokens) nor fewer barriers (128-column steps: 95.8 and 121.2 on
14336x4096, 103.8 and 129.6 on 4096x14336 against 104.3 and 121.4) gained. Both dropped.

Kept: `gq4_k_hdr` converts a super-block's d and dmin from f16 once, not in each of its four
steps, and `gq4_k_dec` takes a step's two sub-block scales out of the header together:
14336x4096 by 32 tokens 102.7 -> 98.9 us, 4096x14336 104.3 -> 98.8, 4096x4096 35.2 -> 33.6;
by 64 tokens 122.0 -> 120.7, 121.4 -> 119.6, 40.5 -> 39.8. CUDA tests pass with and without
`RLI_CUDA_NO_MMA=1` (152 each).

Against llama.cpp after it (`bench --pp N --tg 0 --gpu-layers 999` and `llama-bench -p N -n 0`
in turn, R = 2400 / N repetitions (at most 60), means of two rounds, ours / llama.cpp):
Mistral-7B 8 tokens 558 / 524; 16: 1047 / 1306; 32: 1845 / 2087; 96: 2907 / 3271.
Llama-3.2-1B (±20% that hour) 8: 1910 / 1956; 16: 3250 / 3603; 32: 6679 / 6161; 96: 10036 /
10054. Qwen3-30B-A3B 8: 500 / 537; 16: 658 / 850; 32: 1129 / 1196; 96: 2234 / 1671.

Also tried on that kernel and dropped (same day, same commands). Blocks of 128 tokens (a warp
takes 16 rows by 64 tokens): 14336x4096 by 64 tokens 121.5 -> 158.3 us, 96: 195.0 -> 182.5,
128: 215.2 -> 209.4, 192: 312.8 -> 325.1, 256: 400.1 -> 383.5; a gain only where the 64-token
block's second tile is half empty. And not quantizing results that only the f16 matmuls read
(`quant_x` after a prompt's attention, the warp reductions in `rmsnorm` and `swiglu`): Mistral-7B
32 tokens 1840, 1823 against 1842, 1848 tok/s, 64: 2957, 2918 against 2941, 2936, 128 and 512
the same; the launches around the matmuls are not where a short prompt's time is.

## 2026-11-18: llama.cpp's matmul alone, and a CUDA kernel for chunks of 8 to 32 tokens

llama.cpp's `test-backend-ops perf -o MUL_MAT -b CUDA0` (0.9.0, cd959ba, built with
`-DLLAMA_BUILD_TESTS=ON`) with our shapes added to `make_test_cases_perf`, us a run on the
A100, next to ours from `bench_cuda` (`RLI_DENSE=1` for the f16 kernel) before this change.
Q4_K 14336x4096 by 16 / 24 / 32 / 64 / 96 / 128 tokens: llama.cpp 66.5 / 79.1 / 90.6 / 137.5 /
184.0 / 230.4; ours 91.6 (int8) / 99.5 / 98.9 / 120.7 / 195.0 / 215.2. 4096x4096: 23.2 / 27.0 /
30.7 / 46.8 / 60.7 / 76.4 against 34.1 (int8) / 33.4 / 33.6 / 39.8 / 63.6 / 70.4. Q6_K
14336x4096: 89.4 / 105.2 / 114.5 / 162.2 / 207.5 / 259.9. So its matmul was ahead only under
64 tokens; from 64 the gap is elsewhere. Run in a Mistral-7B prompt four times each
(`bench --pp N`), attention's three launches cost 23 us a layer at 32 tokens of 545 and 99 at
192 of 1670, an RMSNorm 7 and 14.

In our kernels the cost is shared memory, not arithmetic: sharing a row's decoded scales
through it in the 16-token int8 kernel (each lane decodes all eight of its two rows' every
step) made 14336x4096 by 16 tokens 106.8 us from 91.6. `gemm_q4_k_direct` goes the other
way: f16 tensor cores, blocks of 128 rows by 32 tokens, each lane decoding from the matrix
the 16 weights of each of its two rows that its multiplies take; only the tokens are staged.
14336x4096 by 8 / 16 / 24 / 32 tokens: 71.1 / 74.2 / 77.8 / 82.4 us; 4096x14336 72.5 / 76.7 /
80.6 / 85.4; 4096x4096 25.9 / 26.7 / 29.0 / 31.5. Parts: `8 * multiprocessors / tiles`, at
most 16 (14336x4096 by 16 tokens 83.3 us in 4, 74.2 in 8, 77.6 in 12; 4096x14336 131.0 in 4,
85.5 in 8, 76.8 in 16). It takes Q4_K matrices in chunks of 8 to 32 tokens
(`GEMM_DIRECT_TOKENS`); `mm_each` now picks a kernel per matrix, so the Q6_K ones of a group
keep the int8 kernels under 24 tokens. The int8 matmuls of a group are zeroed by one launch.

`bench --pp N --tg 0 --gpu-layers 999`, 60 repetitions, before and after in turn, tok/s.
Mistral-7B 8 tokens 571, 569 -> 617, 620; 12: 820, 796 -> 877, 882; 16: 1079, 1061 -> 1150,
1169; 20: 1090, 1087 -> 1376, 1383; 24: 1432, 1424 -> 1589, 1600; 32: 1837, 1817 -> 1998,
2014. Llama-3.2-1B (±20%) 16: 3392, 3663 -> 3791, 3852; 20: 4172, 3821 -> 4656, 4392.
Qwen3-30B-A3B 8 to 20 tokens: the same within 1% (its experts). Logits of a 16-token prompt
against `--precise`: Mistral-7B 0.21 -> 0.13, Llama-3.2-1B 0.31 -> 0.15. CUDA tests pass with
and without `RLI_CUDA_NO_MMA=1` (152 each).

Against llama.cpp after it (R = 2400 / N repetitions, at most 60; means of two rounds in turn,
ours / llama.cpp): Mistral-7B 8: 612 / 521; 16: 1158 / 1287; 24: 1589 / 1718; 32: 1993 / 2087;
48: 2462 / 2589. Llama-3.2-1B 8: 1957 / 1942; 16: 3430 / 3546; 24: 5066 / 5228; 32: 6979 /
6331; 48: 9127 / 8226. Qwen3-30B-A3B 8: 513 / 539; 16: 660 / 862; 24: 904 / 1010; 32: 1130 /
1207; 48: 1472 / 1402.

## 2026-11-18: the MoE down projection grouped by expert from 12 tokens on CUDA

`RLI_CUDA_PROF=1` on a 16-token Qwen3-30B-A3B prompt: the expert matvecs are 61% of the
time, and the Q6_K ones (the down projection in half the layers: 2048 rows by 768 columns
for each of 128 pairs) take 315 us a launch where the gate and up ones take 138 for as many
weights. Grouped by expert on f16 tensor cores (`gemm_*_id`), as every expert matrix is from
20 tokens: `bench --pp N --tg 0 --gpu-layers 999`, 60 repetitions, in turn, tok/s: 16 tokens
660, 663 -> 727 (the Q6_K layers alone: 705, 701); 12 tokens 604, 613 -> 625; 8 tokens 500,
503 -> 475 (so from 12, `MOE_GROUP_DOWN_TOKENS`). CUDA tests pass with and without
`RLI_CUDA_NO_MMA=1` (152 each). Against llama.cpp after it (means of two rounds in turn,
ours / llama.cpp): 12 tokens 615 / 667; 16: 729 / 862; 20: 774 / 891.

Tried and dropped the same day: `gemm_q4_k_direct` with blocks of 64 tokens for chunks of
33 to 64 (14336x4096 by 40 / 48 / 64 tokens 112.6 / 117.6 / 129.9 us against the 64-row
block's 106.0 / 109.5 / 120.5); the one-launch `attention` kernel in place of the three
tiled ones for short prompts (Mistral-7B 8 tokens 616 -> 633 tok/s, 16 tokens level, 32
tokens 2017 -> 1984, 64 tokens 2950 -> 2807).

## 2026-11-18: the direct CUDA kernel for Q6_K, chunks of 16 to 32 tokens

`gemm_q6_k_direct`: `gemm_q4_k_direct` with Q6_K's bytes, a lane reading 2 bytes of low
nibbles per column pair and 2 of high bits per two pairs, once for the two 64-column steps
that share them. `bench_cuda` with Q6_K shapes (us a matmul, A100; direct / int8 tensor
cores), 14336x4096 by 8 / 16 / 24 / 32 tokens: 113.7 / 119.5 / 124.1 / 128.7 against 114.1 /
126.0 / 140.6 / 153.4; 4096x4096: 38.1 / 39.5 / 41.0 / 42.6 against 38.6 / 42.1 / 47.3 / 51.4.
(llama.cpp, above: 89.4 / 105.2 / 114.5 at 16 / 24 / 32, so Q6_K is still behind it; the
kernel is 60% slower than the Q4_K one for 1.5x the bytes.) `bench --pp N --tg 0
--gpu-layers 999`, 60 repetitions, without and with in turn, tok/s, Mistral-7B: 8 tokens 611,
611 -> 602, 599 (so from 16, `GEMM_DIRECT_Q6_K_MIN`); 16: 1152, 1140 -> 1173, 1147; 24: 1570,
1563 -> 1628, 1662; 32: 2014, 2011 -> 2061, 2090. Llama-3.2-1B: within its noise. On
Qwen3-30B-A3B's 2048-column `attn_v` it is 25.9 us at 16 tokens (int8 26.5) and 29.3 at 32
(the 64-row block 24.9): 0.1 ms of a 28 ms prompt, left alone. CUDA tests pass with and
without `RLI_CUDA_NO_MMA=1` (152 each).

Against llama.cpp after it (means of two rounds in turn, ours / llama.cpp): Mistral-7B 16:
1158 / 1326; 24: 1644 / 1719; 32: 2077 / 2066. Qwen3-30B-A3B 16: 713 / 862; 24: 904 / 1020;
32: 1133 / 1190.

## 2026-11-18: one 8-byte read per row and step in the direct CUDA kernels

`gemm_q4_k_direct` at 16 tokens with a part taken out (`bench_cuda`, us a 14336x4096 matmul):
whole 74.2; tokens staged once, no barriers 67.5; weights not decoded (the bytes still read)
67.0; no `mma.sync` 76.8. So it is the reads of the weights. Each lane took four 2-byte reads
per row and step, as the tensor core's layout has lane gj hold columns 2gj and 2gj + 1 of
every 8. Dealing the columns differently (lane gj's 16 are those of the word at 8gj: multiply
k's columns 8half + 2gj.. are the matrix's 8gj + 4k + 2half.., and the tokens are staged in
that order, four 4-byte stores a thread where it was one of 16) makes it one 8-byte read:
14336x4096 by 16 tokens 74.2 -> 70.1 us, by 32 tokens 82.2 -> 76.3 (llama.cpp 66.5 and 90.6);
4096x14336 by 16 tokens 76.6 -> 70.3. Q6_K (three unaligned 8-byte reads per row and two
steps, and one scale for a lane's columns) 14336x4096 by 16 tokens 119.5 -> 115.3, by 32
tokens 128.7 -> 125.0. CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (152 each).

Against llama.cpp after it (means of two rounds in turn, ours / llama.cpp): Mistral-7B 8:
638 / 518; 16: 1203 / 1293; 24: 1750 / 1753; 32: 2195 / 2094. Llama-3.2-1B 8: 1886 / 1945;
16: 3748 / 3827; 24: 5206 / 5032; 32: 6771 / 6504. Qwen3-30B-A3B 8: 504 / 546; 16: 718 /
850; 24: 908 / 1014; 32: 1126 / 1191.

## 2026-11-18: Q6_K bytes read once per two steps in the in-block CUDA matmul

`RLI_CUDA_PROF=1` on Mistral-7B prompts of 96 and 192 tokens: `gemm_q6_k_dense` took 241 and
345 us a launch where `gemm_q4_k_dense` took 165 and 245 (46% more; llama.cpp's Q6_K matmul
is 13% slower than its Q4_K one). Steps 2n and 2n + 1 of a Q6_K super-block are the two
nibbles of the same 96 bytes, which `gq6_k_load` read in each; it now reads in the first
only (`GEMM_ID` keeps `raw` between steps; the MoE expert kernels are the same macro).
`RLI_DENSE=1 bench_cuda` with Q6_K shapes, us a matmul, 14336x4096 by 48 / 64 / 96 / 128
tokens: 168.0 / 178.9 / 309.3 / 332.6 -> 150.7 / 162.0 / 278.1 / 300.6 (llama.cpp 162.2 at 64,
207.5 at 96, 259.9 at 128); 4096x4096: 52.8 / 56.2 / 91.7 / 100.0 -> 47.7 / 51.5 / 82.9 / 90.7.
CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (152 each).

Against llama.cpp after it (means of two rounds in turn, ours / llama.cpp): Mistral-7B 48:
2512 / 2617; 96: 2990 / 3370; 192: 3597 / 3868. Qwen3-30B-A3B 48: 1502 / 1401; 96: 2260 /
1688; 192: 3303 / 2736.

## 2026-11-18: blocks of 128 tokens for CUDA chunks of 65 to 128 (Q6_K: from 65)

`gemm_*_dense` with a part taken out, 14336x4096 Q4_K by 64 tokens (`RLI_DENSE=1 bench_cuda`,
us): whole 120.6; weights read, decoded and staged in a block's first step only 97.5; read but
not decoded 103.5; tokens staged once 113.6; no `mma.sync` 128.5. Against llama.cpp's matmul
(above) ours has the same slope, 46 us per 32 tokens of this matrix, and loses where a chunk
ends in a tile of few tokens: 96 tokens 195 against 184, though 64 and 128 are ahead. So the
128-token block dropped earlier today (a warp takes 16 rows by 64 tokens; `GEMM_ID`'s T) is
back for the chunks it helps. `bench_cuda`, us a matmul, 64-token blocks -> 128-token blocks,
14336x4096 Q4_K by 72 / 80 / 96 / 112 / 128 / 160 / 192 tokens: 180.1 / 184.6 / 194.9 / 204.2 /
215.1 / 292.8 / 312.5 -> 165.4 / 169.3 / 182.2 / 195.6 / 209.4 / 299.9 / 325.3; Q6_K: 262.0 /
266.7 / 278.2 / 289.6 / 301.1 / 414.8 / 436.9 -> 202.3 / 207.0 / 220.5 / 233.5 / 248.7 / 383.1 /
411.4. So Q4_K takes them for 65 to 128 tokens (`GEMM_DENSE_WIDE_MAX`) and Q6_K from 65 on.
`bench --pp N --tg 0 --gpu-layers 999`, R = 2400 / N, without and with in turn, Mistral-7B
tok/s: 72 tokens 2422, 2434 -> 2705, 2704; 96: 3004, 2986 -> 3245, 3174; 128: 3377, 3412 ->
3560, 3584; 160: 3309, 3275 -> 3329, 3395; 192: 3572, 3669 -> 3583, 3732. CUDA tests pass
with and without `RLI_CUDA_NO_MMA=1` (152 each).

Against llama.cpp after it (means of two rounds in turn, ours / llama.cpp): Mistral-7B 72:
2718 / 2933; 96: 3249 / 3240; 128: 3626 / 3454; 192: 3670 / 3919. Llama-3.2-1B 72: 9608 / 8974;
96: 11261 / 10394; 128: 11185 / 10699; 192: 13813 / 12341.

Tried and dropped the same day (Mistral-7B, same command): dequant + GEMM from 65 tokens in
place of the in-block kernel (80 tokens 2610 -> 1811 tok/s, 96: 2933 -> 2087, 128: 3410 ->
2635, 192: 3601 -> 3146); the in-block kernel past 192 tokens (224: 3449 -> 3544, 256: 3774 ->
3801, 320: 3906 -> 3843, 384: 4373 -> 3735, 512: 4720 -> 3710).

## 2026-11-18: the direct CUDA kernel by expert, for few pairs an expert

A prompt chunk's MoE experts ran `gemm_*_id` (64 rows of an expert by its pairs, 64 at a
time, weights staged) from 20 tokens. But 32 tokens of Qwen3-30B-A3B are 256 pairs over 128
experts: every tile is nearly empty, the case `gemm_*_direct` exists for. `gemm_*_direct_id`
is that kernel taking an expert and its pairs (`GEMM_DIRECT`, one macro for the four now).
`bench --pp N --tg 0 --gpu-layers 999`, R = 2400 / N (at most 60), staged and direct in
turn, Qwen3-30B-A3B tok/s: 16 tokens (its down projection only) 745, 740 -> 767, 758; 24:
926, 923 -> 1042, 1033; 32: 1171, 1163 -> 1277, 1282; 48: 1497, 1526 -> 1575, 1587; 64: 1843,
1860 -> 1886, 1868; 80: 1997, 1992 -> 1972, 1966; 96: 2276, 2248 -> 2173, 2142; 112: 2458,
2429 -> 2321, 2317; 128: 2626, 2709 -> 2500, 2473; 512: 4493, 4490 -> 3514, 3514. So up to 4
pairs an expert on average (`GEMM_DIRECT_ID_PAIRS`: 64 tokens of this model). Grouping gate
and up from 8, 12 or 16 tokens, or down from 8, was still slower than their matvecs (8 tokens
508 -> 403 tok/s, 12 tokens 651 -> 583, 16 tokens 760 -> 735). CUDA tests pass with and
without `RLI_CUDA_NO_MMA=1` (152 each; `matmul_q_id_matches_matvec_q_id` now also runs 9
tokens, which take the new kernel).

Against llama.cpp after it (means of two rounds in turn, ours / llama.cpp): Qwen3-30B-A3B 8:
508 / 527; 16: 767 / 862; 24: 1033 / 1038; 32: 1293 / 1199; 48: 1576 / 1401; 64: 1882 / 1529.
Mistral-7B 8: 640 / 520; 16: 1203 / 1302; 24: 1731 / 1728; 32: 2181 / 2080; 48: 2496 / 2534;
64: 2994 / 2908.

## 2026-11-18: Q6_K's offset once per super-block in the AVX2 row kernel: faster kernel, same decode

The idea left on 2026-11-13: the AVX2 one-row Q6_K kernel takes the format's offset of 32 off
with a second `maddubs` and a subtract per 32 weights; with the sum of each 16 activations
kept in `Q8Act` (16 i16 per 256 values) it is one `madd` per super-block instead, and the
unsigned weights then need four i32 sums a super-block instead of two. Scratch branch
`q6k-offset-once` (deleted), `Benchmark` with `base=main`, one thread (runs 44385171538,
44385181496, 44385192268; two rounds each). `bench_matvec` Q6_K matvec, ms: Xeon 6975P-C
0.490, 0.476 -> 0.441, 0.452 and 0.520, 0.485 -> 0.442, 0.460; EPYC 7763 0.518, 0.501 -> 0.476,
0.473 (and 0.538 -> 0.484 on an EPYC 7F72). Llama-3.2-1B decode, tok/s: Xeon 15.9, 16.2 ->
16.1, 15.7 and 17.2, 16.5 -> 16.3, 16.4; EPYC 7763 17.7, 17.9 -> 18.1, 17.8. Qwen2.5-0.5B
(Q5_0/Q8_0, no Q6_K) the same, as it should be. A kernel 5-10% faster on a third of the
model's weights and no decode gain to show for it: not merged. No Zen 4 runner came up.

For the record of where one-thread decode on Zen 4 stands (table of 2026-11-17 above): 21.8
tok/s against llama.cpp's 23.1 by default, 27.0 with `--repack`. llama.cpp's figure is with
its own repacked copy of the Q4_K weights (553 MiB peak against our 91, 549 with `--repack`),
so at equal memory we are 17% ahead, and 6% behind for a sixth of the memory.

## 2026-11-18: prompt attention on f16 tensor cores (CUDA)

Mistral-7B prompts of 160 to 512 tokens were 3-19% behind llama.cpp (ours / llama.cpp, tok/s,
two rounds in turn: 160: 3391 / 3726; 224: 3370 / 4177; 256: 3857 / 4237; 320: 3961 / 4252;
384: 4382 / 4633; 512: 4744 / 4899), though llama.cpp's own matmul there is no faster than
ours (`test-backend-ops perf`, Q4_K 14336x4096 by 72 / 160 / 192 / 224 / 256 / 384 / 512
tokens: 156.4 / 296.5 / 340.0 / 378.9 / 418.2 / 606.4 / 797.8 us; 4096x4096: 52.7 / 90.4 /
107.2 / 120.8 / 141.4 / 195.1 / 254.4; Q6_K 14336x4096: 190.0 / 357.9 / 385.8 / 434.1 / 469.1 /
678.9 / 892.7; ours at 192 tokens 312.5 and 100.3). At 192 tokens a layer's seven matmuls
come to about 1450 us of llama.cpp's 1530 and 1350 of our 1635: the rest of a layer cost us
285 us and llama.cpp 80. `RLI_CUDA_PROF=1` at 512 tokens: `attn_pv` 252 us a layer,
`attn_scores` 234, `attn_softmax` 67, `swiglu` 110, 14% of the prompt in attention.

`attn_scores_h` and `attn_pv_h` are those two on f16 `mma.sync` with f32 sums, the tiles
staged as f16 (V with a row per value, since the product wants it packed along positions);
not under `precise`, head sizes a multiple of 32. At 512 tokens `attn_scores` 234 -> 74 us a
launch and `attn_pv` 252 -> 152; reading the next step's tiles while this one is multiplied
144; 64 query rows a block, so V is staged half as often, 124 (256 tokens: 94 -> 52). `bench
--pp N --tg 0 --gpu-layers 999`, f32 kernels and these in turn (with the 32-row sums),
Mistral-7B tok/s: 64 tokens 2981, 2969 -> 3025, 3062; 192: 3715, 3738 -> 3695, 3807; 256: 3771,
3701 -> 3837, 3941; 512: 4720, 4779 -> 5045, 5073. Logits of a 900-character prompt against
`--precise`: 0.083 -> 0.063, top-1 the same. CUDA tests pass with and without
`RLI_CUDA_NO_MMA=1` (153 each, with the new `half_attention_matches_f32_attention`).

Against llama.cpp after it (means of two rounds in turn, ours / llama.cpp): Mistral-7B 64:
3026 / 2931; 128: 3678 / 3460; 192: 3754 / 3920; 256: 3895 / 4297; 384: 4683 / 4595; 512: 5155 /
4947. Llama-3.2-1B 64: 10630 / 8740; 128: 10362 / 9992; 192: 15435 / 12671; 256: 15003 / 14992;
384: 17442 / 16135; 512: 20631 / 17701. Qwen3-30B-A3B 64: 1912 / 1529; 128: 2770 / 1770; 192:
3420 / 2724; 256: 3804 / 2915; 384: 4680 / 3860; 512: 5007 / 4255.

## 2026-11-19: f16 activations written by the kernel that makes them (CUDA)

After the tensor-core attention, `RLI_CUDA_PROF=1` on a 256-token Mistral-7B prompt still showed
`to_f16` 500 launches of 17.6 us (four a layer: the input of Q/K/V, of the output projection, of
gate/up and of down), `quant_x` 128 of 19.7 (the attention result quantized for an output
projection that reads f16) and the int8 copies `rmsnorm` and `swiglu` wrote for nobody. With
those launches skipped (wrong results, an upper bound): 256 tokens 3850 -> 4326 tok/s, 512
tokens 5120 -> 5763.

`rmsnorm`, `swiglu` and `attn_pv_h` now write f16 into `Scratch::xh` when the matmuls that
follow read that (`fed` / `fed_add` in rli-models' cuda.rs mirror `mm_each` / `mm_add`), and
quantize only for int8 matmuls. At 256 tokens `to_f16` and `quant_x` are gone from the profile,
`swiglu` 59.8 -> 47.6 us, `rmsnorm` 28.7 -> 22.4. `bench --pp N --tg 0 --gpu-layers 999`, the
build before and this one in turn, Mistral-7B tok/s: 64 tokens 3057, 3054 -> 3035, 3099; 192:
3789, 3821 -> 3789, 3894; 256: 3829, 3803 -> 4075, 3899; 512: 5195, 5160 -> 5391, 5239. Logits
against `--precise` (900-, 300- and 60-character prompts): 0.060, 0.062, 0.085, top-1 the same.
CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (153 each).

Against llama.cpp (means of two rounds in turn, ours / llama.cpp): Mistral-7B 16: 1272 / 1302;
64: 3091 / 2931; 128: 3682 / 3416; 160: 3480 / 3804; 192: 3849 / 3884; 224: 3553 / 4081; 256:
3970 / 4192; 320: 4243 / 4213; 384: 4842 / 4587; 512: 5400 / 4845. Llama-3.2-1B 5-27% ahead at
16 to 512. Qwen3-30B-A3B 16: 786 / 861; 64 to 512: 18-54% ahead. What is left on Mistral-7B
is at lengths that are not whole 64-token blocks (160: 9% behind, 224: 13%).

Tried after it and dropped (Mistral-7B, A100, same command, each pair run in turn twice):
- A tail of 8 to 32 tokens past whole 64-token blocks through `gemm_*_direct`, the rest through
  the in-block kernel (two launches adding into y; with the in-block kernel allowed up to 224
  tokens): 96 tokens 3276, 3305 -> 3002, 2914 tok/s; 160: 3444, 3490 -> 3584, 3444; 224: 3710,
  3775 -> 3518, 3525. A prompt's cost still goes up a step per started 64-token block (128
  tokens 34.8 ms, 160 46.0, 192 49.9, 224 63.0, 256 64.5) where llama.cpp's is a line (37.5,
  42.1, 49.4, 54.9, 61.1): that step is the 9% and 13% at 160 and 224.
- Upper bounds from skipping work (wrong results): no `zero_rows` launches, 256 tokens 4025,
  4019 -> 4037, 3935; 512: 5282, 5381 -> 5747, 5656. `swiglu` storing f16 only: nothing at
  either length.

## 2026-11-19: the in-block CUDA matmul up to 256 tokens; where a 160-token prompt goes

Per matrix (`bench_cuda` with its loops set to these lengths, Q4_K, us; in-block / dequant +
GEMM / llama.cpp's `test-backend-ops perf`): 4096x4096 by 160 tokens 92.7 / 124.9 / 90.4; by
224: 118.1 / 131.9 / 120.8; by 256: 126.2 / 133.5 / 141.4. 14336x4096 by 160: 292.8 / 420.2 /
296.5; by 224: 377.6 / 450.6 / 378.9; by 256: 399.9 / 452.8 / 418.2. 4096x14336 by 160: 285.3 /
391.3; by 224: 372.7 / 398.1; by 256: 400.1 / 395.7. Whole Mistral-7B prompts, `bench --pp N
--tg 0 --reps 30 --gpu-layers 999`, the in-block kernel up to 192 tokens and up to 320 in
turn three times: 224 tokens 3662, 3625, 3651 -> 3878, 3862, 3862 tok/s; 256: 4085, 4030, 4039
-> 4114, 4128, 4090; 320: 4266, 4216, 4238 -> 4163, 4175, 4155. So `GEMM_DENSE_TOKENS` ends at
256 (the trial of 2026-11-18 that found nothing there used runs of 8 to 10).

A fully offloaded chunk that returns no logits no longer downloads the hidden states nobody
reads. With both, previous build and this one in turn twice: 64 tokens 3014, 3010 -> 3104,
3096; 160: 3515, 3516 -> 3547, 3524; 224: 3639, 3656 -> 3832, 3860; 256: 4037, 4052 -> 4097,
4110. CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (153 each). Against llama.cpp
(two rounds in turn, ours / llama.cpp): Mistral-7B 16: 1276 / 1300; 72: 2788 / 2938; 160: 3535 /
3785; 224: 3845 / 4099; 256: 3976 / 4111.

Where a 160-token Mistral-7B prompt goes (44.8 ms; a launch skipped by kernel name, `--reps
400`, so each runs in sequence, not alone as under `RLI_CUDA_PROF`): with every matmul
skipped 4.3 ms, and of that `attn_pv_h` 0.76, `swiglu` 0.55, `rmsnorm` 0.46, `attn_softmax`
0.38, `attn_scores_h` 0.29, `rope` 0.27, `to_f16_zero` 0.18; with those skipped too 0.9 ms
(embeddings up, hidden states down). The matmuls are 39.5 ms, which is also what the
per-matrix times above add up to, for llama.cpp's as for ours.

## 2026-11-19: f16 written by the producing kernel in MoE layers too (CUDA)

`moe` took its chunk through `to_f16` twice a layer: the normalized input for the grouped
gate and up stacks (20+ tokens), and the SwiGLU result for the grouped down (12+). `rmsnorm`
and `swiglu` now write both (and `rmsnorm` no longer quantizes for stacks that read f16).
Qwen3-30B-A3B, `bench --pp N --tg 0 --reps 1600/N --gpu-layers 999`, the build before and this
one in turn twice: 16 tokens 777, 777 -> 790, 789 tok/s; 24: 1050, 1056 -> 1064, 1078; 32: 1325,
1296 -> 1309, 1314; 64: 1875, 1930 -> 1879, 1932; 192: 3348, 3468 -> 3511, 3567. Logits against
`--precise` differ from run to run on this model as before (100-, 60- and 40-character
prompts, two runs each: 0.18, 0.19 / 0.93, 0.69 / 1.59, 1.69 before; 0.18, 0.75 / 0.78, 0.78 /
1.68, 1.51 after). CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (153 each). Against
llama.cpp (two rounds, ours / llama.cpp): 8 tokens 519 / 548; 16: 778 / 861; 24: 1062 / 1026.

Tried and dropped the same day:
- `gemm_*_dense96`, the in-block kernel with 96 tokens a block, for chunks of 65 to 96 and
  the like (Mistral-7B, in turn twice, tok/s): 72 tokens 2788, 2783 -> 2807, 2825; 80: 3015,
  2952 -> 3025, 3021; 96: 3303, 3379 -> 3358, 3295; 160: 3570, 3520 -> 3519, 3552.
- The MoE grouping thresholds again, `--reps 100` (gate/up from 20 and down from 12 / both
  from 8 / down from 8): 16 tokens 783 / 753 / 783; 12: 668 / 601 / 677; 8: 529 / 421 / 521.

Where a 16-token Qwen3-30B-A3B prompt goes (20.1 ms; launches skipped by kernel name,
`--reps 200`): without the experts' int8 gate and up matvecs 10.0 ms, without the tensor-core
matmuls (Q/K/V/O and the grouped down) 13.1, without both 4.3, of which attention 0.95,
`rmsnorm` 0.45, the router 0.45, `moe_topk` 0.3, the rest 0.2 or less each, and 0.9 with every
launch skipped. So half the prompt is two launches a layer of 128 (token, expert) matvecs.

Tried for them and dropped: `mvq_q4_k_grp`, the int8 matvec over pairs grouped by expert, a
row's weights read and decoded once for up to 4 (or 2) of the expert's tokens. Same logits
path as before (a prompt differs by the usual MoE noise). With 4 tokens a pass, old / new
tok/s: 4 tokens 374 / 306; 8: 530 / 463; 12: 682 / 611; 16: 784 / 734. With 2: 8 tokens 529 /
497; 16: 786 / 779, and 126.7 us a launch where the plain matvec takes 125. Reading a third
less weight did not make the launch any shorter: the matvec is not waiting on its weights.

## 2026-11-19: the Q4_K int8 matvec reads 8 super-blocks before using any (CUDA decode)

The grouped trial above said the matvec is not waiting on its weights; what a warp of a
token's matvec does wait on is each step's reads. `mvq_q4_k_step8` issues the reads of both
halves of 8 super-blocks, then does their arithmetic. `bench --pp 0 --tg 256 --reps 3
--gpu-layers 999` on the A100, before and after in turn twice: Llama-3.2-1B 581.6, 583.8 ->
609.9, 608.8 tok/s; Qwen3-30B-A3B 190.3, 192.7 -> 196.1, 198.5; Mistral-7B 170.6, 170.6 ->
173.9, 173.6. Over the 128 (token, expert) rows of a 16-token Qwen3-30B-A3B chunk it was 2%
slower (796 -> 780), so launches of 32 grid rows or more (`MVQ_MANY`) keep 4 a step
(`mvq_q4_k_many`); with that, prompts are as before (Qwen3-30B-A3B 4 / 8 / 16 tokens 378, 530,
785 -> 381, 529, 790; Mistral-7B 4 / 6 tokens 251, 275 -> 257, 276). Decode-path logits against
`--precise` (`dump_logits --per-token`, 200 characters): Llama-3.2-1B 0.31, Mistral-7B 0.17,
top-1 the same. CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (153 each).

## 2026-11-19: CUDA prompts of 2 to 7 tokens

Never compared before: under 8 tokens (`MMQ_MIN_TOKENS`) a chunk ran one int8 matvec a token
and matrix, each reading the matrix again. Ours / llama.cpp, tok/s, two rounds in turn
(`cmp_only.sh`, A100): Mistral-7B 2 tokens 209 / 271; 4: 254 / 424; 6: 276 / 492.
Llama-3.2-1B 2: 575 / 671; 4: 1009 / 1291; 6: 1225 / 1646. Qwen3-30B-A3B 2: 246 / 290; 4: 365 /
405; 6: 436 / 484.

- `mvq_q4_k_multi`: a warp takes a weight row and up to 4 tokens a pass, the row read and
  decoded once. `RLI_CUDA_PROF=1`, Mistral-7B, a launch: 36.4 us for one token, 43.4 for two,
  54.4 for four.
- `gemm_*_direct` takes about as long for 4 tokens as for 16, so it is ahead from 5 tokens
  for Q4_K (`GEMM_DIRECT_TOKENS`, was 8) and from 3 for Q6_K, which has no multi-token
  matvec (`GEMM_DIRECT_Q6_K_FEW`).

`bench --pp N --tg 0 --reps 150 --gpu-layers 999`, tok/s (a matvec a token as before / multi
/ multi and Q6_K direct / the direct kernel for both types): Mistral-7B 2 tokens 213 / 224 /
215 / 177; 3: 239 / 283 / 295 / 267; 4: 258 / 321 / 356 / 346; 5: 270 / 321 / 368 / 434; 6: 278 /
345 / - / 519; 7: 284 / 366 / - / 601. Llama-3.2-1B 2: 633 / 715 / 685 / 566; 3: 954 / 959 / 975 /
857; 4: 1115 / 1217 / 1304 / 1112; 5: 1313 / 1314 / 1406 / 1456; 6: 1402 / 1518 / - / 1612; 7:
1468 / 1722 / - / 1930. CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (153 each; the
int8 test now has a 3-token case beside the 5-token one).

Against llama.cpp after it (two rounds, ours / llama.cpp): Mistral-7B 2: 216 / 272; 3: 288 /
362; 4: 349 / 427; 5: 428 / 469; 6: 518 / 488; 7: 584 / 526. Llama-3.2-1B 2: 646 / 704; 3: 736 /
883; 4: 1037 / 1270; 5: 1344 / 1445; 6: 1538 / 1586; 7: 1648 / 1920. Qwen3-30B-A3B 2: 243 / 295;
3: 310 / 369; 4: 365 / 421; 5: 405 / 445; 6: 455 / 495; 7: 486 / 519. Still behind at 2 to 5
tokens: a 2-token Mistral-7B chunk takes 9.2 ms where a decoded token takes 5.75.

Two more for those chunks, each against the build before it (`--reps 200`, in turn twice):
- A multi-token matvec kernel per token count (`mvq_q4_k_multi2`, `3`, `4`): the one that
  holds four tokens' activations ran two tokens slower. Mistral-7B 2 tokens 223, 224 -> 238,
  239 tok/s; 3: 297, 297 -> 306, 307; 4: 358, 356 -> 357, 358.
- The one `attention` launch under 8 tokens (`ATTN_TILED_MIN`), which also quantizes its
  result, in place of the three tiled ones and `quant_x`: Mistral-7B 2 tokens 237, 241 -> 249,
  248; 3: 308, 304 -> 317, 319; 4: 359, 357 -> 366, 367; 6: 524, 527 -> 534, 535. Llama-3.2-1B
  2: 795, 778 -> 855, 863; 3: 1075, 1054 -> 1128, 1079; 4: 1273, 1303 -> 1416, 1353; 6: 1859,
  1853 -> 1845, 1922.

Where a short Mistral-7B chunk went before these two (launches skipped by kernel name,
`--reps 200`; 1 / 2 / 4 tokens): whole 5.79 / 8.95 / 11.06 ms; the Q4_K matvecs 3.20 / 5.27 /
6.90; Q6_K (matvecs; the direct kernel at 4) 1.26 / 1.93 / 2.15; everything else 1.31 / 1.94 /
2.01, of which attention 0.21 / 0.73 / 0.63. One token is replayed as a CUDA graph with gate
and up in one launch.

CUDA tests pass with and without `RLI_CUDA_NO_MMA=1` (153 each). Against llama.cpp (two
rounds, ours / llama.cpp): Mistral-7B 2: 233 / 266; 3: 311 / 354; 4: 358 / 416; 5: 434 / 463; 6:
514 / 495; 7: 591 / 532. Llama-3.2-1B 2: 642 / 613; 3: 826 / 947; 4: 1015 / 1202; 5: 1308 / 1399;
6: 1543 / 1702; 7: 1885 / 1852. Qwen3-30B-A3B 2: 272 / 294; 3: 327 / 364; 4: 386 / 421; 5: 421 /
455; 6: 471 / 497; 7: 503 / 538.

Tried and dropped: the multi-token matvecs reading 8 super-blocks before using any, as the
one-token kernel does (`--reps 200`, in turn twice, 4 a step / 8): Mistral-7B 2 tokens 254,
257 / 251, 249 tok/s; 3: 322, 322 / 311, 311; 4: 374, 373 / 343, 347. Llama-3.2-1B 2: 832, 856 /
880, 878; 3: 1140, 1135 / 1074, 1072; 4: 1405, 1362 / 1330, 1257. With several tokens a warp
has arithmetic enough per step that it is no longer just waiting for its reads: a second
token adds half again to the Q4_K matvecs' time (3.2 -> 4.7 ms a chunk) with no more weight
read.

Q6_K has the multi-token matvecs too (`mvq_q6_k_multi2`, `3`, `4`), and its direct kernel
starts at 5 tokens, not 3 (`--reps 200`, in turn twice: before / direct from 3 / multi to 4):
Mistral-7B 2 tokens 253, 256 / 267, 267 / 267, 265 tok/s; 3: 325, 323 / 317, 317 / 331, 332; 4:
371, 371 / 371, 371 / 392, 389; 5: 451, 447 / 447, 454 / 446, 448. Llama-3.2-1B within its noise
(2: 844, 837 / 888, 862 / 875, 861; 4: 1429, 1383 / 1389, 1369 / 1414, 1440). CUDA tests pass
both ways (153 each). Against llama.cpp (two rounds, ours / llama.cpp): Mistral-7B 2: 247 /
269; 3: 320 / 355; 4: 378 / 423; 5: 428 / 453. Llama-3.2-1B 2: 678 / 670; 3: 958 / 1014; 4: 1149 /
1141; 5: 1320 / 1460. Qwen3-30B-A3B 2: 269 / 301; 3: 340 / 365; 4: 400 / 421; 5: 418 / 455.

Tried and dropped on the same chunks: the int8 tensor-core matmul from 2 tokens in place of
the matvecs and the direct kernel (Mistral-7B 2 tokens 255 -> 160 tok/s, 4: 376 -> 315, 6:
539 -> 469); the direct kernel staging only the token rows a chunk has, by eights (5 to 24
tokens: within 1% either way).

Chunks of up to 32 tokens (`GRAPH_TOKENS`) are replayed as CUDA graphs, one per chunk
length, as a decoded token is. Launched one by one / replayed (`--reps 200` up to 7 tokens,
1600 / N above; in turn twice): Mistral-7B 2 tokens 264, 264 / 280, 280 tok/s; 4: 391, 391 /
406, 402; 7: 624, 625 / 647, 649; 8: 692, 694 / 725, 731; 16: 1282, 1308 / 1339, 1333; 24: 1837,
1812 / 1878, 1847; 32: 2284, 2249 / 2371, 2363. Llama-3.2-1B 2: 877, 861 / 956, 911; 4: 1477,
1447 / 1528, 1487; 7: 2150, 2143 / 2295, 2308; 8: 2434, 2396 / 2709, 2625; 16: 4197, 4332 /
4871, 4846; 24: 5925, 6100 / 6482, 6233. Qwen3-30B-A3B 2: 280, 283 / 322, 316; 4: 417, 420 /
449, 456; 7: 517, 516 / 546, 545; 8: 531, 533 / 566, 569; 16: 787, 788 / 813, 811; 24: 1072,
1062 / 1127, 1144; 32: 1315, 1314 / 1362, 1375.
**The benchmark repeats one length, which is the best case.** A graph captured anew every
run (launched one by one / captured each time): Mistral-7B 2 tokens 264 / 253; 8: 670 / 626;
32: 1934 / 1779. Llama-3.2-1B 2: 881 / 760; 8: 1730 / 1739. Qwen3-30B-A3B 2: 286 / 252; 8: 515 /
486; 32: 1192 / 1156. So a length gets its graph the second time it comes up: its first chunk
runs as before, its second pays the capture.

CUDA tests pass both ways (154 each, with `short_chunk_repeated_matches_its_first_run`).
Against llama.cpp (two rounds, ours / llama.cpp): Mistral-7B 2: 270 / 274; 3: 323 / 373; 4:
399 / 411; 5: 447 / 444; 6: 542 / 492; 7: 623 / 533; 8: 702 / 517; 16: 1333 / 1300; 24: 1858 /
1716; 32: 2331 / 2083. Llama-3.2-1B 2: 719 / 715; 3: 947 / 910; 4: 1208 / 1191; 5: 1426 / 1358;
6: 1630 / 1485; 7: 1770 / 1912; 8: 2029 / 1940; 16: 4129 / 3760; 24: 5652 / 5264; 32: 8016 /
6098. Qwen3-30B-A3B 2: 290 / 302; 3: 365 / 376; 4: 428 / 420; 5: 461 / 444; 6: 503 / 485; 7:
536 / 529; 8: 550 / 529; 16: 817 / 862; 24: 1125 / 1023; 32: 1379 / 1188.

`RLI_CUDA_PROF=1` on Mistral-7B at 2 / 3 / 4 tokens: the Q4_K multi-token matvec 34.1 / 40.1 /
47.4 us a launch, the Q6_K one 54.5 / 69.4 / 55.6. So Q6_K takes its 4-token kernel for 3
tokens and has none for 3 (`--reps 200`, in turn twice): Mistral-7B 3 tokens 343, 348 ->
359, 361 tok/s; Llama-3.2-1B 1294, 1201 -> 1298, 1277; Qwen3-30B-A3B 392, 386 -> 387, 387.
CUDA tests pass both ways (154 each). Against llama.cpp (two rounds, ours / llama.cpp):
Mistral-7B 2: 267 / 273; 3: 348 / 365; 4: 391 / 422. Llama-3.2-1B 2: 761 / 684; 3: 986 / 923; 4:
1219 / 1200. Qwen3-30B-A3B 2: 301 / 297; 3: 367 / 357; 4: 428 / 421.

Tried and dropped: graphs for chunks of up to 256 tokens (`--reps 30`, in turn twice, at 32 /
at 256): Mistral-7B 48 tokens 2551, 2470 / 2567, 2597 tok/s; 72: 2744, 2822 / 2788, 2832; 160:
3559, 3485 / 3559, 3572; 256: 4120, 4068 / 4094, 4119. Qwen3-30B-A3B 48: 1611, 1610 / 1635,
1687; 72: 1924, 1935 / 1950, 1961; 160: 3173, 3214 / 3256, 3215; 256: 4045, 4048 / 4102, 4112.
One or two percent at most, inside the runs' spread: a chunk that long is its matmuls.

Tried and dropped: the multi-token Q4_K matvec reading each token's activations just before
their products, not all tokens' first (fewer values held at once; `--reps 200`, in turn
twice): Mistral-7B 2 tokens 280, 279 -> 257, 257 tok/s; 3: 356, 362 -> 332, 328; 4: 404, 406 ->
398, 399. With the read-ahead trial above, the kernel sits between the two: more reads in
flight than a step's cost it registers, fewer cost it waits.
