# Benchmark results

Measured numbers, newest first. Each section says how to reproduce it. GitHub runners vary in CPU
from run to run, so compare numbers only within one table.

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
