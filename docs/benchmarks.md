# Benchmark results

Measured numbers, newest first. Each section says how to reproduce it. GitHub runners vary in CPU
from run to run, so compare numbers only within one table.

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

A captured CUDA graph replays the same launches for 2.8 microseconds each where queueing them
takes 4.0 (`bench_cuda`, "launches"; `Gpu::capture` / `Gpu::replay`); decode does not use it yet.
The device stream is now one of our own, since the default stream cannot be captured.

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
