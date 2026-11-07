"""Check the CUDA kernels' weight decoders against the reference `gguf` dequantizer, on the CPU.

No NVIDIA GPU is needed: harness.cpp compiles the `chunk_*` functions of crates/core/src/cuda.cu
as plain C++. This covers the bit layouts only, not the warp/block reductions.

usage: clang++ -O1 -std=c++17 -o /tmp/harness scripts/cuda_cpu_check/harness.cpp
       python scripts/cuda_cpu_check/check.py /tmp/harness <scratch_dir>   (needs gguf, numpy)
"""
import subprocess, sys, numpy as np
from gguf import GGMLQuantizationType as T, GGML_QUANT_SIZES, quants
harness, d = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
rows, cols = 7, 512
# byte offsets of f16 scale fields per block, set to sane values so random bits decode finitely
f16_fields = {"Q4_0": [0], "Q5_0": [0], "Q8_0": [0], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208]}
worst = 0
for name in ["F32", "F16", "Q4_0", "Q5_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"]:
    t = T[name]; bs, bb = GGML_QUANT_SIZES[t]
    nblk = cols // bs
    if name == "F32":
        raw = rng.standard_normal((rows, cols)).astype(np.float32).view(np.uint8)
    elif name == "F16":
        raw = rng.standard_normal((rows, cols)).astype(np.float16).view(np.uint8)
    else:
        raw = rng.integers(0, 256, (rows, nblk, bb), dtype=np.uint8)
        for off in f16_fields[name]:
            raw[:, :, off:off+2] = rng.uniform(0.001, 0.05, (rows, nblk)).astype(np.float16).view(np.uint8).reshape(rows, nblk, 2)
        raw = raw.reshape(rows, nblk * bb)
    raw = np.ascontiguousarray(raw)
    W = quants.dequantize(raw, t).reshape(rows, cols).astype(np.float64)
    x = rng.standard_normal(cols).astype(np.float32)
    raw.tofile(f"{d}/w.bin"); x.tofile(f"{d}/x.f32")
    subprocess.run([harness, name, str(rows), str(cols), str(raw.shape[1]), f"{d}/w.bin", f"{d}/x.f32", f"{d}/y.f32"], check=True)
    y = np.fromfile(f"{d}/y.f32", dtype=np.float32)
    ref = W @ x.astype(np.float64)
    rel = np.max(np.abs(y - ref)) / max(1e-9, np.max(np.abs(ref)))
    worst = max(worst, rel)
    print(f"{name:5} max rel err {rel:.2e}")
sys.exit(0 if worst < 1e-4 else 1)
