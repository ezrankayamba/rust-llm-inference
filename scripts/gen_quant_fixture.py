"""Regenerate crates/core/tests/fixtures/dequant_ref.bin from the reference `gguf` package.

Usage: python scripts/gen_quant_fixture.py   (needs `pip install gguf numpy`)
Layout per record: u32 ggml_type, u32 n_bytes, bytes, u32 n_floats, f32[n_floats] (little-endian).
"""
import struct
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

# (type, block bytes, offsets of f16 scale fields)
TYPES = [(T.Q4_0, 18, [0]), (T.Q4_1, 20, [0, 2]), (T.Q5_0, 22, [0]), (T.Q5_1, 24, [0, 2]), (T.Q8_0, 34, [0]), (T.Q4_K, 144, [0, 2]), (T.Q5_K, 176, [0, 2]), (T.Q6_K, 210, [208]), (T.NVFP4, 36, [])]
rng = np.random.default_rng(1234)
out = bytearray()
for qtype, bb, scale_offs in TYPES:
    blocks = rng.integers(0, 256, size=(3, bb), dtype=np.uint8)
    for off in scale_offs:
        blocks[:, off:off + 2] = rng.uniform(-0.05, 0.05, size=(3, 1)).astype(np.float16).view(np.uint8)
    raw = blocks.tobytes()
    ref = dequantize(np.frombuffer(raw, dtype=np.uint8), qtype).astype(np.float32).ravel()
    out += struct.pack("<II", int(qtype), len(raw)) + raw + struct.pack("<I", ref.size) + ref.tobytes()
Path(__file__).resolve().parent.parent.joinpath("crates/core/tests/fixtures/dequant_ref.bin").write_bytes(out)
print(len(out), "bytes")
