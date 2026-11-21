"""Write tiny random MoE GGUFs (Mixtral-style llama, Qwen2-MoE with a shared expert, and the
Mixtral-style one with NVFP4 weights and their tensor scales) using the reference `gguf` writer,
for crates/models/tests/moe_reference.rs.

usage: python scripts/make_tiny_moe.py <out_dir>    (needs `pip install gguf numpy`)
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as Q, GGUFWriter
from gguf.quants import quantize

out = Path(sys.argv[1])
D, FF, SH_FF, N_EXP, TOP_K, LAYERS, HEADS, KV_HEADS = 64, 96, 64, 4, 2, 2, 4, 2
HD = D // HEADS
byte_chars = [chr(c) for c in range(33, 127)] + [chr(c) for c in range(161, 173)] + [chr(c) for c in range(174, 256)]
extra = [chr(256 + i) for i in range(256 - len(byte_chars))]
VOCAB = byte_chars + extra + ["he"]  # the 256 GPT-2 byte symbols plus one merge


def build(arch: str, path: Path, seed: int, nvfp4: bool = False):
    # NVFP4 blocks are 64 weights, so its expert rows are too.
    ff = 128 if nvfp4 else FF
    rng = np.random.default_rng(seed)
    r = lambda *shape: (rng.standard_normal(shape) * 0.2).astype(np.float32)
    w = GGUFWriter(str(path), arch)
    w.add_block_count(LAYERS)
    w.add_context_length(256)
    w.add_embedding_length(D)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    w.add_expert_count(N_EXP)
    w.add_expert_used_count(TOP_K)
    if arch == "llama":
        w.add_feed_forward_length(ff)
    else:
        w.add_expert_feed_forward_length(ff)
        w.add_expert_shared_feed_forward_length(SH_FF)
    w.add_tokenizer_model("gpt2")
    w.add_token_list(VOCAB)
    w.add_token_merges(["h e"])

    def q8(name, a):  # a has numpy shape (..., cols): GGUF dims are the reverse
        if nvfp4:
            return fp4(name, a.shape)
        w.add_tensor(name, quantize(a, Q.Q8_0), raw_dtype=Q.Q8_0)

    def fp4(name, shape):
        # Random codes; scale bytes 0x18-0x1f (0.016-0.029 once halved) give weights of about
        # the others' size. The tensor scale, one per expert for a stack, multiplies W x.
        *outer, cols = shape
        blocks = rng.integers(0, 256, (*outer, cols // 64, 36), dtype=np.uint8)
        blocks[..., :4] = rng.integers(0x18, 0x20, (*outer, cols // 64, 4), dtype=np.uint8)
        w.add_tensor(name, blocks.reshape(*outer, -1), raw_dtype=Q.NVFP4)
        n = outer[0] if len(outer) == 2 else 1
        w.add_tensor(name.replace(".weight", ".scale"), rng.uniform(0.6, 1.4, n).astype(np.float32))

    w.add_tensor("token_embd.weight", r(len(VOCAB), D))
    w.add_tensor("output_norm.weight", 1 + r(D))
    q8("output.weight", r(len(VOCAB), D))
    for i in range(LAYERS):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", 1 + r(D))
        w.add_tensor(p + "ffn_norm.weight", 1 + r(D))
        q8(p + "attn_q.weight", r(D, D))
        q8(p + "attn_k.weight", r(KV_HEADS * HD, D))
        q8(p + "attn_v.weight", r(KV_HEADS * HD, D))
        q8(p + "attn_output.weight", r(D, D))
        if arch == "qwen2moe":
            for n, rows in (("q", D), ("k", KV_HEADS * HD), ("v", KV_HEADS * HD)):
                w.add_tensor(p + f"attn_{n}.bias", r(rows))
        w.add_tensor(p + "ffn_gate_inp.weight", r(N_EXP, D) * 5)  # sharper routing
        q8(p + "ffn_gate_exps.weight", r(N_EXP, ff, D))
        q8(p + "ffn_up_exps.weight", r(N_EXP, ff, D))
        q8(p + "ffn_down_exps.weight", r(N_EXP, D, ff))
        if arch == "qwen2moe":
            w.add_tensor(p + "ffn_gate_inp_shexp.weight", r(D))
            q8(p + "ffn_gate_shexp.weight", r(SH_FF, D))
            q8(p + "ffn_up_shexp.weight", r(SH_FF, D))
            q8(p + "ffn_down_shexp.weight", r(D, SH_FF))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


build("llama", out / "tiny_mixtral.gguf", 1)
build("qwen2moe", out / "tiny_qwen2moe.gguf", 2)
build("llama", out / "tiny_nvfp4.gguf", 3, nvfp4=True)
