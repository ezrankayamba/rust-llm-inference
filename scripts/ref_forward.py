"""Independent NumPy forward pass for llama/qwen2 GGUFs, to check the engine's logits.

usage: python scripts/ref_forward.py <model.gguf> <comma-separated token ids> <engine_logits.f32> [ref_out.f32]
       (needs `pip install gguf numpy`; get ids + logits from the `dump_logits` example)

Dequantization comes from the reference `gguf` package; the transformer math follows the HF
reference implementations (GGUF llama Q/K are pre-permuted, so llama uses adjacent-pair RoPE).
"""
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

path, ids, engine_path = sys.argv[1], [int(t) for t in sys.argv[2].split(",")], sys.argv[3]
r = GGUFReader(path)
kv = {f.name: f.contents() for f in r.fields.values()}
T = {t.name: t for t in r.tensors}
arch = kv["general.architecture"]
W = lambda n: dequantize(T[n].data, T[n].tensor_type).astype(np.float32).reshape(
    [int(d) for d in reversed(T[n].shape)])

n_head, n_kv = kv[f"{arch}.attention.head_count"], kv[f"{arch}.attention.head_count_kv"]
d = kv[f"{arch}.embedding_length"]
hd = d // n_head
eps = kv[f"{arch}.attention.layer_norm_rms_epsilon"]
theta = kv.get(f"{arch}.rope.freq_base", 10000.0)
neox = arch != "llama"
L = kv[f"{arch}.block_count"]


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w


def rope(x):  # x: [seq, heads, hd]
    seq = x.shape[0]
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.arange(seq)[:, None] * inv[None]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    if neox:
        a, b = x[..., : hd // 2], x[..., hd // 2:]
    else:
        a, b = x[..., 0::2], x[..., 1::2]
    ra, rb = a * cos - b * sin, a * sin + b * cos
    out = np.empty_like(x)
    if neox:
        out[..., : hd // 2], out[..., hd // 2:] = ra, rb
    else:
        out[..., 0::2], out[..., 1::2] = ra, rb
    return out


emb = W("token_embd.weight")
x = emb[ids].astype(np.float64)
seq = len(ids)
mask = np.triu(np.full((seq, seq), -np.inf), 1)
for i in range(L):
    p = f"blk.{i}."
    h = rms(x, W(p + "attn_norm.weight"))
    q, k, v = (h @ W(p + f"attn_{n}.weight").T for n in "qkv")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if p + f"attn_{name}.bias" in T:
            t += W(p + f"attn_{name}.bias")
    q = rope(q.reshape(seq, n_head, hd))
    k = rope(k.reshape(seq, n_kv, hd))
    v = v.reshape(seq, n_kv, hd)
    rep = n_head // n_kv
    k, v = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
    att = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd) + mask
    att = np.exp(att - att.max(-1, keepdims=True))
    att /= att.sum(-1, keepdims=True)
    o = np.einsum("hqk,khd->qhd", att, v).reshape(seq, -1)
    x = x + o @ W(p + "attn_output.weight").T
    h = rms(x, W(p + "ffn_norm.weight"))
    swiglu = lambda h, g, u, d: ((h @ g.T) / (1 + np.exp(-(h @ g.T))) * (h @ u.T)) @ d.T
    if p + "ffn_gate_inp.weight" not in T:
        x = x + swiglu(h, W(p + "ffn_gate.weight"), W(p + "ffn_up.weight"), W(p + "ffn_down.weight"))
        continue
    # MoE: softmax router, top-k experts; Mixtral renormalizes the chosen weights, Qwen2-MoE doesn't.
    k_used = kv[f"{arch}.expert_used_count"]
    ge, ue, de = W(p + "ffn_gate_exps.weight"), W(p + "ffn_up_exps.weight"), W(p + "ffn_down_exps.weight")
    logits = h @ W(p + "ffn_gate_inp.weight").T
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    moe = np.zeros_like(x)
    for t in range(seq):
        top = np.argsort(-probs[t])[:k_used]
        wts = probs[t, top] / (probs[t, top].sum() if arch != "qwen2moe" else 1.0)
        for e, wt in zip(top, wts):
            moe[t] += wt * swiglu(h[t], ge[e], ue[e], de[e])
    if p + "ffn_gate_inp_shexp.weight" in T:
        sg = h @ W(p + "ffn_gate_inp_shexp.weight").reshape(-1)
        moe += (1 / (1 + np.exp(-sg)))[:, None] * swiglu(
            h, W(p + "ffn_gate_shexp.weight"), W(p + "ffn_up_shexp.weight"), W(p + "ffn_down_shexp.weight"))
    x = x + moe
out_w = W("output.weight") if "output.weight" in T else emb
ref = (rms(x[-1], W("output_norm.weight")) @ out_w.T).astype(np.float32)
if len(sys.argv) > 4:
    ref.tofile(sys.argv[4])
eng = np.fromfile(engine_path, dtype=np.float32)
err = np.abs(ref - eng).max()
top_ref, top_eng = np.argsort(-ref)[:5], np.argsort(-eng)[:5]
print(f"max|diff|={err:.4g}  logit range={ref.min():.2f}..{ref.max():.2f}  "
      f"top5 ref={top_ref.tolist()} engine={top_eng.tolist()}")
sys.exit(0 if top_ref[0] == top_eng[0] and err < 0.05 * np.abs(ref).max() else 1)
