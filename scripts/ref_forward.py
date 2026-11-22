"""Independent NumPy forward pass for llama/qwen2/qwen3(-moe)/deepseek2 GGUFs, to check the
engine's logits.

usage: python scripts/ref_forward.py <model.gguf> <comma-separated token ids> <engine_logits.f32> [ref_out.f32]
       (needs `pip install gguf numpy`; get ids + logits from the `dump_logits` example)

Dequantization comes from the reference `gguf` package; the transformer math follows the HF
reference implementations (GGUF llama Q/K are pre-permuted, so llama uses adjacent-pair RoPE).
DeepSeek follows HF's modeling_deepseek (V2 and V3) as written: full keys and values per head,
not the engine's latent cache.
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


def W(n):
    w = dequantize(T[n].data, T[n].tensor_type).astype(np.float32).reshape(
        [int(d) for d in reversed(T[n].shape)])
    # An NVFP4 tensor's second-level scale (one per expert for stacked experts).
    s = n.removesuffix(".weight") + ".scale"
    if s in T:
        w = w * T[s].data.astype(np.float32).reshape((-1,) + (1,) * (w.ndim - 1))
    return w


n_head = kv[f"{arch}.attention.head_count"]
n_kv = kv.get(f"{arch}.attention.head_count_kv", n_head)
d = kv[f"{arch}.embedding_length"]
hd = kv.get(f"{arch}.attention.key_length", d // n_head)  # Qwen3 heads are wider than d / n_head
rope_dim = kv.get(f"{arch}.rope.dimension_count", hd)
eps = kv[f"{arch}.attention.layer_norm_rms_epsilon"]
theta = kv.get(f"{arch}.rope.freq_base", 10000.0)
neox = arch != "llama"
L = kv[f"{arch}.block_count"]


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w


# Llama 3.1+ long-context scaling ships as per-frequency divisors (llama.cpp's rope_freqs).
freq_factors = W("rope_freqs.weight").astype(np.float64) if "rope_freqs.weight" in T else 1.0


def rope(x):  # x: [seq, heads, hd]; rotates the first rope_dim dims
    seq = x.shape[0]
    inv = theta ** (-np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim) / freq_factors
    ang = np.arange(seq)[:, None] * inv[None]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    r, half = x[..., :rope_dim], rope_dim // 2
    if neox:
        a, b = r[..., :half], r[..., half:]
    else:
        a, b = r[..., 0::2], r[..., 1::2]
    ra, rb = a * cos - b * sin, a * sin + b * cos
    out = x.copy()
    if neox:
        out[..., :half], out[..., half:rope_dim] = ra, rb
    else:
        out[..., 0:rope_dim:2], out[..., 1:rope_dim:2] = ra, rb
    return out


def ds_rope(x, pos):
    """HF DeepSeek RoPE (with YaRN frequencies) of x [seq, heads, rd]: the interleaved pairs are
    first regrouped into halves, which it keeps; q and k get the same order."""
    rd = x.shape[-1]
    base = theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd)
    inv = 1.0 / base
    if kv.get(f"{arch}.rope.scaling.type") == "yarn":
        f = kv[f"{arch}.rope.scaling.factor"]
        orig = kv[f"{arch}.rope.scaling.original_context_length"]
        corr = lambda rot: rd * np.log(orig / (rot * 2 * np.pi)) / (2 * np.log(theta))
        lo = max(np.floor(corr(kv.get(f"{arch}.rope.scaling.yarn_beta_fast", 32.0))), 0)
        hi = min(np.ceil(corr(kv.get(f"{arch}.rope.scaling.yarn_beta_slow", 1.0))), rd - 1)
        if lo == hi:
            hi += 0.001
        mask = 1 - np.clip((np.arange(rd // 2) - lo) / (hi - lo), 0, 1)
        inv = inv / f * (1 - mask) + inv * mask
    # Assumes HF's mscale == mscale_all_dim (true of V2-Lite and V3), so cos and sin are unscaled.
    ang = pos[:, None] * inv[None]
    emb = np.concatenate([ang, ang], -1)
    cos, sin = np.cos(emb)[:, None, :], np.sin(emb)[:, None, :]
    s, h = x.shape[0], x.shape[1]
    x = x.reshape(s, h, rd // 2, 2).transpose(0, 1, 3, 2).reshape(s, h, rd)
    rot = np.concatenate([-x[..., rd // 2:], x[..., :rd // 2]], -1)
    return x * cos + rot * sin


def ds_attention(p, h):
    """HF DeepseekV2Attention / DeepseekV3Attention, eager, over the whole prompt."""
    seq = h.shape[0]
    kvl = kv[f"{arch}.attention.kv_lora_rank"]
    rd = kv[f"{arch}.rope.dimension_count"]
    qk = kv.get(f"{arch}.attention.key_length_mla", kv[f"{arch}.attention.key_length"])
    vd = kv.get(f"{arch}.attention.value_length_mla", kv[f"{arch}.attention.value_length"])
    nope = qk - rd
    if p + "attn_q_a.weight" in T:
        q = rms(h @ W(p + "attn_q_a.weight").T, W(p + "attn_q_a_norm.weight")) @ W(p + "attn_q_b.weight").T
    else:
        q = h @ W(p + "attn_q.weight").T
    q = q.reshape(seq, n_head, qk)
    kva = h @ W(p + "attn_kv_a_mqa.weight").T
    c, k_pe = rms(kva[:, :kvl], W(p + "attn_kv_a_norm.weight")), kva[:, None, kvl:]
    if p + "attn_kv_b.weight" in T:
        kv_b = W(p + "attn_kv_b.weight").reshape(n_head, nope + vd, kvl)
    else:  # split and transposed by newer conversions: k_b [head][latent][nope], v_b [head][v][latent]
        kv_b = np.concatenate([W(p + "attn_k_b.weight").transpose(0, 2, 1), W(p + "attn_v_b.weight")], 1)
    kvs = np.einsum("tc,hrc->thr", c, kv_b)
    k_nope, v = kvs[..., :nope], kvs[..., nope:]
    pos = np.arange(seq, dtype=np.float64)
    q_pe, k_pe = ds_rope(q[..., nope:], pos), ds_rope(k_pe, pos)
    qs = np.concatenate([q[..., :nope], q_pe], -1)
    ks = np.concatenate([k_nope, np.repeat(k_pe, n_head, 1)], -1)
    m = 1.0
    if kv.get(f"{arch}.rope.scaling.type") == "yarn":
        m = 1 + kv.get(f"{arch}.rope.scaling.yarn_log_multiplier", 0.0) * np.log(kv[f"{arch}.rope.scaling.factor"])
    att = np.einsum("qhd,khd->hqk", qs, ks) * (m * m / np.sqrt(qk)) + mask
    att = np.exp(att - att.max(-1, keepdims=True))
    att /= att.sum(-1, keepdims=True)
    o = np.einsum("hqk,khd->qhd", att, v).reshape(seq, -1)
    return o @ W(p + "attn_output.weight").T


def ds_route(logits, bias):
    """HF MoEGate: experts and weights per token, for V2 (softmax) and V3 (sigmoid + bias)."""
    k_used = kv[f"{arch}.expert_used_count"]
    groups = kv.get(f"{arch}.expert_group_count", 1)
    used_groups = kv.get(f"{arch}.expert_group_used_count", 1)
    sigmoid = kv.get(f"{arch}.expert_gating_func", 1) == 2
    scores = 1 / (1 + np.exp(-logits)) if sigmoid else np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    rank = scores + bias if bias is not None else scores.copy()
    if groups > 1:
        g = rank.reshape(groups, -1)
        gs = np.sort(g, -1)[:, -2:].sum(-1) if sigmoid else g.max(-1)
        keep = np.argsort(-gs, kind="stable")[:used_groups]
        drop = np.ones(groups, bool)
        drop[keep] = False
        g[drop] = -np.inf
        rank = g.reshape(-1)
    top = np.argsort(-rank, kind="stable")[:k_used]
    w = scores[top]
    if kv.get(f"{arch}.expert_weights_norm", False):
        w = w / (w.sum() + 1e-20)
    return top, w * kv.get(f"{arch}.expert_weights_scale", 1.0)


emb = W("token_embd.weight")
x = emb[ids].astype(np.float64)
seq = len(ids)
mask = np.triu(np.full((seq, seq), -np.inf), 1)
for i in range(L):
    p = f"blk.{i}."
    h = rms(x, W(p + "attn_norm.weight"))
    if arch == "deepseek2":
        x = x + ds_attention(p, h)
        h = rms(x, W(p + "ffn_norm.weight"))
        swiglu = lambda h, g, u, d: ((h @ g.T) / (1 + np.exp(-(h @ g.T))) * (h @ u.T)) @ d.T
        if p + "ffn_gate_inp.weight" not in T:
            x = x + swiglu(h, W(p + "ffn_gate.weight"), W(p + "ffn_up.weight"), W(p + "ffn_down.weight"))
            continue
        ge, ue, de = W(p + "ffn_gate_exps.weight"), W(p + "ffn_up_exps.weight"), W(p + "ffn_down_exps.weight")
        logits = h @ W(p + "ffn_gate_inp.weight").T
        bias = W(p + "exp_probs_b.bias").astype(np.float64) if p + "exp_probs_b.bias" in T else None
        moe = swiglu(h, W(p + "ffn_gate_shexp.weight"), W(p + "ffn_up_shexp.weight"), W(p + "ffn_down_shexp.weight"))
        for t in range(seq):
            for e, wt in zip(*ds_route(logits[t], bias)):
                moe[t] += wt * swiglu(h[t], ge[e], ue[e], de[e])
        x = x + moe
        continue
    q, k, v = (h @ W(p + f"attn_{n}.weight").T for n in "qkv")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if p + f"attn_{name}.bias" in T:
            t += W(p + f"attn_{name}.bias")
    q, k = q.reshape(seq, n_head, hd), k.reshape(seq, n_kv, hd)
    # Qwen3: per-head RMSNorm of q and k, before RoPE.
    if p + "attn_q_norm.weight" in T:
        q, k = rms(q, W(p + "attn_q_norm.weight")), rms(k, W(p + "attn_k_norm.weight"))
    q, k = rope(q), rope(k)
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
