//! Batched prompt processing (CPU, and Metal when built in) must give the same logits, and
//! leave the same KV cache, as running the prompt one token at a time. Uses small random dense models with a prompt
//! longer than one batch chunk, at head_dim 32, 64 and 128 (the batched attention kernel splits
//! keys across lanes differently for each), and MoE models (Mixtral-style, and Qwen2-MoE with a
//! shared expert) whose experts run as gathered per-expert matmuls. The prompt also outgrows the
//! GPU KV cache's initial 256 positions, and one test checks the f16 cache against the f32 one.

use rli_core::GgmlType;
use rli_core::quant::{quantize_q4_0, quantize_q8_0};
use rli_gguf::Gguf;
use rli_gguf::Value;
use rli_gguf::writer::Writer;
use rli_models::Model;

const D: usize = 128;
const FF: usize = 192;
const LAYERS: usize = 2;
const VOCAB: usize = 96;
const EXPERTS: usize = 4;
const EXPERTS_USED: usize = 2;
const SHARED_FF: usize = 64;
/// Longer than the Metal batch chunk (256), so the second chunk reads the first one's cache.
const PROMPT: usize = 300;

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() % 2001) as f32 / 1000.0 - 1.0) * scale)
            .collect()
    }
}

/// Writes a random `arch` model; qwen2/qwen2moe get q/k/v biases (exercising the broadcast
/// add). `moe` makes llama a Mixtral-style MoE; qwen2moe is always MoE with a shared expert.
fn write_model(arch: &str, heads: usize, kv_heads: usize, moe: bool, path: &std::path::Path) {
    let moe = moe || arch == "qwen2moe";
    let mut r = Rng(0x9e37_79b9);
    let hd = D / heads;
    let kv = kv_heads * hd;
    let mut w = Writer::new();
    let key = |k: &str| format!("{arch}.{k}");
    w.kv("general.architecture", Value::String(arch.into()))
        .kv(&key("block_count"), Value::U32(LAYERS as u32))
        .kv(&key("context_length"), Value::U32(1024))
        .kv(&key("embedding_length"), Value::U32(D as u32))
        .kv(&key("feed_forward_length"), Value::U32(FF as u32))
        .kv(&key("attention.head_count"), Value::U32(heads as u32))
        .kv(&key("attention.head_count_kv"), Value::U32(kv_heads as u32))
        .kv(&key("attention.layer_norm_rms_epsilon"), Value::F32(1e-5))
        .kv(&key("rope.freq_base"), Value::F32(10000.0));
    if moe {
        w.kv(&key("expert_count"), Value::U32(EXPERTS as u32))
            .kv(&key("expert_used_count"), Value::U32(EXPERTS_USED as u32))
            .kv(&key("expert_feed_forward_length"), Value::U32(FF as u32));
    }
    // Alternate quant types across matrices so several matmul kernels are exercised.
    let mut quantized = 0;
    // `dims` are GGUF order: columns first; a third dim stacks experts.
    let mut tensor = |w: &mut Writer, r: &mut Rng, name: &str, dims: &[u64]| {
        let data = r.vec(dims.iter().product::<u64>() as usize, 0.3);
        let dims = dims.to_vec();
        quantized += 1;
        match quantized % 3 {
            0 => w.tensor_f32(name, &dims, &data),
            1 => w.tensor(name, &dims, GgmlType::Q8_0, quantize_q8_0(&data)),
            _ => w.tensor(name, &dims, GgmlType::Q4_0, quantize_q4_0(&data)),
        };
    };
    let mut mat = |w: &mut Writer, r: &mut Rng, name: &str, rows: usize, cols: usize| {
        tensor(w, r, name, &[cols as u64, rows as u64])
    };
    mat(&mut w, &mut r, "token_embd.weight", VOCAB, D);
    mat(&mut w, &mut r, "output.weight", VOCAB, D);
    w.tensor_f32("output_norm.weight", &[D as u64], &r.vec(D, 1.0));
    for l in 0..LAYERS {
        let n = |s: &str| format!("blk.{l}.{s}");
        w.tensor_f32(&n("attn_norm.weight"), &[D as u64], &r.vec(D, 1.0));
        w.tensor_f32(&n("ffn_norm.weight"), &[D as u64], &r.vec(D, 1.0));
        mat(&mut w, &mut r, &n("attn_q.weight"), D, D);
        mat(&mut w, &mut r, &n("attn_k.weight"), kv, D);
        mat(&mut w, &mut r, &n("attn_v.weight"), kv, D);
        mat(&mut w, &mut r, &n("attn_output.weight"), D, D);
        if moe {
            let e = EXPERTS as u64;
            let (d, ff) = (D as u64, FF as u64);
            w.tensor_f32(&n("ffn_gate_inp.weight"), &[d, e], &r.vec(D * EXPERTS, 1.0));
            for (t, dims) in [
                ("gate", [d, ff, e]),
                ("up", [d, ff, e]),
                ("down", [ff, d, e]),
            ] {
                let data = r.vec(dims.iter().product::<u64>() as usize, 0.3);
                w.tensor(
                    &n(&format!("ffn_{t}_exps.weight")),
                    &dims,
                    GgmlType::Q8_0,
                    quantize_q8_0(&data),
                );
            }
            if arch == "qwen2moe" {
                w.tensor_f32(&n("ffn_gate_inp_shexp.weight"), &[d], &r.vec(D, 0.5));
                mat(&mut w, &mut r, &n("ffn_gate_shexp.weight"), SHARED_FF, D);
                mat(&mut w, &mut r, &n("ffn_up_shexp.weight"), SHARED_FF, D);
                mat(&mut w, &mut r, &n("ffn_down_shexp.weight"), D, SHARED_FF);
            }
        } else {
            mat(&mut w, &mut r, &n("ffn_gate.weight"), FF, D);
            mat(&mut w, &mut r, &n("ffn_up.weight"), FF, D);
            mat(&mut w, &mut r, &n("ffn_down.weight"), D, FF);
        }
        if arch.starts_with("qwen2") {
            for (b, len) in [("attn_q.bias", D), ("attn_k.bias", kv), ("attn_v.bias", kv)] {
                w.tensor_f32(&n(b), &[len as u64], &r.vec(len, 0.5));
            }
        }
    }
    w.write(path).expect("write model");
}

fn load(path: &std::path::Path, gpu_layers: usize) -> Model {
    load_with(path, gpu_layers, true)
}

/// `precise` also picks the GPU KV cache type: f32 when set, f16 otherwise.
fn load_with(path: &std::path::Path, gpu_layers: usize, precise: bool) -> Model {
    let mut m = Model::load(Gguf::open(path).expect("open")).expect("load");
    m.precise = precise;
    rli_models::backend::attach(&mut m, gpu_layers).expect("attach Metal");
    m
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Returns (logits after the prompt, logits of one more decoded token).
fn run(m: &Model, prompt: &[u32], batched: bool) -> (Vec<f32>, Vec<f32>) {
    let mut s = m.new_session(512);
    let after_prompt = if batched {
        m.forward_batch(&mut s, prompt, true)
    } else {
        let mut l = None;
        for (i, &t) in prompt.iter().enumerate() {
            l = m.forward(&mut s, t, i + 1 == prompt.len());
        }
        l
    }
    .expect("logits");
    let next = m.forward(&mut s, 7, true).expect("logits");
    (after_prompt, next)
}

fn model_path(arch: &str, heads: usize, kv_heads: usize, moe: bool) -> std::path::PathBuf {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("batch_{arch}_{heads}_{kv_heads}_{moe}.gguf"));
    // Tests share these files and run in parallel: rewriting one in place would pull pages from
    // under another test's mapping (SIGBUS), so write a private copy and rename it over.
    let tmp = path.with_extension(format!("{:?}.tmp", std::thread::current().id()));
    write_model(arch, heads, kv_heads, moe, &tmp);
    std::fs::rename(&tmp, &path).expect("rename");
    path
}

fn prompt() -> Vec<u32> {
    let mut r = Rng(42);
    (0..PROMPT).map(|_| r.next() % VOCAB as u32).collect()
}

fn check(arch: &str, heads: usize, kv_heads: usize) {
    check_with(arch, heads, kv_heads, false);
}

fn check_with(arch: &str, heads: usize, kv_heads: usize, moe: bool) {
    let path = model_path(arch, heads, kv_heads, moe);
    let prompt = prompt();

    let cpu_model = load(&path, 0);
    let cpu = run(&cpu_model, &prompt, false);
    let cpu_batched = run(&cpu_model, &prompt, true);
    for (what, a, b) in [
        (
            "CPU batched vs per-token prompt logits",
            &cpu_batched.0,
            &cpu.0,
        ),
        (
            "CPU batched vs per-token next-token logits",
            &cpu_batched.1,
            &cpu.1,
        ),
    ] {
        let d = max_diff(a, b);
        assert!(d < 1e-5, "{arch}: {what}: max diff {d}");
    }
    if !rli_models::backend::HAS_GPU {
        return;
    }
    for gpu_layers in [LAYERS, 1] {
        let m = load(&path, gpu_layers);
        let (seq, bat) = (run(&m, &prompt, false), run(&m, &prompt, true));
        for (what, a, b, tol) in [
            ("batched vs per-token prompt logits", &bat.0, &seq.0, 1e-4),
            (
                "batched vs per-token next-token logits",
                &bat.1,
                &seq.1,
                1e-4,
            ),
            ("batched vs CPU prompt logits", &bat.0, &cpu.0, 1e-3),
            ("batched vs CPU next-token logits", &bat.1, &cpu.1, 1e-3),
        ] {
            let d = max_diff(a, b);
            assert!(
                d < tol,
                "{arch}, {gpu_layers} GPU layers: {what}: max diff {d}"
            );
        }
    }
}

#[test]
fn batched_prefill_matches_per_token_llama_head_dim_32() {
    check("llama", 4, 2);
}

#[test]
fn batched_prefill_matches_per_token_llama_head_dim_64() {
    check("llama", 2, 1);
}

#[test]
fn batched_prefill_matches_per_token_llama_head_dim_128() {
    check("llama", 1, 1);
}

#[test]
fn batched_prefill_matches_per_token_qwen2_with_biases() {
    check("qwen2", 4, 2);
}

#[test]
fn batched_prefill_matches_per_token_mixtral_style_moe() {
    check_with("llama", 4, 2, true);
}

#[test]
fn batched_prefill_matches_per_token_qwen2moe_with_shared_expert() {
    check("qwen2moe", 4, 2);
}

/// With every layer offloaded, `precise` only changes the KV cache type, so this isolates the
/// f16 cache's rounding error (decode and batched attention both read it).
#[test]
fn f16_kv_cache_stays_close_to_f32() {
    if !rli_models::backend::HAS_GPU {
        return;
    }
    let path = model_path("llama", 4, 2, false);
    let prompt = prompt();
    let f32_kv = run(&load_with(&path, LAYERS, true), &prompt, true);
    let f16_model = load_with(&path, LAYERS, false);
    let (f16_seq, f16_bat) = (
        run(&f16_model, &prompt, false),
        run(&f16_model, &prompt, true),
    );
    for (what, a, b, tol) in [
        ("f16 vs f32 prompt logits", &f16_bat.0, &f32_kv.0, 1e-2),
        ("f16 vs f32 next-token logits", &f16_bat.1, &f32_kv.1, 1e-2),
        (
            "f16 batched vs per-token prompt logits",
            &f16_bat.0,
            &f16_seq.0,
            1e-3,
        ),
        (
            "f16 batched vs per-token next-token logits",
            &f16_bat.1,
            &f16_seq.1,
            1e-3,
        ),
    ] {
        let d = max_diff(a, b);
        eprintln!("{what}: max diff {d}");
        assert!(d < tol, "{what}: max diff {d}");
    }
}

/// CPU layers: the KV cache type is fixed when a session is created, so a session made before
/// `precise` is switched on keeps f16 KV while computing in f32, isolating the cache's rounding.
#[test]
fn cpu_f16_kv_cache_stays_close_to_f32_at_half_the_memory() {
    let path = model_path("llama", 4, 2, false);
    let prompt = prompt();
    let mut m = load_with(&path, 0, false);
    let mut s16 = m.new_session(512);
    m.precise = true;
    let mut s32 = m.new_session(512);
    let p16 = m.forward_batch(&mut s16, &prompt, true).expect("logits");
    let p32 = m.forward_batch(&mut s32, &prompt, true).expect("logits");
    let n16 = m.forward(&mut s16, 7, true).expect("logits");
    let n32 = m.forward(&mut s32, 7, true).expect("logits");
    for (what, a, b) in [
        ("f16 vs f32 KV prompt logits", &p16, &p32),
        ("f16 vs f32 KV next-token logits", &n16, &n32),
    ] {
        let d = max_diff(a, b);
        eprintln!("{what}: max diff {d}");
        assert!(d > 0.0 && d < 1e-2, "{what}: max diff {d}");
    }
    assert_eq!(2 * s16.kv_bytes(), s32.kv_bytes());
}

/// Weights split across several Metal buffers (the smallest cap that holds each layer whole)
/// must give exactly the logits of one buffer: the kernels are the same, only the binding
/// differs. Dense, and Qwen2-MoE whose stacked experts run through id-indexed matvecs.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn weights_split_across_metal_buffers_give_identical_logits() {
    for (arch, moe) in [("llama", false), ("qwen2moe", true)] {
        let path = model_path(arch, 4, 2, moe);
        let prompt = &prompt()[..40];
        let single = load(&path, LAYERS);
        let mut cap = 16 * 1024;
        let split = loop {
            let mut m = Model::load(Gguf::open(&path).expect("open")).expect("load");
            m.precise = true;
            if rli_models::backend::attach_capped(&mut m, LAYERS, cap).is_ok() {
                break m;
            }
            cap += 16 * 1024;
        };
        // Below the file size, so the offloaded weights need at least two buffers.
        let file = std::fs::metadata(&path).expect("model file").len() as usize;
        assert!(
            cap < file,
            "{arch}: a {cap}-byte cap can't split a {file}-byte model"
        );
        for batched in [true, false] {
            let want = run(&single, prompt, batched);
            assert_eq!(
                run(&split, prompt, batched),
                want,
                "{arch}, batched={batched}"
            );
        }
    }
}
