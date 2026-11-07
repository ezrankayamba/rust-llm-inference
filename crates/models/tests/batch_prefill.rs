//! Batched prompt processing on Metal must give the same logits, and leave the same KV cache,
//! as running the prompt one token at a time. Uses small random dense models (head_dim 32, so
//! they take the batched path) with a prompt longer than one batch chunk.
#![cfg(all(feature = "metal", target_os = "macos"))]

use rli_core::GgmlType;
use rli_core::quant::{quantize_q4_0, quantize_q8_0};
use rli_gguf::Gguf;
use rli_gguf::Value;
use rli_gguf::writer::Writer;
use rli_models::Model;

const D: usize = 128;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const FF: usize = 192;
const LAYERS: usize = 2;
const VOCAB: usize = 96;
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

/// Writes a random `arch` model; qwen2 gets q/k/v biases (exercising the broadcast add).
fn write_model(arch: &str, path: &std::path::Path) {
    let mut r = Rng(0x9e37_79b9);
    let hd = D / HEADS;
    let kv = KV_HEADS * hd;
    let mut w = Writer::new();
    let key = |k: &str| format!("{arch}.{k}");
    w.kv("general.architecture", Value::String(arch.into()))
        .kv(&key("block_count"), Value::U32(LAYERS as u32))
        .kv(&key("context_length"), Value::U32(1024))
        .kv(&key("embedding_length"), Value::U32(D as u32))
        .kv(&key("feed_forward_length"), Value::U32(FF as u32))
        .kv(&key("attention.head_count"), Value::U32(HEADS as u32))
        .kv(&key("attention.head_count_kv"), Value::U32(KV_HEADS as u32))
        .kv(&key("attention.layer_norm_rms_epsilon"), Value::F32(1e-5))
        .kv(&key("rope.freq_base"), Value::F32(10000.0));
    // Alternate quant types across matrices so several matmul kernels are exercised.
    let mut quantized = 0;
    let mut mat = |w: &mut Writer, r: &mut Rng, name: &str, rows: usize, cols: usize| {
        let data = r.vec(rows * cols, 0.3);
        let dims = [cols as u64, rows as u64];
        quantized += 1;
        match quantized % 3 {
            0 => w.tensor_f32(name, &dims, &data),
            1 => w.tensor(name, &dims, GgmlType::Q8_0, quantize_q8_0(&data)),
            _ => w.tensor(name, &dims, GgmlType::Q4_0, quantize_q4_0(&data)),
        };
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
        mat(&mut w, &mut r, &n("ffn_gate.weight"), FF, D);
        mat(&mut w, &mut r, &n("ffn_up.weight"), FF, D);
        mat(&mut w, &mut r, &n("ffn_down.weight"), D, FF);
        if arch == "qwen2" {
            for (b, len) in [("attn_q.bias", D), ("attn_k.bias", kv), ("attn_v.bias", kv)] {
                w.tensor_f32(&n(b), &[len as u64], &r.vec(len, 0.5));
            }
        }
    }
    w.write(path).expect("write model");
}

fn load(path: &std::path::Path, gpu_layers: usize) -> Model {
    let mut m = Model::load(Gguf::open(path).expect("open")).expect("load");
    m.precise = true;
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

fn check(arch: &str) {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("batch_{arch}.gguf"));
    write_model(arch, &path);
    let mut r = Rng(42);
    let prompt: Vec<u32> = (0..PROMPT).map(|_| r.next() % VOCAB as u32).collect();

    let cpu = run(&load(&path, 0), &prompt, false);
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
fn batched_prefill_matches_per_token_llama() {
    check("llama");
}

#[test]
fn batched_prefill_matches_per_token_qwen2_with_biases() {
    check("qwen2");
}
