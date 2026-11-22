//! Tiny random MoE models (written by scripts/make_tiny_moe.py with the reference `gguf` writer)
//! checked against logits from the independent NumPy forward pass in scripts/ref_forward.py.
//! `tiny_nvfp4` has NVFP4 attention and expert weights, each with its tensor scale (per expert
//! for the stacks). `tiny_deepseek_v2` and `_v3` have DeepSeek's latent attention with YaRN and
//! its MoE: V2 as files converted before llama.cpp split `attn_kv_b`, with softmax routing; V3
//! with low-rank queries, split `attn_k_b` / `attn_v_b`, and sigmoid routing with a bias within
//! expert groups.

use rli_models::Engine;

const PROMPT: &str = "hello moe world, routing!";

fn check(name: &str) {
    check_with(name, 0);
}

fn check_with(name: &str, gpu_layers: usize) {
    check_pass(name, gpu_layers, false);
}

/// `batched` runs the prompt as one pass, else one token at a time.
fn check_pass(name: &str, gpu_layers: usize, batched: bool) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let mut engine = Engine::load(format!("{dir}/{name}.gguf")).expect("load fixture model");
    // Exact f32 activations: this test checks model math, not 8-bit activation rounding.
    engine.model.precise = true;
    rli_models::backend::attach(&mut engine.model, gpu_layers).expect("attach backend");
    let tokens = engine
        .tokenizer
        .encode(PROMPT, engine.tokenizer.add_bos, true);
    let mut sess = engine.new_session(256);
    let mut logits = None;
    if batched {
        logits = engine.model.forward_batch(&mut sess, &tokens, true);
    } else {
        for (i, &t) in tokens.iter().enumerate() {
            logits = engine.model.forward(&mut sess, t, i + 1 == tokens.len());
        }
    }
    let got = logits.expect("non-empty prompt");
    let want: Vec<f32> = std::fs::read(format!("{dir}/{name}.ref.f32"))
        .expect("reference logits")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(got.len(), want.len());
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err < 1e-3, "{name}: max |logit diff| = {err}");
}

#[test]
fn mixtral_style_moe_matches_reference_logits() {
    check("tiny_mixtral");
}

#[test]
fn qwen2moe_with_shared_expert_matches_reference_logits() {
    check("tiny_qwen2moe");
}

#[test]
fn nvfp4_with_tensor_scales_matches_reference_logits() {
    check("tiny_nvfp4");
}

#[test]
fn deepseek_v2_matches_reference_logits() {
    check("tiny_deepseek_v2");
    check_pass("tiny_deepseek_v2", 0, true);
}

#[test]
fn deepseek_v3_matches_reference_logits() {
    check("tiny_deepseek_v3");
    check_pass("tiny_deepseek_v3", 0, true);
}

/// Full and partial (1 of 2 layers) GPU offload must reproduce the same logits. With the `cuda`
/// feature this needs an NVIDIA GPU, so CI doesn't run it.
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
#[test]
fn gpu_offload_matches_reference_logits() {
    for name in ["tiny_mixtral", "tiny_qwen2moe", "tiny_nvfp4"] {
        check_with(name, 2);
        check_with(name, 1);
    }
}

/// DeepSeek's latent attention and MoE on the GPU: every layer and the first only, token by
/// token and batched. V2's softmax experts route on the GPU, V3's sigmoid and groups on the CPU
/// (on CUDA a layer at a time, so its prompts run token by token).
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
#[test]
fn deepseek_on_gpu_matches_reference_logits() {
    for name in ["tiny_deepseek_v2", "tiny_deepseek_v3"] {
        for (layers, batched) in [(3, false), (3, true), (1, false), (1, true)] {
            check_pass(name, layers, batched);
        }
    }
}
