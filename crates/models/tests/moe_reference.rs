//! Tiny random MoE models (written by scripts/make_tiny_moe.py with the reference `gguf` writer)
//! checked against logits from the independent NumPy forward pass in scripts/ref_forward.py.

use rli_models::Engine;

const PROMPT: &str = "hello moe world, routing!";

fn check(name: &str) {
    check_with(name, 0);
}

fn check_with(name: &str, gpu_layers: usize) {
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
    for (i, &t) in tokens.iter().enumerate() {
        logits = engine.model.forward(&mut sess, t, i + 1 == tokens.len());
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

/// Full and partial (1 of 2 layers) GPU offload must reproduce the same logits. With the `cuda`
/// feature this needs an NVIDIA GPU, so CI doesn't run it.
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
#[test]
fn gpu_offload_matches_reference_logits() {
    for name in ["tiny_mixtral", "tiny_qwen2moe"] {
        check_with(name, 2);
        check_with(name, 1);
    }
}
