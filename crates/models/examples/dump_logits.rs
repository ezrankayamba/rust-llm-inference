//! Debug aid: tokenizes a prompt, runs it, and writes the last position's logits.
//! usage: dump_logits <model.gguf> <prompt> <out.f32> [--precise] [--coarse-acts] [--gpu-layers=N]
//!        [--per-token]
//! Prints the token ids. `--precise` keeps CPU activations in f32 so results can be compared
//! tightly with a reference; `--gpu-layers` offloads layers (metal/cuda builds); `--per-token`
//! runs the prompt one token at a time, as decoding does, instead of batched.

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut engine = rli_models::Engine::load(&a[1]).expect("load model");
    for flag in &a[4..] {
        if flag == "--precise" {
            engine.model.precise = true;
        } else if flag == "--coarse-acts" {
            engine.model.coarse_acts = true;
        } else if let Some(n) = flag.strip_prefix("--gpu-layers=") {
            rli_models::backend::attach(&mut engine.model, n.parse().expect("layer count"))
                .expect("attach backend");
        }
    }
    let tokens = engine
        .tokenizer
        .encode(&a[2], engine.tokenizer.add_bos, true);
    println!(
        "{}",
        tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut sess = engine.new_session(4096);
    let logits = if a[4..].iter().any(|f| f == "--per-token") {
        let mut last = None;
        for (i, &t) in tokens.iter().enumerate() {
            last = engine.model.forward(&mut sess, t, i + 1 == tokens.len());
        }
        last
    } else {
        engine.model.forward_batch(&mut sess, &tokens, true)
    };
    let bytes: Vec<u8> = logits
        .expect("non-empty prompt")
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    std::fs::write(&a[3], bytes).expect("write logits");
}
