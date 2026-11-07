//! Debug aid: tokenizes a prompt, runs it, and writes the last position's logits.
//! usage: dump_logits <model.gguf> <prompt> <out.f32> [--precise] [--gpu-layers=N]
//! Prints the token ids. `--precise` keeps CPU activations in f32 so results can be compared
//! tightly with a reference; `--gpu-layers` offloads layers (metal/cuda builds).

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut engine = rli_models::Engine::load(&a[1]).expect("load model");
    for flag in &a[4..] {
        if flag == "--precise" {
            engine.model.precise = true;
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
    let logits = engine.model.forward_batch(&mut sess, &tokens, true);
    let bytes: Vec<u8> = logits
        .expect("non-empty prompt")
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    std::fs::write(&a[3], bytes).expect("write logits");
}
