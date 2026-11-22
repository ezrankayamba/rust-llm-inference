//! How much the fast CPU path's int8 activations change a model's predictions: runs a text one
//! token at a time (the decode path) through the model twice, with f32 activations (`precise`, the
//! reference) and with the default int8 ones, and compares the next-token distributions.
//!
//! usage: eval_quant <model.gguf> <text file> [--tokens 512] [--threads N] [--gpu-layers N]
//! [--batch] [--fp4-acts]
//!
//! Prints perplexity of the text under each, the mean and largest KL divergence of the int8
//! distribution from the reference, and how often their top tokens agree. On the CPU unless
//! `--gpu-layers` offloads layers (metal/cuda builds) of both. `--batch` runs the int8 side
//! as one prompt with the logits of every row, so a GPU's prompt kernels are measured
//! instead of its decode ones.

use rli_models::Engine;

/// `log(softmax(logits))`.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    logits.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let opt = |name: &str| a.iter().position(|x| x == name).map(|i| a[i + 1].as_str());
    let n_max: usize = opt("--tokens").map_or(512, |v| v.parse().expect("--tokens"));
    if let Some(t) = opt("--threads") {
        rli_models::set_threads(t.parse().expect("--threads")).expect("--threads");
    }
    let text = std::fs::read_to_string(&a[2]).expect("read text file");
    let mut fast = Engine::load(&a[1]).expect("load model");
    let mut reference = Engine::load(&a[1]).expect("load model");
    reference.model.precise = true;
    fast.model.precise = false;
    fast.model.fp4_acts = a.iter().any(|x| x == "--fp4-acts");
    if let Some(n) = opt("--gpu-layers") {
        let n = n.parse().expect("--gpu-layers");
        for e in [&mut reference, &mut fast] {
            rli_models::backend::attach(&mut e.model, n).expect("attach backend");
        }
    }
    let mut tokens = fast.tokenizer.encode(&text, fast.tokenizer.add_bos, false);
    tokens.truncate(n_max + 1);
    assert!(tokens.len() > 1, "text too short");
    let n = tokens.len() - 1;
    let (mut sr, mut sf) = (
        reference.model.new_session(tokens.len()),
        fast.model.new_session(tokens.len()),
    );
    let batched = a.iter().any(|x| x == "--batch").then(|| {
        let rows = n.min(fast.model.logit_rows());
        assert_eq!(rows, n, "the backend gives logits for {rows} rows a pass");
        fast.model.forward_rows(&mut sf, &tokens[..n], n).unwrap()
    });
    let (mut nll_ref, mut nll_fast, mut kl_sum, mut kl_max, mut agree) = (0.0, 0.0, 0.0, 0f64, 0);
    for i in 0..n {
        let lr = reference.model.forward(&mut sr, tokens[i], true).unwrap();
        let lf = match &batched {
            Some(all) => all[i * lr.len()..(i + 1) * lr.len()].to_vec(),
            None => fast.model.forward(&mut sf, tokens[i], true).unwrap(),
        };
        let (pr, pf) = (log_softmax(&lr), log_softmax(&lf));
        let next = tokens[i + 1] as usize;
        nll_ref -= pr[next];
        nll_fast -= pf[next];
        let kl: f64 = pr.iter().zip(&pf).map(|(r, f)| r.exp() * (r - f)).sum();
        kl_sum += kl;
        kl_max = kl_max.max(kl);
        agree += (argmax(&lr) == argmax(&lf)) as usize;
    }
    println!(
        "{} tokens | perplexity f32 {:.4}, int8 {:.4} ({:+.3}%) | KL mean {:.6}, max {:.4} | top-1 agree {:.2}%",
        n,
        (nll_ref / n as f64).exp(),
        (nll_fast / n as f64).exp(),
        100.0 * (((nll_fast - nll_ref) / n as f64).exp() - 1.0),
        kl_sum / n as f64,
        kl_max,
        100.0 * agree as f64 / n as f64
    );
}
