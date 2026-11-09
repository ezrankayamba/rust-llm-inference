//! Fixed-length throughput benchmark, independent of EOS and sampling.
//!
//! usage: bench <model.gguf> [--pp 512] [--tg 128] [--depth 0] [--reps 3] [--gpu-layers N]
//!        [--threads N] [--precise]
//!
//! `pp`: prompt processing of N tokens into an empty session (logits for the last one only).
//! `tg`: N greedy decode steps, ignoring EOS, after `depth` prompt tokens (prefilled untimed),
//! so decode can be measured at a long context. Each test runs once as a
//! warm-up (page in the mapping, compile GPU pipelines), then `reps` timed runs.

use rli_models::sampler::argmax;
use rli_models::{Engine, Session};
use std::time::Instant;

const TEXT: &str = "The history of computing is a story of steadily shrinking machines and \
growing ambitions. Early computers filled rooms and consumed enormous power, yet performed \
fewer operations per second than a modern wristwatch. Over the decades, transistors replaced \
vacuum tubes, integrated circuits replaced discrete components, and software grew from \
hand-written machine code into vast layered systems. ";

fn stats(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let opt = |name: &str, default: usize| {
        a.iter()
            .position(|x| x == name)
            .map_or(default, |i| a[i + 1].parse().expect("numeric flag value"))
    };
    let (pp, tg, reps) = (opt("--pp", 512), opt("--tg", 128), opt("--reps", 3));
    let depth = opt("--depth", 0);
    let gpu_layers = opt("--gpu-layers", 0);
    if a.iter().any(|x| x == "--threads") {
        rli_models::set_threads(opt("--threads", 0)).expect("--threads");
    }
    let mut engine = Engine::load(&a[1]).expect("load model");
    engine.model.precise = a.iter().any(|x| x == "--precise");
    rli_models::backend::attach(&mut engine.model, gpu_layers).expect("attach backend");

    let one = engine.tokenizer.encode(TEXT, false, false);
    let start = engine.tokenizer.bos.unwrap_or(one[0]);
    let mut prompt = vec![start];
    while prompt.len() < pp.max(depth) {
        prompt.extend(&one);
    }
    let ctx = pp.max(depth + tg) + 1;
    let m = &engine.model;
    let new_session = || -> Session { m.new_session(ctx) };

    let run_pp = || {
        let mut s = new_session();
        let t = Instant::now();
        m.forward_batch(&mut s, &prompt[..pp], true);
        pp as f64 / t.elapsed().as_secs_f64()
    };
    let run_tg = || {
        let mut s = new_session();
        let mut tok = start;
        if depth > 0 {
            tok = argmax(
                &m.forward_batch(&mut s, &prompt[..depth], true)
                    .expect("logits"),
            );
        }
        let t = Instant::now();
        for _ in 0..tg {
            tok = argmax(&m.forward(&mut s, tok, true).expect("logits"));
        }
        tg as f64 / t.elapsed().as_secs_f64()
    };

    let backend = m
        .accelerator()
        .map_or("cpu".to_string(), |(n, l)| format!("{n}, {l} layers"));
    let threads = rayon::current_num_threads();
    println!(
        "model: {} | backend: {backend} | cpu threads: {threads} | precise: {}",
        engine.model_name, m.precise
    );
    let tg_name = if depth > 0 {
        format!("tg@d{depth} ")
    } else {
        "tg".into()
    };
    for (name, n, f) in [
        ("pp", pp, &run_pp as &dyn Fn() -> f64),
        (tg_name.as_str(), tg, &run_tg),
    ] {
        if n == 0 {
            continue;
        }
        f();
        let xs: Vec<f64> = (0..reps).map(|_| f()).collect();
        let (mean, sd) = stats(&xs);
        println!("{name}{n}: {mean:8.1} ± {sd:5.1} tok/s");
    }
}
