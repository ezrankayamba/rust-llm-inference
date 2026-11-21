//! How often a draft model's top token matches a target model's, over the target's replies to
//! tools requests, at the positions the target samples (the text a call's schema forces is
//! left out): what speculative decoding would accept.
//!
//! usage: spec_accept <target.gguf> <draft.gguf> <cases.json> [--max-tokens N]
//! cases.json: {"system": ..., "tools": [...OpenAI tools...], "texts": [user text, ...]}

use rli_models::tools::{Constraint, Grammar, TokenTable, parse_tools};
use rli_models::{Engine, GenParams, Message, RenderOpts, SamplerConfig};
use std::sync::Arc;

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let max_tokens = a
        .iter()
        .position(|s| s == "--max-tokens")
        .map_or(256, |i| a[i + 1].parse().unwrap());
    let load = |p: &str| {
        let mut e = Engine::load(p).expect("load model");
        rli_models::backend::attach(&mut e.model, usize::MAX).expect("attach backend");
        e
    };
    let (target, draft) = (load(&a[1]), load(&a[2]));
    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&a[3]).unwrap()).unwrap();
    let tools = parse_tools(&cases["tools"]).unwrap();
    let grammar = Arc::new(Grammar::new(&tools));
    let opts = RenderOpts {
        tools: tools.iter().map(|t| t.to_json()).collect(),
        no_think: true,
    };
    let table = TokenTable::new(&draft.tokenizer);
    let (mut sampled, mut hits, mut forced) = (0, 0, 0);
    // Accepted drafts in a row from each sampled position, to simulate rounds of k drafts.
    let mut runs: Vec<Vec<bool>> = Vec::new();
    for text in cases["texts"].as_array().unwrap() {
        let msgs = [
            Message {
                role: "system".into(),
                content: cases["system"].as_str().unwrap().into(),
            },
            Message {
                role: "user".into(),
                content: text.as_str().unwrap().into(),
            },
        ];
        let prompt = target.chat_tokens_with(&msgs, &opts);
        let mut sess = target.new_session(4096);
        let params = GenParams {
            max_tokens,
            sampler: SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.0,
                repeat_last_n: 64,
                seed: 0,
            },
            tools: Some(grammar.clone()),
            ..Default::default()
        };
        target.generate(&mut sess, &prompt, &params, || false, |_| true);
        let reply = sess.tokens[prompt.len()..].to_vec();

        // The draft, teacher-forced over the prompt and the reply.
        let mut ds = draft.new_session(4096);
        let mut logits = draft
            .model
            .forward_batch(&mut ds, &prompt, true)
            .expect("logits");
        let mut c = Constraint::new(&grammar, &table);
        let mut matched = Vec::new();
        let mut i = 0;
        while i < reply.len() {
            c.mask(&mut logits);
            let ok = argmax(&logits) == reply[i];
            matched.push(ok);
            sampled += 1;
            hits += usize::from(ok);
            c.advance(reply[i]);
            let jump = c.jump(|s| draft.tokenizer.encode(s, false, false), usize::MAX);
            let n = jump.len().min(reply.len() - i - 1);
            assert_eq!(&jump[..n], &reply[i + 1..i + 1 + n], "forced text differs");
            forced += n;
            logits = draft
                .model
                .forward_batch(&mut ds, &reply[i..=i + n], true)
                .expect("logits");
            i += 1 + n;
        }
        println!(
            "{:3} sampled, {:3} forced, {:5.1}% matched  {}",
            matched.len(),
            reply.len() - matched.len(),
            100.0 * matched.iter().filter(|&&m| m).count() as f64 / matched.len().max(1) as f64,
            text.as_str().unwrap()
        );
        runs.push(matched);
    }
    println!(
        "\n{sampled} sampled tokens, {forced} forced; draft matched {:.1}%",
        100.0 * hits as f64 / sampled as f64
    );
    // Rounds of k drafts: each round yields the drafts accepted in a row plus the target's own.
    for k in [2, 3, 4, 6, 8] {
        let mut rounds = 0;
        for m in &runs {
            let mut p = 0;
            while p < m.len() {
                let acc = m[p..].iter().take(k).take_while(|&&x| x).count();
                p += acc + 1;
                rounds += 1;
            }
        }
        println!(
            "k={k}: {rounds} target passes for {sampled} tokens ({:.2} tokens a pass)",
            sampled as f64 / rounds as f64
        );
    }
}
