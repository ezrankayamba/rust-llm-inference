//! Cost of constrained tool-call decoding on a real vocabulary: building the token table (once
//! per model), then masking the logits for each token of a typical reply, with the per-state
//! cache cold (a request's first call) and warm.
//! usage: bench_tool_grammar <model.gguf>

use rli_models::Engine;
use rli_models::tools::{Constraint, Grammar, TokenTable, parse_tools};
use std::time::Instant;

const TOOLS: &str = r#"[
 {"type": "function", "function": {"name": "navigate", "description": "Open a page.",
  "parameters": {"type": "object", "properties": {"page": {"type": "string", "enum": ["home", "loans", "profile"]}},
  "required": ["page"], "additionalProperties": false}, "strict": true}},
 {"type": "function", "function": {"name": "fill_field", "description": "Fill one field of a form.",
  "parameters": {"type": "object", "properties": {
   "form_id": {"type": "string", "enum": ["loan_application"]},
   "field_id": {"type": "string", "enum": ["amount", "full_name", "phone"]},
   "value": {"type": "string"}},
  "required": ["form_id", "field_id", "value"], "additionalProperties": false}, "strict": true}}
]"#;

const REPLY: &str = "<tool_call>\n{\"name\": \"navigate\", \"arguments\": {\"page\": \"loans\"}}\n</tool_call>\n\
<tool_call>\n{\"name\": \"fill_field\", \"arguments\": {\"form_id\": \"loan_application\", \"field_id\": \"full_name\", \"value\": \"Asha Juma\"}}\n</tool_call>\n\
<tool_call>\n{\"name\": \"fill_field\", \"arguments\": {\"form_id\": \"loan_application\", \"field_id\": \"amount\", \"value\": \"500000\"}}\n</tool_call>";

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bench_tool_grammar <model.gguf>");
    let e = Engine::load(&path).expect("load model");
    let t0 = Instant::now();
    let table = TokenTable::new(&e.tokenizer);
    let build = t0.elapsed();
    assert!(
        table.has_call_tokens(),
        "no <tool_call> tokens in this vocabulary"
    );
    let tools = parse_tools(&serde_json::from_str(TOOLS).unwrap()).unwrap();
    let grammar = Grammar::new(&tools);
    let reply = e.tokenizer.encode(REPLY, false, true);
    let n_vocab = e.tokenizer.vocab_size();
    println!(
        "vocabulary {n_vocab}, token table built in {:.1} ms",
        build.as_secs_f64() * 1e3
    );
    let mut logits = vec![0.0f32; n_vocab];
    for pass in ["cold", "warm"] {
        let mut c = Constraint::new(&grammar, &table);
        let mut in_call = 0;
        let mut total = 0.0;
        let mut worst: f64 = 0.0;
        // Warm: one pass first fills the cache, the second is timed.
        let rounds = if pass == "warm" { 2 } else { 1 };
        for round in 0..rounds {
            for &id in &reply {
                logits.fill(0.0);
                let t = Instant::now();
                c.mask(&mut logits);
                let dt = t.elapsed().as_secs_f64();
                if c.in_call() && round == rounds - 1 {
                    assert!(logits[id as usize].is_finite(), "reply token {id} masked");
                    in_call += 1;
                    total += dt;
                    worst = worst.max(dt);
                }
                c.advance(id);
            }
        }
        println!(
            "{pass}: {in_call} constrained tokens of {}, mean {:.0} us, worst {:.0} us, total {:.2} ms",
            reply.len(),
            total / in_call as f64 * 1e6,
            worst * 1e6,
            total * 1e3
        );
    }
}
