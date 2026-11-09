//! Debug aid: prints the chat prompt rendered for each conversation of a JSON file, as a JSON array.
//! usage: render_chat <model.gguf> <conversations.json>
//! A conversation is an array of `{"role": ..., "content": ...}` objects.

use serde_json::Value;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let e = rli_models::Engine::load(&a[1]).expect("load model");
    let conversations: Vec<Vec<Value>> =
        serde_json::from_str(&std::fs::read_to_string(&a[2]).expect("read")).expect("json");
    let rendered: Vec<String> = conversations
        .iter()
        .map(|c| {
            let field = |m: &Value, k: &str| m[k].as_str().expect("string field").to_owned();
            let msgs: Vec<_> = c
                .iter()
                .map(|m| rli_models::Message::new(&field(m, "role"), &field(m, "content")))
                .collect();
            e.template.render(&msgs).text
        })
        .collect();
    println!("{}", serde_json::to_string(&rendered).expect("json"));
}
