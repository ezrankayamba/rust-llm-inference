//! Debug aid: encodes each string of a JSON array file and prints a JSON array of id lists.
//! usage: tokenize <model.gguf> <strings.json>

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = rli_gguf::Gguf::open(&a[1]).expect("open gguf");
    let tok = rli_models::tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
    let texts: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(&a[2]).expect("read")).expect("json");
    let ids: Vec<Vec<u32>> = texts.iter().map(|t| tok.encode(t, false, true)).collect();
    for (t, i) in texts.iter().zip(&ids) {
        let d = tok.decode(i);
        if d != *t && d != format!(" {t}") {
            eprintln!("decode roundtrip mismatch: {t:?} -> {d:?}");
        }
    }
    println!("{}", serde_json::to_string(&ids).expect("json"));
}
