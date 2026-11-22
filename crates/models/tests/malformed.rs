//! Corrupted model files must load with an error, or load and run; never panic. Corrupts the
//! header (metadata and tensor table) of the tiny fixtures with seeded random edits and runs what
//! still loads on the CPU. GPU backends aren't exercised: a bad dispatch could hang the machine.

use rli_models::{Engine, GenParams, Message, SamplerConfig};
use std::panic::{AssertUnwindSafe, catch_unwind};

const EDITS_PER_FIXTURE: u64 = 4000;
const MIN_RUNS: u64 = 100;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}.gguf", env!("CARGO_MANIFEST_DIR"))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A little-endian integer field in the file: (byte offset, width).
type Field = (usize, usize);

/// Walks a well-formed header and returns its numeric fields: the counts, metadata numbers and
/// lengths, and each tensor's dimensions, type and offset. Random bytes mostly land in token text
/// or break the file outright; these are the values that can load yet disagree with each other.
fn header_fields(b: &[u8]) -> Vec<Field> {
    fn u32_at(b: &[u8], at: usize) -> usize {
        u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes")) as usize
    }
    fn u64_at(b: &[u8], at: usize) -> usize {
        u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes")) as usize
    }
    fn value(b: &[u8], at: &mut usize, ty: usize, fields: &mut Vec<Field>) {
        match ty {
            8 => {
                fields.push((*at, 8));
                *at += 8 + u64_at(b, *at);
            }
            9 => {
                let elem = u32_at(b, *at);
                fields.push((*at, 4));
                fields.push((*at + 4, 8));
                let n = u64_at(b, *at + 4);
                *at += 12;
                // Only the first few strings: a vocabulary would crowd out everything else.
                let before = fields.len();
                for _ in 0..n {
                    value(b, at, elem, fields);
                }
                fields.truncate(before + 4);
            }
            _ => {
                let width = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8][ty];
                fields.push((*at, width));
                *at += width;
            }
        }
    }
    let n_tensors = u64_at(b, 8);
    let n_kv = u64_at(b, 16);
    let mut fields = vec![(4, 4), (8, 8), (16, 8)];
    let mut at = 24;
    for _ in 0..n_kv {
        at += 8 + u64_at(b, at);
        let ty = u32_at(b, at);
        fields.push((at, 4));
        at += 4;
        value(b, &mut at, ty, &mut fields);
    }
    for _ in 0..n_tensors {
        at += 8 + u64_at(b, at);
        let n_dims = u32_at(b, at);
        fields.push((at, 4));
        at += 4;
        for _ in 0..n_dims {
            fields.push((at, 8));
            at += 8;
        }
        fields.push((at, 4));
        fields.push((at + 4, 8));
        at += 12;
    }
    fields
}

/// Sets one to three fields to values near the original or at the extremes.
fn corrupt(bytes: &mut [u8], fields: &[Field], r: &mut Rng) {
    for _ in 0..=r.below(3) {
        let (at, width) = fields[r.below(fields.len())];
        let mut le = [0u8; 8];
        le[..width].copy_from_slice(&bytes[at..at + width]);
        let v = u64::from_le_bytes(le);
        let new = match r.below(12) {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => v / 2,
            5 => v.wrapping_mul(2),
            6 => v.wrapping_add(1),
            7 => v.wrapping_sub(1),
            8 => 32,
            9 => 1 << r.below(64),
            10 => u64::MAX,
            _ => r.next(),
        };
        bytes[at..at + width].copy_from_slice(&new.to_le_bytes()[..width]);
    }
}

/// Everything a binary does with a freshly loaded model.
fn exercise(path: &std::path::Path) -> bool {
    let Ok(e) = Engine::load(path) else {
        return false;
    };
    let tokens = e.chat_tokens(&[Message::new("user", "hello <|im_end|> world")]);
    e.tokenizer.decode(&tokens);
    for temperature in [0.0, 1.0] {
        let params = GenParams {
            max_tokens: 2,
            stop: vec!["x".into()],
            sampler: SamplerConfig {
                temperature,
                top_k: 4,
                top_p: 0.9,
                repeat_penalty: 1.1,
                repeat_last_n: 8,
                seed: 1,
            },
            tools: None,
        };
        let mut sess = e.new_session(32);
        e.generate(&mut sess, &tokens, &params, || false, |_| true);
    }
    true
}

fn check(name: &str) {
    let original = std::fs::read(fixture(name)).expect("read fixture");
    let fields = header_fields(&original);
    let path =
        std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("malformed_{name}.gguf"));
    let mut failures = Vec::new();
    let mut ran = 0;
    for seed in 1..=EDITS_PER_FIXTURE {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut bytes = original.clone();
        corrupt(&mut bytes, &fields, &mut r);
        std::fs::write(&path, &bytes).expect("write");
        match catch_unwind(AssertUnwindSafe(|| exercise(&path))) {
            Ok(loaded) => ran += u64::from(loaded),
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                failures.push(format!("seed {seed}: {msg}"));
            }
        }
    }
    let _ = std::fs::remove_file(path);
    assert!(
        failures.is_empty(),
        "{name}: {} of {EDITS_PER_FIXTURE} panicked, first few:\n{}",
        failures.len(),
        failures[..failures.len().min(25)].join("\n")
    );
    // Otherwise this test could pass by no longer reaching the code it is here to exercise.
    assert!(ran >= MIN_RUNS, "{name}: only {ran} corrupted files loaded");
}

#[test]
fn corrupted_mixtral_header_never_panics() {
    check("tiny_mixtral");
}

#[test]
fn corrupted_qwen2moe_header_never_panics() {
    check("tiny_qwen2moe");
}

#[test]
fn corrupted_nvfp4_header_never_panics() {
    check("tiny_nvfp4");
}

#[test]
fn corrupted_deepseek_v2_header_never_panics() {
    check("tiny_deepseek_v2");
}

#[test]
fn corrupted_deepseek_v3_header_never_panics() {
    check("tiny_deepseek_v3");
}

/// The byte offset just past the first occurrence of `needle` in a header.
fn after(bytes: &[u8], needle: &str) -> usize {
    let n = needle.as_bytes();
    bytes
        .windows(n.len())
        .position(|w| w == n)
        .expect("in header")
        + n.len()
}

fn set_u32_metadata(bytes: &mut [u8], key: &str, v: u32) {
    let at = after(bytes, key) + 4;
    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Sets dimension `dim` of the tensor whose name is the first match of `name` in the table.
fn set_dim(bytes: &mut [u8], name: &str, dim: usize, v: u64) {
    let at = after(bytes, name) + 4 + dim * 8;
    bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn load_error(name: &str, edit: impl Fn(&mut Vec<u8>)) -> String {
    load_error_of("tiny_mixtral", name, edit)
}

fn load_error_of(fixture_name: &str, name: &str, edit: impl Fn(&mut Vec<u8>)) -> String {
    let mut bytes = std::fs::read(fixture(fixture_name)).expect("read fixture");
    edit(&mut bytes);
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("bad_{name}.gguf"));
    std::fs::write(&path, &bytes).expect("write");
    let r = Engine::load(&path);
    let _ = std::fs::remove_file(path);
    r.err().map_or_else(|| "loaded".into(), |e| e.to_string())
}

fn assert_load_error(name: &str, want: &str, edit: impl Fn(&mut Vec<u8>)) {
    let got = load_error(name, edit);
    assert!(got.contains(want), "{name}: {got}");
}

#[test]
fn unedited_fixture_loads() {
    assert_eq!(load_error("none", |_| {}), "loaded");
}

#[test]
fn rejects_zero_heads() {
    assert_load_error("heads", "attention.head_count is 0", |b| {
        set_u32_metadata(b, "llama.attention.head_count", 0)
    });
}

#[test]
fn rejects_kv_heads_that_do_not_divide_heads() {
    assert_load_error("kv-heads", "KV heads", |b| {
        set_u32_metadata(b, "llama.attention.head_count_kv", 3)
    });
}

#[test]
fn rejects_more_experts_used_than_exist() {
    assert_load_error("experts", "experts used", |b| {
        set_u32_metadata(b, "llama.expert_used_count", 5)
    });
}

fn set_f32_metadata(bytes: &mut [u8], key: &str, v: f32) {
    let at = after(bytes, key) + 4;
    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

#[test]
fn rejects_non_finite_or_non_positive_float_metadata() {
    // NaN here makes every hidden state NaN; on Metal that once routed to no expert at all.
    for (key, v) in [
        ("llama.attention.layer_norm_rms_epsilon", f32::NAN),
        ("llama.attention.layer_norm_rms_epsilon", -1e30),
        ("llama.rope.freq_base", f32::INFINITY),
        ("llama.rope.freq_base", 0.0),
    ] {
        assert_load_error("float-meta", key.trim_start_matches("llama."), |b| {
            set_f32_metadata(b, key, v)
        });
    }
}

#[test]
fn rejects_embedding_size_that_disagrees_with_tensors() {
    // Corrupting this alone used to size buffers from the wrong value.
    assert_load_error("embd", "expected", |b| {
        set_u32_metadata(b, "llama.embedding_length", 1 << 30)
    });
}

#[test]
fn rejects_attention_tensor_of_the_wrong_shape() {
    assert_load_error("attn-q", "attn_q", |b| {
        set_dim(b, "blk.0.attn_q.weight", 1, 32)
    });
}

#[test]
fn rejects_vocabulary_that_disagrees_with_embeddings() {
    assert_load_error("embd-rows", "output", |b| {
        set_dim(b, "token_embd.weight", 1, 256)
    });
}

#[test]
fn rejects_nvfp4_tensor_scales_of_the_wrong_size() {
    // One per expert for a stack of experts, one for any other matrix.
    for (tensor, n) in [("blk.0.ffn_up_exps.scale", 3), ("blk.0.attn_q.scale", 2)] {
        let got = load_error_of("tiny_nvfp4", "nvfp4-scale-size", |b| {
            set_dim(b, tensor, 0, n)
        });
        assert!(got.contains(tensor), "{tensor}: {got}");
    }
}

#[test]
fn rejects_non_finite_nvfp4_tensor_scales() {
    let g = rli_gguf::Gguf::open(fixture("tiny_nvfp4")).expect("open fixture");
    let at = g.tensor("blk.1.attn_k.scale").expect("scale tensor").offset;
    for v in [f32::NAN, f32::INFINITY] {
        let got = load_error_of("tiny_nvfp4", "nvfp4-scale-value", |b| {
            b[at..at + 4].copy_from_slice(&v.to_le_bytes())
        });
        assert!(
            got.contains("blk.1.attn_k.scale") && got.contains("not finite"),
            "{got}"
        );
    }
}

#[test]
fn rejects_latent_attention_that_disagrees_with_its_tensors() {
    for (fixture_name, want, edit) in [
        (
            "tiny_deepseek_v2",
            "attn_kv_b",
            &(|b: &mut Vec<u8>| set_dim(b, "blk.1.attn_kv_b.weight", 1, 64))
                as &dyn Fn(&mut Vec<u8>),
        ),
        ("tiny_deepseek_v2", "attn_kv_a_mqa", &|b| {
            set_u32_metadata(b, "deepseek2.attention.kv_lora_rank", 64)
        }),
        ("tiny_deepseek_v3", "attn_k_b", &|b| {
            set_dim(b, "blk.2.attn_k_b.weight", 2, 3)
        }),
        ("tiny_deepseek_v3", "attn_q_a", &|b| {
            set_u32_metadata(b, "deepseek2.attention.q_lora_rank", 16)
        }),
        ("tiny_deepseek_v3", "groups", &|b| {
            set_u32_metadata(b, "deepseek2.expert_group_count", 3)
        }),
        ("tiny_deepseek_v3", "exp_probs_b", &|b| {
            set_dim(b, "blk.1.exp_probs_b.bias", 0, 7)
        }),
    ] {
        let got = load_error_of(fixture_name, "mla", edit);
        assert!(got.contains(want), "{want}: {got}");
    }
}
