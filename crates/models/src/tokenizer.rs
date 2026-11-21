//! Tokenizers reconstructed from GGUF `tokenizer.ggml.*` metadata.
//!
//! * `llama` model: SentencePiece-style BPE driven by token scores, with `<0xXX>` byte fallback.
//! * `gpt2` model: byte-level BPE driven by merge ranks, after a regex-style pre-tokenizer
//!   (hand-written here to avoid a regex engine with look-ahead).

use rli_gguf::{Error, Gguf, Result, Value};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;

const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;
const TYPE_BYTE: i32 = 6;
const SPM_SPACE: char = '\u{2581}';

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreTokenizer {
    /// GPT-2 pattern. `single_digits` splits every digit on its own first (SmolLM).
    Gpt2 { single_digits: bool },
    /// Llama 3 pattern; Qwen2 is the same with `max_digits == 1`.
    Llama3 { max_digits: usize },
}

enum Model {
    Spm {
        add_space_prefix: bool,
    },
    Bpe {
        pre: PreTokenizer,
        merges: HashMap<(u32, u32), (u32, u32)>,
        /// Token id of each byte's GPT-2 unicode symbol (every byte-level vocab has all 256).
        byte_ids: Vec<Option<u32>>,
        char_byte: HashMap<char, u8>,
    },
}

pub struct Tokenizer {
    model: Model,
    tokens: Vec<String>,
    token_id: HashMap<String, u32>,
    scores: Vec<f32>,
    types: Vec<i32>,
    /// Control / user-defined tokens, longest first, matched verbatim when parsing specials.
    specials: Vec<(String, u32)>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
}

/// GPT-2's reversible byte -> printable-unicode mapping.
fn bytes_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).expect("valid scalar")
        } else {
            n += 1;
            char::from_u32(255 + n).expect("valid scalar")
        };
    }
    map
}

fn str_array(g: &Gguf, key: &str) -> Result<Vec<String>> {
    let arr = g
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Missing(key.into()))?;
    arr.iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| Error::Format(format!("{key}: non-string")))
        })
        .collect()
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let tokens = str_array(g, "tokenizer.ggml.tokens")?;
        let n = tokens.len();
        let num_array = |key: &str| -> Option<Vec<f32>> {
            g.get(key)?
                .as_array()
                .map(|a| a.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect())
        };
        let mut scores = num_array("tokenizer.ggml.scores").unwrap_or_else(|| vec![0.0; n]);
        // Some conversions store all-zero scores. SentencePiece BPE vocabularies are sorted by
        // merge priority, so the id order recovers it (the real scores are -(id - offset)).
        if scores.windows(2).all(|w| w[0] == w[1]) {
            scores
                .iter_mut()
                .enumerate()
                .for_each(|(i, s)| *s = -(i as f32));
        }
        let types: Vec<i32> = num_array("tokenizer.ggml.token_type")
            .map(|v| v.into_iter().map(|t| t as i32).collect())
            .unwrap_or_else(|| vec![1; n]);
        let token_id: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        if scores.len() != n || types.len() != n {
            return Err(Error::Format(format!(
                "{n} tokens but {} scores and {} types",
                scores.len(),
                types.len()
            )));
        }
        let opt_id = |key: &str| -> Result<Option<u32>> {
            match g.get(key).and_then(Value::as_u64) {
                Some(id) if id < n as u64 => Ok(Some(id as u32)),
                Some(id) => Err(Error::Format(format!(
                    "{key} {id} is not in the vocabulary"
                ))),
                None => Ok(None),
            }
        };
        let opt_bool = |key: &str| g.get(key).and_then(Value::as_bool);

        let model = match g.str("tokenizer.ggml.model")? {
            "llama" => Model::Spm {
                add_space_prefix: opt_bool("tokenizer.ggml.add_space_prefix").unwrap_or(true),
            },
            "gpt2" => {
                let pre = match g
                    .get("tokenizer.ggml.pre")
                    .and_then(Value::as_str)
                    .unwrap_or("default")
                {
                    "llama3" | "llama-bpe" | "smaug-bpe" | "dbrx" | "falcon3" => {
                        PreTokenizer::Llama3 { max_digits: 3 }
                    }
                    "qwen2" | "deepseek-r1-qwen" => PreTokenizer::Llama3 { max_digits: 1 },
                    "smollm" => PreTokenizer::Gpt2 {
                        single_digits: true,
                    },
                    _ => PreTokenizer::Gpt2 {
                        single_digits: false,
                    },
                };
                let mut merges = HashMap::new();
                for (rank, m) in str_array(g, "tokenizer.ggml.merges")?.iter().enumerate() {
                    let Some((a, b)) = m.split_once(' ') else {
                        continue;
                    };
                    if let (Some(&ia), Some(&ib), Some(&im)) = (
                        token_id.get(a),
                        token_id.get(b),
                        token_id.get(&format!("{a}{b}")),
                    ) {
                        merges.entry((ia, ib)).or_insert((rank as u32, im));
                    }
                }
                let byte_char = bytes_to_unicode();
                let char_byte = byte_char
                    .iter()
                    .enumerate()
                    .map(|(b, &c)| (c, b as u8))
                    .collect();
                let byte_ids = byte_char
                    .iter()
                    .map(|c| token_id.get(&c.to_string()).copied())
                    .collect();
                Model::Bpe {
                    pre,
                    merges,
                    byte_ids,
                    char_byte,
                }
            }
            other => {
                return Err(Error::Format(format!(
                    "unsupported tokenizer model {other:?}"
                )));
            }
        };

        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|&(i, t)| matches!(types[i], TYPE_CONTROL | TYPE_USER_DEFINED) && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));

        let is_spm = matches!(model, Model::Spm { .. });
        Ok(Self {
            model,
            tokens,
            token_id,
            scores,
            types,
            specials,
            bos: opt_id("tokenizer.ggml.bos_token_id")?,
            eos: opt_id("tokenizer.ggml.eos_token_id")?,
            add_bos: opt_bool("tokenizer.ggml.add_bos_token").unwrap_or(is_spm),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_str(&self, id: u32) -> &str {
        &self.tokens[id as usize]
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.token_id.get(s).copied()
    }

    pub fn is_control(&self, id: u32) -> bool {
        self.types.get(id as usize) == Some(&TYPE_CONTROL)
    }

    /// Encodes `text`. With `parse_special`, control tokens written literally (e.g. `<|im_end|>`)
    /// become their single ids; otherwise they are tokenized as plain text.
    pub fn encode(&self, text: &str, add_bos: bool, parse_special: bool) -> Vec<u32> {
        self.encode_where(text, add_bos, |_, _| parse_special)
    }

    /// Encodes `text` with specials parsed, except that control tokens touching a byte range in
    /// `untrusted` stay plain text, so client-supplied content can't forge a turn boundary.
    /// User-defined tokens still parse there: a model's replies contain them (Qwen3's `<think>`),
    /// and replaying a reply must give back the tokens that were generated.
    pub fn encode_untrusted(
        &self,
        text: &str,
        add_bos: bool,
        untrusted: &[Range<usize>],
    ) -> Vec<u32> {
        self.encode_rendered(text, add_bos, untrusted, &[])
    }

    /// [`encode_untrusted`](Self::encode_untrusted), and no special token of any kind parses
    /// where it touches a range in `opaque` (tool definitions: `<tool_call>` is user-defined).
    pub fn encode_rendered(
        &self,
        text: &str,
        add_bos: bool,
        untrusted: &[Range<usize>],
        opaque: &[Range<usize>],
    ) -> Vec<u32> {
        let touches = |span: &Range<usize>, ranges: &[Range<usize>]| {
            ranges
                .iter()
                .any(|r| span.start < r.end && r.start < span.end)
        };
        self.encode_where(text, add_bos, |span, id| {
            !touches(&span, opaque) && (!self.is_control(id) || !touches(&span, untrusted))
        })
    }

    fn encode_where(
        &self,
        text: &str,
        add_bos: bool,
        allow: impl Fn(Range<usize>, u32) -> bool,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos && let Some(b) = self.bos {
            out.push(b);
        }
        let mut from = 0;
        // SPM prefixes a space only at the very start of the text, not after each special token.
        let mut at_start = true;
        let mut found = vec![Found::Unknown; self.specials.len()];
        while from < text.len() {
            let special = self.find_special(text, from, &allow, &mut found);
            let plain = &text[from..special.as_ref().map_or(text.len(), |(span, _)| span.start)];
            if !plain.is_empty() {
                self.encode_plain(plain, at_start, &mut out);
            }
            at_start = false;
            let Some((span, id)) = special else { break };
            out.push(id);
            from = span.end;
        }
        out
    }

    /// The earliest allowed special at or after `from` (the longest, if several start there).
    /// `found` remembers each special's next allowed match from earlier calls, whose `from` was
    /// no later: searching every special from `from` each time was quadratic in the text.
    fn find_special(
        &self,
        text: &str,
        from: usize,
        allow: &impl Fn(Range<usize>, u32) -> bool,
        found: &mut [Found],
    ) -> Option<(Range<usize>, u32)> {
        let mut best: Option<(Range<usize>, u32)> = None;
        for ((s, id), f) in self.specials.iter().zip(found.iter_mut()) {
            let pos = match *f {
                Found::None => continue,
                Found::At(pos) if pos >= from => pos,
                _ => {
                    *f = Found::None;
                    let mut start = from;
                    while let Some(i) = text[start..].find(s.as_str()) {
                        let pos = start + i;
                        if allow(pos..pos + s.len(), *id) {
                            *f = Found::At(pos);
                            break;
                        }
                        // Occurrences can overlap: step one character, not past the whole match.
                        start = pos + text[pos..].chars().next().map_or(1, char::len_utf8);
                    }
                    let Found::At(pos) = *f else { continue };
                    pos
                }
            };
            if best
                .as_ref()
                .is_none_or(|(b, _)| pos < b.start || (pos == b.start && pos + s.len() > b.end))
            {
                best = Some((pos..pos + s.len(), *id));
            }
        }
        best
    }

    fn encode_plain(&self, text: &str, at_start: bool, out: &mut Vec<u32>) {
        match &self.model {
            Model::Spm { add_space_prefix } => {
                let mut s = String::with_capacity(text.len() + 3);
                if *add_space_prefix && at_start {
                    s.push(SPM_SPACE);
                }
                s.extend(text.chars().map(|c| if c == ' ' { SPM_SPACE } else { c }));
                for word in spm_words(&s) {
                    self.spm_encode(word, out);
                }
            }
            Model::Bpe {
                pre,
                merges,
                byte_ids,
                ..
            } => {
                for word in pre_tokenize(text, *pre) {
                    let mut syms: Vec<u32> = Vec::with_capacity(word.len());
                    for &b in word.as_bytes() {
                        if let Some(id) = byte_ids[b as usize] {
                            syms.push(id);
                        }
                    }
                    bpe_merge(&mut syms, merges);
                    out.extend(syms);
                }
            }
        }
    }

    /// Merges the adjacent pair whose union is the highest-scoring token, the leftmost on a tie,
    /// until none is a token. Symbols are byte spans of `text`; a heap holds the candidate pairs,
    /// so a long word costs n log n, not n² (one request of a long word could hold a core).
    fn spm_encode(&self, text: &str, out: &mut Vec<u32>) {
        // Symbol i is text[start[i]..end[i]]; next[i] == n ends the list.
        let start: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        let n = start.len();
        let mut end: Vec<usize> = start[1..].iter().copied().chain([text.len()]).collect();
        let mut next: Vec<usize> = (1..=n).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive = vec![true; n];
        // (score, leftmost first, left, right, end of right when pushed)
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<_>, l: usize, r: usize, end: &[usize]| {
            if let Some(&id) = self.token_id.get(&text[start[l]..end[r]]) {
                let score = self.scores[id as usize];
                heap.push((TotalF32(score), Reverse(l), r, end[r]));
            }
        };
        for i in 1..n {
            push(&mut heap, i - 1, i, &end);
        }
        while let Some((_, Reverse(l), r, r_end)) = heap.pop() {
            if !alive[l] || next[l] != r || end[r] != r_end {
                continue;
            }
            end[l] = end[r];
            alive[r] = false;
            next[l] = next[r];
            if next[l] < n {
                prev[next[l]] = l;
                push(&mut heap, l, next[l], &end);
            }
            if prev[l] < n {
                push(&mut heap, prev[l], l, &end);
            }
        }
        let syms = (0..n)
            .filter(|&i| alive[i])
            .map(|i| &text[start[i]..end[i]]);
        for s in syms {
            match self.token_id.get(s) {
                Some(&id) => out.push(id),
                None => {
                    for b in s.bytes() {
                        if let Some(&id) = self.token_id.get(&format!("<0x{b:02X}>")) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Raw bytes of one token. Multi-byte UTF-8 characters may span several tokens, so callers
    /// streaming text should accumulate bytes and decode complete sequences.
    pub fn decode_bytes(&self, id: u32, out: &mut Vec<u8>) {
        let Some(tok) = self.tokens.get(id as usize) else {
            return;
        };
        match &self.model {
            Model::Spm { .. } => {
                if self.types[id as usize] == TYPE_BYTE
                    && let Some(hex) = tok.strip_prefix("<0x").and_then(|t| t.strip_suffix('>'))
                    && let Ok(b) = u8::from_str_radix(hex, 16)
                {
                    out.push(b);
                } else if self.types[id as usize] != TYPE_CONTROL {
                    out.extend(tok.replace(SPM_SPACE, " ").as_bytes());
                }
            }
            Model::Bpe { char_byte, .. } => {
                if matches!(self.types[id as usize], TYPE_CONTROL | TYPE_USER_DEFINED) {
                    if self.types[id as usize] == TYPE_USER_DEFINED {
                        out.extend(tok.as_bytes());
                    }
                    return;
                }
                for c in tok.chars() {
                    match char_byte.get(&c) {
                        Some(&b) => out.push(b),
                        None => out.extend(c.to_string().as_bytes()),
                    }
                }
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            self.decode_bytes(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// SentencePiece word splitting, as observed from the reference library: the last `▁` of a run
/// starts the following word and the rest of the run is a whitespace-only piece, so
/// `a▁▁▁b▁▁` splits into `a`, `▁▁`, `▁b`, `▁▁`. BPE never merges across words.
fn spm_words(s: &str) -> Vec<&str> {
    const W: usize = SPM_SPACE.len_utf8();
    let mut words = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < s.len() {
        if !s[i..].starts_with(SPM_SPACE) {
            i += s[i..].chars().next().map_or(1, char::len_utf8);
            continue;
        }
        let run_start = i;
        while s[i..].starts_with(SPM_SPACE) {
            i += W;
        }
        let word_start = if i < s.len() { i - W } else { i };
        for (a, b) in [(start, run_start), (run_start, word_start)] {
            if b > a {
                words.push(&s[a..b]);
            }
        }
        start = word_start;
    }
    if start < s.len() {
        words.push(&s[start..]);
    }
    words
}

/// A special token's next allowed match, as far as `find_special` has searched.
#[derive(Clone, Copy)]
enum Found {
    Unknown,
    None,
    At(usize),
}

/// `f32` ordered by `total_cmp`, for the SPM merge heap (scores come from the file).
struct TotalF32(f32);
impl PartialEq for TotalF32 {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for TotalF32 {}
impl PartialOrd for TotalF32 {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for TotalF32 {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Applies BPE merges in rank order, the leftmost pair on a tie, until none apply. A heap of
/// candidate pairs over a linked list of symbols: n log n for a word of n bytes, not n².
fn bpe_merge(syms: &mut Vec<u32>, merges: &HashMap<(u32, u32), (u32, u32)>) {
    let n = syms.len();
    let mut next: Vec<usize> = (1..=n).collect();
    let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
    let mut alive = vec![true; n];
    // Reverse((rank, left, right, the pair's symbols when pushed))
    let mut heap = BinaryHeap::new();
    let push = |heap: &mut BinaryHeap<_>, syms: &[u32], l: usize, r: usize| {
        if let Some(&(rank, _)) = merges.get(&(syms[l], syms[r])) {
            heap.push(Reverse((rank, l, r, syms[l], syms[r])));
        }
    };
    for i in 1..n {
        push(&mut heap, syms, i - 1, i);
    }
    while let Some(Reverse((_, l, r, a, b))) = heap.pop() {
        if !alive[l] || next[l] != r || syms[l] != a || syms[r] != b {
            continue;
        }
        syms[l] = merges[&(a, b)].1;
        alive[r] = false;
        next[l] = next[r];
        if next[l] < n {
            prev[next[l]] = l;
            push(&mut heap, syms, l, next[l]);
        }
        if prev[l] < n {
            push(&mut heap, syms, prev[l], l);
        }
    }
    let mut i = 0;
    syms.retain(|_| {
        i += 1;
        alive[i - 1]
    });
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_number(c: char) -> bool {
    c.is_numeric()
}
fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Length in chars of an English contraction (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`) at `i`.
fn contraction_len(cs: &[char], i: usize, ignore_case: bool) -> usize {
    if cs[i] != '\'' {
        return 0;
    }
    let lower = |k: usize| {
        cs.get(i + k).map(|c| {
            if ignore_case {
                c.to_ascii_lowercase()
            } else {
                *c
            }
        })
    };
    match (lower(1), lower(2)) {
        (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => 3,
        (Some('s' | 't' | 'm' | 'd'), _) => 2,
        _ => 0,
    }
}

/// Length of a whitespace match at `i` per `\s+(?!\S)|\s+`: a run of whitespace, leaving its last
/// char for the following word when the run is followed by non-whitespace.
fn whitespace_len(cs: &[char], i: usize) -> usize {
    let mut j = i;
    while j < cs.len() && cs[j].is_whitespace() {
        j += 1;
    }
    let run = j - i;
    if j < cs.len() && run > 1 {
        run - 1
    } else {
        run
    }
}

fn pre_tokenize(text: &str, pre: PreTokenizer) -> Vec<String> {
    let cs: Vec<char> = text.chars().collect();
    let mut words = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let len = match pre {
            PreTokenizer::Llama3 { max_digits } => llama3_match(&cs, i, max_digits),
            PreTokenizer::Gpt2 { single_digits } => gpt2_match(&cs, i, single_digits),
        }
        .max(1);
        words.push(cs[i..i + len].iter().collect());
        i += len;
    }
    words
}

fn llama3_match(cs: &[char], i: usize, max_digits: usize) -> usize {
    let n = cs.len();
    let c = cs[i];
    let k = contraction_len(cs, i, true);
    if k > 0 {
        return k;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    let start = if !is_newline(c)
        && !is_letter(c)
        && !is_number(c)
        && cs.get(i + 1).is_some_and(|&d| is_letter(d))
    {
        i + 1
    } else {
        i
    };
    if is_letter(cs[start]) {
        let mut j = start;
        while j < n && is_letter(cs[j]) {
            j += 1;
        }
        return j - i;
    }
    // \p{N}{1,max}
    if is_number(c) {
        let mut j = i;
        while j < n && j - i < max_digits && is_number(cs[j]) {
            j += 1;
        }
        return j - i;
    }
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    let s = if c == ' ' { i + 1 } else { i };
    let punct = |d: char| !d.is_whitespace() && !is_letter(d) && !is_number(d);
    if s < n && punct(cs[s]) {
        let mut j = s;
        while j < n && punct(cs[j]) {
            j += 1;
        }
        while j < n && is_newline(cs[j]) {
            j += 1;
        }
        return j - i;
    }
    // \s*[\r\n]+  -> the longest whitespace prefix ending in a newline
    let mut j = i;
    let mut last_nl = None;
    while j < n && cs[j].is_whitespace() {
        if is_newline(cs[j]) {
            last_nl = Some(j);
        }
        j += 1;
    }
    if let Some(k) = last_nl {
        return k + 1 - i;
    }
    whitespace_len(cs, i)
}

fn gpt2_match(cs: &[char], i: usize, single_digits: bool) -> usize {
    let n = cs.len();
    let c = cs[i];
    if single_digits && is_number(c) {
        return 1;
    }
    let k = contraction_len(cs, i, false);
    if k > 0 {
        return k;
    }
    let s = if c == ' ' && i + 1 < n { i + 1 } else { i };
    let run = |pred: &dyn Fn(char) -> bool| {
        let mut j = s;
        while j < n && pred(cs[j]) {
            j += 1;
        }
        j - i
    };
    if is_letter(cs[s]) {
        return run(&is_letter);
    }
    if is_number(cs[s]) {
        // With single_digits a digit is split out on its own, so a leading space stays alone.
        return if single_digits {
            s - i
        } else {
            run(&is_number)
        };
    }
    let punct = |d: char| !d.is_whitespace() && !is_letter(d) && !is_number(d);
    if punct(cs[s]) {
        return run(&punct);
    }
    whitespace_len(cs, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str, pre: PreTokenizer) -> Vec<String> {
        pre_tokenize(text, pre)
    }

    #[test]
    fn llama3_pretokenizer_follows_reference_regex() {
        let pre = PreTokenizer::Llama3 { max_digits: 3 };
        assert_eq!(
            split("Hello world! I'M 12345 ok\n\n  x", pre),
            [
                "Hello", " world", "!", " I", "'M", " ", "123", "45", " ok", "\n\n", " ", " x"
            ]
        );
        assert_eq!(split("a,b  \n c", pre), ["a", ",b", "  \n", " c"]);
    }

    #[test]
    fn qwen2_pretokenizer_splits_single_digits() {
        assert_eq!(
            split("x 2024", PreTokenizer::Llama3 { max_digits: 1 }),
            ["x", " ", "2", "0", "2", "4"]
        );
    }

    #[test]
    fn gpt2_pretokenizer_follows_reference_regex() {
        let pre = PreTokenizer::Gpt2 {
            single_digits: false,
        };
        assert_eq!(
            split("Hi there's 42 ...  x", pre),
            ["Hi", " there", "'s", " 42", " ...", " ", " x"]
        );
        let smol = PreTokenizer::Gpt2 {
            single_digits: true,
        };
        assert_eq!(
            split("in 1990s", smol),
            ["in", " ", "1", "9", "9", "0", "s"]
        );
    }

    #[test]
    fn pretokenizers_match_python_regex_reference() {
        // Generated with the `regex` package from the patterns llama.cpp uses for each pre type.
        let v: serde_json::Value =
            serde_json::from_str(include_str!("../tests/pretok_ref.json")).unwrap();
        let cases: Vec<&str> = v["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap())
            .collect();
        for (name, pre) in [
            ("llama3", PreTokenizer::Llama3 { max_digits: 3 }),
            ("qwen2", PreTokenizer::Llama3 { max_digits: 1 }),
            (
                "gpt2",
                PreTokenizer::Gpt2 {
                    single_digits: false,
                },
            ),
            (
                "smollm",
                PreTokenizer::Gpt2 {
                    single_digits: true,
                },
            ),
        ] {
            for (case, want) in cases.iter().zip(v[name].as_array().unwrap()) {
                let want: Vec<&str> = want
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|w| w.as_str().unwrap())
                    .collect();
                assert_eq!(split(case, pre), want, "{name} on {case:?}");
            }
        }
    }

    #[test]
    fn bytes_to_unicode_is_a_bijection() {
        let m = bytes_to_unicode();
        let set: std::collections::HashSet<char> = m.iter().copied().collect();
        assert_eq!(set.len(), 256);
        assert_eq!(m[b' ' as usize], '\u{120}'); // 'Ġ'
    }

    #[test]
    fn spm_words_start_at_each_whitespace_run() {
        assert_eq!(spm_words("▁▁▁ab▁c▁▁"), ["▁▁", "▁ab", "▁c", "▁▁"]);
        assert_eq!(spm_words("x▁y"), ["x", "▁y"]);
        assert_eq!(spm_words("é▁▁\n"), ["é", "▁", "▁\n"]);
    }

    #[test]
    fn bpe_merge_applies_lowest_rank_first() {
        // a=0 b=1 c=2 ab=3 bc=4; "bc" outranks "ab", so "abc" -> [a, bc].
        let merges = HashMap::from([((0, 1), (5, 3)), ((1, 2), (1, 4))]);
        let mut syms = vec![0, 1, 2];
        bpe_merge(&mut syms, &merges);
        assert_eq!(syms, [0, 4]);
    }

    #[test]
    fn bpe_merge_breaks_rank_ties_leftmost() {
        // a=0 aa=1: "aaa" -> [aa, a].
        let merges = HashMap::from([((0, 0), (0, 1))]);
        let mut syms = vec![0, 0, 0];
        bpe_merge(&mut syms, &merges);
        assert_eq!(syms, [1, 0]);
    }

    /// A word of 200k bytes: the rescan-per-merge loop this replaced took hours here, and the
    /// server tokenizes client text before checking its length.
    #[test]
    fn bpe_merge_of_a_long_word_is_not_quadratic() {
        // a=0 aa=1 aaaa=2
        let merges = HashMap::from([((0, 0), (0, 1)), ((1, 1), (1, 2))]);
        let mut syms = vec![0; 200_000];
        bpe_merge(&mut syms, &merges);
        assert_eq!(syms, vec![2; 50_000]);
    }

    /// `tok()` plus `a` and `aa`, `aa` scoring higher.
    fn tok_with_a() -> Tokenizer {
        let mut t = tok();
        for (s, score) in [("a", -2.0), ("aa", -1.0)] {
            t.token_id.insert(s.into(), t.tokens.len() as u32);
            t.tokens.push(s.into());
            t.types.push(1);
            t.scores.push(score);
        }
        t
    }

    #[test]
    fn spm_merge_breaks_score_ties_leftmost() {
        let t = tok_with_a();
        let (a, aa) = (t.token_id("a").unwrap(), t.token_id("aa").unwrap());
        assert_eq!(t.encode("aaa", false, false), [aa, a]);
    }

    #[test]
    fn spm_encode_of_a_long_word_is_not_quadratic() {
        let t = tok_with_a();
        let aa = t.token_id("aa").unwrap();
        assert_eq!(
            t.encode(&"a".repeat(200_000), false, false),
            vec![aa; 100_000]
        );
    }

    #[test]
    fn many_specials_in_a_long_text_are_not_quadratic() {
        let t = tok();
        let text = "<think>".repeat(100_000);
        assert_eq!(t.encode(&text, false, true), vec![THINK; 100_000]);
        // Forged control tokens are all rejected, each searched for once.
        let text = "<|end|>".repeat(100_000);
        assert!(!encode_one(&t, &text, 0..text.len()).contains(&END));
    }

    /// A byte-level SPM vocabulary with one control token and one user-defined token.
    fn tok() -> Tokenizer {
        let mut tokens = vec!["<|end|>".to_owned(), "<think>".to_owned()];
        let mut types = vec![TYPE_CONTROL, TYPE_USER_DEFINED];
        for b in 0..=255u8 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(TYPE_BYTE);
        }
        let token_id = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        Tokenizer {
            model: Model::Spm {
                add_space_prefix: false,
            },
            scores: vec![0.0; tokens.len()],
            specials: vec![("<|end|>".to_owned(), 0), ("<think>".to_owned(), 1)],
            tokens,
            token_id,
            types,
            bos: None,
            eos: None,
            add_bos: false,
        }
    }

    const END: u32 = 0;
    const THINK: u32 = 1;

    /// Encodes `text` with one untrusted byte range.
    fn encode_one(t: &Tokenizer, text: &str, content: Range<usize>) -> Vec<u32> {
        t.encode_untrusted(text, false, std::slice::from_ref(&content))
    }

    #[test]
    fn encode_untrusted_keeps_control_tokens_in_content_as_plain_text() {
        let t = tok();
        let text = "<|end|>a<|end|>b<|end|>";
        let ids = encode_one(&t, text, 7..15);
        assert_eq!(ids.iter().filter(|&&i| i == END).count(), 2);
        assert_eq!((ids[0], ids[ids.len() - 1]), (END, END));
        assert_eq!(t.decode(&ids), "a<|end|>b");
    }

    #[test]
    fn encode_untrusted_rejects_a_control_token_straddling_content() {
        let t = tok();
        let ids = encode_one(&t, "<|end|>", 0..3);
        assert!(!ids.contains(&END));
    }

    #[test]
    fn encode_untrusted_finds_the_template_token_after_a_forged_one() {
        let t = tok();
        let ids = encode_one(&t, "<|end|><|end|>", 0..7);
        assert_eq!(ids.iter().filter(|&&i| i == END).count(), 1);
        assert_eq!(ids.last(), Some(&END));
    }

    #[test]
    fn encode_untrusted_still_parses_user_defined_tokens_in_content() {
        let t = tok();
        assert_eq!(encode_one(&t, "<think>", 0..7), [THINK]);
    }

    #[test]
    fn encode_rendered_parses_no_special_token_in_opaque_ranges() {
        let t = tok();
        let text = "<think><|end|><think>";
        // The middle and last tokens are tool text: neither parses, user-defined or not.
        let ids = t.encode_rendered(text, false, &[], std::slice::from_ref(&(7..text.len())));
        assert_eq!(ids[0], THINK);
        assert!(!ids[1..].contains(&THINK) && !ids.contains(&END), "{ids:?}");
        assert_eq!(t.decode(&ids), text);
    }

    #[test]
    fn encode_untrusted_matches_encode_without_specials_in_content() {
        let t = tok();
        let text = "<|end|>hello<think>world<|end|>";
        assert_eq!(
            t.encode_untrusted(text, false, &[7..12, 19..24]),
            t.encode(text, false, true)
        );
    }
}
