//! Next-token sampling: repetition penalty, temperature, top-k, top-p.

#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// `<= 0` means greedy (argmax).
    pub temperature: f32,
    /// `0` disables top-k.
    pub top_k: usize,
    /// `>= 1` disables nucleus sampling.
    pub top_p: f32,
    /// `1.0` disables the penalty.
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 0x5eed,
        }
    }
}

/// A distribution over tokens: candidate ids with weights, not normalized (masked ones weigh 0).
pub type Dist = Vec<(u32, f32)>;

/// Candidates sorted on the first top-p pass; more are sorted only if they fall short.
const NUCLEUS_FIRST: usize = 64;

pub struct Sampler {
    cfg: SamplerConfig,
    state: u64,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let state = cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        Self { cfg, state }
    }

    fn uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Picks the next token. `history` is the token sequence so far (for repetition penalty).
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        self.penalize(logits, history);
        if self.cfg.temperature <= 0.0 {
            return argmax(logits);
        }
        let cand = self.weights(logits);
        self.pick(&cand)
    }

    /// The distribution `sample` draws from at temperature > 0: candidate ids with weights
    /// (not normalized; masked tokens weigh 0), after the penalty, top-k, temperature and top-p.
    pub fn probs(&self, logits: &mut [f32], history: &[u32]) -> Dist {
        self.penalize(logits, history);
        self.weights(logits)
    }

    /// Draws a token from `cand` (as [`Sampler::probs`] returns) with this sampler's stream.
    pub fn pick(&mut self, cand: &[(u32, f32)]) -> u32 {
        // Masked tokens (-inf, weight 0) are never picked while any other remains: not by a
        // draw of exactly 0, nor by the fallback when rounding leaves `r` above 0 at the end.
        let total: f32 = cand.iter().map(|c| c.1).sum();
        let mut r = self.uniform() * total;
        for &(id, p) in cand {
            r -= p;
            if p > 0.0 && r <= 0.0 {
                return id;
            }
        }
        cand.iter()
            .rev()
            .find(|c| c.1 > 0.0)
            .or(cand.last())
            .expect("at least one candidate")
            .0
    }

    /// One step of speculative sampling: `x` was drawn from `q`, and the target distribution
    /// is `p` (both as [`Sampler::probs`] returns). `x` stands with probability
    /// min(1, p(x)/q(x)) (`None`); otherwise the token returned is drawn from max(0, p - q),
    /// normalized, so that whichever is taken is distributed as `p`, whatever `q` is.
    pub fn verify(&mut self, p: &[(u32, f32)], q: &[(u32, f32)], x: u32) -> Option<u32> {
        let mass = |d: &[(u32, f32)], id: u32| d.iter().find(|c| c.0 == id).map_or(0.0, |c| c.1);
        let p_sum: f32 = p.iter().map(|c| c.1).sum();
        let q_sum: f32 = q.iter().map(|c| c.1).sum();
        let (px, qx) = (mass(p, x) / p_sum, mass(q, x) / q_sum);
        if self.uniform() * qx < px {
            return None;
        }
        let mut qn: Vec<(u32, f32)> = q.iter().map(|&(id, w)| (id, w / q_sum)).collect();
        qn.sort_unstable_by_key(|c| c.0);
        let residual: Vec<(u32, f32)> = p
            .iter()
            .map(|&(id, w)| {
                let qi = qn
                    .binary_search_by_key(&id, |c| c.0)
                    .map_or(0.0, |i| qn[i].1);
                (id, (w / p_sum - qi).max(0.0))
            })
            .collect();
        // A rejection means q(x) > p(x), so the residual has mass q(x) - p(x) or more; only
        // rounding can leave it none, with p and q all but equal, and then p is what it is.
        if residual.iter().any(|c| c.1 > 0.0) {
            Some(self.pick(&residual))
        } else {
            Some(self.pick(p))
        }
    }

    fn penalize(&self, logits: &mut [f32], history: &[u32]) {
        if self.cfg.repeat_penalty == 1.0 {
            return;
        }
        let start = history.len().saturating_sub(self.cfg.repeat_last_n);
        let mut seen = std::collections::HashSet::new();
        for &t in &history[start..] {
            if seen.insert(t)
                && let Some(l) = logits.get_mut(t as usize)
            {
                // Same rule as llama.cpp / CTRL: shrink positive logits, push negative ones down.
                *l = if *l > 0.0 {
                    *l / self.cfg.repeat_penalty
                } else {
                    *l * self.cfg.repeat_penalty
                };
            }
        }
    }

    /// Top-k, temperature and top-p over penalized logits (temperature > 0).
    fn weights(&self, logits: &[f32]) -> Dist {
        let mut cand: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, l))
            .collect();
        let k = if self.cfg.top_k == 0 {
            cand.len()
        } else {
            self.cfg.top_k.min(cand.len())
        };
        let desc = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, desc);
            cand.truncate(k);
        }

        let max = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for c in cand.iter_mut() {
            c.1 = ((c.1 - max) / self.cfg.temperature).exp();
            sum += c.1;
        }
        if self.cfg.top_p < 1.0 {
            // The nucleus is usually a few dozen tokens: sort only the leading `m` candidates,
            // doubling `m` until they reach the mass, instead of sorting the whole vocabulary.
            let target = self.cfg.top_p * sum;
            let mut m = NUCLEUS_FIRST.min(cand.len());
            loop {
                if m < cand.len() {
                    cand.select_nth_unstable_by(m - 1, desc);
                }
                cand[..m].sort_unstable_by(desc);
                let mut cum = 0.0;
                if let Some(i) = cand[..m].iter().position(|c| {
                    cum += c.1;
                    cum >= target
                }) {
                    cand.truncate(i + 1);
                    break;
                }
                if m == cand.len() {
                    break; // rounding left the total just short of the target: keep them all
                }
                m = (2 * m).min(cand.len());
            }
        }
        cand
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    // The largest value first, 16 running maxima at a time (which the compiler vectorizes; a
    // fold that carries the index doesn't, and cost 100 us over a 128k vocabulary), then
    // where it first occurs.
    let mut best = [f32::NEG_INFINITY; 16];
    let (chunks, rest) = x.as_chunks::<16>();
    for c in chunks {
        for (b, &v) in best.iter_mut().zip(c) {
            if v > *b {
                *b = v;
            }
        }
    }
    let max = best
        .iter()
        .chain(rest)
        .fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    x.iter().position(|&v| v == max).unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_is_the_first_largest_value_at_any_length() {
        for len in [1, 15, 16, 17, 40, 1000] {
            for at in [0, len / 2, len - 1] {
                let mut x: Vec<f32> = (0..len).map(|i| -(i as f32) - 1.0).collect();
                x[at] = 5.0;
                assert_eq!(argmax(&x), at as u32, "len {len}");
                x[len - 1] = 5.0; // a tie goes to the first
                assert_eq!(argmax(&x), at as u32, "len {len}");
                x[0] = f32::NAN;
                assert_eq!(argmax(&x), if at == 0 { len as u32 - 1 } else { at as u32 });
            }
        }
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..Default::default()
        });
        assert_eq!(s.sample(&mut [0.1, 3.0, -1.0, 2.9], &[]), 1);
    }

    #[test]
    fn top_k_one_always_picks_the_max() {
        let mut s = Sampler::new(SamplerConfig {
            top_k: 1,
            temperature: 5.0,
            repeat_penalty: 1.0,
            ..Default::default()
        });
        for _ in 0..50 {
            assert_eq!(s.sample(&mut [0.0, 1.0, 0.5, 0.9], &[]), 1);
        }
    }

    #[test]
    fn top_p_excludes_the_tail() {
        // p = softmax([10, 10, 0, 0]) ~ [0.5, 0.5, 2e-5, 2e-5]: top_p 0.9 keeps only tokens 0 and 1.
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            repeat_penalty: 1.0,
            seed: 9,
            ..Default::default()
        };
        let mut s = Sampler::new(cfg);
        let mut seen = [0; 4];
        for _ in 0..500 {
            seen[s.sample(&mut [10.0, 10.0, 0.0, 0.0], &[]) as usize] += 1;
        }
        assert_eq!(seen[2] + seen[3], 0);
        assert!(seen[0] > 150 && seen[1] > 150, "{seen:?}");
    }

    #[test]
    fn top_p_nucleus_wider_than_first_pass_is_exact() {
        // Logits fall linearly, so the nucleus is a prefix of token ids; at top_p 0.9 it is
        // ~230 tokens, so the partial sort has to widen past its first pass.
        let logits: Vec<f32> = (0..2000).map(|i| -0.01 * i as f32).collect();
        let w: Vec<f64> = logits.iter().map(|&l| (l as f64).exp()).collect();
        let total: f64 = w.iter().sum();
        let mut cum = 0.0;
        let cutoff = w.iter().position(|&x| {
            cum += x / total;
            cum >= 0.9
        });
        let cutoff = cutoff.expect("nucleus within vocab");
        assert!(cutoff > NUCLEUS_FIRST);
        let mut s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            repeat_penalty: 1.0,
            seed: 3,
            ..Default::default()
        });
        let picks: Vec<u32> = (0..3000)
            .map(|_| s.sample(&mut logits.clone(), &[]))
            .collect();
        assert!(
            picks.iter().all(|&p| p as usize <= cutoff),
            "sampled outside the nucleus"
        );
        assert!(
            picks.iter().any(|&p| p as usize > 2 * NUCLEUS_FIRST),
            "nucleus was cut short"
        );
    }

    #[test]
    fn masked_tokens_are_never_sampled() {
        // One allowed token at the far end of the vocabulary; the rest masked out. The fallback
        // used to return the last candidate, and a zero draw the first, whatever their weight.
        for seed in 0..200 {
            let mut s = Sampler::new(SamplerConfig {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.0,
                seed,
                ..Default::default()
            });
            let mut logits = vec![f32::NEG_INFINITY; 1000];
            logits[500] = 0.0;
            assert_eq!(s.sample(&mut logits, &[]), 500);
        }
    }

    #[test]
    fn repeat_penalty_demotes_recent_tokens() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            ..Default::default()
        });
        assert_eq!(s.sample(&mut [3.0, 2.0], &[0]), 1);
    }

    #[test]
    fn verify_of_draws_from_q_is_distributed_as_p() {
        // q favours token 0, which p barely wants; p wants token 3, which q never proposes, and
        // token 4 is masked in both. Whatever q proposes, what is taken must follow p.
        let p = [(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.0)];
        let q = [(4, 0.0), (2, 1.0), (0, 6.0), (1, 2.0), (5, 1.0)];
        let cfg = |seed| SamplerConfig {
            seed,
            ..Default::default()
        };
        let (mut target, mut draft) = (Sampler::new(cfg(1)), Sampler::new(cfg(2)));
        let n = 200_000;
        let mut seen = [0usize; 6];
        for _ in 0..n {
            let x = draft.pick(&q);
            let t = target.verify(&p, &q, x).unwrap_or(x);
            seen[t as usize] += 1;
        }
        assert_eq!(seen[4] + seen[5], 0, "{seen:?}");
        for &(id, w) in &p[..4] {
            let (got, w) = (seen[id as usize] as f64 / n as f64, w as f64);
            // 4 standard deviations of a share of n draws.
            let sd = (w * (1.0 - w) / n as f64).sqrt();
            assert!(
                (got - w).abs() < 4.0 * sd,
                "token {id}: {got} for {w}, {seen:?}"
            );
        }
    }

    #[test]
    fn verify_keeps_a_guess_p_wants_more_than_q() {
        let mut s = Sampler::new(SamplerConfig::default());
        let (p, q) = ([(0, 3.0), (1, 1.0)], [(0, 1.0), (1, 1.0)]);
        for _ in 0..100 {
            assert_eq!(s.verify(&p, &q, 0), None);
        }
    }
}
