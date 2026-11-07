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
        if self.cfg.repeat_penalty != 1.0 {
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
        if self.cfg.temperature <= 0.0 {
            return argmax(logits);
        }

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
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            cand.truncate(k);
        }
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let max = cand[0].1;
        let mut sum = 0.0;
        for c in cand.iter_mut() {
            c.1 = ((c.1 - max) / self.cfg.temperature).exp();
            sum += c.1;
        }
        let mut cum = 0.0;
        let mut keep = None;
        for (i, c) in cand.iter_mut().enumerate() {
            c.1 /= sum;
            cum += c.1;
            if cum >= self.cfg.top_p && keep.is_none() {
                keep = Some(i + 1);
            }
        }
        if let Some(k) = keep {
            cand.truncate(k);
        }

        let total: f32 = cand.iter().map(|c| c.1).sum();
        let mut r = self.uniform() * total;
        for &(id, p) in &cand {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        cand.last().expect("at least one candidate").0
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
        )
        .0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn repeat_penalty_demotes_recent_tokens() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            ..Default::default()
        });
        assert_eq!(s.sample(&mut [3.0, 2.0], &[0]), 1);
    }
}
