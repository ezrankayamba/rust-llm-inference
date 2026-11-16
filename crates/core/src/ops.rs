//! Elementwise and normalization ops used by decoder layers.

pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * scale * w;
    }
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `gate[i] = silu(gate[i]) * up[i]`: the SwiGLU activation, in place.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = silu(*g) * u;
    }
}

pub fn add_assign(acc: &mut [f32], x: &[f32]) {
    for (a, &v) in acc.iter_mut().zip(x) {
        *a += v;
    }
}

/// Rotary position embedding, applied to `n_heads` contiguous heads of `head_dim` in place.
///
/// `neox` selects the pairing: GPT-NeoX style rotates `(i, i + rope_dim/2)`; the original
/// (GGUF "llama") style rotates adjacent `(2i, 2i+1)`. Dims past `rope_dim` are left untouched.
/// `freq_factors` (Llama 3.1 `rope_freqs.weight`) divides each frequency when present.
#[derive(Clone, Copy, Debug)]
pub struct Rope<'a> {
    pub head_dim: usize,
    pub rope_dim: usize,
    pub theta: f32,
    pub neox: bool,
    pub freq_factors: Option<&'a [f32]>,
}

impl Rope<'_> {
    pub fn apply(&self, x: &mut [f32], n_heads: usize, pos: usize) {
        let half = self.rope_dim / 2;
        // The angles are the same for every head: found per head, they were 3% of a CPU
        // prompt on four threads, all of it on one.
        let trig: Vec<(f32, f32)> = (0..half)
            .map(|i| {
                let mut freq = self.theta.powf(-2.0 * i as f32 / self.rope_dim as f32);
                if let Some(ff) = self.freq_factors {
                    freq /= ff[i];
                }
                (pos as f32 * freq).sin_cos()
            })
            .collect();
        for h in 0..n_heads {
            let head = &mut x[h * self.head_dim..(h + 1) * self.head_dim];
            for (i, &(sin, cos)) in trig.iter().enumerate() {
                let (a, b) = if self.neox {
                    (i, i + half)
                } else {
                    (2 * i, 2 * i + 1)
                };
                let (x0, x1) = (head[a], head[b]);
                head[a] = x0 * cos - x1 * sin;
                head[b] = x0 * sin + x1 * cos;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_is_shift_invariant() {
        let mut a = vec![1.0, 2.0, 3.0, 1000.0];
        let mut b = vec![-999.0, -998.0, -997.0, 0.0];
        softmax(&mut a);
        softmax(&mut b);
        assert!((a.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn rmsnorm_unit_weight_gives_unit_rms() {
        let x = [3.0, -4.0, 12.0, 0.5];
        let mut out = [0.0; 4];
        rmsnorm(&mut out, &x, &[1.0; 4], 0.0);
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_pair_norms_and_is_identity_at_pos_zero() {
        let x: Vec<f32> = (0..16).map(|i| i as f32 - 7.5).collect();
        for neox in [false, true] {
            let rope = Rope {
                head_dim: 8,
                rope_dim: 8,
                theta: 10000.0,
                neox,
                freq_factors: None,
            };
            let mut y0 = x.clone();
            rope.apply(&mut y0, 2, 0);
            assert_eq!(y0, x);
            let mut y = x.clone();
            rope.apply(&mut y, 2, 37);
            let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
            assert!((norm(&y) - norm(&x)).abs() < 1e-3);
        }
    }

    #[test]
    fn rope_neox_and_normal_differ_only_in_pairing() {
        // Permuting adjacent pairs into halves must turn the normal rotation into the NeoX one.
        let x: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        let perm: Vec<f32> = (0..4)
            .map(|i| x[2 * i])
            .chain((0..4).map(|i| x[2 * i + 1]))
            .collect();
        let base = Rope {
            head_dim: 8,
            rope_dim: 8,
            theta: 10000.0,
            neox: false,
            freq_factors: None,
        };
        let mut a = x.clone();
        base.apply(&mut a, 1, 5);
        let mut b = perm.clone();
        Rope { neox: true, ..base }.apply(&mut b, 1, 5);
        for i in 0..4 {
            assert!((a[2 * i] - b[i]).abs() < 1e-6);
            assert!((a[2 * i + 1] - b[i + 4]).abs() < 1e-6);
        }
    }
}
