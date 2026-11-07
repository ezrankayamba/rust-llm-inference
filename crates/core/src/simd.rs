//! SIMD f32 dot products, also against f16 weights. NEON is architecturally guaranteed on
//! aarch64; on x86_64 AVX2+FMA is detected at runtime. Every path has the scalar version as its
//! fallback and test reference.

pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; lengths checked above.
        unsafe { dot_neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx2_fma() {
            // SAFETY: CPU support verified at runtime; lengths checked above.
            unsafe { x86::dot_avx2(a, b) }
        } else {
            dot_scalar(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_scalar(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    // SAFETY: every load stays within `n`, which both slices share.
    unsafe {
        while i + 16 <= n {
            for (k, slot) in acc.iter_mut().enumerate() {
                let off = i + 4 * k;
                *slot = vfmaq_f32(*slot, vld1q_f32(pa.add(off)), vld1q_f32(pb.add(off)));
            }
            i += 16;
        }
        while i + 4 <= n {
            acc[0] = vfmaq_f32(acc[0], vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            i += 4;
        }
        let mut s = vaddvq_f32(vaddq_f32(
            vaddq_f32(acc[0], acc[1]),
            vaddq_f32(acc[2], acc[3]),
        ));
        while i < n {
            s += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        s
    }
}

/// `Σ w·x` with `w` as little-endian IEEE halves (`2 * x.len()` bytes).
#[inline]
pub fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    assert_eq!(w.len(), 2 * x.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; lengths checked above.
        unsafe { dot_f16_neon(w, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_f16_scalar(w, x)
    }
}

pub fn dot_f16_scalar(w: &[u8], x: &[f32]) -> f32 {
    w.as_chunks::<2>()
        .0
        .iter()
        .zip(x)
        .map(|(h, &v)| crate::half::f16_to_f32(u16::from_le_bytes(*h)) * v)
        .sum()
}

/// Halves to f32 eight at a time with `fcvtl`/`fcvtl2` (base AArch64; Rust's f16 vector
/// intrinsics are not stable yet), fused into the multiply-adds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f16_neon(w: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    #[inline(always)]
    fn widen(h: uint16x8_t) -> (float32x4_t, float32x4_t) {
        let (lo, hi);
        // SAFETY: register-only conversions.
        unsafe {
            std::arch::asm!(
                "fcvtl {lo:v}.4s, {h:v}.4h",
                "fcvtl2 {hi:v}.4s, {h:v}.8h",
                h = in(vreg) h,
                // `out`, not `lateout`: `h` is still read after `lo` is written.
                lo = out(vreg) lo,
                hi = lateout(vreg) hi,
                options(pure, nomem, nostack),
            );
        }
        (lo, hi)
    }
    let n = x.len();
    let (pw, px) = (w.as_ptr() as *const u16, x.as_ptr());
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    // SAFETY: every load stays within `n` elements, which `w` (as halves) and `x` share; u16
    // loads need no alignment beyond a byte for `vld1q_u16`.
    unsafe {
        while i + 16 <= n {
            let (a, b) = widen(vld1q_u16(pw.add(i)));
            let (c, d) = widen(vld1q_u16(pw.add(i + 8)));
            acc[0] = vfmaq_f32(acc[0], a, vld1q_f32(px.add(i)));
            acc[1] = vfmaq_f32(acc[1], b, vld1q_f32(px.add(i + 4)));
            acc[2] = vfmaq_f32(acc[2], c, vld1q_f32(px.add(i + 8)));
            acc[3] = vfmaq_f32(acc[3], d, vld1q_f32(px.add(i + 12)));
            i += 16;
        }
        let mut s = vaddvq_f32(vaddq_f32(
            vaddq_f32(acc[0], acc[1]),
            vaddq_f32(acc[2], acc[3]),
        ));
        while i < n {
            s += crate::half::f16_to_f32(*pw.add(i)) * *px.add(i);
            i += 1;
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::sync::OnceLock;

    pub fn has_avx2_fma() -> bool {
        static HAS: OnceLock<bool> = OnceLock::new();
        *HAS.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        // SAFETY: every load stays within `n`, which both slices share.
        unsafe {
            while i + 16 <= n {
                acc0 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 8)),
                    _mm256_loadu_ps(pb.add(i + 8)),
                    acc1,
                );
                i += 16;
            }
            let acc = _mm256_add_ps(acc0, acc1);
            let lo = _mm256_castps256_ps128(acc);
            let hi = _mm256_extractf128_ps(acc, 1);
            let s4 = _mm_add_ps(lo, hi);
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            let mut s = _mm_cvtss_f32(s1);
            while i < n {
                s += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    #[test]
    fn dot_f16_matches_scalar_for_all_tail_lengths() {
        let mut rng = Rng::new(8);
        // Long rows too: many 16-wide iterations, where a register clash in the conversion
        // shows up (one did, as `lateout` let `lo` overwrite the input).
        for n in (0..70).chain([512, 1000]) {
            let w: Vec<u8> = rng
                .vec(n)
                .iter()
                .flat_map(|&v| crate::half::f32_to_f16(v).to_le_bytes())
                .collect();
            let x = rng.vec(n);
            let (fast, slow) = (dot_f16(&w, &x), dot_f16_scalar(&w, &x));
            assert!(
                (fast - slow).abs() <= 1e-4 * (1.0 + slow.abs()),
                "n={n} {fast} vs {slow}"
            );
        }
    }

    #[test]
    fn dot_matches_scalar_for_all_tail_lengths() {
        let mut rng = Rng::new(7);
        for n in 0..70 {
            let a = rng.vec(n);
            let b = rng.vec(n);
            let (fast, slow) = (dot(&a, &b), dot_scalar(&a, &b));
            assert!(
                (fast - slow).abs() <= 1e-4 * (1.0 + slow.abs()),
                "n={n} {fast} vs {slow}"
            );
        }
    }
}
