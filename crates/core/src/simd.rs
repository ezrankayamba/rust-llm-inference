//! SIMD f32 dot products, also against f16 weights. NEON is architecturally guaranteed on
//! aarch64; on x86_64 AVX2+FMA+F16C is detected at runtime. Every path has the scalar version as
//! its fallback and test reference.

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
        if x86::has_avx2() {
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
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx2() {
            // SAFETY: CPU support verified at runtime; lengths checked above.
            unsafe { x86::dot_f16_avx2(w, x) }
        } else {
            dot_f16_scalar(w, x)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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

/// Eight halves to two f32 vectors with `fcvtl`/`fcvtl2` (base AArch64; Rust's f16 vector
/// intrinsics are not stable yet).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn widen(
    h: std::arch::aarch64::uint16x8_t,
) -> (
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
) {
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

/// `out[t] = x · row_t`, row `t` being `x.len()` little-endian halves at byte `t * stride` of
/// `rows` (attention's scores against cached keys). Each equals [`dot_f16`] of that row exactly;
/// one call for all rows saves a dispatch per row that cost more than a 64-128 element dot.
pub fn dot_f16_rows(x: &[f32], rows: &[u8], stride: usize, out: &mut [f32]) {
    if let Some(last) = out.len().checked_sub(1) {
        assert!(rows.len() >= last * stride + 2 * x.len(), "rows too short");
    }
    let row = |t: usize| {
        // SAFETY: checked above: row `t < out.len()` ends within `rows`.
        unsafe { rows.get_unchecked(t * stride..t * stride + 2 * x.len()) }
    };
    #[cfg(target_arch = "aarch64")]
    for (t, o) in out.iter_mut().enumerate() {
        // SAFETY: NEON is mandatory on aarch64; the row holds `x.len()` halves.
        *o = unsafe { dot_f16_neon(row(t), x) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx2() {
            // SAFETY: CPU support verified at runtime; bounds checked above.
            unsafe { x86::dot_f16_rows_avx2(x, rows, stride, out) }
        } else {
            for (t, o) in out.iter_mut().enumerate() {
                *o = dot_f16_scalar(row(t), x);
            }
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for (t, o) in out.iter_mut().enumerate() {
        *o = dot_f16_scalar(row(t), x);
    }
}

/// The f16 dot, converting eight halves per step and fusing them into the multiply-adds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_f16_neon(w: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len();
    let (pw, px) = (w.as_ptr() as *const u16, x.as_ptr());
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    // SAFETY: every load stays within `n` elements, which `w` (as halves) and `x` share. `w`
    // may be only byte-aligned: `vld1q_u16` doesn't require alignment, and the scalar tail
    // reads unaligned.
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
            s += crate::half::f16_to_f32(pw.add(i).read_unaligned()) * *px.add(i);
            i += 1;
        }
        s
    }
}

/// `y += a * w` with `w` as little-endian IEEE halves (`2 * y.len()` bytes).
#[inline]
pub fn axpy_f16(y: &mut [f32], a: f32, w: &[u8]) {
    assert_eq!(w.len(), 2 * y.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; lengths checked above.
        unsafe { axpy_f16_neon(y, a, w) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx2() {
            // SAFETY: CPU support verified at runtime; lengths checked above.
            unsafe { x86::axpy_f16_avx2(y, a, w) }
        } else {
            axpy_f16_scalar(y, a, w)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        axpy_f16_scalar(y, a, w)
    }
}

pub fn axpy_f16_scalar(y: &mut [f32], a: f32, w: &[u8]) {
    for (o, h) in y.iter_mut().zip(w.as_chunks::<2>().0) {
        *o += a * crate::half::f16_to_f32(u16::from_le_bytes(*h));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_f16_neon(y: &mut [f32], a: f32, w: &[u8]) {
    use std::arch::aarch64::*;
    let n = y.len();
    let (pw, py) = (w.as_ptr() as *const u16, y.as_mut_ptr());
    let mut i = 0;
    // SAFETY: every access stays within `n` elements, which `w` (as halves) and `y` share. `w`
    // may be only byte-aligned: `vld1q_u16` doesn't require alignment, and the scalar tail
    // reads unaligned.
    unsafe {
        while i + 8 <= n {
            let (lo, hi) = widen(vld1q_u16(pw.add(i)));
            vst1q_f32(py.add(i), vfmaq_n_f32(vld1q_f32(py.add(i)), lo, a));
            vst1q_f32(py.add(i + 4), vfmaq_n_f32(vld1q_f32(py.add(i + 4)), hi, a));
            i += 8;
        }
        while i < n {
            *py.add(i) += a * crate::half::f16_to_f32(pw.add(i).read_unaligned());
            i += 1;
        }
    }
}

/// `out = Σ_t w[t] · row_t`, row `t` being `out.len()` little-endian halves at byte `t * stride`
/// of `rows` (attention's weighted sum of cached values). Bit-identical to zeroing `out` and
/// calling [`axpy_f16`] per row, but the SIMD paths keep the outputs in registers across all rows
/// instead of loading and storing them for each one.
pub fn weighted_rows_f16(out: &mut [f32], w: &[f32], rows: &[u8], stride: usize) {
    if let Some(last) = w.len().checked_sub(1) {
        assert!(
            rows.len() >= last * stride + 2 * out.len(),
            "rows too short"
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; bounds checked above.
        unsafe { weighted_rows_f16_neon(out, w, rows, stride) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx2() {
            // SAFETY: CPU support verified at runtime; bounds checked above.
            unsafe { x86::weighted_rows_f16_avx2(out, w, rows, stride) }
        } else {
            weighted_rows_f16_scalar(out, w, rows, stride)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        weighted_rows_f16_scalar(out, w, rows, stride)
    }
}

pub fn weighted_rows_f16_scalar(out: &mut [f32], w: &[f32], rows: &[u8], stride: usize) {
    out.fill(0.0);
    for (t, &a) in w.iter().enumerate() {
        axpy_f16_scalar(out, a, &rows[t * stride..t * stride + 2 * out.len()]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn weighted_rows_f16_neon(out: &mut [f32], w: &[f32], rows: &[u8], stride: usize) {
    use std::arch::aarch64::*;
    /// Columns `c..c + 4 * N` over every row, accumulated in `N` registers.
    #[inline(always)]
    unsafe fn cols<const N: usize>(
        out: *mut f32,
        w: &[f32],
        rows: *const u8,
        stride: usize,
        c: usize,
    ) {
        // SAFETY: the caller checked that every row holds these columns.
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); N];
            for (t, &a) in w.iter().enumerate() {
                let r = rows.add(t * stride + 2 * c) as *const u16;
                for j in 0..N / 2 {
                    let (lo, hi) = widen(vld1q_u16(r.add(8 * j)));
                    acc[2 * j] = vfmaq_n_f32(acc[2 * j], lo, a);
                    acc[2 * j + 1] = vfmaq_n_f32(acc[2 * j + 1], hi, a);
                }
            }
            for (j, v) in acc.iter().enumerate() {
                vst1q_f32(out.add(c + 4 * j), *v);
            }
        }
    }
    let n = out.len();
    let (po, pr) = (out.as_mut_ptr(), rows.as_ptr());
    let mut c = 0;
    // SAFETY: `weighted_rows_f16` checked the bounds; `rows` may be only byte-aligned, and the
    // loads are unaligned-safe.
    unsafe {
        while c + 32 <= n {
            cols::<8>(po, w, pr, stride, c);
            c += 32;
        }
        while c + 8 <= n {
            cols::<2>(po, w, pr, stride, c);
            c += 8;
        }
        for i in c..n {
            let mut s = 0.0;
            for (t, &a) in w.iter().enumerate() {
                s += a * crate::half::f16_to_f32(
                    (pr.add(t * stride + 2 * i) as *const u16).read_unaligned(),
                );
            }
            *po.add(i) = s;
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    /// Every CPU with AVX2 has FMA and F16C too, so all the x86 kernels share one check.
    pub fn has_avx2() -> bool {
        static HAS: OnceLock<bool> = OnceLock::new();
        *HAS.get_or_init(|| {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
        })
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn hsum(v: __m256) -> f32 {
        let s4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
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
            let mut s = hsum(_mm256_add_ps(acc0, acc1));
            while i < n {
                s += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            s
        }
    }

    /// Eight halves at `p`, which may be only byte-aligned, as f32s.
    #[target_feature(enable = "avx2,f16c")]
    #[inline]
    unsafe fn load_halves(p: *const u16) -> __m256 {
        // SAFETY: the caller guarantees eight readable halves; the load is unaligned-safe.
        _mm256_cvtph_ps(unsafe { _mm_loadu_si128(p.cast()) })
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    pub unsafe fn dot_f16_avx2(w: &[u8], x: &[f32]) -> f32 {
        let n = x.len();
        let (pw, px) = (w.as_ptr() as *const u16, x.as_ptr());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        // SAFETY: every load stays within `n` elements, which `w` (as halves) and `x` share.
        unsafe {
            while i + 16 <= n {
                acc0 = _mm256_fmadd_ps(load_halves(pw.add(i)), _mm256_loadu_ps(px.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(
                    load_halves(pw.add(i + 8)),
                    _mm256_loadu_ps(px.add(i + 8)),
                    acc1,
                );
                i += 16;
            }
            let mut s = hsum(_mm256_add_ps(acc0, acc1));
            while i < n {
                s += crate::half::f16_to_f32(pw.add(i).read_unaligned()) * *px.add(i);
                i += 1;
            }
            s
        }
    }

    /// [`super::dot_f16_rows`], with the per-row dot inlined.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16_rows_avx2(x: &[f32], rows: &[u8], stride: usize, out: &mut [f32]) {
        for (t, o) in out.iter_mut().enumerate() {
            // SAFETY: the caller checked that row `t` ends within `rows`.
            *o = unsafe {
                dot_f16_avx2(rows.get_unchecked(t * stride..t * stride + 2 * x.len()), x)
            };
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn axpy_f16_avx2(y: &mut [f32], a: f32, w: &[u8]) {
        let n = y.len();
        let (pw, py) = (w.as_ptr() as *const u16, y.as_mut_ptr());
        let va = _mm256_set1_ps(a);
        let mut i = 0;
        // SAFETY: every access stays within `n` elements, which `w` (as halves) and `y` share.
        unsafe {
            while i + 8 <= n {
                let v = _mm256_fmadd_ps(load_halves(pw.add(i)), va, _mm256_loadu_ps(py.add(i)));
                _mm256_storeu_ps(py.add(i), v);
                i += 8;
            }
            while i < n {
                *py.add(i) += a * crate::half::f16_to_f32(pw.add(i).read_unaligned());
                i += 1;
            }
        }
    }

    /// Columns `c..c + 8 * N` over every row, accumulated in `N` registers.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    unsafe fn weighted_cols<const N: usize>(
        out: *mut f32,
        w: &[f32],
        rows: *const u8,
        stride: usize,
        c: usize,
    ) {
        // SAFETY: the caller checked that every row holds these columns.
        unsafe {
            let mut acc = [_mm256_setzero_ps(); N];
            for (t, &a) in w.iter().enumerate() {
                let r = rows.add(t * stride + 2 * c) as *const u16;
                let va = _mm256_set1_ps(a);
                for (j, slot) in acc.iter_mut().enumerate() {
                    *slot = _mm256_fmadd_ps(load_halves(r.add(8 * j)), va, *slot);
                }
            }
            for (j, v) in acc.iter().enumerate() {
                _mm256_storeu_ps(out.add(c + 8 * j), *v);
            }
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn weighted_rows_f16_avx2(out: &mut [f32], w: &[f32], rows: &[u8], stride: usize) {
        let n = out.len();
        let (po, pr) = (out.as_mut_ptr(), rows.as_ptr());
        let mut c = 0;
        // SAFETY: `weighted_rows_f16` checked the bounds.
        unsafe {
            while c + 64 <= n {
                weighted_cols::<8>(po, w, pr, stride, c);
                c += 64;
            }
            while c + 8 <= n {
                weighted_cols::<1>(po, w, pr, stride, c);
                c += 8;
            }
            for i in c..n {
                let mut s = 0.0;
                for (t, &a) in w.iter().enumerate() {
                    s += a * crate::half::f16_to_f32(
                        (pr.add(t * stride + 2 * i) as *const u16).read_unaligned(),
                    );
                }
                *po.add(i) = s;
            }
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
    fn axpy_f16_matches_scalar_for_all_tail_lengths() {
        let mut rng = Rng::new(9);
        for n in (0..40).chain([128, 1000]) {
            let w: Vec<u8> = rng
                .vec(n)
                .iter()
                .flat_map(|&v| crate::half::f32_to_f16(v).to_le_bytes())
                .collect();
            let y0 = rng.vec(n);
            let (mut fast, mut slow) = (y0.clone(), y0);
            axpy_f16(&mut fast, 0.37, &w);
            axpy_f16_scalar(&mut slow, 0.37, &w);
            for (f, s) in fast.iter().zip(&slow) {
                assert!((f - s).abs() <= 1e-6 * (1.0 + s.abs()), "n={n} {f} vs {s}");
            }
        }
    }

    #[test]
    fn dot_f16_rows_equals_per_row_dot_exactly() {
        let mut rng = Rng::new(12);
        for (n, rows) in [
            (0, 3),
            (5, 0),
            (7, 4),
            (16, 9),
            (64, 300),
            (128, 17),
            (70, 5),
        ] {
            let stride = 2 * n + 6; // rows padded apart, as K/V heads are inside a cache row
            let bytes: Vec<u8> = rng
                .vec(rows * stride / 2 + n)
                .iter()
                .flat_map(|&v| crate::half::f32_to_f16(v).to_le_bytes())
                .collect();
            let x = rng.vec(n);
            let mut got = vec![1.0; rows];
            dot_f16_rows(&x, &bytes, stride, &mut got);
            let want: Vec<f32> = (0..rows)
                .map(|t| dot_f16(&bytes[t * stride..t * stride + 2 * n], &x))
                .collect();
            assert_eq!(got, want, "n={n} rows={rows}");
        }
    }

    #[test]
    fn weighted_rows_f16_equals_per_row_axpy_exactly() {
        let mut rng = Rng::new(11);
        for (n, rows) in [
            (0, 3),
            (5, 0),
            (7, 4),
            (8, 9),
            (40, 33),
            (64, 300),
            (128, 17),
            (70, 5),
        ] {
            let stride = 2 * n + 6; // rows padded apart, as K/V heads are inside a cache row
            let bytes: Vec<u8> = rng
                .vec(rows * stride / 2 + n)
                .iter()
                .flat_map(|&v| crate::half::f32_to_f16(v).to_le_bytes())
                .collect();
            let w = rng.vec(rows);
            let mut got = vec![1.0; n];
            weighted_rows_f16(&mut got, &w, &bytes, stride);
            let mut want = vec![0.0; n];
            for (t, &a) in w.iter().enumerate() {
                axpy_f16(&mut want, a, &bytes[t * stride..t * stride + 2 * n]);
            }
            assert_eq!(got, want, "n={n} rows={rows}");
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
