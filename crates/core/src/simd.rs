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
#[inline]
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

/// `out[t] = x · row_t`, row `t` being `x.len()` values at `t * stride` of `rows` (attention's
/// scores against cached keys). Each equals [`dot`] of that row exactly; one call for all rows
/// saves a dispatch per row that cost more than a 64-128 element dot.
pub fn dot_rows(x: &[f32], rows: &[f32], stride: usize, out: &mut [f32]) {
    if let Some(last) = out.len().checked_sub(1) {
        assert!(rows.len() >= last * stride + x.len(), "rows too short");
    }
    let row = |t: usize| {
        // SAFETY: checked above: row `t < out.len()` ends within `rows`.
        unsafe { rows.get_unchecked(t * stride..t * stride + x.len()) }
    };
    #[cfg(target_arch = "aarch64")]
    for (t, o) in out.iter_mut().enumerate() {
        // SAFETY: NEON is mandatory on aarch64; the row holds `x.len()` values.
        *o = unsafe { dot_neon(row(t), x) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx2() {
            // SAFETY: CPU support verified at runtime; bounds checked above.
            unsafe { x86::dot_rows_avx2(x, rows, stride, out) }
        } else {
            for (t, o) in out.iter_mut().enumerate() {
                *o = dot_scalar(row(t), x);
            }
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for (t, o) in out.iter_mut().enumerate() {
        *o = dot_scalar(row(t), x);
    }
}

/// Rows of `width` little-endian halves, row `t` at byte `t * stride` of `rows`, converted to f32
/// and packed into `out` (`out.len() / width` rows). Exact: every half is an f32.
pub fn f16_rows_to_f32(rows: &[u8], stride: usize, width: usize, out: &mut [f32]) {
    if width == 0 {
        return;
    }
    let n = out.len() / width;
    assert_eq!(out.len(), n * width, "out holds whole rows");
    if let Some(last) = n.checked_sub(1) {
        assert!(rows.len() >= last * stride + 2 * width, "rows too short");
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; bounds checked above.
        unsafe { f16_rows_to_f32_neon(rows, stride, width, out) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx2() {
            // SAFETY: CPU support verified at runtime; bounds checked above.
            unsafe { x86::f16_rows_to_f32_avx2(rows, stride, width, out) }
        } else {
            f16_rows_to_f32_scalar(rows, stride, width, out)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        f16_rows_to_f32_scalar(rows, stride, width, out)
    }
}

pub fn f16_rows_to_f32_scalar(rows: &[u8], stride: usize, width: usize, out: &mut [f32]) {
    for (t, o) in out.chunks_exact_mut(width).enumerate() {
        let r = &rows[t * stride..][..2 * width];
        for (v, h) in o.iter_mut().zip(r.as_chunks::<2>().0) {
            *v = crate::half::f16_to_f32(u16::from_le_bytes(*h));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn f16_rows_to_f32_neon(rows: &[u8], stride: usize, width: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let po = out.as_mut_ptr();
    for t in 0..out.len() / width {
        // SAFETY: the caller checked that row `t` holds `width` halves and `out` `width` values
        // for it; `rows` may be only byte-aligned, and the loads are unaligned-safe.
        unsafe {
            let (r, o) = (
                rows.as_ptr().add(t * stride) as *const u16,
                po.add(t * width),
            );
            let mut i = 0;
            while i + 8 <= width {
                let (lo, hi) = widen(vld1q_u16(r.add(i)));
                vst1q_f32(o.add(i), lo);
                vst1q_f32(o.add(i + 4), hi);
                i += 8;
            }
            while i < width {
                *o.add(i) = crate::half::f16_to_f32(r.add(i).read_unaligned());
                i += 1;
            }
        }
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

/// `out += Σ_t w[t] · row_t`, row `t` being `out.len()` values at `t * stride` of `rows`
/// (attention's weighted sum of cached values). The SIMD paths keep `out` in registers across all
/// rows instead of loading and storing it for each one.
pub fn weighted_rows(out: &mut [f32], w: &[f32], rows: &[f32], stride: usize) {
    if let Some(last) = w.len().checked_sub(1) {
        assert!(rows.len() >= last * stride + out.len(), "rows too short");
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; bounds checked above.
        unsafe { weighted_rows_neon(out, w, rows, stride) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx2() {
            // SAFETY: CPU support verified at runtime; bounds checked above.
            unsafe { x86::weighted_rows_avx2(out, w, rows, stride) }
        } else {
            weighted_rows_scalar(out, w, rows, stride)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        weighted_rows_scalar(out, w, rows, stride)
    }
}

pub fn weighted_rows_scalar(out: &mut [f32], w: &[f32], rows: &[f32], stride: usize) {
    let n = out.len();
    for (t, &a) in w.iter().enumerate() {
        for (o, &v) in out.iter_mut().zip(&rows[t * stride..][..n]) {
            *o += a * v;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn weighted_rows_neon(out: &mut [f32], w: &[f32], rows: &[f32], stride: usize) {
    use std::arch::aarch64::*;
    /// Columns `c..c + 4 * N` over every row, accumulated in `N` registers.
    #[inline(always)]
    unsafe fn cols<const N: usize>(
        out: *mut f32,
        w: &[f32],
        rows: *const f32,
        stride: usize,
        c: usize,
    ) {
        // SAFETY: the caller checked that every row holds these columns.
        unsafe {
            let mut acc: [float32x4_t; N] = std::array::from_fn(|j| vld1q_f32(out.add(c + 4 * j)));
            for (t, &a) in w.iter().enumerate() {
                let r = rows.add(t * stride + c);
                for (j, slot) in acc.iter_mut().enumerate() {
                    *slot = vfmaq_n_f32(*slot, vld1q_f32(r.add(4 * j)), a);
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
    // SAFETY: `weighted_rows` checked the bounds.
    unsafe {
        while c + 32 <= n {
            cols::<8>(po, w, pr, stride, c);
            c += 32;
        }
        while c + 4 <= n {
            cols::<1>(po, w, pr, stride, c);
            c += 4;
        }
        for i in c..n {
            let mut s = *po.add(i);
            for (t, &a) in w.iter().enumerate() {
                s += a * *pr.add(t * stride + i);
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
    #[inline]
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

    /// [`super::dot_rows`], with the per-row dot inlined.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_rows_avx2(x: &[f32], rows: &[f32], stride: usize, out: &mut [f32]) {
        for (t, o) in out.iter_mut().enumerate() {
            // SAFETY: the caller checked that row `t` ends within `rows`.
            *o = unsafe { dot_avx2(rows.get_unchecked(t * stride..t * stride + x.len()), x) };
        }
    }

    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn f16_rows_to_f32_avx2(rows: &[u8], stride: usize, width: usize, out: &mut [f32]) {
        let po = out.as_mut_ptr();
        for t in 0..out.len() / width {
            // SAFETY: the caller checked that row `t` holds `width` halves and `out` `width`
            // values for it.
            unsafe {
                let (r, o) = (
                    rows.as_ptr().add(t * stride) as *const u16,
                    po.add(t * width),
                );
                let mut i = 0;
                while i + 8 <= width {
                    _mm256_storeu_ps(o.add(i), load_halves(r.add(i)));
                    i += 8;
                }
                while i < width {
                    *o.add(i) = crate::half::f16_to_f32(r.add(i).read_unaligned());
                    i += 1;
                }
            }
        }
    }

    /// Columns `c..c + 8 * N` over every row, accumulated in `N` registers.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn weighted_cols<const N: usize>(
        out: *mut f32,
        w: &[f32],
        rows: *const f32,
        stride: usize,
        c: usize,
    ) {
        // SAFETY: the caller checked that every row holds these columns.
        unsafe {
            let mut acc: [__m256; N] = std::array::from_fn(|j| _mm256_loadu_ps(out.add(c + 8 * j)));
            for (t, &a) in w.iter().enumerate() {
                let r = rows.add(t * stride + c);
                let va = _mm256_set1_ps(a);
                for (j, slot) in acc.iter_mut().enumerate() {
                    *slot = _mm256_fmadd_ps(_mm256_loadu_ps(r.add(8 * j)), va, *slot);
                }
            }
            for (j, v) in acc.iter().enumerate() {
                _mm256_storeu_ps(out.add(c + 8 * j), *v);
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn weighted_rows_avx2(out: &mut [f32], w: &[f32], rows: &[f32], stride: usize) {
        let n = out.len();
        let (po, pr) = (out.as_mut_ptr(), rows.as_ptr());
        let mut c = 0;
        // SAFETY: `weighted_rows` checked the bounds.
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
                let mut s = *po.add(i);
                for (t, &a) in w.iter().enumerate() {
                    s += a * *pr.add(t * stride + i);
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

    /// Rows padded apart, as K/V heads are inside a cache row.
    const SIZES: [(usize, usize); 8] = [
        (0, 3),
        (5, 0),
        (7, 4),
        (8, 9),
        (40, 33),
        (64, 300),
        (128, 17),
        (70, 5),
    ];

    #[test]
    fn f16_rows_to_f32_is_exact() {
        let mut rng = Rng::new(12);
        for (n, rows) in SIZES {
            let stride = 2 * n + 6;
            let bytes: Vec<u8> = rng
                .vec(rows * stride / 2 + n)
                .iter()
                .flat_map(|&v| crate::half::f32_to_f16(v).to_le_bytes())
                .collect();
            let mut got = vec![1.0; rows * n];
            f16_rows_to_f32(&bytes, stride, n, &mut got);
            let mut want = vec![0.0; rows * n];
            if n > 0 {
                f16_rows_to_f32_scalar(&bytes, stride, n, &mut want);
            }
            assert_eq!(got, want, "n={n} rows={rows}");
        }
    }

    #[test]
    fn dot_rows_equals_per_row_dot_exactly() {
        let mut rng = Rng::new(13);
        for (n, rows) in SIZES {
            let stride = n + 3;
            let m = rng.vec(rows * stride + n);
            let x = rng.vec(n);
            let mut got = vec![1.0; rows];
            dot_rows(&x, &m, stride, &mut got);
            let want: Vec<f32> = (0..rows)
                .map(|t| dot(&m[t * stride..t * stride + n], &x))
                .collect();
            assert_eq!(got, want, "n={n} rows={rows}");
        }
    }

    #[test]
    fn weighted_rows_adds_to_out_and_matches_scalar() {
        let mut rng = Rng::new(11);
        for (n, rows) in SIZES {
            let stride = n + 3;
            let m = rng.vec(rows * stride + n);
            let w = rng.vec(rows);
            let start = rng.vec(n);
            let (mut got, mut want) = (start.clone(), start);
            weighted_rows(&mut got, &w, &m, stride);
            weighted_rows_scalar(&mut want, &w, &m, stride);
            for (g, s) in got.iter().zip(&want) {
                assert!((g - s).abs() <= 1e-4 * (1.0 + s.abs()), "n={n} {g} vs {s}");
            }
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
