//! Integer dot products of quantized weight rows with 8-bit-quantized activations.
//!
//! The activation vector is quantized once per matvec into 32-element blocks (scale `d`, i8
//! values, and `d * Σq` for the min/offset terms). Each weight format then unpacks 32 (or 16)
//! weights at a time to i8 and accumulates exact integer products, applying scales per
//! sub-block. This is the fast path; `quant::dot_row` (f32) is its reference.

use crate::half::f16_to_f32;
use crate::quant::{GgmlType, QK_K};

const QK: usize = 32;

/// Activations quantized to Q8 in blocks of 32.
#[derive(Clone)]
pub struct Q8Act {
    d: Vec<f32>,
    /// `d * Σ q` per block: the block's (quantized) sum, for formats with a min/offset term.
    sum: Vec<f32>,
    qs: Vec<i8>,
}

impl Q8Act {
    pub fn new(x: &[f32]) -> Self {
        assert!(
            x.len().is_multiple_of(QK),
            "activation length must be a multiple of {QK}"
        );
        let n = x.len() / QK;
        let mut a = Q8Act {
            d: Vec::with_capacity(n),
            sum: Vec::with_capacity(n),
            qs: vec![0; x.len()],
        };
        for (blk, qs) in x
            .as_chunks::<QK>()
            .0
            .iter()
            .zip(a.qs.as_chunks_mut::<QK>().0.iter_mut())
        {
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut s = 0i32;
            for (q, &v) in qs.iter_mut().zip(blk) {
                *q = (v * id).round() as i8;
                s += *q as i32;
            }
            a.d.push(d);
            a.sum.push(d * s as f32);
        }
        a
    }
}

/// Whether `ty` has an integer kernel (half/full float rows use the f32 path).
pub fn supports(ty: GgmlType) -> bool {
    !matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

#[inline(always)]
fn idot(a: &[i8], b: &[i8]) -> i32 {
    // Written so LLVM vectorizes it to widening multiply-accumulates (smull/sadalp on NEON,
    // pmaddubsw-style sequences on AVX2).
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

#[inline]
fn le16(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    let (s, m) = if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    };
    (s as f32, m as f32)
}

/// Dot product of one encoded row with quantized activations. Panics for float types.
pub fn dot_row_q8(ty: GgmlType, row: &[u8], a: &Q8Act) -> f32 {
    let mut out = [0.0];
    dot_rows_q8(ty, row, std::slice::from_ref(a), &mut out);
    out[0]
}

/// Most tokens [`dot_rows_q8`] handles per call (unpacking each weight block once for them).
pub const MAX_TOKENS: usize = 4;

/// Dot products of one encoded row with the activations of up to [`MAX_TOKENS`] tokens, into
/// `out` (one per token). Each token's result equals [`dot_row_q8`] on it exactly.
pub fn dot_rows_q8(ty: GgmlType, row: &[u8], acts: &[Q8Act], out: &mut [f32]) {
    assert!(acts.len() == out.len() && acts.len() <= MAX_TOKENS);
    #[cfg(target_arch = "aarch64")]
    if neon::has_dotprod() {
        // SAFETY: dotprod support checked at runtime; sizes are asserted inside.
        let done = unsafe {
            match acts.len() {
                1 => neon::dot_rows::<1>(ty, row, acts, out),
                2 => neon::dot_rows::<2>(ty, row, acts, out),
                3 => neon::dot_rows::<3>(ty, row, acts, out),
                4 => neon::dot_rows::<4>(ty, row, acts, out),
                _ => true,
            }
        };
        if done {
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    if crate::simd::x86::has_avx2() {
        // SAFETY: AVX2, FMA and F16C checked at runtime; sizes are asserted inside.
        let done = unsafe {
            match acts.len() {
                1 => avx2::dot_rows::<1>(ty, row, acts, out),
                2 => avx2::dot_rows::<2>(ty, row, acts, out),
                3 => avx2::dot_rows::<3>(ty, row, acts, out),
                4 => avx2::dot_rows::<4>(ty, row, acts, out),
                _ => true,
            }
        };
        if done {
            return;
        }
    }
    for (o, a) in out.iter_mut().zip(acts) {
        *o = dot_row_q8_portable(ty, row, a);
    }
}

/// Dot products of a block of encoded rows (`rows`, `row_bytes` each) with two or more tokens
/// in `acts`, into `out` as `[row][token]`, on the i8mm matrix-multiply instructions (2 rows x
/// 2 tokens per instruction). A result does not depend on how tokens are grouped, but differs
/// from [`dot_row_q8`] by float rounding: decode keeps that kernel, which streams one row at a
/// time faster than any pairing. `false` (nothing written) for a single token, when the CPU
/// lacks i8mm, or when the type has no kernel; callers then use [`dot_rows_q8`].
pub fn dot_block(
    ty: GgmlType,
    rows: &[u8],
    row_bytes: usize,
    acts: &[Q8Act],
    out: &mut [f32],
) -> bool {
    #[cfg(target_arch = "aarch64")]
    if acts.len() >= 2 && neon::has_i8mm() && neon::mm_supports(ty) {
        // SAFETY: i8mm (and dotprod, which every i8mm CPU has) checked at runtime; sizes are
        // asserted inside.
        unsafe { neon::dot_block(ty, rows, row_bytes, acts, out) };
        return true;
    }
    let _ = (ty, rows, row_bytes, acts, out);
    false
}

/// Portable kernels; the reference for the SIMD versions.
pub fn dot_row_q8_portable(ty: GgmlType, row: &[u8], a: &Q8Act) -> f32 {
    let bb = ty.block_bytes();
    let mut w = [0i8; QK];
    let mut sum = 0f32;
    match ty {
        GgmlType::Q8_0 => {
            for (i, b) in row.chunks_exact(bb).enumerate() {
                // SAFETY: i8 and u8 have identical layout.
                let q: &[i8] = unsafe { std::slice::from_raw_parts(b[2..].as_ptr().cast(), QK) };
                sum += le16(b, 0) * a.d[i] * idot(q, &a.qs[i * QK..(i + 1) * QK]) as f32;
            }
        }
        GgmlType::Q4_0 | GgmlType::Q4_1 => {
            let (qs_at, offset) = if ty == GgmlType::Q4_0 { (2, 8) } else { (4, 0) };
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let qs = &b[qs_at..qs_at + 16];
                for j in 0..16 {
                    w[j] = (qs[j] & 0xf) as i8 - offset;
                    w[j + 16] = (qs[j] >> 4) as i8 - offset;
                }
                sum += le16(b, 0) * a.d[i] * idot(&w, &a.qs[i * QK..(i + 1) * QK]) as f32;
                if ty == GgmlType::Q4_1 {
                    sum += le16(b, 2) * a.sum[i];
                }
            }
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            let (at, offset) = if ty == GgmlType::Q5_0 {
                (2, 16)
            } else {
                (4, 0)
            };
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let qh = u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
                let qs = &b[at + 4..at + 20];
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    w[j] = ((qs[j] & 0xf) | h0) as i8 - offset;
                    w[j + 16] = ((qs[j] >> 4) | h1) as i8 - offset;
                }
                sum += le16(b, 0) * a.d[i] * idot(&w, &a.qs[i * QK..(i + 1) * QK]) as f32;
                if ty == GgmlType::Q5_1 {
                    sum += le16(b, 2) * a.sum[i];
                }
            }
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let five = ty == GgmlType::Q5_K;
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let (d, dmin) = (le16(b, 0), le16(b, 2));
                let scales = &b[4..16];
                let (qh, ql) = if five {
                    (&b[16..48], &b[48..176])
                } else {
                    (&b[0..0], &b[16..144])
                };
                for g in 0..4 {
                    let q = &ql[32 * g..32 * g + 32];
                    for half in 0..2 {
                        let sub = 2 * g + half;
                        let xb = i * (QK_K / QK) + sub;
                        for l in 0..32 {
                            let mut v = if half == 0 { q[l] & 0xf } else { q[l] >> 4 };
                            if five && qh[l] & (1 << sub) != 0 {
                                v += 16;
                            }
                            w[l] = v as i8;
                        }
                        let (sc, m) = scale_min_k4(sub, scales);
                        let dot = idot(&w, &a.qs[xb * QK..(xb + 1) * QK]) as f32;
                        sum += d * sc * a.d[xb] * dot - dmin * m * a.sum[xb];
                    }
                }
            }
        }
        GgmlType::Q6_K => {
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let d = le16(b, 208);
                let sc = &b[192..208];
                for n in 0..2 {
                    let (ql, qh) = (&b[64 * n..64 * n + 64], &b[128 + 32 * n..128 + 32 * n + 32]);
                    // Quarter k covers elements [32k, 32k+32) of this 128-half.
                    for k in 0..4 {
                        let xb = i * (QK_K / QK) + 4 * n + k;
                        for l in 0..32 {
                            // Quarters 0,1 are the low nibbles of ql[0..32], ql[32..64]; 2,3 the high.
                            let byte = ql[l + 32 * (k % 2)];
                            let lo = if k < 2 { byte & 0xf } else { byte >> 4 };
                            let hi = (qh[l] >> (2 * k)) & 3;
                            w[l] = (lo | (hi << 4)) as i8 - 32;
                        }
                        let xq = &a.qs[xb * QK..(xb + 1) * QK];
                        let s0 = sc[8 * n + 2 * k] as i8 as f32;
                        let s1 = sc[8 * n + 2 * k + 1] as i8 as f32;
                        let dot = s0 * idot(&w[..16], &xq[..16]) as f32
                            + s1 * idot(&w[16..], &xq[16..]) as f32;
                        sum += d * a.d[xb] * dot;
                    }
                }
            }
        }
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => panic!("{ty:?} has no integer kernel"),
    }
    sum
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Q8Act, QK, le16, scale_min_k4};
    use crate::quant::{GgmlType, QK_K};
    use std::arch::aarch64::*;
    use std::sync::OnceLock;

    pub fn has_dotprod() -> bool {
        static HAS: OnceLock<bool> = OnceLock::new();
        *HAS.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
    }

    /// Σ w·x over 32 lanes given as two 16-lane halves, left as 4 partial sums.
    #[inline(always)]
    unsafe fn dot32x4(w0: int8x16_t, w1: int8x16_t, x: *const i8) -> int32x4_t {
        unsafe {
            vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(x)),
                w1,
                vld1q_s8(x.add(16)),
            )
        }
    }

    /// Expands bits `[8*byte_lo .. 8*byte_lo + 16)` of `qh` to lanes holding 0x10 where set.
    #[inline(always)]
    unsafe fn high_bits(qh: u32, byte_lo: usize) -> uint8x16_t {
        unsafe {
            const MASK: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let b = qh.to_le_bytes();
            let src = vcombine_u8(vdup_n_u8(b[byte_lo]), vdup_n_u8(b[byte_lo + 1]));
            vandq_u8(vtstq_u8(src, vld1q_u8(MASK.as_ptr())), vdupq_n_u8(0x10))
        }
    }

    /// Dot products of one encoded row with the activations of `N` tokens, into `out`. Each
    /// weight block is unpacked (and its scales decoded) once for all `N` tokens, and every
    /// token's arithmetic is the same whatever `N` is, so a matmul matches per-token matvecs
    /// bit for bit. `false` for types without a NEON kernel.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_rows<const N: usize>(
        ty: GgmlType,
        row: &[u8],
        acts: &[Q8Act],
        out: &mut [f32],
    ) -> bool {
        let bb = ty.block_bytes();
        let n_blocks = row.len() / bb;
        assert!(acts.len() == N && out.len() == N);
        for a in acts {
            assert_eq!(n_blocks * ty.block_size(), a.qs.len());
        }
        let xq: [*const i8; N] = std::array::from_fn(|t| acts[t].qs.as_ptr());
        let ad: [&[f32]; N] = std::array::from_fn(|t| &acts[t].d[..]);
        let m4 = vdupq_n_u8(0xf);
        let mut acc = [vdupq_n_f32(0.0); N];
        // Scalar terms (the K-quant mins), kept apart from the vector accumulators.
        let mut extra = [0f32; N];
        // SAFETY (all blocks below): every pointer offset stays inside `row` (n_blocks * bb bytes)
        // or each token's `qs` (n_blocks * block_size values), as asserted above.
        unsafe {
            match ty {
                GgmlType::Q8_0 => {
                    for i in 0..n_blocks {
                        let w = row.as_ptr().add(i * bb + 2) as *const i8;
                        let (w0, w1) = (vld1q_s8(w), vld1q_s8(w.add(16)));
                        let d = le16(&row[i * bb..], 0);
                        for t in 0..N {
                            let s = dot32x4(w0, w1, xq[t].add(i * QK));
                            acc[t] = vmlaq_n_f32(acc[t], vcvtq_f32_s32(s), d * ad[t][i]);
                        }
                    }
                }
                GgmlType::Q4_0 => {
                    let eight = vdupq_n_s8(8);
                    for i in 0..n_blocks {
                        let v = vld1q_u8(row.as_ptr().add(i * bb + 2));
                        let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), eight);
                        let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
                        let d = le16(&row[i * bb..], 0);
                        for t in 0..N {
                            let s = dot32x4(lo, hi, xq[t].add(i * QK));
                            acc[t] = vmlaq_n_f32(acc[t], vcvtq_f32_s32(s), d * ad[t][i]);
                        }
                    }
                }
                GgmlType::Q5_0 => {
                    let sixteen = vdupq_n_s8(16);
                    for i in 0..n_blocks {
                        let b = &row[i * bb..];
                        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                        let v = vld1q_u8(b.as_ptr().add(6));
                        let lo = vorrq_u8(vandq_u8(v, m4), high_bits(qh, 0));
                        let hi = vorrq_u8(vshrq_n_u8(v, 4), high_bits(qh, 2));
                        let lo = vsubq_s8(vreinterpretq_s8_u8(lo), sixteen);
                        let hi = vsubq_s8(vreinterpretq_s8_u8(hi), sixteen);
                        let d = le16(b, 0);
                        for t in 0..N {
                            let s = dot32x4(lo, hi, xq[t].add(i * QK));
                            acc[t] = vmlaq_n_f32(acc[t], vcvtq_f32_s32(s), d * ad[t][i]);
                        }
                    }
                }
                GgmlType::Q4_K | GgmlType::Q5_K => {
                    let five = ty == GgmlType::Q5_K;
                    let ql_at = if five { 48 } else { 16 };
                    let sixteen = vdupq_n_u8(16);
                    for i in 0..n_blocks {
                        let b = &row[i * bb..(i + 1) * bb];
                        let (d, dmin) = (le16(b, 0), le16(b, 2));
                        let scales = &b[4..16];
                        let (qh0, qh1) = if five {
                            (vld1q_u8(b.as_ptr().add(16)), vld1q_u8(b.as_ptr().add(32)))
                        } else {
                            (vdupq_n_u8(0), vdupq_n_u8(0))
                        };
                        for g in 0..4 {
                            let q0 = vld1q_u8(b.as_ptr().add(ql_at + 32 * g));
                            let q1 = vld1q_u8(b.as_ptr().add(ql_at + 32 * g + 16));
                            for half in 0..2 {
                                let sub = 2 * g + half;
                                let (mut w0, mut w1) = if half == 0 {
                                    (vandq_u8(q0, m4), vandq_u8(q1, m4))
                                } else {
                                    (vshrq_n_u8(q0, 4), vshrq_n_u8(q1, 4))
                                };
                                if five {
                                    let bit = vdupq_n_u8(1 << sub);
                                    w0 = vorrq_u8(w0, vandq_u8(vtstq_u8(qh0, bit), sixteen));
                                    w1 = vorrq_u8(w1, vandq_u8(vtstq_u8(qh1, bit), sixteen));
                                }
                                let (w0, w1) = (vreinterpretq_s8_u8(w0), vreinterpretq_s8_u8(w1));
                                let xb = i * (QK_K / QK) + sub;
                                let (sc, m) = scale_min_k4(sub, scales);
                                let (dsc, dm) = (d * sc, dmin * m);
                                for t in 0..N {
                                    let s = dot32x4(w0, w1, xq[t].add(xb * QK));
                                    acc[t] = vmlaq_n_f32(acc[t], vcvtq_f32_s32(s), dsc * ad[t][xb]);
                                    extra[t] -= dm * acts[t].sum[xb];
                                }
                            }
                        }
                    }
                }
                GgmlType::Q6_K => {
                    let three = vdupq_n_u8(3);
                    let off = vdupq_n_s8(32);
                    for i in 0..n_blocks {
                        let b = &row[i * bb..(i + 1) * bb];
                        let d = le16(b, 208);
                        let sc = &b[192..208];
                        for n in 0..2 {
                            let p = b.as_ptr();
                            let ql = [0, 1, 2, 3].map(|j| vld1q_u8(p.add(64 * n + 16 * j)));
                            let qh = [
                                vld1q_u8(p.add(128 + 32 * n)),
                                vld1q_u8(p.add(128 + 32 * n + 16)),
                            ];
                            for k in 0..4 {
                                let (l0, l1) = if k % 2 == 0 {
                                    (ql[0], ql[1])
                                } else {
                                    (ql[2], ql[3])
                                };
                                let (l0, l1) = if k < 2 {
                                    (vandq_u8(l0, m4), vandq_u8(l1, m4))
                                } else {
                                    (vshrq_n_u8(l0, 4), vshrq_n_u8(l1, 4))
                                };
                                let shift = vdupq_n_s8(-(2 * k as i8));
                                let h0 = vshlq_n_u8(vandq_u8(vshlq_u8(qh[0], shift), three), 4);
                                let h1 = vshlq_n_u8(vandq_u8(vshlq_u8(qh[1], shift), three), 4);
                                let w0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(l0, h0)), off);
                                let w1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(l1, h1)), off);
                                let xb = i * (QK_K / QK) + 4 * n + k;
                                // The two 16-element halves have their own i8 scales; apply them
                                // in integer arithmetic (exact) before the one float multiply.
                                let s0 = sc[8 * n + 2 * k] as i8 as i32;
                                let s1 = sc[8 * n + 2 * k + 1] as i8 as i32;
                                for t in 0..N {
                                    let x = xq[t].add(xb * QK);
                                    let p0 = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(x));
                                    let p1 = vdotq_s32(vdupq_n_s32(0), w1, vld1q_s8(x.add(16)));
                                    let s = vmlaq_n_s32(vmulq_n_s32(p0, s0), p1, s1);
                                    acc[t] = vmlaq_n_f32(acc[t], vcvtq_f32_s32(s), d * ad[t][xb]);
                                }
                            }
                        }
                    }
                }
                _ => return false,
            }
        }
        for t in 0..N {
            out[t] = vaddvq_f32(acc[t]) + extra[t];
        }
        true
    }

    pub fn has_i8mm() -> bool {
        static HAS: OnceLock<bool> = OnceLock::new();
        *HAS.get_or_init(|| {
            std::arch::is_aarch64_feature_detected!("i8mm")
                && std::arch::is_aarch64_feature_detected!("dotprod")
        })
    }

    /// `acc + a · bᵀ` with `a` = two rows of 8 i8 and `b` = two columns of 8: the result is
    /// `[a0·b0, a0·b1, a1·b0, a1·b1]`. (`vmmlaq_s32` is not stable yet.)
    #[target_feature(enable = "neon,i8mm")]
    #[inline]
    unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut r = acc;
        // SAFETY: register-only instruction, available per the target feature.
        unsafe {
            std::arch::asm!(
                "smmla {r:v}.4s, {a:v}.16b, {b:v}.16b",
                r = inout(vreg) r,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack),
            );
        }
        r
    }

    const T_Q8_0: u8 = 0;
    const T_Q4_0: u8 = 1;
    const T_Q5_0: u8 = 2;
    const T_Q4_K: u8 = 3;
    const T_Q5_K: u8 = 4;
    const T_Q6_K: u8 = 5;

    fn mm_type(ty: GgmlType) -> Option<u8> {
        Some(match ty {
            GgmlType::Q8_0 => T_Q8_0,
            GgmlType::Q4_0 => T_Q4_0,
            GgmlType::Q5_0 => T_Q5_0,
            GgmlType::Q4_K => T_Q4_K,
            GgmlType::Q5_K => T_Q5_K,
            GgmlType::Q6_K => T_Q6_K,
            _ => return None,
        })
    }

    pub fn mm_supports(ty: GgmlType) -> bool {
        mm_type(ty).is_some()
    }

    /// One 32-weight chunk of a row as i8 (`lo` = weights 0..16, `hi` = 16..32), with its
    /// float scale `d`, its min coefficient `m` (weight = d·q − m; 0 for symmetric types), and
    /// for Q6_K the i8 scales of its two 16-weight halves.
    struct Chunk {
        lo: int8x16_t,
        hi: int8x16_t,
        d: f32,
        m: f32,
        half_scales: [i32; 2],
    }

    /// Chunk `j` of `row`, encoded as type `T`.
    #[inline(always)]
    unsafe fn chunk<const T: u8>(row: &[u8], j: usize) -> Chunk {
        // SAFETY: the caller passes `j < row.len() / 32 weights`, so every offset is in `row`.
        unsafe {
            let m4 = vdupq_n_u8(0xf);
            let s8 = |v| vreinterpretq_s8_u8(v);
            match T {
                T_Q8_0 => {
                    let b = &row[j * 34..];
                    let p = b.as_ptr().add(2) as *const i8;
                    Chunk {
                        lo: vld1q_s8(p),
                        hi: vld1q_s8(p.add(16)),
                        d: le16(b, 0),
                        m: 0.0,
                        half_scales: [1, 1],
                    }
                }
                T_Q4_0 => {
                    let b = &row[j * 18..];
                    let v = vld1q_u8(b.as_ptr().add(2));
                    let eight = vdupq_n_s8(8);
                    Chunk {
                        lo: vsubq_s8(s8(vandq_u8(v, m4)), eight),
                        hi: vsubq_s8(s8(vshrq_n_u8(v, 4)), eight),
                        d: le16(b, 0),
                        m: 0.0,
                        half_scales: [1, 1],
                    }
                }
                T_Q5_0 => {
                    let b = &row[j * 22..];
                    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                    let v = vld1q_u8(b.as_ptr().add(6));
                    let sixteen = vdupq_n_s8(16);
                    let lo = vorrq_u8(vandq_u8(v, m4), high_bits(qh, 0));
                    let hi = vorrq_u8(vshrq_n_u8(v, 4), high_bits(qh, 2));
                    Chunk {
                        lo: vsubq_s8(s8(lo), sixteen),
                        hi: vsubq_s8(s8(hi), sixteen),
                        d: le16(b, 0),
                        m: 0.0,
                        half_scales: [1, 1],
                    }
                }
                T_Q4_K | T_Q5_K => {
                    let five = T == T_Q5_K;
                    let (bb, ql_at) = if five { (176, 48) } else { (144, 16) };
                    let b = &row[(j / 8) * bb..(j / 8 + 1) * bb];
                    let sub = j % 8;
                    let q = b.as_ptr().add(ql_at + 32 * (sub / 2));
                    let (q0, q1) = (vld1q_u8(q), vld1q_u8(q.add(16)));
                    let (mut w0, mut w1) = if sub.is_multiple_of(2) {
                        (vandq_u8(q0, m4), vandq_u8(q1, m4))
                    } else {
                        (vshrq_n_u8(q0, 4), vshrq_n_u8(q1, 4))
                    };
                    if five {
                        let bit = vdupq_n_u8(1 << sub);
                        let sixteen = vdupq_n_u8(16);
                        let (h0, h1) = (vld1q_u8(b.as_ptr().add(16)), vld1q_u8(b.as_ptr().add(32)));
                        w0 = vorrq_u8(w0, vandq_u8(vtstq_u8(h0, bit), sixteen));
                        w1 = vorrq_u8(w1, vandq_u8(vtstq_u8(h1, bit), sixteen));
                    }
                    let (sc, mn) = scale_min_k4(sub, &b[4..16]);
                    Chunk {
                        lo: s8(w0),
                        hi: s8(w1),
                        d: le16(b, 0) * sc,
                        m: le16(b, 2) * mn,
                        half_scales: [1, 1],
                    }
                }
                _ => {
                    let b = &row[(j / 8) * 210..(j / 8 + 1) * 210];
                    let (n, k) = ((j % 8) / 4, j % 4);
                    let p = b.as_ptr();
                    let ql = p.add(64 * n + 32 * (k % 2));
                    let (l0, l1) = (vld1q_u8(ql), vld1q_u8(ql.add(16)));
                    let (l0, l1) = if k < 2 {
                        (vandq_u8(l0, m4), vandq_u8(l1, m4))
                    } else {
                        (vshrq_n_u8(l0, 4), vshrq_n_u8(l1, 4))
                    };
                    let qh = p.add(128 + 32 * n);
                    let shift = vdupq_n_s8(-(2 * k as i8));
                    let three = vdupq_n_u8(3);
                    let h0 = vshlq_n_u8(vandq_u8(vshlq_u8(vld1q_u8(qh), shift), three), 4);
                    let h1 = vshlq_n_u8(vandq_u8(vshlq_u8(vld1q_u8(qh.add(16)), shift), three), 4);
                    let off = vdupq_n_s8(32);
                    Chunk {
                        lo: vsubq_s8(s8(vorrq_u8(l0, h0)), off),
                        hi: vsubq_s8(s8(vorrq_u8(l1, h1)), off),
                        d: le16(b, 208),
                        m: 0.0,
                        half_scales: [
                            b[192 + 8 * n + 2 * k] as i8 as i32,
                            b[192 + 8 * n + 2 * k + 1] as i8 as i32,
                        ],
                    }
                }
            }
        }
    }

    /// Rows `r0`, `r1` (type `T`) against the token pairs `acts[2p]`, `acts[2p + 1]` for
    /// `p < NP`; `out[p]` = `[r0·t0, r0·t1, r1·t0, r1·t1]`. Every lane does the same
    /// operations in the same order for its row and token, whatever it is paired with.
    #[target_feature(enable = "neon,dotprod,i8mm")]
    unsafe fn rows2<const T: u8, const NP: usize>(
        r0: &[u8],
        r1: &[u8],
        acts: &[&Q8Act; 8],
        out: &mut [[f32; 4]; 4],
    ) {
        let n_chunks = acts[0].qs.len() / QK;
        let mut acc = [vdupq_n_f32(0.0); NP];
        let mut ex = [vdupq_n_f32(0.0); NP];
        let pair = |a: f32, b: f32| vcombine_f32(vdup_n_f32(a), vdup_n_f32(b));
        // SAFETY: `j < n_chunks`, and the caller checked that rows and activations hold
        // `n_chunks` chunks.
        unsafe {
            for j in 0..n_chunks {
                let (c0, c1) = (chunk::<T>(r0, j), chunk::<T>(r1, j));
                let zip = |a: int8x16_t, b: int8x16_t, hi: bool| {
                    let (a, b) = (vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b));
                    vreinterpretq_s8_s64(if hi {
                        vzip2q_s64(a, b)
                    } else {
                        vzip1q_s64(a, b)
                    })
                };
                // Weights k..k+8 of both rows, for k = 0, 8, 16, 24.
                let w = [
                    zip(c0.lo, c1.lo, false),
                    zip(c0.lo, c1.lo, true),
                    zip(c0.hi, c1.hi, false),
                    zip(c0.hi, c1.hi, true),
                ];
                let vd = pair(c0.d, c1.d);
                let vm = pair(c0.m, c1.m);
                let hs = |h: usize| {
                    vcombine_s32(vdup_n_s32(c0.half_scales[h]), vdup_n_s32(c1.half_scales[h]))
                };
                for p in 0..NP {
                    let (a0, a1) = (acts[2 * p], acts[2 * p + 1]);
                    let (x0, x1) = (a0.qs.as_ptr().add(j * QK), a1.qs.as_ptr().add(j * QK));
                    let x = |k: usize| vcombine_s8(vld1_s8(x0.add(8 * k)), vld1_s8(x1.add(8 * k)));
                    let zero = vdupq_n_s32(0);
                    let s = if T == T_Q6_K {
                        let s0 = smmla(smmla(zero, w[0], x(0)), w[1], x(1));
                        let s1 = smmla(smmla(zero, w[2], x(2)), w[3], x(3));
                        vmlaq_s32(vmulq_s32(s0, hs(0)), s1, hs(1))
                    } else {
                        smmla(
                            smmla(smmla(smmla(zero, w[0], x(0)), w[1], x(1)), w[2], x(2)),
                            w[3],
                            x(3),
                        )
                    };
                    let ad = vcombine_f32(
                        vld1_f32([a0.d[j], a1.d[j]].as_ptr()),
                        vld1_f32([a0.d[j], a1.d[j]].as_ptr()),
                    );
                    acc[p] = vfmaq_f32(acc[p], vcvtq_f32_s32(s), vmulq_f32(vd, ad));
                    if T == T_Q4_K || T == T_Q5_K {
                        let su = vld1_f32([a0.sum[j], a1.sum[j]].as_ptr());
                        ex[p] = vfmsq_f32(ex[p], vm, vcombine_f32(su, su));
                    }
                }
            }
            for p in 0..NP {
                vst1q_f32(out[p].as_mut_ptr(), vaddq_f32(acc[p], ex[p]));
            }
        }
    }

    /// [`super::dot_block`]: rows in pairs and tokens in groups of up to 8, an odd last row or
    /// token paired with itself.
    #[target_feature(enable = "neon,dotprod,i8mm")]
    pub unsafe fn dot_block(ty: GgmlType, rows: &[u8], rb: usize, acts: &[Q8Act], out: &mut [f32]) {
        let t = mm_type(ty).expect("checked by mm_supports");
        let (n_rows, nt) = (rows.len() / rb, acts.len());
        assert!(nt > 0 && out.len() == n_rows * nt && rows.len() == n_rows * rb);
        for a in acts {
            assert_eq!(rb / ty.block_bytes() * ty.block_size(), a.qs.len());
        }
        for rp in (0..n_rows).step_by(2) {
            let r0 = &rows[rp * rb..(rp + 1) * rb];
            let two = rp + 1 < n_rows;
            let r1 = if two {
                &rows[(rp + 1) * rb..(rp + 2) * rb]
            } else {
                r0
            };
            for t0 in (0..nt).step_by(8) {
                let k = (nt - t0).min(8);
                let group: [&Q8Act; 8] = std::array::from_fn(|i| &acts[t0 + i.min(k - 1)]);
                let mut res = [[0.0f32; 4]; 4];
                macro_rules! run {
                    ($($t:ident),*) => {
                        match (t, k.div_ceil(2)) {
                            $(
                                ($t, 1) => rows2::<$t, 1>(r0, r1, &group, &mut res),
                                ($t, 2) => rows2::<$t, 2>(r0, r1, &group, &mut res),
                                ($t, 3) => rows2::<$t, 3>(r0, r1, &group, &mut res),
                                ($t, _) => rows2::<$t, 4>(r0, r1, &group, &mut res),
                            )*
                            _ => unreachable!(),
                        }
                    };
                }
                // SAFETY: sizes asserted above; features per this function's contract.
                unsafe { run!(T_Q8_0, T_Q4_0, T_Q5_0, T_Q4_K, T_Q5_K, T_Q6_K) };
                for i in 0..k {
                    let (p, lane) = (i / 2, i % 2);
                    out[rp * nt + t0 + i] = res[p][lane];
                    if two {
                        out[(rp + 1) * nt + t0 + i] = res[p][2 + lane];
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Q8Act, QK, scale_min_k4};
    use crate::quant::{GgmlType, QK_K};
    use crate::simd::x86::hsum;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,f16c")]
    #[inline]
    fn le16(b: &[u8], at: usize) -> f32 {
        let h = u16::from_le_bytes([b[at], b[at + 1]]);
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)))
    }

    /// Eight partial sums of `w · x` over 32 lanes, the low four covering lanes 0..16. `maddubs`
    /// needs one unsigned side, so `w`'s sign moves onto `x`, not the other way round: a Q8_0
    /// weight can be -128, whose absolute value still reads correctly as an unsigned 128, whereas
    /// negating it as the signed side would leave -128. `x` is within ±127, so a lane pair
    /// reaches at most 2 · 128 · 127 = 32512 and the i16 sums can't saturate.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot32(w: __m256i, x: *const i8) -> __m256i {
        // SAFETY: the caller guarantees 32 readable values at `x`.
        let x = unsafe { _mm256_loadu_si256(x.cast()) };
        let pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
        _mm256_madd_epi16(pairs, _mm256_set1_epi16(1))
    }

    /// [`dot32`] for weights known to be non-negative, which skips moving the sign.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot32_unsigned(w: __m256i, x: *const i8) -> __m256i {
        // SAFETY: the caller guarantees 32 readable values at `x`.
        let x = unsafe { _mm256_loadu_si256(x.cast()) };
        _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1))
    }

    /// Each byte's high nibble, in the low four bits of the byte.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn high_nibbles(v: __m256i) -> __m256i {
        _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0xf))
    }

    /// Sixteen bytes as 32 lanes: the low nibbles, then the high nibbles.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn unpack_nibbles(p: *const u8) -> __m256i {
        // SAFETY: the caller guarantees 16 readable bytes at `p`.
        let v = unsafe { _mm_loadu_si128(p.cast()) };
        let m4 = _mm_set1_epi8(0xf);
        let lo = _mm_and_si128(v, m4);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
        _mm256_set_m128i(hi, lo)
    }

    /// Lane `i` holds 0x10 if bit `i` of `qh` is set, else 0.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn high_bits(qh: u32) -> __m256i {
        #[rustfmt::skip]
        let spread = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        );
        let bits = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), spread);
        let set = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bits), bits);
        _mm256_and_si256(set, _mm256_set1_epi8(0x10))
    }

    /// Dot products of one encoded row with the activations of `N` tokens, into `out`, unpacking
    /// each weight block once for all of them. Every token's arithmetic is the same whatever `N`
    /// is, so a matmul matches per-token matvecs bit for bit. `false` for types without a kernel.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_rows<const N: usize>(
        ty: GgmlType,
        row: &[u8],
        acts: &[Q8Act],
        out: &mut [f32],
    ) -> bool {
        let bb = ty.block_bytes();
        let n_blocks = row.len() / bb;
        assert!(acts.len() == N && out.len() == N);
        for a in acts {
            assert_eq!(n_blocks * ty.block_size(), a.qs.len());
        }
        let xq: [*const i8; N] = std::array::from_fn(|t| acts[t].qs.as_ptr());
        let mut acc = [_mm256_setzero_ps(); N];
        // Scalar terms (the K-quant mins), kept apart from the vector accumulators.
        let mut extra = [0f32; N];
        let add = |acc: __m256, s: __m256i, scale: f32| -> __m256 {
            _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(scale), acc)
        };
        // SAFETY (all blocks below): every pointer offset stays inside `row` (n_blocks * bb bytes)
        // or each token's `qs` (n_blocks * block_size values), as asserted above.
        unsafe {
            match ty {
                GgmlType::Q8_0 => {
                    for i in 0..n_blocks {
                        let b = &row[i * bb..];
                        let w = _mm256_loadu_si256(b.as_ptr().add(2).cast());
                        let d = le16(b, 0);
                        for t in 0..N {
                            let s = dot32(w, xq[t].add(i * QK));
                            acc[t] = add(acc[t], s, d * acts[t].d[i]);
                        }
                    }
                }
                GgmlType::Q4_0 => {
                    let eight = _mm256_set1_epi8(8);
                    for i in 0..n_blocks {
                        let b = &row[i * bb..];
                        let w = _mm256_sub_epi8(unpack_nibbles(b.as_ptr().add(2)), eight);
                        let d = le16(b, 0);
                        for t in 0..N {
                            let s = dot32(w, xq[t].add(i * QK));
                            acc[t] = add(acc[t], s, d * acts[t].d[i]);
                        }
                    }
                }
                GgmlType::Q5_0 => {
                    let sixteen = _mm256_set1_epi8(16);
                    for i in 0..n_blocks {
                        let b = &row[i * bb..];
                        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                        let w = _mm256_or_si256(unpack_nibbles(b.as_ptr().add(6)), high_bits(qh));
                        let w = _mm256_sub_epi8(w, sixteen);
                        let d = le16(b, 0);
                        for t in 0..N {
                            let s = dot32(w, xq[t].add(i * QK));
                            acc[t] = add(acc[t], s, d * acts[t].d[i]);
                        }
                    }
                }
                GgmlType::Q4_K | GgmlType::Q5_K => {
                    let five = ty == GgmlType::Q5_K;
                    let ql_at = if five { 48 } else { 16 };
                    let m4 = _mm256_set1_epi8(0xf);
                    let sixteen = _mm256_set1_epi8(16);
                    for i in 0..n_blocks {
                        let b = &row[i * bb..(i + 1) * bb];
                        let (d, dmin) = (le16(b, 0), le16(b, 2));
                        let scales = &b[4..16];
                        let qh = if five {
                            _mm256_loadu_si256(b.as_ptr().add(16).cast())
                        } else {
                            _mm256_setzero_si256()
                        };
                        for g in 0..4 {
                            let q = _mm256_loadu_si256(b.as_ptr().add(ql_at + 32 * g).cast());
                            for half in 0..2 {
                                let sub = 2 * g + half;
                                let mut w = if half == 0 {
                                    _mm256_and_si256(q, m4)
                                } else {
                                    high_nibbles(q)
                                };
                                if five {
                                    let bit = _mm256_set1_epi8(1 << sub);
                                    let set = _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit);
                                    w = _mm256_or_si256(w, _mm256_and_si256(set, sixteen));
                                }
                                let xb = i * (QK_K / QK) + sub;
                                let (sc, m) = scale_min_k4(sub, scales);
                                let (dsc, dm) = (d * sc, dmin * m);
                                for t in 0..N {
                                    let s = dot32_unsigned(w, xq[t].add(xb * QK));
                                    acc[t] = add(acc[t], s, dsc * acts[t].d[xb]);
                                    extra[t] -= dm * acts[t].sum[xb];
                                }
                            }
                        }
                    }
                }
                GgmlType::Q6_K => {
                    let m4 = _mm256_set1_epi8(0xf);
                    let three = _mm256_set1_epi8(3);
                    let off = _mm256_set1_epi8(32);
                    for i in 0..n_blocks {
                        let b = &row[i * bb..(i + 1) * bb];
                        let d = le16(b, 208);
                        let sc = &b[192..208];
                        let p = b.as_ptr();
                        for n in 0..2 {
                            let ql = [
                                _mm256_loadu_si256(p.add(64 * n).cast()),
                                _mm256_loadu_si256(p.add(64 * n + 32).cast()),
                            ];
                            let qh = _mm256_loadu_si256(p.add(128 + 32 * n).cast());
                            for k in 0..4 {
                                let l = ql[k % 2];
                                let lo = if k < 2 {
                                    _mm256_and_si256(l, m4)
                                } else {
                                    high_nibbles(l)
                                };
                                let shift = _mm_cvtsi32_si128(2 * k as i32);
                                let hi = _mm256_and_si256(_mm256_srl_epi16(qh, shift), three);
                                let w = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 4));
                                let w = _mm256_sub_epi8(w, off);
                                let xb = i * (QK_K / QK) + 4 * n + k;
                                // Each half's i8 scale goes on in integer arithmetic (exact)
                                // before the one float multiply; the halves are the two 128-bit
                                // lanes of the partial sums.
                                let scale = _mm256_set_m128i(
                                    _mm_set1_epi32(sc[8 * n + 2 * k + 1] as i8 as i32),
                                    _mm_set1_epi32(sc[8 * n + 2 * k] as i8 as i32),
                                );
                                for t in 0..N {
                                    let s = dot32(w, xq[t].add(xb * QK));
                                    let s = _mm256_mullo_epi32(s, scale);
                                    acc[t] = add(acc[t], s, d * acts[t].d[xb]);
                                }
                            }
                        }
                    }
                }
                _ => return false,
            }
        }
        for t in 0..N {
            out[t] = hsum(acc[t]) + extra[t];
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;
    use crate::quant::{dequantize_row, dot_row};
    use crate::simd::dot_scalar;
    use crate::testutil::Rng;

    fn random_row(ty: GgmlType, n: usize, rng: &mut Rng) -> Vec<u8> {
        let mut row: Vec<u8> = (0..ty.row_bytes(n)).map(|_| rng.next_u32() as u8).collect();
        let offs: &[usize] = match ty {
            GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => &[0],
            GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
            GgmlType::Q6_K => &[208],
            _ => unreachable!(),
        };
        for blk in row.chunks_exact_mut(ty.block_bytes()) {
            for &o in offs {
                blk[o..o + 2].copy_from_slice(&f32_to_f16(rng.f32() * 0.01).to_le_bytes());
            }
        }
        row
    }

    #[test]
    fn multi_token_dots_equal_single_token_dots_exactly() {
        let mut rng = Rng::new(12);
        let n = 512;
        for ty in [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ] {
            let row = random_row(ty, n, &mut rng);
            let acts: Vec<Q8Act> = (0..MAX_TOKENS).map(|_| Q8Act::new(&rng.vec(n))).collect();
            for k in 1..=MAX_TOKENS {
                let mut out = vec![0.0; k];
                dot_rows_q8(ty, &row, &acts[..k], &mut out);
                for (t, &got) in out.iter().enumerate() {
                    assert_eq!(
                        got,
                        dot_row_q8(ty, &row, &acts[t]),
                        "{ty:?} k={k} token {t}"
                    );
                }
            }
        }
    }

    const INT_TYPES: [GgmlType; 8] = [
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ];

    /// Exact dot of a row with the dequantized Q8 activations, and the scale of its terms.
    fn exact_dot(ty: GgmlType, row: &[u8], a: &Q8Act) -> (f32, f32) {
        let n = a.qs.len();
        let xq: Vec<f32> = (0..n).map(|i| a.qs[i] as f32 * a.d[i / QK]).collect();
        let mut w = vec![0.0; n];
        dequantize_row(ty, row, &mut w);
        let scale = w.iter().zip(&xq).map(|(a, b)| (a * b).abs()).sum::<f32>();
        (dot_scalar(&w, &xq), scale)
    }

    #[test]
    fn block_kernel_matches_exact_dots_however_rows_and_tokens_are_grouped() {
        let mut rng = Rng::new(13);
        let n = 512;
        // Odd row and token counts exercise the self-paired last row and token, and 11 tokens
        // span a full group of 8 plus a partial one.
        let (n_rows, n_tok) = (5, 11);
        for ty in INT_TYPES {
            let rows: Vec<u8> = (0..n_rows)
                .flat_map(|_| random_row(ty, n, &mut rng))
                .collect();
            let rb = ty.row_bytes(n);
            let acts: Vec<Q8Act> = (0..n_tok).map(|_| Q8Act::new(&rng.vec(n))).collect();
            let mut out = vec![0.0; n_rows * n_tok];
            if !dot_block(ty, &rows, rb, &acts, &mut out) {
                continue; // no i8mm on this CPU, or no kernel for the type
            }
            for r in 0..n_rows {
                let row = &rows[r * rb..(r + 1) * rb];
                for (t, a) in acts.iter().enumerate() {
                    let got = out[r * n_tok + t];
                    // Regrouped: this row alone (paired with itself) and the token in a pair.
                    let pair = [a.clone(), a.clone()];
                    let mut two = [0.0; 2];
                    assert!(dot_block(ty, row, rb, &pair, &mut two));
                    assert_eq!(got, two[0], "{ty:?} row {r} token {t} regrouped");
                    let (exact, scale) = exact_dot(ty, row, a);
                    assert!(
                        (got - exact).abs() <= 1e-4 * (1.0 + scale),
                        "{ty:?} row {r} token {t}: {got} vs {exact}"
                    );
                }
            }
        }
    }

    /// With unit scales and integer activations every float sum is exact, so any SIMD kernel must
    /// equal the portable one bit for bit. Rounding can't hide a wrong integer here.
    #[test]
    fn simd_kernels_equal_portable_kernels_exactly_when_sums_are_exact() {
        let mut rng = Rng::new(21);
        let n = 256;
        for ty in INT_TYPES {
            for _ in 0..50 {
                let mut row = random_row(ty, n, &mut rng);
                let offs: &[usize] = match ty {
                    GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => &[0],
                    GgmlType::Q6_K => &[208],
                    _ => &[0, 2],
                };
                for blk in row.chunks_exact_mut(ty.block_bytes()) {
                    for &o in offs {
                        blk[o..o + 2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
                    }
                }
                let x: Vec<f32> = (0..n)
                    .map(|i| {
                        if i % QK == 0 {
                            127.0 // sets the activation block's scale to exactly 1
                        } else {
                            (rng.next_u32() % 255) as f32 - 127.0
                        }
                    })
                    .collect();
                let a = Q8Act::new(&x);
                assert_eq!(
                    dot_row_q8(ty, &row, &a),
                    dot_row_q8_portable(ty, &row, &a),
                    "{ty:?}"
                );
            }
        }
    }

    /// The extremes of the widening multiply: -128 weights, which can't be negated in an i8,
    /// against ±127 activations, the largest products there are.
    #[test]
    fn q8_kernel_handles_the_most_negative_weight_without_saturating() {
        let n = 256;
        let ty = GgmlType::Q8_0;
        let mut row = vec![0x80u8; ty.row_bytes(n)];
        for blk in row.chunks_exact_mut(ty.block_bytes()) {
            blk[..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        }
        for (sign, want) in [(127.0, -1.0), (-127.0, 1.0)] {
            let a = Q8Act::new(&vec![sign; n]);
            let expected = want * (128 * 127 * n as i32) as f32;
            assert_eq!(dot_row_q8(ty, &row, &a), expected, "activations {sign}");
            assert_eq!(dot_row_q8_portable(ty, &row, &a), expected);
        }
    }

    #[test]
    fn integer_kernels_match_f32_reference_within_activation_quantization_error() {
        let mut rng = Rng::new(11);
        let n = 512;
        for ty in [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ] {
            for _ in 0..20 {
                let row = random_row(ty, n, &mut rng);
                let x = rng.vec(n);
                let a = Q8Act::new(&x);
                // Exact reference for the quantized activations isolates kernel bugs from rounding.
                let xq: Vec<f32> =
                    a.qs.iter()
                        .enumerate()
                        .map(|(i, &q)| q as f32 * a.d[i / QK])
                        .collect();
                let mut w = vec![0.0; n];
                dequantize_row(ty, &row, &mut w);
                let exact = dot_scalar(&w, &xq);
                let got = dot_row_q8(ty, &row, &a);
                let portable = dot_row_q8_portable(ty, &row, &a);
                let scale = w.iter().zip(&xq).map(|(a, b)| (a * b).abs()).sum::<f32>();
                assert!(
                    (got - exact).abs() <= 1e-4 * (1.0 + scale),
                    "{ty:?}: {got} vs {exact}"
                );
                assert!(
                    (portable - exact).abs() <= 1e-4 * (1.0 + scale),
                    "{ty:?} portable: {portable} vs {exact}"
                );
                // And close to the unquantized-activation result.
                let f = dot_row(ty, &row, &x);
                assert!(
                    (got - f).abs() <= 0.02 * (1.0 + scale),
                    "{ty:?}: {got} vs f32 {f}"
                );
            }
        }
    }
}
