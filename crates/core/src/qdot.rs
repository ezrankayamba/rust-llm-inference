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
    #[cfg(target_arch = "aarch64")]
    if neon::has_dotprod() {
        // SAFETY: dotprod support checked at runtime; row/activation sizes are validated by
        // the caller (QMatrix) and asserted inside.
        if let Some(s) = unsafe { neon::dot_row(ty, row, a) } {
            return s;
        }
    }
    dot_row_q8_portable(ty, row, a)
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

    #[inline(always)]
    unsafe fn dot32(w0: int8x16_t, w1: int8x16_t, x: *const i8) -> i32 {
        unsafe { vaddvq_s32(dot32x4(w0, w1, x)) }
    }

    #[inline(always)]
    unsafe fn dot16(w: int8x16_t, x: *const i8) -> i32 {
        unsafe { vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(x))) }
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

    /// `None` for types without a NEON kernel (the caller falls back to the portable one).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_row(ty: GgmlType, row: &[u8], a: &Q8Act) -> Option<f32> {
        let bb = ty.block_bytes();
        let n_blocks = row.len() / bb;
        assert_eq!(n_blocks * ty.block_size(), a.qs.len());
        let xq = a.qs.as_ptr();
        let m4 = vdupq_n_u8(0xf);
        let mut sum = 0f32;
        // SAFETY (all blocks below): every pointer offset stays inside `row` (n_blocks * bb bytes)
        // or `a.qs` (n_blocks * block_size values), as asserted above.
        unsafe {
            match ty {
                GgmlType::Q8_0 => {
                    let mut acc = vdupq_n_f32(0.0);
                    for i in 0..n_blocks {
                        let w = row.as_ptr().add(i * bb + 2) as *const i8;
                        let s = dot32x4(vld1q_s8(w), vld1q_s8(w.add(16)), xq.add(i * QK));
                        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), le16(&row[i * bb..], 0) * a.d[i]);
                    }
                    sum = vaddvq_f32(acc);
                }
                GgmlType::Q4_0 => {
                    let eight = vdupq_n_s8(8);
                    let mut acc = vdupq_n_f32(0.0);
                    for i in 0..n_blocks {
                        let v = vld1q_u8(row.as_ptr().add(i * bb + 2));
                        let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), eight);
                        let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
                        let s = dot32x4(lo, hi, xq.add(i * QK));
                        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), le16(&row[i * bb..], 0) * a.d[i]);
                    }
                    sum = vaddvq_f32(acc);
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
                        sum += le16(b, 0) * a.d[i] * dot32(lo, hi, xq.add(i * QK)) as f32;
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
                                let xb = i * (QK_K / QK) + sub;
                                let s = dot32(
                                    vreinterpretq_s8_u8(w0),
                                    vreinterpretq_s8_u8(w1),
                                    xq.add(xb * QK),
                                );
                                let (sc, m) = scale_min_k4(sub, scales);
                                sum += d * sc * a.d[xb] * s as f32 - dmin * m * a.sum[xb];
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
                                let x = xq.add(xb * QK);
                                let s0 = sc[8 * n + 2 * k] as i8 as f32;
                                let s1 = sc[8 * n + 2 * k + 1] as i8 as f32;
                                let s = s0 * dot16(w0, x) as f32 + s1 * dot16(w1, x.add(16)) as f32;
                                sum += d * a.d[xb] * s;
                            }
                        }
                    }
                }
                _ => return None,
            }
        }
        Some(sum)
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
