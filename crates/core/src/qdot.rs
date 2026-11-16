//! Integer dot products of quantized weight rows with 8-bit-quantized activations.
//!
//! The activation vector is quantized once per matvec into 32-element blocks (scale `d`, i8
//! values, and `d * Σq` for the min/offset terms). Each weight format then unpacks 32 (or 16)
//! weights at a time to i8 and accumulates exact integer products, applying scales per
//! sub-block. This is the fast path; `quant::dot_row` (f32) is its reference.

use crate::half::f16_to_f32;
use crate::quant::{GgmlType, QK_K};

const QK: usize = 32;

/// Activation blocks per group that shares a base scale: a K-quant super-block's worth.
const GROUP: usize = QK_K / QK;

/// A block's scale is its group's largest block scale times a whole number of this many parts,
/// so a kernel can put the block's share on as an integer and leave one float multiply per
/// group. A block much quieter than its group's loudest gets a slightly coarser scale than its
/// own (at most 2% coarser at a fifth of the loudest, up to 2x below 1/256 of it), where its
/// error is far below the loud block's anyway. A power of two keeps the group's part exact.
const RATIO_MAX: f32 = 256.0;

/// Activations quantized to Q8 in blocks of 32.
#[derive(Clone)]
pub struct Q8Act {
    d: Vec<f32>,
    /// Each block's scale in parts of its group's: `d = unit * ratio`, 1 to [`RATIO_MAX`].
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code)
    )] // only the AVX2 and NEON kernels read it
    ratio: Vec<i16>,
    /// One per group of [`GROUP`] blocks: its largest block scale over [`RATIO_MAX`].
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code)
    )]
    unit: Vec<f32>,
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
            ratio: Vec::with_capacity(n),
            unit: Vec::with_capacity(n.div_ceil(GROUP)),
            sum: Vec::with_capacity(n),
            qs: vec![0; x.len()],
        };
        let groups = x.chunks(QK * GROUP).zip(a.qs.chunks_mut(QK * GROUP));
        for (xg, qg) in groups {
            // Each block's own scale, found once: the group's base is the largest of them.
            let blocks = xg.as_chunks::<QK>().0;
            let mut exact = [0f32; GROUP];
            for (e, blk) in exact.iter_mut().zip(blocks) {
                *e = blk.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            }
            let base = exact.iter().fold(0f32, |m, &e| m.max(e));
            a.unit.push(base / RATIO_MAX);
            for ((blk, qs), exact) in blocks.iter().zip(qg.as_chunks_mut::<QK>().0).zip(exact) {
                // The smallest ratio whose scale still covers the block's largest value.
                let mut r = if base != 0.0 {
                    (exact / base * RATIO_MAX).ceil().clamp(1.0, RATIO_MAX)
                } else {
                    1.0
                };
                if base * (r / RATIO_MAX) < exact && r < RATIO_MAX {
                    r += 1.0;
                }
                let d = base * (r / RATIO_MAX);
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let mut s = 0i32;
                for (q, &v) in qs.iter_mut().zip(blk) {
                    *q = (v * id).round() as i8;
                    s += *q as i32;
                }
                a.d.push(d);
                a.ratio.push(r as i16);
                a.sum.push(d * s as f32);
            }
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

/// All eight 6-bit scales and mins of a Q4_K/Q5_K super-block from its 12 packed bytes, the
/// same values [`scale_min_k4`] gives one at a time, as a little-endian u64 of bytes each.
#[inline]
fn scales_mins_k4(q: &[u8]) -> (u64, u64) {
    let u = |i: usize| u32::from_le_bytes([q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]]);
    let (u0, u1, u2) = (u(0), u(1), u(2));
    let (low6, low4, top2) = (0x3f3f_3f3f, 0x0f0f_0f0f, 0x0303_0303);
    let sc = [u0 & low6, (u2 & low4) | (((u0 >> 6) & top2) << 4)];
    let mn = [u1 & low6, ((u2 >> 4) & low4) | (((u1 >> 6) & top2) << 4)];
    let pack = |v: [u32; 2]| v[0] as u64 | (v[1] as u64) << 32;
    (pack(sc), pack(mn))
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
        let (rb, pf) = (row.len(), crate::simd::x86::prefetch_rows());
        // SAFETY: AVX2, FMA and F16C checked at runtime; sizes are asserted inside.
        let done = unsafe {
            match acts.len() {
                1 => avx2::dot_rows::<1>(ty, row, rb, acts, out, pf),
                2 => avx2::dot_rows::<2>(ty, row, rb, acts, out, pf),
                3 => avx2::dot_rows::<3>(ty, row, rb, acts, out, pf),
                4 => avx2::dot_rows::<4>(ty, row, rb, acts, out, pf),
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

#[cfg(target_arch = "x86_64")]
pub use avx2::BlockActs;
#[cfg(target_arch = "aarch64")]
pub use neon::BlockActs;

/// Fewest tokens a matmul hands to [`dot_block`]. The x86 kernel decodes its weights into a
/// scratch panel, which only pays off once enough tokens share it.
pub const BLOCK_MIN_TOKENS: usize = if cfg!(target_arch = "x86_64") { 8 } else { 2 };

/// No block kernel on this architecture: [`BlockActs::new`] always says no.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub struct BlockActs;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
impl BlockActs {
    pub fn new(_: GgmlType, _: &[Q8Act]) -> Option<Self> {
        None
    }
}

/// Dot products of a block of encoded rows (`rows`, `row_bytes` each) with the tokens in `acts`,
/// into `out` as `[row][token]`: on ARM with the i8mm matrix instructions (2 rows x 2 tokens
/// each), on x86 over 8 rows at once decoded into an interleaved panel. A result does not depend
/// on how rows and tokens are grouped, but differs from [`dot_row_q8`] by float rounding: decode
/// keeps that kernel, which streams one row at a time faster.
pub fn dot_block(ty: GgmlType, rows: &[u8], row_bytes: usize, acts: &BlockActs, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `BlockActs::new` checked for i8mm (and dotprod, which every i8mm CPU has); sizes
    // are asserted inside.
    unsafe {
        neon::dot_block(ty, rows, row_bytes, acts, out)
    };
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `BlockActs::new` checked for AVX2, FMA and F16C; sizes are asserted inside.
    unsafe {
        avx2::dot_block(ty, rows, row_bytes, acts, out)
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (ty, rows, row_bytes, acts, out);
        unreachable!("BlockActs::new is always None here");
    }
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
    use super::{Q8Act, QK, le16, scales_mins_k4};
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

    /// A Q5_0 block's 32 weights as i8, -16 to 15, in two halves. The four bytes of high bits
    /// are loaded into every lane and spread by a table, staying in vector registers: moving
    /// them over from general ones made this the slowest of the decoders.
    #[inline(always)]
    unsafe fn q5_0(b: &[u8]) -> (int8x16_t, int8x16_t) {
        const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        const BYTES: [u8; 32] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        ];
        assert!(b.len() >= 22);
        // SAFETY: the block holds 22 bytes, as asserted.
        unsafe {
            let qh = vreinterpretq_u8_u32(vld1q_dup_u32(b.as_ptr().add(2) as *const u32));
            let v = vld1q_u8(b.as_ptr().add(6));
            let (bit, f0) = (vld1q_u8(BIT.as_ptr()), vdupq_n_u8(0xf0));
            // 0xf0 where the high bit is clear: or-ed with the nibble, that is the weight
            // less 16.
            let high = |half: usize| {
                let bytes = vqtbl1q_u8(qh, vld1q_u8(BYTES.as_ptr().add(16 * half)));
                vbicq_u8(f0, vtstq_u8(bytes, bit))
            };
            (
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v, vdupq_n_u8(0xf)), high(0))),
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(v, 4), high(1))),
            )
        }
    }

    /// [`dot_rows`] for Q4_K (`FIVE` false) and Q5_K, as the AVX2 kernel does it. A super-block
    /// stays in integers: each sub-block's 6-bit scale times its activations' ratio multiplies
    /// that sub-block's dot product, the eight add up as i32, and one float multiply-add per
    /// super-block puts on the weights' scale and the activation group's. The i32 sums can't
    /// overflow: a lane of a Q4_K sub-block's dot product reaches 8 · 15 · 127, a multiplier
    /// 63 · 256, and eight of those are 1.97e9. Q5_K's weights reach 31, so its two 16-weight
    /// halves keep a sum each (4 · 31 · 127 a lane, 2.03e9 in all).
    #[target_feature(enable = "neon,dotprod")]
    #[inline]
    unsafe fn k_rows<const N: usize, const FIVE: bool>(
        row: &[u8],
        acts: &[Q8Act],
        acc: &mut [[float32x4_t; 4]; N],
        extra: &mut [[float32x4_t; 2]; N],
    ) {
        let (bb, ql_at) = if FIVE { (176, 48) } else { (144, 16) };
        // SAFETY: per `dot_rows`, whose block count bounds every token's activations.
        unsafe {
            let m4 = vdupq_n_u8(0xf);
            let sixteen = vdupq_n_u8(16);
            let zero = vdupq_n_s32(0);
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let (d, dmin) = (le16(b, 0), le16(b, 2));
                let (sc, mn) = scales_mins_k4(&b[4..16]);
                let sc = vreinterpretq_s16_u16(vmovl_u8(vcreate_u8(sc)));
                let mn = vmovl_u8(vcreate_u8(mn));
                let mins = [
                    vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(mn))), dmin),
                    vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(mn)), dmin),
                ];
                let xb = i * (QK_K / QK);
                // Each token's eight multipliers, and its two sums.
                let mut mult = [[0i32; 8]; N];
                let mut sum = [[zero; 2]; N];
                for t in 0..N {
                    let xs = acts[t].sum.as_ptr().add(xb);
                    extra[t][0] = vfmsq_f32(extra[t][0], mins[0], vld1q_f32(xs));
                    extra[t][1] = vfmsq_f32(extra[t][1], mins[1], vld1q_f32(xs.add(4)));
                    let m = vmulq_s16(sc, vld1q_s16(acts[t].ratio.as_ptr().add(xb)));
                    vst1q_s32(mult[t].as_mut_ptr(), vmovl_s16(vget_low_s16(m)));
                    vst1q_s32(mult[t].as_mut_ptr().add(4), vmovl_high_s16(m));
                }
                let (qh0, qh1) = if FIVE {
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
                        if FIVE {
                            let bit = vdupq_n_u8(1 << sub);
                            w0 = vorrq_u8(w0, vandq_u8(vtstq_u8(qh0, bit), sixteen));
                            w1 = vorrq_u8(w1, vandq_u8(vtstq_u8(qh1, bit), sixteen));
                        }
                        let (w0, w1) = (vreinterpretq_s8_u8(w0), vreinterpretq_s8_u8(w1));
                        for t in 0..N {
                            let x = acts[t].qs.as_ptr().add((xb + sub) * QK);
                            let m = vld1q_dup_s32(mult[t].as_ptr().add(sub));
                            if FIVE {
                                let p0 = vdotq_s32(zero, w0, vld1q_s8(x));
                                let p1 = vdotq_s32(zero, w1, vld1q_s8(x.add(16)));
                                sum[t][0] = vaddq_s32(sum[t][0], vmulq_s32(p0, m));
                                sum[t][1] = vaddq_s32(sum[t][1], vmulq_s32(p1, m));
                            } else {
                                // Sub-blocks in turn into the two sums, so neither waits on
                                // the one before.
                                let p = dot32x4(w0, w1, x);
                                sum[t][half] = vaddq_s32(sum[t][half], vmulq_s32(p, m));
                            }
                        }
                    }
                }
                for t in 0..N {
                    let scale = vdupq_n_f32(d * acts[t].unit[i]);
                    if FIVE {
                        acc[t][0] = vfmaq_f32(acc[t][0], vcvtq_f32_s32(sum[t][0]), scale);
                        acc[t][1] = vfmaq_f32(acc[t][1], vcvtq_f32_s32(sum[t][1]), scale);
                    } else {
                        let s = vaddq_s32(sum[t][0], sum[t][1]);
                        acc[t][0] = vfmaq_f32(acc[t][0], vcvtq_f32_s32(s), scale);
                    }
                }
            }
        }
    }

    /// Dot products of one encoded row with the activations of `N` tokens, into `out`. Each
    /// weight block is unpacked (and its scales decoded) once for all `N` tokens, and every
    /// token's arithmetic is the same whatever `N` is, so a matmul matches per-token matvecs
    /// bit for bit. `false` for types without a NEON kernel. The 32-weight types use only the
    /// first of a token's sums: dealing their blocks over four (indexed by block) ran 5-12%
    /// slower.
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
        let zero = vdupq_n_f32(0.0);
        let mut acc = [[zero; 4]; N];
        // The K-quant mins' terms.
        let mut extra = [[zero; 2]; N];
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
                            let a = &mut acc[t][0];
                            *a = vmlaq_n_f32(*a, vcvtq_f32_s32(s), d * ad[t][i]);
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
                            let a = &mut acc[t][0];
                            *a = vmlaq_n_f32(*a, vcvtq_f32_s32(s), d * ad[t][i]);
                        }
                    }
                }
                GgmlType::Q5_0 => {
                    for i in 0..n_blocks {
                        let b = &row[i * bb..];
                        let (lo, hi) = q5_0(b);
                        let d = le16(b, 0);
                        for t in 0..N {
                            let s = dot32x4(lo, hi, xq[t].add(i * QK));
                            let a = &mut acc[t][0];
                            *a = vmlaq_n_f32(*a, vcvtq_f32_s32(s), d * ad[t][i]);
                        }
                    }
                }
                GgmlType::Q4_K => k_rows::<N, false>(row, acts, &mut acc, &mut extra),
                GgmlType::Q5_K => k_rows::<N, true>(row, acts, &mut acc, &mut extra),
                GgmlType::Q6_K => {
                    let off = vdupq_n_s8(32);
                    let (m6, m2) = (vdupq_n_u8(0x3f), vdupq_n_u8(0x30));
                    // Chunk `k`'s weights from 16 bytes of low nibbles and of high bit pairs,
                    // in two or three instructions: the shifts insert (`sli`) where they can.
                    let w6 = |k: usize, l: uint8x16_t, h: uint8x16_t| {
                        let w = match k {
                            0 => vandq_u8(vsliq_n_u8::<4>(l, h), m6),
                            1 => vandq_u8(vsliq_n_u8::<4>(l, vshrq_n_u8(h, 2)), m6),
                            2 => vorrq_u8(vshrq_n_u8(l, 4), vandq_u8(h, m2)),
                            _ => vsliq_n_u8::<4>(vshrq_n_u8(l, 4), vshrq_n_u8(h, 6)),
                        };
                        vsubq_s8(vreinterpretq_s8_u8(w), off)
                    };
                    for i in 0..n_blocks {
                        let b = &row[i * bb..(i + 1) * bb];
                        let d = le16(b, 208);
                        // The 16 scales as i32, to load straight into every lane.
                        let mut sc = [0i32; 16];
                        let s8 = vld1q_s8(b.as_ptr().add(192) as *const i8);
                        let (lo, hi) = (vmovl_s8(vget_low_s8(s8)), vmovl_high_s8(s8));
                        vst1q_s32(sc.as_mut_ptr(), vmovl_s16(vget_low_s16(lo)));
                        vst1q_s32(sc.as_mut_ptr().add(4), vmovl_high_s16(lo));
                        vst1q_s32(sc.as_mut_ptr().add(8), vmovl_s16(vget_low_s16(hi)));
                        vst1q_s32(sc.as_mut_ptr().add(12), vmovl_high_s16(hi));
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
                                let (w0, w1) = (w6(k, l0, qh[0]), w6(k, l1, qh[1]));
                                let xb = i * (QK_K / QK) + 4 * n + k;
                                // The two 16-element halves have their own i8 scales; apply them
                                // in integer arithmetic (exact) before the one float multiply.
                                let s0 = vld1q_dup_s32(sc.as_ptr().add(8 * n + 2 * k));
                                let s1 = vld1q_dup_s32(sc.as_ptr().add(8 * n + 2 * k + 1));
                                for t in 0..N {
                                    let x = xq[t].add(xb * QK);
                                    let p0 = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(x));
                                    let p1 = vdotq_s32(vdupq_n_s32(0), w1, vld1q_s8(x.add(16)));
                                    let s = vmlaq_s32(vmulq_s32(p0, s0), p1, s1);
                                    // Two sums in turn, so a chunk doesn't wait on the last.
                                    let a = &mut acc[t][k % 2];
                                    *a = vfmaq_n_f32(*a, vcvtq_f32_s32(s), d * ad[t][xb]);
                                }
                            }
                        }
                    }
                }
                _ => return false,
            }
        }
        for t in 0..N {
            let a = vaddq_f32(
                vaddq_f32(acc[t][0], acc[t][1]),
                vaddq_f32(acc[t][2], acc[t][3]),
            );
            out[t] = vaddvq_f32(a) + vaddvq_f32(vaddq_f32(extra[t][0], extra[t][1]));
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

    /// Two or more tokens' activations laid out for the matrix instructions: tokens in pairs
    /// (an odd last one paired with itself), so each instruction's inputs are one 16-byte load.
    pub struct BlockActs {
        n_tokens: usize,
        n_chunks: usize,
        /// `[pair][chunk][k < 4]`: 8 values of the pair's first token, then 8 of its second.
        qs: Vec<i8>,
        /// `[pair][chunk]`: `[t0, t1, t0, t1]`, the lanes of an `smmla` result.
        d: Vec<[f32; 4]>,
        sum: Vec<[f32; 4]>,
    }

    impl BlockActs {
        /// `None` for a single token, when the CPU lacks i8mm, or when `ty` has no kernel.
        pub fn new(ty: GgmlType, acts: &[Q8Act]) -> Option<Self> {
            if acts.len() < 2 || !has_i8mm() || mm_type(ty).is_none() {
                return None;
            }
            let n_chunks = acts[0].qs.len() / QK;
            let n_pairs = acts.len().div_ceil(2);
            let mut b = BlockActs {
                n_tokens: acts.len(),
                n_chunks,
                qs: Vec::with_capacity(n_pairs * n_chunks * 2 * QK),
                d: Vec::with_capacity(n_pairs * n_chunks),
                sum: Vec::with_capacity(n_pairs * n_chunks),
            };
            for p in acts.chunks(2) {
                let (a0, a1) = (&p[0], &p[p.len() - 1]);
                assert_eq!(a0.qs.len(), a1.qs.len());
                for (q0, q1) in a0
                    .qs
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(a1.qs.as_chunks::<8>().0)
                {
                    b.qs.extend_from_slice(q0);
                    b.qs.extend_from_slice(q1);
                }
                for j in 0..n_chunks {
                    b.d.push([a0.d[j], a1.d[j], a0.d[j], a1.d[j]]);
                    b.sum.push([a0.sum[j], a1.sum[j], a0.sum[j], a1.sum[j]]);
                }
            }
            Some(b)
        }
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
                    let (lo, hi) = q5_0(b);
                    Chunk {
                        lo,
                        hi,
                        d: le16(b, 0),
                        m: 0.0,
                        half_scales: [1, 1],
                    }
                }
                T_Q4_K | T_Q5_K => {
                    let five = T == T_Q5_K;
                    let (bb, ql_at) = if five { (176, 48) } else { (144, 16) };
                    let b = row.as_ptr().add((j / 8) * bb);
                    let sub = j % 8;
                    let q = b.add(ql_at + 32 * (sub / 2));
                    let (q0, q1) = (vld1q_u8(q), vld1q_u8(q.add(16)));
                    let (mut w0, mut w1) = if sub.is_multiple_of(2) {
                        (vandq_u8(q0, m4), vandq_u8(q1, m4))
                    } else {
                        (vshrq_n_u8(q0, 4), vshrq_n_u8(q1, 4))
                    };
                    if five {
                        let bit = vdupq_n_u8(1 << sub);
                        let sixteen = vdupq_n_u8(16);
                        let (h0, h1) = (vld1q_u8(b.add(16)), vld1q_u8(b.add(32)));
                        w0 = vorrq_u8(w0, vandq_u8(vtstq_u8(h0, bit), sixteen));
                        w1 = vorrq_u8(w1, vandq_u8(vtstq_u8(h1, bit), sixteen));
                    }
                    // `rows2` decodes these once per block: see `k_scales`.
                    Chunk {
                        lo: s8(w0),
                        hi: s8(w1),
                        d: 0.0,
                        m: 0.0,
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

    /// For the Q4_K/Q5_K block holding chunk `j` of both rows, each chunk's scale and min as
    /// `[r0, r0, r1, r1]`, the lanes of an `smmla` result.
    #[target_feature(enable = "neon")]
    #[inline]
    fn k_scales(
        block_bytes: usize,
        r0: &[u8],
        r1: &[u8],
        j: usize,
    ) -> ([float32x4_t; 8], [float32x4_t; 8]) {
        let at = (j / 8) * block_bytes;
        // A row's eight scales and mins as floats, times its d and dmin.
        let row = |r: &[u8]| {
            let b = &r[at..at + block_bytes];
            let (sc, mn) = scales_mins_k4(&b[4..16]);
            let f = |v: u64, by: f32| {
                let v = vmovl_u8(vcreate_u8(v));
                [
                    vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v))), by),
                    vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(v)), by),
                ]
            };
            (f(sc, le16(b, 0)), f(mn, le16(b, 2)))
        };
        let ((d0, m0), (d1, m1)) = (row(r0), row(r1));
        // [a0, a1, a2, a3] and [b0, ..] to [a0, a0, b0, b0], [a1, a1, b1, b1], ...
        let spread = |a: [float32x4_t; 2], b: [float32x4_t; 2]| {
            let mut out = [vdupq_n_f32(0.0); 8];
            for h in 0..2 {
                let (al, ah) = (vzip1q_f32(a[h], a[h]), vzip2q_f32(a[h], a[h]));
                let (bl, bh) = (vzip1q_f32(b[h], b[h]), vzip2q_f32(b[h], b[h]));
                out[4 * h] = vcombine_f32(vget_low_f32(al), vget_low_f32(bl));
                out[4 * h + 1] = vcombine_f32(vget_high_f32(al), vget_high_f32(bl));
                out[4 * h + 2] = vcombine_f32(vget_low_f32(ah), vget_low_f32(bh));
                out[4 * h + 3] = vcombine_f32(vget_high_f32(ah), vget_high_f32(bh));
            }
            out
        };
        (spread(d0, d1), spread(m0, m1))
    }

    /// Rows `r0`, `r1` (type `T`) against token pairs `first..first + NP`; `out[p]` =
    /// `[r0·t0, r0·t1, r1·t0, r1·t1]`. Every lane does the same operations in the same order
    /// for its row and token, whatever it is paired with.
    #[target_feature(enable = "neon,dotprod,i8mm")]
    unsafe fn rows2<const T: u8, const NP: usize>(
        r0: &[u8],
        r1: &[u8],
        acts: &BlockActs,
        first: usize,
        out: &mut [[f32; 4]; 8],
    ) {
        let n_chunks = acts.n_chunks;
        let mut acc = [vdupq_n_f32(0.0); NP];
        let mut ex = [vdupq_n_f32(0.0); NP];
        let pair = |a: f32, b: f32| vcombine_f32(vdup_n_f32(a), vdup_n_f32(b));
        // SAFETY: `first + NP` pairs and `j < n_chunks` are inside `acts`, per the caller, whose
        // rows also hold `n_chunks` chunks.
        unsafe {
            let qs = acts.qs.as_ptr().add(first * n_chunks * 2 * QK);
            let ds = acts.d.as_ptr().add(first * n_chunks) as *const f32;
            let sums = acts.sum.as_ptr().add(first * n_chunks) as *const f32;
            let is_k = T == T_Q4_K || T == T_Q5_K;
            let block_bytes = if T == T_Q5_K { 176 } else { 144 };
            let (mut k_d, mut k_m) = ([vdupq_n_f32(0.0); 8], [vdupq_n_f32(0.0); 8]);
            for j in 0..n_chunks {
                if is_k && j.is_multiple_of(8) {
                    (k_d, k_m) = k_scales(block_bytes, r0, r1, j);
                }
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
                let (vd, vm) = if is_k {
                    (k_d[j % 8], k_m[j % 8])
                } else {
                    (pair(c0.d, c1.d), pair(c0.m, c1.m))
                };
                let hs = |h: usize| {
                    vcombine_s32(vdup_n_s32(c0.half_scales[h]), vdup_n_s32(c1.half_scales[h]))
                };
                for p in 0..NP {
                    let at = p * n_chunks + j;
                    let x = qs.add(at * 2 * QK);
                    let x = |k: usize| vld1q_s8(x.add(16 * k));
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
                    let ad = vld1q_f32(ds.add(at * 4));
                    acc[p] = vfmaq_f32(acc[p], vcvtq_f32_s32(s), vmulq_f32(vd, ad));
                    if is_k {
                        ex[p] = vfmsq_f32(ex[p], vm, vld1q_f32(sums.add(at * 4)));
                    }
                }
            }
            for p in 0..NP {
                vst1q_f32(out[p].as_mut_ptr(), vaddq_f32(acc[p], ex[p]));
            }
        }
    }

    /// [`super::dot_block`]: rows in pairs and tokens in groups of up to 16, an odd last row
    /// paired with itself. A row pair's weights are decoded once per group, and 16 tokens of
    /// activations still fit L1 beside them; groups of 8 ran Q6_K 28% slower.
    #[target_feature(enable = "neon,dotprod,i8mm")]
    pub unsafe fn dot_block(
        ty: GgmlType,
        rows: &[u8],
        rb: usize,
        acts: &BlockActs,
        out: &mut [f32],
    ) {
        let t = mm_type(ty).expect("checked by BlockActs::new");
        let (n_rows, nt) = (rows.len() / rb, acts.n_tokens);
        assert!(out.len() == n_rows * nt && rows.len() == n_rows * rb);
        assert_eq!(rb / ty.block_bytes() * ty.block_size(), acts.n_chunks * QK);
        // Tokens outside rows: a group's activations stay in L1 while it meets every row, where
        // the other way round each row would cycle through all the tokens' activations.
        for t0 in (0..nt).step_by(16) {
            let k = (nt - t0).min(16);
            for rp in (0..n_rows).step_by(2) {
                let r0 = &rows[rp * rb..(rp + 1) * rb];
                let two = rp + 1 < n_rows;
                let r1 = if two {
                    &rows[(rp + 1) * rb..(rp + 2) * rb]
                } else {
                    r0
                };
                let mut res = [[0.0f32; 4]; 8];
                macro_rules! run {
                    ($($t:ident),*) => {
                        match (t, k.div_ceil(2)) {
                            $(
                                ($t, 1) => rows2::<$t, 1>(r0, r1, acts, t0 / 2, &mut res),
                                ($t, 2) => rows2::<$t, 2>(r0, r1, acts, t0 / 2, &mut res),
                                ($t, 3) => rows2::<$t, 3>(r0, r1, acts, t0 / 2, &mut res),
                                ($t, 4) => rows2::<$t, 4>(r0, r1, acts, t0 / 2, &mut res),
                                ($t, 5) => rows2::<$t, 5>(r0, r1, acts, t0 / 2, &mut res),
                                ($t, 6) => rows2::<$t, 6>(r0, r1, acts, t0 / 2, &mut res),
                                ($t, 7) => rows2::<$t, 7>(r0, r1, acts, t0 / 2, &mut res),
                                ($t, _) => rows2::<$t, 8>(r0, r1, acts, t0 / 2, &mut res),
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

/// [`dot_row_q8`] for consecutive rows of `rb` bytes, one result per row. On x86 this spares each
/// row the kernel's setup, which [`dot_rows_q8`] repeats per call.
pub fn matvec_q8(ty: GgmlType, rows: &[u8], rb: usize, act: &Q8Act, out: &mut [f32]) {
    assert!(rb > 0 && rows.len() == rb * out.len());
    #[cfg(target_arch = "x86_64")]
    if crate::simd::x86::has_avx2() {
        let (acts, pf) = (std::slice::from_ref(act), crate::simd::x86::prefetch_rows());
        // SAFETY: AVX2, FMA and F16C checked at runtime; sizes are asserted inside.
        if unsafe { avx2::dot_rows::<1>(ty, rows, rb, acts, out, pf) } {
            return;
        }
    }
    for (o, row) in out.iter_mut().zip(rows.chunks_exact(rb)) {
        *o = dot_row_q8(ty, row, act);
    }
}

/// Rows per group of a [`pack_rows`] copy.
pub const PACK_ROWS: usize = 8;

/// Bytes of one super-block of [`PACK_ROWS`] packed Q4_K rows: their f16 scales (16 bytes) and
/// min scales (16), each sub-block's eight 6-bit scales as bytes (64) and mins (64), then the
/// 4-bit weights (1024): for each pair of sub-blocks, the rows' 4 bytes for weights 0..4, then
/// for 4..8 and so on, the pair's first sub-block in the low nibbles as in the file.
const PACK_BLOCK: usize = 160 + PACK_ROWS * (QK_K / 2);
const PACK_SCALES: usize = 32;
const PACK_MINS: usize = 96;
const PACK_QS: usize = 160;

/// Whether a matrix of `ty` gains from a [`pack_rows`] copy on this CPU: Q4_K with AVX2. The
/// portable [`matvec_packed`] is slower than the row kernels.
pub fn packs(ty: GgmlType) -> bool {
    #[cfg(target_arch = "x86_64")]
    return ty == GgmlType::Q4_K && crate::simd::x86::has_avx2();
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ty;
        false
    }
}

/// Bytes of a [`pack_rows`] copy of [`PACK_ROWS`] rows of `cols` weights.
pub fn pack_bytes(cols: usize) -> usize {
    cols / QK_K * PACK_BLOCK
}

/// A copy of Q4_K `rows` (`rb` bytes each) with every [`PACK_ROWS`] of them interleaved, so that
/// [`matvec_packed`] loads the same four weights of all of them at once and multiplies them by
/// four activations it loaded once. Rows left over after the last full group aren't copied. It
/// takes 2.8% more memory than the rows it copies, on top of them.
pub fn pack_rows(rows: &[u8], rb: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let bb = GgmlType::Q4_K.block_bytes();
    assert!(rb > 0 && rb.is_multiple_of(bb) && rows.len().is_multiple_of(rb));
    let gb = rb / bb * PACK_BLOCK;
    let mut out = vec![0u8; rows.len() / rb / PACK_ROWS * gb];
    out.par_chunks_mut(gb)
        .zip(rows.par_chunks(rb * PACK_ROWS))
        .for_each(|(out, group)| {
            for (i, o) in out.as_chunks_mut::<PACK_BLOCK>().0.iter_mut().enumerate() {
                for r in 0..PACK_ROWS {
                    let b = &group[r * rb + i * bb..][..bb];
                    o[2 * r..2 * r + 2].copy_from_slice(&b[0..2]);
                    o[16 + 2 * r..16 + 2 * r + 2].copy_from_slice(&b[2..4]);
                    let (sc, mn) = scales_mins_k4(&b[4..16]);
                    for sub in 0..GROUP {
                        o[PACK_SCALES + PACK_ROWS * sub + r] = (sc >> (8 * sub)) as u8;
                        o[PACK_MINS + PACK_ROWS * sub + r] = (mn >> (8 * sub)) as u8;
                    }
                    for (c, q) in b[16..].as_chunks::<4>().0.iter().enumerate() {
                        o[PACK_QS + 4 * (PACK_ROWS * c + r)..][..4].copy_from_slice(q);
                    }
                }
            }
        });
    out
}

/// [`matvec_q8`] over a [`pack_rows`] copy of rows of `cols` weights: one result per packed row.
/// A result differs from the row kernel's by float rounding, as the sums are grouped differently.
pub fn matvec_packed(packed: &[u8], cols: usize, act: &Q8Act, out: &mut [f32]) {
    let gb = pack_bytes(cols);
    assert!(
        cols.is_multiple_of(QK_K) && act.qs.len() == cols && out.len().is_multiple_of(PACK_ROWS)
    );
    assert!(packed.len() == gb * (out.len() / PACK_ROWS));
    #[cfg(target_arch = "x86_64")]
    if crate::simd::x86::has_avx2() {
        let pf = crate::simd::x86::prefetch_rows();
        // SAFETY: AVX2, FMA and F16C checked at runtime; sizes are asserted above and inside.
        return unsafe { avx2::packed_k4(packed, gb, act, out, pf) };
    }
    for (group, out) in packed
        .chunks_exact(gb.max(1))
        .zip(out.as_chunks_mut::<PACK_ROWS>().0)
    {
        *out = packed_portable(group, act);
    }
}

fn packed_portable(group: &[u8], a: &Q8Act) -> [f32; PACK_ROWS] {
    let mut out = [0.0; PACK_ROWS];
    for (i, b) in group.as_chunks::<PACK_BLOCK>().0.iter().enumerate() {
        for (r, out) in out.iter_mut().enumerate() {
            let (d, dmin) = (le16(b, 2 * r), le16(b, 16 + 2 * r));
            for sub in 0..GROUP {
                let blk = i * GROUP + sub;
                let x = &a.qs[blk * QK..(blk + 1) * QK];
                let mut s = 0i32;
                for (k, x) in x.as_chunks::<4>().0.iter().enumerate() {
                    let q = &b[PACK_QS + 4 * (PACK_ROWS * (8 * (sub / 2) + k) + r)..][..4];
                    for (q, &x) in q.iter().zip(x) {
                        let w = if sub % 2 == 0 { q & 0xf } else { q >> 4 };
                        s += w as i32 * x as i32;
                    }
                }
                let scale = b[PACK_SCALES + PACK_ROWS * sub + r] as f32;
                let min = b[PACK_MINS + PACK_ROWS * sub + r] as f32;
                *out += d * scale * a.d[blk] * s as f32 - dmin * min * a.sum[blk];
            }
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{GROUP, PACK_BLOCK, PACK_MINS, PACK_QS, PACK_ROWS, PACK_SCALES};
    use super::{Q8Act, QK, scale_min_k4, scales_mins_k4};
    use crate::quant::{GgmlType, QK_K};
    use crate::simd::x86::hsum;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,f16c")]
    #[inline]
    fn le16(b: &[u8], at: usize) -> f32 {
        let v = u16::from_le_bytes([b[at], b[at + 1]]) as u32;
        let x: __m128i;
        // A move written out, because the compiler turns every way of loading 16 bits into a
        // vector register (a wider load included) into an insert into whatever that register
        // last held, and the conversion then waits for it. On Zen 4 that wait was the hottest
        // spot of the Q6_K loop.
        // SAFETY: a register-to-register move; AVX is checked by the callers' callers.
        unsafe {
            std::arch::asm!(
                "vmovd {x}, {v:e}",
                v = in(reg) v,
                x = lateout(xmm_reg) x,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        _mm_cvtss_f32(_mm_cvtph_ps(x))
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

    /// [`dot_rows`] for Q4_K (`FIVE` false) and Q5_K. A super-block stays in integers: each
    /// sub-block's 6-bit scale times its activations' ratio is the `madd` multiplier, the eight
    /// products add up as i32, and one float multiply-add per super-block puts on the weights'
    /// scale and the activation group's. The mins are one vector multiply-add per super-block.
    /// The i32 sums can't overflow: a pair of products reaches 2 · 31 · 127, a multiplier
    /// 63 · 256, and a lane adds two pairs from each of eight sub-blocks, 2.04e9 in all.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    unsafe fn k_rows<const N: usize, const FIVE: bool>(
        row: &[u8],
        xq: [*const i8; N],
        xr: [*const i16; N],
        xu: [*const f32; N],
        xs: [*const f32; N],
        acc: &mut [__m256; N],
        extra: &mut [__m256; N],
    ) {
        let bb = if FIVE { 176 } else { 144 };
        let ql_at = if FIVE { 48 } else { 16 };
        let m4 = _mm256_set1_epi8(0xf);
        let sixteen = _mm256_set1_epi8(16);
        let picks = lane_picks();
        // SAFETY: per `dot_rows`, whose block count bounds every token's activations.
        unsafe {
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let (sc, mn) = scales_mins_k4(&b[4..16]);
                let sc = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(sc as i64));
                let mn = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(mn as i64)));
                let dmin = _mm256_mul_ps(mn, _mm256_set1_ps(le16(b, 2)));
                let d = le16(b, 0);
                let xb = i * (QK_K / QK);
                // Each token's eight multipliers as 16-bit lanes, in both halves of the register
                // for the shuffles below.
                let mut mult = [_mm256_setzero_si256(); N];
                let mut sum = [_mm256_setzero_si256(); N];
                for t in 0..N {
                    extra[t] = _mm256_fnmadd_ps(dmin, _mm256_loadu_ps(xs[t].add(xb)), extra[t]);
                    let ratio = _mm_loadu_si128(xr[t].add(xb).cast());
                    mult[t] = _mm256_broadcastsi128_si256(_mm_mullo_epi16(sc, ratio));
                }
                let qh = if FIVE {
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
                        if FIVE {
                            // Bit `sub` of each byte, moved to bit 4. The 16-bit shifts carry
                            // a neighbouring byte's bits in, which the mask drops.
                            let bit = if sub <= 4 {
                                _mm256_sll_epi16(qh, _mm_cvtsi32_si128(4 - sub as i32))
                            } else {
                                _mm256_srl_epi16(qh, _mm_cvtsi32_si128(sub as i32 - 4))
                            };
                            w = _mm256_or_si256(w, _mm256_and_si256(bit, sixteen));
                        }
                        for t in 0..N {
                            let x = _mm256_loadu_si256(xq[t].add((xb + sub) * QK).cast());
                            let m = _mm256_shuffle_epi8(mult[t], picks[sub]);
                            let s = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), m);
                            sum[t] = _mm256_add_epi32(sum[t], s);
                        }
                    }
                }
                for t in 0..N {
                    let scale = _mm256_set1_ps(d * *xu[t].add(i));
                    acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum[t]), scale, acc[t]);
                }
            }
        }
    }

    /// Shuffle masks that copy 16-bit lane `j` of each half of a register to all of that half's
    /// lanes, for `j` in 0..8. They go through `black_box` so each stays one byte shuffle: as
    /// visible constants, each compiles to a pair of word and dword shuffles on the ports the
    /// kernels are shortest of.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn lane_picks() -> [__m256i; 8] {
        std::hint::black_box(std::array::from_fn(|j| {
            _mm256_set1_epi16((((2 * j + 1) << 8) | (2 * j)) as i16)
        }))
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

    /// How far ahead of a row's start [`prefetch_ahead`] begins. The best of 1152 to 18432 bytes,
    /// and of starting at the next row instead, on Xeon 6973P-C and 6975P-C.
    const PREFETCH_BYTES: usize = 1152;

    /// Asks for the `row.len()` bytes that start `PREFETCH_BYTES` after `row` does: weights the
    /// kernel reaches soon, in this row or the rows after it. Decode reads every weight once per
    /// token, and without this an Intel core read them at a third of the speed it manages in
    /// cache. May name addresses past the matrix, which a prefetch ignores.
    #[inline]
    pub fn prefetch_ahead(row: &[u8]) {
        for off in (0..row.len()).step_by(64) {
            let p = row.as_ptr().wrapping_add(PREFETCH_BYTES + off);
            // SAFETY: a prefetch never faults, whatever the address; SSE is in the x86-64 baseline.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(p.cast()) };
        }
    }

    /// [`super::matvec_packed`]: groups of `gb` bytes, eight results each. Four activations are
    /// broadcast against the eight rows' matching weights, so a row's products land in its own
    /// 32-bit lane as two 16-bit sums. Those take a sub-block's eight steps without saturating
    /// (8 · 2 · 15 · 127 = 30480), then `madd` puts on the sub-block's scale times the
    /// activations' ratio. Two sub-blocks' sums fit an i32 (2 · 2 · 30480 · 63 · 256 = 1.97e9),
    /// a whole super-block's wouldn't, so they go to float a pair at a time.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn packed_k4(
        packed: &[u8],
        gb: usize,
        act: &Q8Act,
        out: &mut [f32],
        prefetch: bool,
    ) {
        let nb = gb / PACK_BLOCK;
        assert!(gb > 0 && gb.is_multiple_of(PACK_BLOCK) && out.len().is_multiple_of(PACK_ROWS));
        assert!(packed.len() == gb * (out.len() / PACK_ROWS));
        assert!(act.qs.len() == nb * QK_K && act.d.len() == nb * GROUP);
        assert!(act.sum.len() == nb * GROUP && act.ratio.len() == nb * GROUP);
        assert!(act.unit.len() == nb);
        let (xq, xr) = (act.qs.as_ptr(), act.ratio.as_ptr());
        let (xu, xs) = (act.unit.as_ptr(), act.sum.as_ptr());
        let m4 = _mm256_set1_epi8(0xf);
        for (group, out) in packed
            .chunks_exact(gb)
            .zip(out.as_chunks_mut::<PACK_ROWS>().0)
        {
            if prefetch {
                prefetch_ahead(group);
            }
            let mut acc = _mm256_setzero_ps();
            let mut extra = _mm256_setzero_ps();
            // SAFETY: every offset stays inside the group's current block of `PACK_BLOCK` bytes or
            // the activations' `nb` groups, as asserted above.
            unsafe {
                for i in 0..nb {
                    let p = group.as_ptr().add(i * PACK_BLOCK);
                    let xb = i * GROUP;
                    let d = _mm256_cvtph_ps(_mm_loadu_si128(p.cast()));
                    let dmin = _mm256_cvtph_ps(_mm_loadu_si128(p.add(16).cast()));
                    let scale = _mm256_mul_ps(d, _mm256_set1_ps(*xu.add(i)));
                    let mut mins = _mm256_setzero_ps();
                    for g in 0..GROUP / 2 {
                        let x = xq.add((xb + 2 * g) * QK);
                        let mut pairs = [_mm256_setzero_si256(); 2];
                        for k in 0..8 {
                            let q = _mm256_loadu_si256(p.add(PACK_QS + 32 * (8 * g + k)).cast());
                            let w = [_mm256_and_si256(q, m4), high_nibbles(q)];
                            for half in 0..2 {
                                let x4 = x.add(half * QK + 4 * k).cast::<i32>().read_unaligned();
                                let s = _mm256_maddubs_epi16(w[half], _mm256_set1_epi32(x4));
                                pairs[half] = _mm256_add_epi16(pairs[half], s);
                            }
                        }
                        let mut sum = _mm256_setzero_si256();
                        for (half, pairs) in pairs.into_iter().enumerate() {
                            let sub = 2 * g + half;
                            let at = PACK_ROWS * sub;
                            let sc = _mm_loadl_epi64(p.add(PACK_SCALES + at).cast());
                            let ratio = _mm256_set1_epi32(*xr.add(xb + sub) as i32);
                            let m = _mm256_mullo_epi16(_mm256_cvtepu8_epi32(sc), ratio);
                            let m = _mm256_or_si256(m, _mm256_slli_epi32::<16>(m));
                            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(pairs, m));
                            let mn = _mm_loadl_epi64(p.add(PACK_MINS + at).cast());
                            let mn = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(mn));
                            mins = _mm256_fmadd_ps(mn, _mm256_set1_ps(*xs.add(xb + sub)), mins);
                        }
                        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), scale, acc);
                    }
                    extra = _mm256_fnmadd_ps(dmin, mins, extra);
                }
                _mm256_storeu_ps(out.as_mut_ptr(), _mm256_add_ps(acc, extra));
            }
        }
    }

    /// Dot products of consecutive encoded rows of `rb` bytes with the activations of `N` tokens,
    /// into `out` (`N` per row), unpacking each weight block once for all of them. Every token's
    /// arithmetic is the same whatever `N` is, so a matmul matches per-token matvecs bit for bit.
    /// `prefetch` asks for the weights ahead of each row ([`prefetch_ahead`]). `false` for types
    /// without a kernel.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_rows<const N: usize>(
        ty: GgmlType,
        rows: &[u8],
        rb: usize,
        acts: &[Q8Act],
        out: &mut [f32],
        prefetch: bool,
    ) -> bool {
        macro_rules! of {
            ($($t:ident)*) => {
                match ty {
                    // SAFETY: the caller's.
                    $(GgmlType::$t => unsafe {
                        rows_of::<N, { GgmlType::$t as u8 }>(rows, rb, acts, out, prefetch)
                    },)*
                    _ => return false,
                }
            };
        }
        of!(Q8_0 Q4_0 Q5_0 Q4_K Q5_K Q6_K);
        true
    }

    const fn ty_of(t: u8) -> GgmlType {
        const TYPES: [GgmlType; 6] = [
            GgmlType::Q8_0,
            GgmlType::Q4_0,
            GgmlType::Q5_0,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ];
        let mut i = 0;
        while i < TYPES.len() {
            if TYPES[i] as u8 == t {
                return TYPES[i];
            }
            i += 1;
        }
        panic!("not a type with a kernel")
    }

    /// [`dot_rows`] for weight type `T` (a [`GgmlType`] as a number). The type is a constant here
    /// so that everything that depends only on it and on the row length is worked out once, not
    /// per row: with it inside the row loop, a 2048-weight Q4_K row spent about a tenth of its
    /// time there on Zen 4.
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn rows_of<const N: usize, const T: u8>(
        rows: &[u8],
        rb: usize,
        acts: &[Q8Act],
        out: &mut [f32],
        prefetch: bool,
    ) {
        let ty = const { ty_of(T) };
        let bb = ty.block_bytes();
        let n_blocks = rb / bb;
        assert!(acts.len() == N && rb > 0 && rows.len() == rb * (out.len() / N));
        assert!(out.len().is_multiple_of(N));
        for a in acts {
            assert_eq!(n_blocks * ty.block_size(), a.qs.len());
        }
        let xq: [*const i8; N] = std::array::from_fn(|t| acts[t].qs.as_ptr());
        // Activation scales and sums, read unchecked in the loops below: indexing them cost a
        // bounds check per token every 32 weights.
        let xd: [*const f32; N] = std::array::from_fn(|t| acts[t].d.as_ptr());
        let xs: [*const f32; N] = std::array::from_fn(|t| acts[t].sum.as_ptr());
        let xr: [*const i16; N] = std::array::from_fn(|t| acts[t].ratio.as_ptr());
        let xu: [*const f32; N] = std::array::from_fn(|t| acts[t].unit.as_ptr());
        for a in acts {
            assert!(a.d.len() * QK == a.qs.len() && a.sum.len() == a.d.len());
            assert!(a.ratio.len() == a.d.len() && a.unit.len() == a.d.len().div_ceil(GROUP));
        }
        for (row, out) in rows.chunks_exact(rb).zip(out.as_chunks_mut::<N>().0) {
            if prefetch {
                prefetch_ahead(row);
            }
            let mut acc = [_mm256_setzero_ps(); N];
            // The K-quant mins' terms, kept apart from the weights' sums.
            let mut extra = [_mm256_setzero_ps(); N];
            // SAFETY (all blocks below): every pointer offset stays inside the current block of `row`
            // or each token's activations (n_blocks * block_size values and n_blocks * block_size / 32
            // scales and sums), as asserted above.
            unsafe {
                // The 32-weight formats, whose blocks start with their f16 scale: `$weights` decodes
                // a block's weights, as stored values that stand for themselves minus `$off`. Four
                // running sums take the blocks in turn, so a block's multiply-add doesn't wait for
                // the one before it (a single sum held the loop to one multiply-add latency per
                // block). Four blocks' scales are converted together: one f16 at a time compiles to
                // an insert into a register an earlier sum last wrote, which chains the sums again.
                // With an offset, the weights stay unsigned for `maddubs` and the offset comes off
                // through the activations' sums, which saves moving each block's signs.
                macro_rules! blocks32 {
                    ($off:literal, $weights:expr) => {{
                        let weights = $weights;
                        let mut acc4 = [[_mm256_setzero_ps(); 4]; N];
                        let mut sums = [_mm_setzero_ps(); N];
                        let ones = _mm256_set1_epi16(1);
                        let h = |b: *const u8| b.cast::<u16>().read_unaligned().to_le() as u64;
                        for i in (0..n_blocks / 4).map(|i| 4 * i) {
                            let b = row.as_ptr().add(i * bb);
                            let d = h(b)
                                | h(b.add(bb)) << 16
                                | h(b.add(2 * bb)) << 32
                                | h(b.add(3 * bb)) << 48;
                            let d = _mm_cvtph_ps(_mm_cvtsi64_si128(d as i64));
                            let mut scales = [_mm256_setzero_ps(); N];
                            for t in 0..N {
                                let s = _mm_mul_ps(d, _mm_loadu_ps(xd[t].add(i)));
                                scales[t] = _mm256_castps128_ps256(s);
                                if $off != 0 {
                                    sums[t] = _mm_fmadd_ps(d, _mm_loadu_ps(xs[t].add(i)), sums[t]);
                                }
                            }
                            for k in 0..4 {
                                let w: __m256i = weights(b.add(k * bb + 2));
                                let lane = _mm256_set1_epi32(k as i32);
                                for t in 0..N {
                                    let x = xq[t].add((i + k) * QK);
                                    let s = if $off != 0 {
                                        let x = _mm256_loadu_si256(x.cast());
                                        _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), ones)
                                    } else {
                                        dot32(w, x)
                                    };
                                    let scale = _mm256_permutevar8x32_ps(scales[t], lane);
                                    acc4[t][k] =
                                        _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), scale, acc4[t][k]);
                                }
                            }
                        }
                        for i in n_blocks / 4 * 4..n_blocks {
                            let b = row.as_ptr().add(i * bb);
                            let w: __m256i = weights(b.add(2));
                            let w = _mm256_sub_epi8(w, _mm256_set1_epi8($off));
                            let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi64_si128(h(b) as i64)));
                            for t in 0..N {
                                let s = dot32(w, xq[t].add(i * QK));
                                let scale = _mm256_set1_ps(d * *xd[t].add(i));
                                acc4[t][0] =
                                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), scale, acc4[t][0]);
                            }
                        }
                        for t in 0..N {
                            let a = acc4[t];
                            acc[t] =
                                _mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3]));
                            let off = _mm_mul_ps(sums[t], _mm_set1_ps(-($off as f32)));
                            extra[t] = _mm256_zextps128_ps256(off);
                        }
                    }};
                }
                match ty {
                    GgmlType::Q8_0 => blocks32!(0, |q: *const u8| _mm256_loadu_si256(q.cast())),
                    GgmlType::Q4_0 => blocks32!(8, |q: *const u8| unpack_nibbles(q)),
                    GgmlType::Q5_0 => blocks32!(16, |q: *const u8| {
                        let qh = q.cast::<u32>().read_unaligned().to_le();
                        _mm256_or_si256(unpack_nibbles(q.add(4)), high_bits(qh))
                    }),
                    GgmlType::Q4_K => k_rows::<N, false>(row, xq, xr, xu, xs, &mut acc, &mut extra),
                    GgmlType::Q5_K => k_rows::<N, true>(row, xq, xr, xu, xs, &mut acc, &mut extra),
                    GgmlType::Q6_K => {
                        let m4 = _mm256_set1_epi8(0xf);
                        let three = _mm256_set1_epi8(3);
                        let off = _mm256_set1_epi8(32);
                        // The 16 scales with the even ones first, so that as 16-bit lanes each
                        // half of the register holds one half-block's scale of every pair.
                        let split =
                            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
                        let picks = lane_picks();
                        for (i, b) in row.chunks_exact(bb).enumerate() {
                            let d = le16(b, 208);
                            let p = b.as_ptr();
                            let sc = _mm_shuffle_epi8(_mm_loadu_si128(p.add(192).cast()), split);
                            let sc = _mm256_cvtepi8_epi16(sc);
                            // As in `k_rows`, each scale times its block's activation ratio is the
                            // multiplier, and the products add up as i32. Here a multiplier reaches
                            // 128 · 256 and a pair of products 2 · 32 · 127, so a lane holds four
                            // blocks (2.13e9) and each half of the super-block has its own sum.
                            let mut mult = [_mm256_setzero_si256(); N];
                            let mut sum = [[_mm256_setzero_si256(); N]; 2];
                            for t in 0..N {
                                let ratio = _mm_loadu_si128(xr[t].add(i * (QK_K / QK)).cast());
                                mult[t] =
                                    _mm256_mullo_epi16(sc, _mm256_broadcastsi128_si256(ratio));
                            }
                            for (n, sum) in sum.iter_mut().enumerate() {
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
                                    let j = 4 * n + k;
                                    let xb = i * (QK_K / QK) + j;
                                    // Each half's i8 scale goes on in integer arithmetic (exact)
                                    // before the one float multiply, as the `madd` multiplier: the
                                    // halves are the two 128-bit lanes of the partial sums.
                                    for t in 0..N {
                                        // Weights are stored plus 32 and `maddubs` wants them
                                        // unsigned, so the 32 comes off as its own product: pair
                                        // sums reach 2 · 63 · 127 and 2 · 32 · 127, inside an i16.
                                        let x = _mm256_loadu_si256(xq[t].add(xb * QK).cast());
                                        let pairs = _mm256_sub_epi16(
                                            _mm256_maddubs_epi16(w, x),
                                            _mm256_maddubs_epi16(off, x),
                                        );
                                        let m = _mm256_shuffle_epi8(mult[t], picks[j]);
                                        let s = _mm256_madd_epi16(pairs, m);
                                        sum[t] = _mm256_add_epi32(sum[t], s);
                                    }
                                }
                            }
                            for t in 0..N {
                                let halves = _mm256_add_ps(
                                    _mm256_cvtepi32_ps(sum[0][t]),
                                    _mm256_cvtepi32_ps(sum[1][t]),
                                );
                                let scale = _mm256_set1_ps(d * *xu[t].add(i));
                                acc[t] = _mm256_fmadd_ps(halves, scale, acc[t]);
                            }
                        }
                    }
                    _ => unreachable!("dot_rows names the types"),
                }
            }
            for t in 0..N {
                out[t] = hsum(_mm256_add_ps(acc[t], extra[t]));
            }
        }
    }

    /// Tokens' activations for [`dot_block`], each field in one buffer, token after token.
    pub struct BlockActs {
        n_tokens: usize,
        n_chunks: usize,
        qs: Vec<i8>,
        d: Vec<f32>,
        sum: Vec<f32>,
        /// For the AVX-512 kernel (empty without it): 128 times the sum of each half chunk's
        /// 16 values, which is what biasing signed weights by 128 adds to a half's products.
        bias: Vec<i32>,
    }

    impl BlockActs {
        /// `None` for a single token, when the CPU lacks AVX2, or when `ty` has no kernel.
        pub fn new(ty: GgmlType, acts: &[Q8Act]) -> Option<Self> {
            if acts.len() < 2 || !crate::simd::x86::has_avx2() || kind(ty).is_none() {
                return None;
            }
            let n_chunks = acts[0].d.len();
            let mut b = BlockActs {
                n_tokens: acts.len(),
                n_chunks,
                qs: Vec::with_capacity(acts.len() * n_chunks * QK),
                d: Vec::with_capacity(acts.len() * n_chunks),
                sum: Vec::with_capacity(acts.len() * n_chunks),
                bias: Vec::new(),
            };
            for a in acts {
                assert!(a.d.len() == n_chunks && a.qs.len() == n_chunks * QK);
                b.qs.extend_from_slice(&a.qs);
                b.d.extend_from_slice(&a.d);
                b.sum.extend_from_slice(&a.sum);
            }
            if crate::simd::x86::has_avx512_vnni() && kind(ty) != Some(OFFSET) {
                let halves = b.qs.as_chunks::<{ QK / 2 }>().0;
                b.bias = halves
                    .iter()
                    .map(|h| 128 * h.iter().map(|&q| q as i32).sum::<i32>())
                    .collect();
            }
            Some(b)
        }
    }

    /// Q8_0: signed weights.
    const SIGNED: u8 = 0;
    /// Q4_0, Q5_0, Q4_K, Q5_K: unsigned weights `q`, with the offset or min as a separate term:
    /// weight = d·q − m.
    const OFFSET: u8 = 1;
    /// Q6_K: signed weights with an integer scale per 16.
    const HALVES: u8 = 2;

    fn kind(ty: GgmlType) -> Option<u8> {
        Some(match ty {
            GgmlType::Q8_0 => SIGNED,
            GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q4_K | GgmlType::Q5_K => OFFSET,
            GgmlType::Q6_K => HALVES,
            _ => return None,
        })
    }

    /// Chunk `j` (weights `32j..32j + 32`) of `row` as 32 bytes (i8 for [`SIGNED`] and
    /// [`HALVES`], u8 for [`OFFSET`]), with its float scale `d`, its offset `m`, and the
    /// integer scales of its two halves.
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn decode(ty: GgmlType, row: &[u8], j: usize) -> (__m256i, f32, f32, [i16; 2]) {
        let bb = ty.block_bytes();
        let m4 = _mm256_set1_epi8(0xf);
        // SAFETY: the caller passes `j` below the row's chunk count, so every block read is in
        // `row` (sliced, so a bad `j` panics rather than reading out of bounds).
        unsafe {
            match ty {
                GgmlType::Q8_0 => {
                    let b = &row[j * bb..(j + 1) * bb];
                    let w = _mm256_loadu_si256(b.as_ptr().add(2).cast());
                    (w, le16(b, 0), 0.0, [1, 1])
                }
                GgmlType::Q4_0 => {
                    let b = &row[j * bb..(j + 1) * bb];
                    let d = le16(b, 0);
                    (unpack_nibbles(b.as_ptr().add(2)), d, 8.0 * d, [1, 1])
                }
                GgmlType::Q5_0 => {
                    let b = &row[j * bb..(j + 1) * bb];
                    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                    let w = _mm256_or_si256(unpack_nibbles(b.as_ptr().add(6)), high_bits(qh));
                    let d = le16(b, 0);
                    (w, d, 16.0 * d, [1, 1])
                }
                GgmlType::Q4_K | GgmlType::Q5_K => {
                    let five = ty == GgmlType::Q5_K;
                    let ql_at = if five { 48 } else { 16 };
                    let b = &row[(j / 8) * bb..(j / 8 + 1) * bb];
                    let sub = j % 8;
                    let q = _mm256_loadu_si256(b.as_ptr().add(ql_at + 32 * (sub / 2)).cast());
                    let mut w = if sub.is_multiple_of(2) {
                        _mm256_and_si256(q, m4)
                    } else {
                        high_nibbles(q)
                    };
                    if five {
                        let qh = _mm256_loadu_si256(b.as_ptr().add(16).cast());
                        let bit = _mm256_set1_epi8(1 << sub);
                        let set = _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit);
                        w = _mm256_or_si256(w, _mm256_and_si256(set, _mm256_set1_epi8(16)));
                    }
                    let (sc, mn) = scale_min_k4(sub, &b[4..16]);
                    (w, le16(b, 0) * sc, le16(b, 2) * mn, [1, 1])
                }
                GgmlType::Q6_K => {
                    let b = &row[(j / 8) * bb..(j / 8 + 1) * bb];
                    let (n, k) = ((j % 8) / 4, j % 4);
                    let p = b.as_ptr();
                    let l = _mm256_loadu_si256(p.add(64 * n + 32 * (k % 2)).cast());
                    let lo = if k < 2 {
                        _mm256_and_si256(l, m4)
                    } else {
                        high_nibbles(l)
                    };
                    let qh = _mm256_loadu_si256(p.add(128 + 32 * n).cast());
                    let shift = _mm_cvtsi32_si128(2 * k as i32);
                    let hi = _mm256_and_si256(_mm256_srl_epi16(qh, shift), _mm256_set1_epi8(3));
                    let w = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 4));
                    let w = _mm256_sub_epi8(w, _mm256_set1_epi8(32));
                    let sc = &b[192..208];
                    let hs = [
                        sc[8 * n + 2 * k] as i8 as i16,
                        sc[8 * n + 2 * k + 1] as i8 as i16,
                    ];
                    (w, le16(b, 208), 0.0, hs)
                }
                _ => unreachable!("checked by BlockActs::new"),
            }
        }
    }

    /// Chunks of eight rows decoded and interleaved, so that one 4-byte group of a token's
    /// activations, broadcast, meets the matching group of all eight rows: vector `8c + k` holds,
    /// in its i32 lane `r`, weights `4k..4k + 4` of chunk `c` of row `r`.
    struct Panel {
        w: Vec<__m256i>,
        /// Per chunk, each row's `d` and `m`.
        d: Vec<__m256>,
        m: Vec<__m256>,
        /// Per chunk and half, each row's integer scale (twice, as the i16 pairs `madd` takes).
        hs: Vec<[__m256i; 2]>,
    }

    impl Panel {
        #[target_feature(enable = "avx2,f16c")]
        unsafe fn fill(&mut self, ty: GgmlType, rows: [&[u8]; 8], chunks: std::ops::Range<usize>) {
            self.w.clear();
            self.d.clear();
            self.m.clear();
            self.hs.clear();
            for j in chunks {
                let mut lanes = [[0i32; 8]; 8];
                let (mut d, mut m, mut hs) = ([0f32; 8], [0f32; 8], [[0i16; 16]; 2]);
                for (r, row) in rows.iter().enumerate() {
                    // SAFETY: `j` is a chunk of the row, per the caller.
                    let (w, rd, rm, rh) = unsafe { decode(ty, row, j) };
                    let mut bytes = [0i32; 8];
                    // SAFETY: 32 writable bytes.
                    unsafe { _mm256_storeu_si256(bytes.as_mut_ptr().cast(), w) };
                    for (k, v) in bytes.into_iter().enumerate() {
                        lanes[k][r] = v;
                    }
                    (d[r], m[r]) = (rd, rm);
                    for h in 0..2 {
                        hs[h][2 * r] = rh[h];
                        hs[h][2 * r + 1] = rh[h];
                    }
                }
                // SAFETY: each source holds 32 bytes.
                unsafe {
                    for l in &lanes {
                        self.w.push(_mm256_loadu_si256(l.as_ptr().cast()));
                    }
                    self.d.push(_mm256_loadu_ps(d.as_ptr()));
                    self.m.push(_mm256_loadu_ps(m.as_ptr()));
                    self.hs
                        .push(hs.map(|h| _mm256_loadu_si256(h.as_ptr().cast())));
                }
            }
        }
    }

    /// Panel chunks `0..n` (chunks `j0..j0 + n` of the rows) against tokens `t0..t0 + NT`,
    /// accumulating into `acc[t]` (the scaled integer sums) and `ex[t]` (the offsets), lane `r`
    /// for row `r`. Each lane does the same operations in the same order for its row and token,
    /// whatever else is in the panel or the group.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run<const K: u8, const NT: usize>(
        p: &Panel,
        j0: usize,
        n: usize,
        acts: &BlockActs,
        t0: usize,
        acc: &mut [__m256],
        ex: &mut [__m256],
    ) {
        let nc = acts.n_chunks;
        let ones = _mm256_set1_epi16(1);
        let mut a: [__m256; NT] = std::array::from_fn(|i| acc[t0 + i]);
        let mut e: [__m256; NT] = std::array::from_fn(|i| ex[t0 + i]);
        // SAFETY: tokens `t0..t0 + NT` and chunks `j0..j0 + n` are inside `acts`, and the panel
        // holds `n` chunks, per the caller.
        unsafe {
            let x: [*const i8; NT] =
                std::array::from_fn(|i| acts.qs.as_ptr().add(((t0 + i) * nc + j0) * QK));
            for c in 0..n {
                let mut isum = [_mm256_setzero_si256(); NT];
                for k in 0..8 {
                    let w = *p.w.get_unchecked(8 * c + k);
                    let aw = if K == OFFSET { w } else { _mm256_abs_epi8(w) };
                    let mult = if K == HALVES {
                        p.hs.get_unchecked(c)[k / 4]
                    } else {
                        ones
                    };
                    for i in 0..NT {
                        let g = x[i].add(c * QK + 4 * k).cast::<i32>().read_unaligned();
                        let xb = _mm256_set1_epi32(g);
                        // `maddubs` needs one unsigned side; see `dot32` for why the sign moves
                        // onto the activations.
                        let xs = if K == OFFSET {
                            xb
                        } else {
                            _mm256_sign_epi8(xb, w)
                        };
                        let pairs = _mm256_maddubs_epi16(aw, xs);
                        isum[i] = _mm256_add_epi32(isum[i], _mm256_madd_epi16(pairs, mult));
                    }
                }
                let at = |i: usize| (t0 + i) * nc + j0 + c;
                for i in 0..NT {
                    let xd = _mm256_set1_ps(*acts.d.get_unchecked(at(i)));
                    let scale = _mm256_mul_ps(*p.d.get_unchecked(c), xd);
                    a[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum[i]), scale, a[i]);
                    if K == OFFSET {
                        let xs = _mm256_set1_ps(*acts.sum.get_unchecked(at(i)));
                        e[i] = _mm256_fnmadd_ps(*p.m.get_unchecked(c), xs, e[i]);
                    }
                }
            }
        }
        acc[t0..t0 + NT].copy_from_slice(&a);
        ex[t0..t0 + NT].copy_from_slice(&e);
    }

    /// Chunks per panel: 8 KB of weights, with a token's matching activations (1 KB), stays
    /// in L1 while every token passes over it.
    const PANEL_CHUNKS: usize = 32;

    /// [`super::dot_block`], on the widest kernel the CPU has.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_block(
        ty: GgmlType,
        rows: &[u8],
        rb: usize,
        acts: &BlockActs,
        out: &mut [f32],
    ) {
        // SAFETY: the caller's guarantees, plus the AVX-512 check for that kernel.
        unsafe {
            if crate::simd::x86::has_avx512_vnni() {
                dot_block16(ty, rows, rb, acts, out)
            } else {
                dot_block8(ty, rows, rb, acts, out)
            }
        }
    }

    /// The AVX2 block kernel: rows in eights (the last row repeated to fill the final one) and
    /// tokens in fours.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn dot_block8(
        ty: GgmlType,
        rows: &[u8],
        rb: usize,
        acts: &BlockActs,
        out: &mut [f32],
    ) {
        let k = kind(ty).expect("checked by BlockActs::new");
        let (n_rows, nt, nc) = (rows.len() / rb, acts.n_tokens, acts.n_chunks);
        assert!(n_rows > 0 && out.len() == n_rows * nt && rows.len() == n_rows * rb);
        assert_eq!(rb / ty.block_bytes() * ty.block_size(), nc * QK);
        let mut panel = Panel {
            w: Vec::with_capacity(8 * PANEL_CHUNKS),
            d: Vec::with_capacity(PANEL_CHUNKS),
            m: Vec::with_capacity(PANEL_CHUNKS),
            hs: Vec::with_capacity(PANEL_CHUNKS),
        };
        let mut acc = vec![_mm256_setzero_ps(); nt];
        let mut ex = vec![_mm256_setzero_ps(); nt];
        for r0 in (0..n_rows).step_by(8) {
            let eight: [&[u8]; 8] = std::array::from_fn(|r| {
                let r = (r0 + r).min(n_rows - 1);
                &rows[r * rb..(r + 1) * rb]
            });
            acc.fill(_mm256_setzero_ps());
            ex.fill(_mm256_setzero_ps());
            for j0 in (0..nc).step_by(PANEL_CHUNKS) {
                let n = PANEL_CHUNKS.min(nc - j0);
                // SAFETY: chunks `j0..j0 + n` are in every row (sizes asserted above), and the
                // token ranges below are inside `acts`.
                unsafe {
                    panel.fill(ty, eight, j0..j0 + n);
                    let mut t = 0;
                    macro_rules! go {
                        ($nt:literal) => {
                            match k {
                                SIGNED => {
                                    run::<SIGNED, $nt>(&panel, j0, n, acts, t, &mut acc, &mut ex)
                                }
                                OFFSET => {
                                    run::<OFFSET, $nt>(&panel, j0, n, acts, t, &mut acc, &mut ex)
                                }
                                _ => run::<HALVES, $nt>(&panel, j0, n, acts, t, &mut acc, &mut ex),
                            }
                        };
                    }
                    while t + 4 <= nt {
                        go!(4);
                        t += 4;
                    }
                    while t < nt {
                        go!(1);
                        t += 1;
                    }
                }
            }
            for t in 0..nt {
                let mut lanes = [0f32; 8];
                // SAFETY: 8 writable floats.
                unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(acc[t], ex[t])) };
                for (r, &v) in lanes.iter().enumerate().take(n_rows - r0) {
                    out[(r0 + r) * nt + t] = v;
                }
            }
        }
    }

    /// [`Panel`] for sixteen rows: vector `8c + k` holds, in its i32 lane `r`, weights
    /// `4k..4k + 4` of chunk `c` of row `r`. Signed weights are stored plus 128, as the unsigned
    /// bytes `vpdpbusd` takes.
    struct Panel16 {
        w: Vec<__m512i>,
        d: Vec<__m512>,
        m: Vec<__m512>,
        /// Per chunk and half, each row's integer scale as an i32.
        hs: Vec<[__m512i; 2]>,
    }

    impl Panel16 {
        #[target_feature(enable = "avx512f,avx2,f16c")]
        unsafe fn fill(&mut self, ty: GgmlType, rows: [&[u8]; 16], chunks: std::ops::Range<usize>) {
            self.w.clear();
            self.d.clear();
            self.m.clear();
            self.hs.clear();
            // Adding 128 to a signed byte flips its top bit.
            let flip = if kind(ty) == Some(OFFSET) {
                0
            } else {
                0x80u8 as i8
            };
            let flip = _mm256_set1_epi8(flip);
            for j in chunks {
                let mut lanes = [[0i32; 16]; 8];
                let (mut d, mut m, mut hs) = ([0f32; 16], [0f32; 16], [[0i32; 16]; 2]);
                for (r, row) in rows.iter().enumerate() {
                    // SAFETY: `j` is a chunk of the row, per the caller.
                    let (w, rd, rm, rh) = unsafe { decode(ty, row, j) };
                    let mut bytes = [0i32; 8];
                    // SAFETY: 32 writable bytes.
                    unsafe {
                        _mm256_storeu_si256(bytes.as_mut_ptr().cast(), _mm256_xor_si256(w, flip))
                    };
                    for (k, v) in bytes.into_iter().enumerate() {
                        lanes[k][r] = v;
                    }
                    (d[r], m[r]) = (rd, rm);
                    for h in 0..2 {
                        hs[h][r] = rh[h] as i32;
                    }
                }
                // SAFETY: each source holds 64 bytes.
                unsafe {
                    for l in &lanes {
                        self.w.push(_mm512_loadu_si512(l.as_ptr().cast()));
                    }
                    self.d.push(_mm512_loadu_ps(d.as_ptr()));
                    self.m.push(_mm512_loadu_ps(m.as_ptr()));
                    self.hs
                        .push(hs.map(|h| _mm512_loadu_si512(h.as_ptr().cast())));
                }
            }
        }
    }

    /// [`run`] over a [`Panel16`]. `vpdpbusd` multiplies four unsigned bytes by four signed
    /// ones and adds the products to an i32 in one step, where AVX2 takes `maddubs`, `madd` and
    /// an add. Signed weights come biased by 128, and each half chunk's sum gives back 128
    /// times its activations' sum ([`BlockActs::bias`]), the same for every row. The integer
    /// sums are the ones [`run`] gets and the float steps are the same in the same order, so
    /// the two kernels agree bit for bit.
    #[target_feature(enable = "avx512f,avx512vnni")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run16<const K: u8, const NT: usize>(
        p: &Panel16,
        j0: usize,
        n: usize,
        acts: &BlockActs,
        t0: usize,
        acc: &mut [__m512],
        ex: &mut [__m512],
    ) {
        let nc = acts.n_chunks;
        let zero = _mm512_setzero_si512();
        let mut a: [__m512; NT] = std::array::from_fn(|i| acc[t0 + i]);
        let mut e: [__m512; NT] = std::array::from_fn(|i| ex[t0 + i]);
        // SAFETY: tokens `t0..t0 + NT` and chunks `j0..j0 + n` are inside `acts`, and the panel
        // holds `n` chunks, per the caller.
        unsafe {
            let x: [*const i8; NT] =
                std::array::from_fn(|i| acts.qs.as_ptr().add(((t0 + i) * nc + j0) * QK));
            for c in 0..n {
                let mut isum = [zero; NT];
                // A half is 16 weights: Q6_K's integer scale goes on each half's sum.
                for h in 0..2 {
                    let mut half = [zero; NT];
                    for k in 4 * h..4 * h + 4 {
                        let w = *p.w.get_unchecked(8 * c + k);
                        for i in 0..NT {
                            let g = x[i].add(c * QK + 4 * k).cast::<i32>().read_unaligned();
                            half[i] = _mm512_dpbusd_epi32(half[i], w, _mm512_set1_epi32(g));
                        }
                    }
                    for i in 0..NT {
                        let mut v = half[i];
                        if K != OFFSET {
                            let at = 2 * ((t0 + i) * nc + j0 + c) + h;
                            v = _mm512_sub_epi32(
                                v,
                                _mm512_set1_epi32(*acts.bias.get_unchecked(at)),
                            );
                        }
                        if K == HALVES {
                            v = _mm512_mullo_epi32(v, p.hs.get_unchecked(c)[h]);
                        }
                        isum[i] = _mm512_add_epi32(isum[i], v);
                    }
                }
                let at = |i: usize| (t0 + i) * nc + j0 + c;
                for i in 0..NT {
                    let xd = _mm512_set1_ps(*acts.d.get_unchecked(at(i)));
                    let scale = _mm512_mul_ps(*p.d.get_unchecked(c), xd);
                    a[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum[i]), scale, a[i]);
                    if K == OFFSET {
                        let xs = _mm512_set1_ps(*acts.sum.get_unchecked(at(i)));
                        e[i] = _mm512_fnmadd_ps(*p.m.get_unchecked(c), xs, e[i]);
                    }
                }
            }
        }
        acc[t0..t0 + NT].copy_from_slice(&a);
        ex[t0..t0 + NT].copy_from_slice(&e);
    }

    /// The AVX-512 VNNI block kernel: [`dot_block8`] with rows in sixteens.
    #[target_feature(enable = "avx512f,avx512vnni,avx2,fma,f16c")]
    pub(super) unsafe fn dot_block16(
        ty: GgmlType,
        rows: &[u8],
        rb: usize,
        acts: &BlockActs,
        out: &mut [f32],
    ) {
        let k = kind(ty).expect("checked by BlockActs::new");
        let (n_rows, nt, nc) = (rows.len() / rb, acts.n_tokens, acts.n_chunks);
        assert!(n_rows > 0 && out.len() == n_rows * nt && rows.len() == n_rows * rb);
        assert_eq!(rb / ty.block_bytes() * ty.block_size(), nc * QK);
        assert!(k == OFFSET || acts.bias.len() == 2 * nt * nc);
        let mut panel = Panel16 {
            w: Vec::with_capacity(8 * PANEL_CHUNKS),
            d: Vec::with_capacity(PANEL_CHUNKS),
            m: Vec::with_capacity(PANEL_CHUNKS),
            hs: Vec::with_capacity(PANEL_CHUNKS),
        };
        let mut acc = vec![_mm512_setzero_ps(); nt];
        let mut ex = vec![_mm512_setzero_ps(); nt];
        for r0 in (0..n_rows).step_by(16) {
            let sixteen: [&[u8]; 16] = std::array::from_fn(|r| {
                let r = (r0 + r).min(n_rows - 1);
                &rows[r * rb..(r + 1) * rb]
            });
            acc.fill(_mm512_setzero_ps());
            ex.fill(_mm512_setzero_ps());
            for j0 in (0..nc).step_by(PANEL_CHUNKS) {
                let n = PANEL_CHUNKS.min(nc - j0);
                // SAFETY: chunks `j0..j0 + n` are in every row (sizes asserted above), and the
                // token ranges below are inside `acts`.
                unsafe {
                    panel.fill(ty, sixteen, j0..j0 + n);
                    let mut t = 0;
                    macro_rules! go {
                        ($nt:literal) => {
                            match k {
                                SIGNED => {
                                    run16::<SIGNED, $nt>(&panel, j0, n, acts, t, &mut acc, &mut ex)
                                }
                                OFFSET => {
                                    run16::<OFFSET, $nt>(&panel, j0, n, acts, t, &mut acc, &mut ex)
                                }
                                _ => {
                                    run16::<HALVES, $nt>(&panel, j0, n, acts, t, &mut acc, &mut ex)
                                }
                            }
                        };
                    }
                    while t + 4 <= nt {
                        go!(4);
                        t += 4;
                    }
                    while t < nt {
                        go!(1);
                        t += 1;
                    }
                }
            }
            for t in 0..nt {
                let mut lanes = [0f32; 16];
                // SAFETY: 16 writable floats.
                unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), _mm512_add_ps(acc[t], ex[t])) };
                for (r, &v) in lanes.iter().enumerate().take(n_rows - r0) {
                    out[(r0 + r) * nt + t] = v;
                }
            }
        }
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
    fn packed_k4_scales_match_one_at_a_time() {
        let mut rng = Rng::new(5);
        for _ in 0..1000 {
            let q: Vec<u8> = (0..12).map(|_| rng.next_u32() as u8).collect();
            let (sc, mn) = scales_mins_k4(&q);
            for j in 0..8 {
                let want = scale_min_k4(j, &q);
                let got = ((sc >> (8 * j)) as u8 as f32, (mn >> (8 * j)) as u8 as f32);
                assert_eq!(got, want, "sub-block {j} of {q:?}");
            }
        }
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

    /// Whether this CPU has a block kernel for `ty`.
    fn has_block(ty: GgmlType) -> bool {
        let a = Q8Act::new(&[0.0; QK]);
        BlockActs::new(ty, &[a.clone(), a]).is_some()
    }

    /// The block kernel's result for one row and token (given as a pair of the same token).
    fn block_dot(ty: GgmlType, row: &[u8], a: &Q8Act) -> Option<f32> {
        let b = BlockActs::new(ty, &[a.clone(), a.clone()])?;
        let mut out = [0.0; 2];
        dot_block(ty, row, row.len(), &b, &mut out);
        assert_eq!(out[0], out[1]);
        Some(out[0])
    }

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
        // Odd row and token counts exercise the self-paired last row and token, 11 tokens span
        // a full group of 8 plus a partial one, and 13 rows a full eight plus a partial one.
        let (n_rows, n_tok) = (13, 11);
        for ty in INT_TYPES {
            let rows: Vec<u8> = (0..n_rows)
                .flat_map(|_| random_row(ty, n, &mut rng))
                .collect();
            let rb = ty.row_bytes(n);
            let acts: Vec<Q8Act> = (0..n_tok).map(|_| Q8Act::new(&rng.vec(n))).collect();
            let mut out = vec![0.0; n_rows * n_tok];
            let Some(block) = BlockActs::new(ty, &acts) else {
                continue; // no i8mm on this CPU, or no kernel for the type
            };
            dot_block(ty, &rows, rb, &block, &mut out);
            for r in 0..n_rows {
                let row = &rows[r * rb..(r + 1) * rb];
                for (t, a) in acts.iter().enumerate() {
                    let got = out[r * n_tok + t];
                    // Regrouped: this row alone (paired with itself) and the token in a pair.
                    let pair = [a.clone(), a.clone()];
                    let mut two = [0.0; 2];
                    let pair = BlockActs::new(ty, &pair).unwrap();
                    dot_block(ty, row, rb, &pair, &mut two);
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

    /// The AVX-512 block kernel does the AVX2 one's arithmetic sixteen rows at a time, so they
    /// must agree bit for bit, whatever the row count leaves in the last group.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_block_kernel_equals_avx2_block_kernel_exactly() {
        if !crate::simd::x86::has_avx512_vnni() {
            eprintln!("skipped: no AVX-512 VNNI on this CPU");
            return;
        }
        let mut rng = Rng::new(31);
        let (n, n_tok) = (512, 11);
        for ty in INT_TYPES {
            let acts: Vec<Q8Act> = (0..n_tok).map(|_| Q8Act::new(&rng.vec(n))).collect();
            let Some(block) = BlockActs::new(ty, &acts) else {
                continue;
            };
            let rb = ty.row_bytes(n);
            for n_rows in [1, 13, 16, 17, 35] {
                let rows: Vec<u8> = (0..n_rows)
                    .flat_map(|_| random_row(ty, n, &mut rng))
                    .collect();
                let (mut wide, mut narrow) = (vec![0.0; n_rows * n_tok], vec![0.0; n_rows * n_tok]);
                // SAFETY: AVX-512 VNNI checked above, AVX2 by `BlockActs::new`.
                unsafe {
                    super::avx2::dot_block16(ty, &rows, rb, &block, &mut wide);
                    super::avx2::dot_block8(ty, &rows, rb, &block, &mut narrow);
                }
                assert_eq!(wide, narrow, "{ty:?} with {n_rows} rows");
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
                let want = dot_row_q8_portable(ty, &row, &a);
                assert_eq!(dot_row_q8(ty, &row, &a), want, "{ty:?}");
                assert_eq!(block_dot(ty, &row, &a), has_block(ty).then_some(want));
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
            if has_block(ty) {
                assert_eq!(block_dot(ty, &row, &a), Some(expected));
            }
            assert_eq!(dot_row_q8_portable(ty, &row, &a), expected);
        }
    }

    #[test]
    fn packed_rows_match_row_dots_within_float_rounding() {
        let mut rng = Rng::new(17);
        let (ty, cols, rows) = (GgmlType::Q4_K, 768, 2 * PACK_ROWS + 3);
        let rb = ty.row_bytes(cols);
        let data: Vec<u8> = (0..rows)
            .flat_map(|_| random_row(ty, cols, &mut rng))
            .collect();
        let packed = pack_rows(&data, rb);
        assert_eq!(packed.len(), 2 * pack_bytes(cols));
        let a = Q8Act::new(&rng.vec(cols));
        let mut got = [0.0; 2 * PACK_ROWS];
        matvec_packed(&packed, cols, &a, &mut got);
        for (r, (row, got)) in data.chunks_exact(rb).zip(got).enumerate() {
            let want = dot_row_q8_portable(ty, row, &a);
            let gb = pack_bytes(cols);
            let portable = packed_portable(&packed[r / PACK_ROWS * gb..][..gb], &a);
            let mut w = vec![0.0; cols];
            dequantize_row(ty, row, &mut w);
            let scale: f32 = w
                .iter()
                .zip(&a.qs)
                .enumerate()
                .map(|(i, (w, &q))| (w * q as f32 * a.d[i / QK]).abs())
                .sum();
            for got in [got, portable[r % PACK_ROWS]] {
                assert!(
                    (got - want).abs() <= 1e-5 * (1.0 + scale),
                    "row {r}: {got} vs {want}"
                );
            }
        }
    }

    /// Weights, scales and activations at the extremes that the packed kernel's 16- and 32-bit
    /// sums have to hold.
    #[test]
    fn packed_kernel_holds_the_largest_sums() {
        let cols = 512;
        let mut row = vec![0xffu8; GgmlType::Q4_K.row_bytes(cols)];
        for b in row.as_chunks_mut::<144>().0 {
            b[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
            b[2..4].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
        }
        let data = row.repeat(PACK_ROWS);
        let a = Q8Act::new(&vec![1.0; cols]);
        let mut got = [0.0; PACK_ROWS];
        matvec_packed(&pack_rows(&data, row.len()), cols, &a, &mut got);
        let want = 15.0 * 63.0 * cols as f32;
        for got in got {
            assert!((got - want).abs() <= 1e-5 * want, "{got} vs {want}");
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
