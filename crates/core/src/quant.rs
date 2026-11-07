//! GGML tensor element types and their block layouts (little-endian, as stored in GGUF).
//!
//! Layouts follow ggml's `block_*` structs exactly; the dequantizers are the scalar reference
//! that every fused/SIMD kernel is tested against.

use crate::half::{bf16_to_f32, f16_to_f32};
use crate::simd;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
}

pub const QK_K: usize = 256;

impl GgmlType {
    /// Maps the on-disk `ggml_type` id. `None` for types this engine does not implement.
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            30 => Self::BF16,
            _ => return None,
        })
    }

    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::BF16 => 30,
        }
    }

    /// Elements per block.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K => QK_K,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 2 + 16,
            Self::Q4_1 => 2 + 2 + 16,
            Self::Q5_0 => 2 + 4 + 16,
            Self::Q5_1 => 2 + 2 + 4 + 16,
            Self::Q8_0 => 2 + 32,
            Self::Q4_K => 2 + 2 + 12 + 128,
            Self::Q5_K => 2 + 2 + 12 + 32 + 128,
            Self::Q6_K => 128 + 64 + 16 + 2,
        }
    }

    /// Bytes for `n` elements. `n` must be a multiple of the block size.
    pub fn row_bytes(self, n: usize) -> usize {
        assert!(
            n.is_multiple_of(self.block_size()),
            "{n} elements is not a whole number of {self:?} blocks"
        );
        n / self.block_size() * self.block_bytes()
    }
}

#[inline]
fn le16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

/// K-quant 6-bit scale/min unpacking (ggml `get_scale_min_k4`).
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantizes one block of `ty` from `b` into `y` (`y.len() == ty.block_size()`).
pub fn dequantize_block(ty: GgmlType, b: &[u8], y: &mut [f32]) {
    match ty {
        GgmlType::F32 => y[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        GgmlType::F16 => y[0] = f16_to_f32(le16(b, 0)),
        GgmlType::BF16 => y[0] = bf16_to_f32(le16(b, 0)),
        GgmlType::Q4_0 => {
            let d = f16_to_f32(le16(b, 0));
            let qs = &b[2..18];
            for j in 0..16 {
                y[j] = ((qs[j] & 0xf) as i32 - 8) as f32 * d;
                y[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
            }
        }
        GgmlType::Q4_1 => {
            let (d, m) = (f16_to_f32(le16(b, 0)), f16_to_f32(le16(b, 2)));
            let qs = &b[4..20];
            for j in 0..16 {
                y[j] = (qs[j] & 0xf) as f32 * d + m;
                y[j + 16] = (qs[j] >> 4) as f32 * d + m;
            }
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            // Q5_0: d, qh, qs with values offset by -16. Q5_1: d, m, qh, qs, unsigned plus m.
            let d = f16_to_f32(le16(b, 0));
            let (m, at) = if ty == GgmlType::Q5_1 {
                (f16_to_f32(le16(b, 2)), 4)
            } else {
                (0.0, 2)
            };
            let offset = if ty == GgmlType::Q5_0 { 16 } else { 0 };
            let qh = u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
            let qs = &b[at + 4..at + 20];
            for j in 0..16 {
                let h0 = ((qh >> j) << 4) & 0x10;
                let h1 = (qh >> (j + 12)) & 0x10;
                y[j] = (((qs[j] & 0xf) as u32 | h0) as i32 - offset) as f32 * d + m;
                y[j + 16] = (((qs[j] >> 4) as u32 | h1) as i32 - offset) as f32 * d + m;
            }
        }
        GgmlType::Q8_0 => {
            let d = f16_to_f32(le16(b, 0));
            for (o, &q) in y.iter_mut().zip(&b[2..34]) {
                *o = q as i8 as f32 * d;
            }
        }
        GgmlType::Q4_K => {
            let d = f16_to_f32(le16(b, 0));
            let dmin = f16_to_f32(le16(b, 2));
            let scales = &b[4..16];
            let qs = &b[16..144];
            for (g, (q, out)) in qs
                .as_chunks::<32>()
                .0
                .iter()
                .zip(y.as_chunks_mut::<64>().0.iter_mut())
                .enumerate()
            {
                let (sc1, m1) = scale_min_k4(2 * g, scales);
                let (sc2, m2) = scale_min_k4(2 * g + 1, scales);
                let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
                let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
                for l in 0..32 {
                    out[l] = d1 * (q[l] & 0xf) as f32 - m1;
                    out[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
                }
            }
        }
        GgmlType::Q5_K => {
            let d = f16_to_f32(le16(b, 0));
            let dmin = f16_to_f32(le16(b, 2));
            let scales = &b[4..16];
            let qh = &b[16..48];
            let ql = &b[48..176];
            for (g, (q, out)) in ql
                .as_chunks::<32>()
                .0
                .iter()
                .zip(y.as_chunks_mut::<64>().0.iter_mut())
                .enumerate()
            {
                let (sc1, m1) = scale_min_k4(2 * g, scales);
                let (sc2, m2) = scale_min_k4(2 * g + 1, scales);
                let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
                let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
                let (u1, u2) = (1u8 << (2 * g), 2u8 << (2 * g));
                for l in 0..32 {
                    let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    out[l] = d1 * ((q[l] & 0xf) + hi1) as f32 - m1;
                    out[l + 32] = d2 * ((q[l] >> 4) + hi2) as f32 - m2;
                }
            }
        }
        GgmlType::Q6_K => {
            let ql = &b[0..128];
            let qh = &b[128..192];
            let sc = &b[192..208];
            let d = f16_to_f32(le16(b, 208));
            for n in 0..2 {
                let (ql, qh, sc) = (&ql[64 * n..], &qh[32 * n..], &sc[8 * n..]);
                let out = &mut y[128 * n..128 * (n + 1)];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    out[l] = d * (sc[is] as i8 as f32) * q1 as f32;
                    out[l + 32] = d * (sc[is + 2] as i8 as f32) * q2 as f32;
                    out[l + 64] = d * (sc[is + 4] as i8 as f32) * q3 as f32;
                    out[l + 96] = d * (sc[is + 6] as i8 as f32) * q4 as f32;
                }
            }
        }
    }
}

/// Dequantizes a full row of `y.len()` elements.
pub fn dequantize_row(ty: GgmlType, src: &[u8], y: &mut [f32]) {
    if let Some(row) = as_f32(ty, src) {
        y.copy_from_slice(&row[..y.len()]);
        return;
    }
    let (bs, bb) = (ty.block_size(), ty.block_bytes());
    for (b, out) in src.chunks_exact(bb).zip(y.chunks_exact_mut(bs)) {
        dequantize_block(ty, b, out);
    }
}

fn as_f32(ty: GgmlType, src: &[u8]) -> Option<&[f32]> {
    if ty != GgmlType::F32 {
        return None;
    }
    // SAFETY: any bit pattern is a valid f32; alignment is checked by `align_to`.
    let (pre, mid, _) = unsafe { src.align_to::<f32>() };
    pre.is_empty().then_some(mid)
}

/// Dot product of one encoded row with f32 activations `x`.
pub fn dot_row(ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
    if let Some(w) = as_f32(ty, row) {
        return simd::dot(&w[..x.len()], x);
    }
    // Half-precision rows are converted in chunks so the dot itself stays vectorized.
    let (bs, bb) = match ty {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => (64, 64 * ty.block_bytes()),
        _ => (ty.block_size(), ty.block_bytes()),
    };
    let mut tmp = [0f32; QK_K];
    let mut sum = 0.0;
    for (b, xs) in row.chunks(bb).zip(x.chunks(bs)) {
        let t = &mut tmp[..xs.len()];
        match ty {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => dequantize_row(ty, b, t),
            _ => dequantize_block(ty, b, t),
        }
        sum += simd::dot(t, xs);
    }
    sum
}

/// Reference quantizer for Q8_0 (ggml `quantize_row_q8_0_ref`).
pub fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    use crate::half::f32_to_f16;
    let mut out = Vec::with_capacity(GgmlType::Q8_0.row_bytes(x.len()));
    for blk in x.as_chunks::<32>().0 {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend(blk.iter().map(|v| (v * id).round() as i8 as u8));
    }
    out
}

/// Reference quantizer for Q4_0 (ggml `quantize_row_q4_0_ref`).
pub fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    use crate::half::f32_to_f16;
    let mut out = Vec::with_capacity(GgmlType::Q4_0.row_bytes(x.len()));
    for blk in x.as_chunks::<32>().0 {
        let max = blk
            .iter()
            .copied()
            .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        let q = |v: f32| ((v * id + 8.5) as i32).min(15) as u8;
        out.extend((0..16).map(|j| q(blk[j]) | (q(blk[j + 16]) << 4)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;
    use crate::testutil::Rng;

    /// Random block bytes with small, finite f16 scale fields so values stay well-conditioned.
    fn random_row(ty: GgmlType, n: usize, rng: &mut Rng) -> Vec<u8> {
        let mut row: Vec<u8> = (0..ty.row_bytes(n)).map(|_| rng.next_u32() as u8).collect();
        let bb = ty.block_bytes();
        let scale_offsets: &[usize] = match ty {
            GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q5_0 => &[0],
            GgmlType::Q4_1 | GgmlType::Q5_1 => &[0, 2],
            GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
            GgmlType::Q6_K => &[208],
            GgmlType::F16 => {
                let v: Vec<f32> = rng.vec(n);
                return v
                    .iter()
                    .flat_map(|&f| f32_to_f16(f).to_le_bytes())
                    .collect();
            }
            GgmlType::BF16 => {
                let v: Vec<f32> = rng.vec(n);
                return v
                    .iter()
                    .flat_map(|&f| ((f.to_bits() >> 16) as u16).to_le_bytes())
                    .collect();
            }
            GgmlType::F32 => {
                return rng.vec(n).iter().flat_map(|f| f.to_le_bytes()).collect();
            }
        };
        for blk in row.chunks_exact_mut(bb) {
            for &o in scale_offsets {
                let s = f32_to_f16(rng.f32() * 0.01);
                blk[o..o + 2].copy_from_slice(&s.to_le_bytes());
            }
        }
        row
    }

    const ALL: [GgmlType; 11] = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ];

    #[test]
    fn dot_row_matches_dequantized_scalar_dot_for_every_type() {
        let mut rng = Rng::new(42);
        let n = 512;
        for ty in ALL {
            let row = random_row(ty, n, &mut rng);
            let x = rng.vec(n);
            let mut deq = vec![0.0; n];
            dequantize_row(ty, &row, &mut deq);
            let reference = simd::dot_scalar(&deq, &x);
            let got = dot_row(ty, &row, &x);
            assert!(
                (got - reference).abs() <= 1e-4 * (1.0 + reference.abs()),
                "{ty:?}: {got} vs {reference}"
            );
        }
    }

    #[test]
    fn q8_0_roundtrip_error_is_within_half_a_step() {
        let mut rng = Rng::new(1);
        let x = rng.vec(256);
        let mut y = vec![0.0; 256];
        dequantize_row(GgmlType::Q8_0, &quantize_q8_0(&x), &mut y);
        for (blk_x, blk_y) in x.chunks(32).zip(y.chunks(32)) {
            let step = blk_x.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            for (a, b) in blk_x.iter().zip(blk_y) {
                assert!((a - b).abs() <= step * 0.51 + 1e-3 * step, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn q4_0_roundtrip_error_is_within_one_step() {
        let mut rng = Rng::new(2);
        let x = rng.vec(256);
        let mut y = vec![0.0; 256];
        dequantize_row(GgmlType::Q4_0, &quantize_q4_0(&x), &mut y);
        for (blk_x, blk_y) in x.chunks(32).zip(y.chunks(32)) {
            let step = blk_x.iter().fold(0f32, |m, v| m.max(v.abs())) / 8.0;
            for (a, b) in blk_x.iter().zip(blk_y) {
                assert!((a - b).abs() <= step * 1.01, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn q4_k_layout_orders_low_nibbles_before_high_nibbles_per_64() {
        // d=1, dmin=0, every 6-bit scale = 1, mins = 0: output equals the raw nibble.
        let mut b = vec![0u8; GgmlType::Q4_K.block_bytes()];
        b[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        for s in &mut b[4..8] {
            *s = 1; // scales 0..3
        }
        for s in &mut b[12..16] {
            *s = 0x01; // scales 4..7 low nibble = 1, mins 4..7 = 0
        }
        for (i, q) in b[16..144].iter_mut().enumerate() {
            *q = ((i % 16) as u8) | (((15 - i % 16) as u8) << 4);
        }
        let mut y = vec![0.0; QK_K];
        dequantize_block(GgmlType::Q4_K, &b, &mut y);
        for g in 0..4 {
            for l in 0..32 {
                assert_eq!(y[64 * g + l], (l % 16) as f32);
                assert_eq!(y[64 * g + 32 + l], (15 - l % 16) as f32);
            }
        }
    }
}
