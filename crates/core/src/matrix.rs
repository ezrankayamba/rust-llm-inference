//! A borrowed, possibly-quantized weight matrix and its CPU matrix-vector product.

use crate::qdot::{self, Q8Act};
use crate::quant::{GgmlType, dequantize_row, dot_row};
use rayon::prelude::*;

/// Row-major matrix of `rows` x `cols` stored in `ty` encoding. Each row is `cols` contiguous
/// elements (GGUF `ne[0]` = cols, `ne[1]` = rows), so `y = W x` is one dot product per row.
#[derive(Clone, Copy, Debug)]
pub struct QMatrix<'a> {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

/// Below this many weight elements a matvec runs on the calling thread; scheduling costs more.
const PARALLEL_MIN_ELEMS: usize = 1 << 16;
const ROWS_PER_TASK: usize = 16;

impl<'a> QMatrix<'a> {
    pub fn new(ty: GgmlType, rows: usize, cols: usize, data: &'a [u8]) -> Self {
        assert_eq!(
            data.len(),
            ty.row_bytes(cols) * rows,
            "matrix byte length mismatch"
        );
        Self {
            ty,
            rows,
            cols,
            data,
        }
    }

    pub fn row_bytes(&self) -> usize {
        self.ty.row_bytes(self.cols)
    }

    pub fn row(&self, r: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        &self.data[r * rb..(r + 1) * rb]
    }

    /// Rows `[start, start + n)` as their own matrix, e.g. one expert of a stacked 3-D tensor.
    pub fn rows_slice(&self, start: usize, n: usize) -> QMatrix<'a> {
        let rb = self.row_bytes();
        QMatrix {
            ty: self.ty,
            rows: n,
            cols: self.cols,
            data: &self.data[start * rb..(start + n) * rb],
        }
    }

    pub fn dequantize_row_into(&self, r: usize, out: &mut [f32]) {
        dequantize_row(self.ty, self.row(r), &mut out[..self.cols]);
    }

    /// `y = W x` on the CPU. Quantized weights use the integer path against activations
    /// quantized once to 8 bits (as llama.cpp does).
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        self.matvec_with(x, y, false);
    }

    /// `precise` keeps activations in f32 (slower): the reference for validating model math.
    pub fn matvec_with(&self, x: &[f32], y: &mut [f32], precise: bool) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        let rb = self.row_bytes();
        let ty = self.ty;
        let act = (!precise && qdot::supports(ty)).then(|| Q8Act::new(x));
        let dot = |ys: &mut [f32], rows: &[u8]| match &act {
            Some(a) => qdot::matvec_q8(ty, rows, rb, a, ys),
            None => {
                for (out, row) in ys.iter_mut().zip(rows.chunks_exact(rb)) {
                    *out = dot_row(ty, row, x);
                }
            }
        };
        if self.rows * self.cols < PARALLEL_MIN_ELEMS {
            return dot(y, self.data);
        }
        y.par_chunks_mut(ROWS_PER_TASK)
            .zip(self.data.par_chunks(rb * ROWS_PER_TASK))
            .for_each(|(ys, rows)| dot(ys, rows));
    }

    /// [`QMatrix::matvec`] reading `packed`, this matrix's [`qdot::pack_rows`] copy, for all but
    /// the rows after its last full group.
    pub fn matvec_packed(&self, packed: &[u8], x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        let (rb, cols) = (self.row_bytes(), self.cols);
        let act = Q8Act::new(x);
        let full = self.rows - self.rows % qdot::PACK_ROWS;
        let (y, tail) = y.split_at_mut(full);
        qdot::matvec_q8(self.ty, &self.data[full * rb..], rb, &act, tail);
        let groups = ROWS_PER_TASK / qdot::PACK_ROWS;
        if self.rows * cols < PARALLEL_MIN_ELEMS {
            return qdot::matvec_packed(packed, cols, &act, y);
        }
        y.par_chunks_mut(groups * qdot::PACK_ROWS)
            .zip(packed.par_chunks(groups * qdot::pack_bytes(cols)))
            .for_each(|(ys, p)| qdot::matvec_packed(p, cols, &act, ys));
    }

    /// `Y = X W^T` for `n` tokens: `xs` is `n x cols` and `ys` is `n x rows`, row-major. Each
    /// parallel task takes a block of weight rows and runs it against every token while those
    /// rows are in cache, so the weights are read from memory once per call, not once per token.
    /// With quantized activations a token's result can differ from [`QMatrix::matvec_with`] by
    /// float rounding (the i8mm kernel sums in a different order); `precise` results match.
    pub fn matmul_with(&self, xs: &[f32], ys: &mut [f32], n: usize, precise: bool) {
        self.matmul_impl(xs, ys, n, precise, false);
    }

    /// [`QMatrix::matmul_with`] over activations quantized with one scale per 256 values
    /// ([`Q8Act::coarse`]): less accurate, and faster for K-quants on ARM.
    pub fn matmul_coarse(&self, xs: &[f32], ys: &mut [f32], n: usize) {
        self.matmul_impl(xs, ys, n, false, true);
    }

    fn matmul_impl(&self, xs: &[f32], ys: &mut [f32], n: usize, precise: bool, coarse: bool) {
        assert_eq!(xs.len(), n * self.cols);
        assert_eq!(ys.len(), n * self.rows);
        if n == 1 {
            return self.matvec_with(xs, ys, precise);
        }
        let (rb, ty, cols, rows) = (self.row_bytes(), self.ty, self.cols, self.rows);
        let acts: Option<Vec<Q8Act>> = (!precise && qdot::supports(ty)).then(|| {
            let quantize = if coarse { Q8Act::coarse } else { Q8Act::new };
            xs.par_chunks_exact(cols).map(quantize).collect()
        });
        let block = acts
            .as_deref()
            .filter(|_| n >= qdot::BLOCK_MIN_TOKENS)
            .and_then(|a| qdot::BlockActs::new(ty, a));
        // Computed as [row][token] so each task owns a contiguous slice, then transposed.
        let mut yt = vec![0.0f32; rows * n];
        yt.par_chunks_mut(ROWS_PER_TASK * n)
            .zip(self.data.par_chunks(rb * ROWS_PER_TASK))
            .for_each(|(out, rows)| {
                // The whole block on the i8mm matrix instructions when the CPU has them.
                if let Some(b) = &block {
                    return qdot::dot_block(ty, rows, rb, b, out);
                }
                let Some(a) = &acts else {
                    for (o, row) in out.chunks_exact_mut(n).zip(rows.chunks_exact(rb)) {
                        for (t, v) in o.iter_mut().enumerate() {
                            *v = dot_row(ty, row, &xs[t * cols..(t + 1) * cols]);
                        }
                    }
                    return;
                };
                // `MAX_TOKENS` tokens per kernel call, so each weight block is unpacked once per
                // group. Tokens outside rows: a group's activations stay in L1 while it meets
                // every row, where the other way round each row would cycle through them all.
                for (g, ac) in a.chunks(qdot::MAX_TOKENS).enumerate() {
                    let t0 = g * qdot::MAX_TOKENS;
                    for (r, row) in rows.chunks_exact(rb).enumerate() {
                        let o = &mut out[r * n + t0..r * n + t0 + ac.len()];
                        qdot::dot_rows_q8(ty, row, ac, o);
                    }
                }
            });
        ys.par_chunks_mut(rows).enumerate().for_each(|(t, y)| {
            for (v, col) in y.iter_mut().zip(yt.chunks_exact(n)) {
                *v = col[t];
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::quantize_q8_0;
    use crate::testutil::Rng;

    #[test]
    fn parallel_matvec_matches_sequential_row_dots() {
        let mut rng = Rng::new(3);
        let (rows, cols) = (300, 512); // large enough to take the parallel path, odd task tail
        let w: Vec<f32> = rng.vec(rows * cols);
        let data = quantize_q8_0(&w);
        let m = QMatrix::new(GgmlType::Q8_0, rows, cols, &data);
        let x = rng.vec(cols);
        let mut y = vec![0.0; rows];
        m.matvec(&x, &mut y);
        let a = Q8Act::new(&x);
        for (r, &got) in y.iter().enumerate() {
            assert_eq!(got, qdot::dot_row_q8(GgmlType::Q8_0, m.row(r), &a));
        }
    }

    #[test]
    fn packed_matvec_matches_matvec_including_rows_past_the_last_group() {
        let mut rng = Rng::new(9);
        let (ty, rows, cols) = (GgmlType::Q4_K, 301, 256); // parallel path, 5 rows left over
        let mut data: Vec<u8> = (0..rows * ty.row_bytes(cols))
            .map(|_| rng.next_u32() as u8)
            .collect();
        for block in data.as_chunks_mut::<144>().0 {
            let d = crate::half::f32_to_f16(0.01).to_le_bytes();
            block[..4].copy_from_slice(&[d[0], d[1], d[0], d[1]]);
        }
        let m = QMatrix::new(ty, rows, cols, &data);
        let packed = qdot::pack_rows(&data, m.row_bytes());
        let x = rng.vec(cols);
        let (mut want, mut got) = (vec![0.0; rows], vec![0.0; rows]);
        m.matvec(&x, &mut want);
        m.matvec_packed(&packed, &x, &mut got);
        // A row's terms reach about 0.01 * 63 * 15 * |x| each and largely cancel.
        let scale = 0.01 * 63.0 * 15.0 * x.iter().map(|v| v.abs()).sum::<f32>();
        for (r, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-5 * scale, "row {r}: {g} vs {w}");
        }
    }

    #[test]
    fn matmul_matches_per_token_matvec() {
        let mut rng = Rng::new(5);
        let (rows, cols, n) = (37, 256, 5);
        let data = quantize_q8_0(&rng.vec(rows * cols));
        let m = QMatrix::new(GgmlType::Q8_0, rows, cols, &data);
        let xs = rng.vec(n * cols);
        for precise in [false, true] {
            let mut ys = vec![0.0; n * rows];
            m.matmul_with(&xs, &mut ys, n, precise);
            for t in 0..n {
                let mut y = vec![0.0; rows];
                m.matvec_with(&xs[t * cols..(t + 1) * cols], &mut y, precise);
                let got = &ys[t * rows..(t + 1) * rows];
                if precise {
                    assert_eq!(got, &y[..], "precise token {t}");
                } else {
                    for (g, w) in got.iter().zip(&y) {
                        assert!(
                            (g - w).abs() <= 1e-5 * (1.0 + w.abs()),
                            "token {t}: {g} vs {w}"
                        );
                    }
                }
            }
        }
    }
}
