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
        let dot = |row: &[u8]| match &act {
            Some(a) => qdot::dot_row_q8(ty, row, a),
            None => dot_row(ty, row, x),
        };
        if self.rows * self.cols < PARALLEL_MIN_ELEMS {
            for (out, row) in y.iter_mut().zip(self.data.chunks_exact(rb)) {
                *out = dot(row);
            }
            return;
        }
        y.par_chunks_mut(ROWS_PER_TASK)
            .zip(self.data.par_chunks(rb * ROWS_PER_TASK))
            .for_each(|(ys, rows)| {
                for (out, row) in ys.iter_mut().zip(rows.chunks_exact(rb)) {
                    *out = dot(row);
                }
            });
    }

    /// `Y = X W^T` for `n` tokens: `xs` is `n x cols` and `ys` is `n x rows`, row-major. Each
    /// parallel task takes a block of weight rows and runs it against every token while those
    /// rows are in cache, so the weights are read from memory once per call, not once per token.
    pub fn matmul_with(&self, xs: &[f32], ys: &mut [f32], n: usize, precise: bool) {
        assert_eq!(xs.len(), n * self.cols);
        assert_eq!(ys.len(), n * self.rows);
        if n == 1 {
            return self.matvec_with(xs, ys, precise);
        }
        let (rb, ty, cols, rows) = (self.row_bytes(), self.ty, self.cols, self.rows);
        let acts: Option<Vec<Q8Act>> = (!precise && qdot::supports(ty))
            .then(|| xs.chunks_exact(cols).map(Q8Act::new).collect());
        // One row against every token; quantized rows go `MAX_TOKENS` tokens per kernel call,
        // so each weight block is unpacked once per group rather than once per token.
        let dots = |row: &[u8], o: &mut [f32]| match &acts {
            Some(a) => {
                for (oc, ac) in o
                    .chunks_mut(qdot::MAX_TOKENS)
                    .zip(a.chunks(qdot::MAX_TOKENS))
                {
                    qdot::dot_rows_q8(ty, row, ac, oc);
                }
            }
            None => {
                for (t, v) in o.iter_mut().enumerate() {
                    *v = dot_row(ty, row, &xs[t * cols..(t + 1) * cols]);
                }
            }
        };
        // Computed as [row][token] so each task owns a contiguous slice, then transposed.
        let mut yt = vec![0.0f32; rows * n];
        yt.par_chunks_mut(ROWS_PER_TASK * n)
            .zip(self.data.par_chunks(rb * ROWS_PER_TASK))
            .for_each(|(out, block)| {
                for (o, row) in out.chunks_exact_mut(n).zip(block.chunks_exact(rb)) {
                    dots(row, o);
                }
            });
        for (r, col) in yt.chunks_exact(n).enumerate() {
            for (t, &v) in col.iter().enumerate() {
                ys[t * rows + r] = v;
            }
        }
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
                assert_eq!(
                    &ys[t * rows..(t + 1) * rows],
                    &y[..],
                    "precise={precise} token {t}"
                );
            }
        }
    }
}
