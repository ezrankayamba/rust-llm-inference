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
}
