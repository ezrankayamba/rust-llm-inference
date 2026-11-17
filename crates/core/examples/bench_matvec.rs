//! Matvec (decode) and 64-token matmul (prompt, also with one activation scale per 256 values:
//! "coarse") throughput per weight type on a
//! TinyLlama-FFN-sized matrix (5632 x 2048).
//! usage: cargo run --release -p rli-core --example bench_matvec

use rli_core::{GgmlType, QMatrix};
use std::time::Instant;

fn main() {
    let (rows, cols) = (5632, 2048);
    let mut s = 1u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s as u8
    };
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i * 7919) % 200) as f32 / 100.0 - 1.0)
        .collect();
    let mut y = vec![0.0; rows];
    for ty in [
        GgmlType::F16,
        GgmlType::Q8_0,
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ] {
        let mut data: Vec<u8> = (0..ty.row_bytes(cols) * rows).map(|_| rnd()).collect();
        // Keep f16 fields finite: zero the high byte of every 16-bit word (small positive values).
        for c in data.as_chunks_mut::<2>().0 {
            c[1] &= 0x3b;
        }
        let m = QMatrix::new(ty, rows, cols, &data);
        m.matvec(&x, &mut y);
        let iters = 50;
        let t = Instant::now();
        for _ in 0..iters {
            m.matvec(&x, &mut y);
        }
        let secs = t.elapsed().as_secs_f64() / iters as f64;
        let n = 64;
        let xs: Vec<f32> = (0..n).flat_map(|_| x.iter().copied()).collect();
        let mut ys = vec![0.0; n * rows];
        m.matmul_with(&xs, &mut ys, n, false);
        let t = Instant::now();
        for _ in 0..5 {
            m.matmul_with(&xs, &mut ys, n, false);
        }
        let mm = t.elapsed().as_secs_f64() / 5.0;
        let t = Instant::now();
        for _ in 0..5 {
            m.matmul_coarse(&xs, &mut ys, n);
        }
        let coarse = t.elapsed().as_secs_f64() / 5.0;
        println!(
            "{:>5?}: matvec {:7.3} ms  {:6.1} GB/s  ({:.1} MB) | matmul n={n} {:7.2} ms  {:6.1} GOP/s | coarse {:7.2} ms",
            ty,
            secs * 1e3,
            data.len() as f64 / secs / 1e9,
            data.len() as f64 / 1e6,
            mm * 1e3,
            2.0 * (rows * cols * n) as f64 / mm / 1e9,
            coarse * 1e3
        );
    }
}
