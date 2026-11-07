//! Metal matvec latency (kernel throughput on a big matrix vs. fixed per-call overhead), and
//! batched matmul throughput for prompt processing.
//! usage: cargo run --release -p rli-core --features metal --example bench_metal

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rli_core::GgmlType;
    use rli_core::metal::Gpu;
    use std::time::Instant;
    let page = 16 * 1024;
    let layout = std::alloc::Layout::from_size_align(64 << 20, page).unwrap();
    let mem = unsafe { std::slice::from_raw_parts_mut(std::alloc::alloc_zeroed(layout), 64 << 20) };
    let gpu = unsafe { Gpu::new(mem) }.unwrap();
    for (ty, rows, cols) in [
        (GgmlType::Q4_0, 5632, 2048),
        (GgmlType::Q6_K, 32000, 2048),
        (GgmlType::Q4_0, 64, 64),
    ] {
        let x = vec![0.5; cols];
        let mut y = vec![0.0; rows];
        gpu.matvec(ty, rows, cols, 0, &x, &mut y);
        let iters = 100;
        let t = Instant::now();
        for _ in 0..iters {
            gpu.matvec(ty, rows, cols, 0, &x, &mut y);
        }
        let s = t.elapsed().as_secs_f64() / iters as f64;
        let bytes = ty.row_bytes(cols) * rows;
        println!(
            "{ty:?} {rows}x{cols}: {:.3} ms  {:.1} GB/s",
            s * 1e3,
            bytes as f64 / s / 1e9
        );
    }
    for (ty, rows, cols, n) in [
        (GgmlType::Q4_0, 5632, 2048, 256),
        (GgmlType::Q4_K, 4864, 1024, 256),
    ] {
        let x = vec![0.5; n * cols];
        let mut y = vec![0.0; n * rows];
        gpu.matmul(ty, rows, cols, 0, &x, &mut y, n);
        let iters = 10;
        let t = Instant::now();
        for _ in 0..iters {
            gpu.matmul(ty, rows, cols, 0, &x, &mut y, n);
        }
        let s = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "matmul {ty:?} {rows}x{cols} n={n}: {:.3} ms  {:.0} GFLOP/s  ({:.0} tok/s for this matrix)",
            s * 1e3,
            2.0 * (rows * cols * n) as f64 / s / 1e9,
            n as f64 / s
        );
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {}
