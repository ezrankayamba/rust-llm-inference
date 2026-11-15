//! CUDA matvec throughput, back to back over distinct matrices as in decode: the f32 kernel
//! and, where there is one, the int8 kernel over quantized activations.
//! usage: cargo run --release -p rli-core --features cuda --example bench_cuda

#[cfg(feature = "cuda")]
fn main() {
    use rli_core::GgmlType;
    use rli_core::cuda::{At, Gpu};
    use std::time::Instant;
    // 1 GB of weights, more than the caches hold. Zeroed: the kernels' time does not depend
    // on the values.
    let mem = vec![0u8; 1 << 30];
    let gpu = Gpu::new(&mem, &[(0, mem.len())]).expect("CUDA device");
    let (x, y) = (gpu.alloc(14336), gpu.alloc(14336));
    let xq = gpu.alloc_q(14336);
    println!("{}", gpu.name());
    for (ty, rows, cols) in [
        (GgmlType::Q4_K, 4096, 4096),
        (GgmlType::Q4_K, 14336, 4096),
        (GgmlType::Q4_K, 4096, 14336),
        (GgmlType::Q6_K, 4096, 4096),
        (GgmlType::Q6_K, 4096, 14336),
        (GgmlType::Q4_K, 2048, 2048),
        (GgmlType::Q4_K, 512, 2048),
        (GgmlType::Q4_K, 256, 256),
        (GgmlType::Q8_0, 4096, 4096),
        (GgmlType::F16, 4096, 4096),
    ] {
        let bytes = ty.row_bytes(cols) * rows;
        let count = (mem.len() / bytes).min(4096);
        let time = |int8: bool| {
            let run = || {
                let mut b = gpu.batch();
                for i in 0..count {
                    if int8 {
                        b.matmul_q(ty, rows, cols, i * bytes, &xq, At(&y, 0), 1);
                    } else {
                        b.matvec(ty, rows, cols, i * bytes, At(&x, 0), At(&y, 0));
                    }
                }
                b.finish();
            };
            run();
            let best = (0..5)
                .map(|_| {
                    let t = Instant::now();
                    run();
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::MAX, f64::min);
            format!(
                "{:7.1} us, {:6.1} GB/s",
                best / count as f64 * 1e6,
                (count * bytes) as f64 / best / 1e9
            )
        };
        let q = if gpu.has_int8(ty, cols, 0) {
            format!("   int8 {}", time(true))
        } else {
            String::new()
        };
        println!("{ty:?} {rows}x{cols}: f32 {}{q}", time(false));
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}
