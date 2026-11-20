//! CUDA matvec throughput, back to back over distinct matrices as in decode: the f32 kernel
//! and, where there is one, the int8 kernel over quantized activations. Then decode attention.
//! usage: cargo run --release -p rli-core --features cuda --example bench_cuda

#[cfg(feature = "cuda")]
fn main() {
    use rli_core::GgmlType;
    use rli_core::cuda::{At, AttnParams, Batch, Gpu};
    use std::time::Instant;
    // 1 GB of weights, more than the caches hold. Zeroed: the kernels' time does not depend
    // on the values.
    let mem = vec![0u8; 1 << 30];
    let gpu = Gpu::new(&mem, &[(0, mem.len())]).expect("CUDA device");
    let (x, y) = (gpu.alloc(14336), gpu.alloc(131072));
    let xq = gpu.alloc_q(14336);
    println!("{}", gpu.name());
    for (ty, rows, cols) in [
        (GgmlType::Q4_K, 4096, 4096),
        (GgmlType::Q4_K, 14336, 4096),
        (GgmlType::Q4_K, 4096, 14336),
        (GgmlType::Q6_K, 4096, 4096),
        (GgmlType::Q6_K, 4096, 14336),
        (GgmlType::Q6_K, 128256, 2048),
        (GgmlType::Q4_K, 2048, 2048),
        (GgmlType::Q4_K, 512, 2048),
        (GgmlType::Q4_K, 256, 256),
        (GgmlType::Q8_0, 4096, 4096),
        (GgmlType::F16, 4096, 4096),
        (GgmlType::F32, 128, 2048),
    ] {
        let bytes = ty.row_bytes(cols) * rows;
        let count = (mem.len() / bytes).min(4096);
        let time = |int8: bool| {
            let run = || {
                let mut b = gpu.batch();
                for i in 0..count {
                    if int8 {
                        b.matmul_q(ty, rows, cols, i * bytes, &xq, At(&y, 0), 1, false);
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
    // The f16 GEMM of long prompts: writing a matrix out as f16, and multiplying it by 512
    // and 128 tokens (the shapes of Mistral-7B and Llama-3.2-1B). Random activations, so the
    // products are ordinary numbers.
    for n_tok in [512, 256] {
        if !gpu.batch().has_gemm(4096, n_tok) {
            continue;
        }
        let mut seed = 5u32;
        let xs: Vec<f32> = (0..n_tok * 14336)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / 8388608.0 - 1.0
            })
            .collect();
        let xf = gpu.upload(&xs);
        let (wh, xh) = (gpu.alloc_half(14336 * 4096), gpu.alloc_half(n_tok * 14336));
        let yb = gpu.alloc(n_tok * 14336);
        let mut b = gpu.batch();
        b.to_f16(At(&xf, 0), n_tok * 14336, &xh);
        b.finish();
        for (ty, rows, cols) in [
            (GgmlType::Q4_K, 4096, 4096),
            (GgmlType::Q4_K, 14336, 4096),
            (GgmlType::Q4_K, 4096, 14336),
            (GgmlType::Q4_K, 2048, 2048),
            (GgmlType::Q4_K, 8192, 2048),
            (GgmlType::Q4_K, 2048, 8192),
        ] {
            let bytes = ty.row_bytes(cols) * rows;
            let count = (mem.len() / bytes).min(16);
            // `gemm`: 1 for the product, 2 for the one that adds into y, where it differs.
            let time = |dequant: bool, gemm: u8| {
                let run = || {
                    let mut b = gpu.batch();
                    for i in 0..count {
                        if dequant {
                            b.dequant(ty, rows, cols, i * bytes, &wh);
                        }
                        match gemm {
                            1 => b.gemm(rows, cols, &wh, &xh, At(&yb, 0), n_tok),
                            2 => b.gemm_add(rows, cols, &wh, &xh, At(&yb, 0), n_tok),
                            _ => {}
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
                best / count as f64 * 1e6
            };
            let (d, g) = (time(true, 0), time(false, 1));
            let tflops = 2.0 * (rows * cols * n_tok) as f64 / g / 1e6;
            let add = if gpu.batch().has_gemm_add(rows, cols, n_tok) {
                format!(", adding {:7.1} us", time(false, 2))
            } else {
                String::new()
            };
            println!(
                "gemm {ty:?} {rows}x{cols} x {n_tok} tokens: dequant {d:7.1} us, gemm {g:7.1} us ({tflops:.0} TFLOPS){add}"
            );
        }
    }
    for n_tok in [32, 64, 128, 256, 512] {
        let xf = gpu.alloc(n_tok * 14336);
        let xqm = gpu.alloc_q(n_tok * 14336);
        let yb = gpu.alloc(n_tok * 14336);
        let mut b = gpu.batch();
        b.quantize(At(&xf, 0), n_tok * 14336, &xqm);
        b.finish();
        for (ty, rows, cols) in [
            (GgmlType::Q4_K, 4096, 4096),
            (GgmlType::Q4_K, 14336, 4096),
            (GgmlType::Q4_K, 4096, 14336),
            (GgmlType::Q6_K, 4096, 4096),
            (GgmlType::Q6_K, 4096, 14336),
            (GgmlType::Q4_K, 2048, 2048),
            (GgmlType::Q4_K, 8192, 2048),
            (GgmlType::Q4_K, 2048, 8192),
        ] {
            let bytes = ty.row_bytes(cols) * rows;
            let count = (mem.len() / bytes).min(16);
            let xh = gpu.alloc_half(n_tok * 14336);
            // In int8, or (RLI_DENSE) on f16 tensor cores with the weights decoded in the block.
            let dense = std::env::var_os("RLI_DENSE").is_some();
            let run = || {
                let mut b = gpu.batch();
                for i in 0..count {
                    if dense {
                        b.gemm_dense(ty, rows, cols, i * bytes, &xh, At(&yb, 0), n_tok, true);
                    } else {
                        b.matmul_q(ty, rows, cols, i * bytes, &xqm, At(&yb, 0), n_tok, false);
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
            println!(
                "gemm int8 {ty:?} {rows}x{cols} x {n_tok} tokens: {:7.1} us",
                best / count as f64 * 1e6
            );
        }
    }
    // Launch cost: small matvecs queued one by one, and replayed as a captured graph.
    for (rows, cols) in [(256, 256), (2048, 2048), (4096, 4096)] {
        let ty = GgmlType::Q4_K;
        let bytes = ty.row_bytes(cols) * rows;
        let count = (mem.len() / bytes).min(1024);
        let queue = |b: &mut Batch<'_>| {
            for i in 0..count {
                b.matmul_q(ty, rows, cols, i * bytes, &xq, At(&y, 0), 1, false);
            }
        };
        let mut graph = None;
        let best = |f: &mut dyn FnMut()| {
            f();
            (0..5)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::MAX, f64::min)
                / count as f64
                * 1e6
        };
        let direct = best(&mut || {
            let mut b = gpu.batch();
            queue(&mut b);
            b.finish();
        });
        let replayed = best(&mut || {
            gpu.run_graph(&mut graph, queue);
            gpu.sync();
        });
        println!("launches {rows}x{cols}: direct {direct:.2} us, graph {replayed:.2} us");
    }
    // Decode attention over 16 layers' caches in turn, by context length.
    for (n_head, n_kv, head_dim) in [(32, 8, 64), (32, 8, 128), (32, 4, 128)] {
        let max_ctx = 4096;
        let kv_dim = n_kv * head_dim;
        let kv: Vec<_> = (0..32).map(|_| gpu.alloc(max_ctx * kv_dim)).collect();
        let (q, out) = (gpu.alloc(n_head * head_dim), gpu.alloc(n_head * head_dim));
        let scores = gpu.alloc(Batch::attention_scratch(n_head, head_dim, max_ctx));
        print!("attention {n_head}/{n_kv} heads of {head_dim}, us:");
        for n_pos in [16, 64, 128, 512, 2048, 4096] {
            let p = AttnParams {
                n_head: n_head as u32,
                n_kv: n_kv as u32,
                head_dim: head_dim as u32,
                n_tok: 1,
                n_pos,
                kv_dim: kv_dim as u32,
                max_ctx: max_ctx as u32,
                scale: 0.1,
                half: false,
                pos_dev: None,
            };
            let run = || {
                let mut b = gpu.batch();
                for _ in 0..16 {
                    for l in 0..16 {
                        b.attention(
                            &q,
                            &kv[2 * l],
                            &kv[2 * l + 1],
                            &out,
                            &scores,
                            &p,
                            None,
                            None,
                        );
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
            print!("  {n_pos}: {:.1}", best / 256.0 * 1e6);
        }
        println!();
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}
