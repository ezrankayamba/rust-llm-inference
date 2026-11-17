//! What each piece of an Intel AMX step costs: the tile loads (activations at the stride the
//! block kernel reads them with, or packed), the multiply and the store.
//! usage: cargo run --release -p rli-core --example bench_amx   (Linux on an AMX Xeon)

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn main() {
    use std::arch::asm;
    use std::arch::x86_64::__cpuid_count;
    use std::time::Instant;

    let ret: i64;
    // SAFETY: arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA), two integers in.
    unsafe {
        asm!("syscall", inlateout("rax") 158i64 => ret, in("rdi") 0x1023, in("rsi") 18,
            lateout("rcx") _, lateout("r11") _, options(nostack));
    }
    if __cpuid_count(7, 0).edx >> 24 & 3 != 3 || ret != 0 {
        println!("skipped: no AMX on this CPU");
        return;
    }
    #[repr(C, align(64))]
    struct Cfg([u8; 64]);
    // tmm0: sums, tmm1: activations (16 tokens), tmm2: weights (four per row and tile row).
    let cfg = |cols: usize| {
        let mut c = [0u8; 64];
        c[0] = 1;
        for (tile, (colsb, rows)) in [(64, 16), (cols, 16), (64, cols / 4)].iter().enumerate() {
            c[16 + 2 * tile] = *colsb as u8;
            c[48 + tile] = *rows as u8;
        }
        Cfg(c)
    };
    const CHUNKS: usize = 64;
    let w = vec![1u8; CHUNKS * 1024];
    let mut sums = [[0i32; 16]; 16];
    let iters = 200_000;
    for cols in [32usize, 64] {
        let c = cfg(cols);
        // SAFETY: AMX checked above; every pointer below stays inside its buffer.
        unsafe { asm!("ldtilecfg [{}]", in(reg) &c, options(nostack, readonly)) };
        let time = |name: &str, f: &mut dyn FnMut(usize)| {
            for i in 0..1000 {
                f(i);
            }
            let t = Instant::now();
            for i in 0..iters {
                f(i);
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            println!("{cols:2} columns | {name:44} {ns:7.1} ns");
        };
        let x = vec![1i8; 16 * 8192 + 64];
        let (xp, wp, sp) = (x.as_ptr(), w.as_ptr(), sums.as_mut_ptr());
        unsafe {
            asm!("tileloadd tmm1, [{x} + {s}]", "tileloadd tmm2, [{w} + {l}]",
                x = in(reg) xp, s = in(reg) cols, w = in(reg) wp, l = in(reg) 64usize,
                out("tmm1") _, out("tmm2") _, options(nostack));
        }
        time("zero, multiply", &mut |_| unsafe {
            asm!("tilezero tmm0", "tdpbsud tmm0, tmm1, tmm2", out("tmm0") _, options(nostack));
        });
        time("zero, multiply, store", &mut |_| unsafe {
            asm!("tilezero tmm0", "tdpbsud tmm0, tmm1, tmm2", "tilestored [{o} + {l}], tmm0",
                o = in(reg) sp, l = in(reg) 64usize, out("tmm0") _, options(nostack));
        });
        time("load weights, zero, multiply", &mut |i| unsafe {
            asm!("tileloadd tmm2, [{w} + {l}]", "tilezero tmm0", "tdpbsud tmm0, tmm1, tmm2",
                w = in(reg) wp.add(i % CHUNKS * 1024), l = in(reg) 64usize,
                out("tmm0") _, out("tmm2") _, options(nostack));
        });
        for stride in [cols, 2048, 4096, 8192] {
            let name = format!("load activations (stride {stride}), zero, multiply");
            time(&name, &mut |i| unsafe {
                asm!("tileloadd tmm1, [{x} + {s}]", "tilezero tmm0", "tdpbsud tmm0, tmm1, tmm2",
                    x = in(reg) xp.add(i % (stride / cols).min(CHUNKS) * cols), s = in(reg) stride,
                    out("tmm0") _, out("tmm1") _, options(nostack));
            });
        }
        for stride in [cols, 2048, 8192] {
            let name = format!("whole step (stride {stride}), sums read at once");
            let mut total = 0i32;
            time(&name, &mut |i| unsafe {
                asm!("tileloadd tmm2, [{w} + {l}]", "tilezero tmm0",
                    "tileloadd tmm1, [{x} + {s}]", "tdpbsud tmm0, tmm1, tmm2",
                    "tilestored [{o} + {l}], tmm0",
                    w = in(reg) wp.add(i % CHUNKS * 1024), l = in(reg) 64usize,
                    x = in(reg) xp.add(i % (stride / cols).min(CHUNKS) * cols), s = in(reg) stride,
                    o = in(reg) sp, out("tmm0") _, out("tmm1") _, out("tmm2") _, options(nostack));
                total = total.wrapping_add(sp.read_volatile()[0]);
            });
            std::hint::black_box(total);
        }
    }
    // SAFETY: AMX, as above.
    unsafe { asm!("tilerelease", options(nostack, nomem)) };
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() {
    println!("skipped: needs Linux on x86-64");
}
