//! Metal matvec/matmul kernels against the CPU reference (f32 activations) for every supported
//! weight type.
#![cfg(all(feature = "metal", target_os = "macos"))]

use rli_core::half::f32_to_f16;
use rli_core::metal::Gpu;
use rli_core::{GgmlType, QMatrix};
use std::alloc::{Layout, alloc_zeroed, dealloc};

const PAGE: usize = 16 * 1024;

/// Page-aligned, page-rounded memory, like a file mapping.
struct Pages(*mut u8, Layout);
impl Pages {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len.div_ceil(PAGE) * PAGE, PAGE).unwrap();
        Pages(unsafe { alloc_zeroed(layout) }, layout)
    }
    fn bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0, self.1.size()) }
    }
}
impl Drop for Pages {
    fn drop(&mut self) {
        unsafe { dealloc(self.0, self.1) }
    }
}

const ROWS: usize = 37; // not a multiple of the 4-row threadgroup
const COLS: usize = 512;
const TYPES: [GgmlType; 8] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q4_0,
    GgmlType::Q5_0,
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
];

/// Random `ROWS x COLS` matrices of every type, back to back at 32-byte-aligned offsets (as in a
/// GGUF), with small finite f16 scales.
fn fixture() -> (Pages, Vec<usize>) {
    let (rows, cols, types) = (ROWS, COLS, TYPES);
    // Lay all matrices out back to back at 32-byte-aligned offsets, as in a GGUF.
    let mut offsets = Vec::new();
    let mut end = 0usize;
    for ty in types {
        end = end.div_ceil(32) * 32;
        offsets.push(end);
        end += ty.row_bytes(cols) * rows;
    }
    let mut mem = Pages::new(end);
    let buf = mem.bytes();
    let mut seed = 0x1234_5678u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for (&ty, &off) in types.iter().zip(&offsets) {
        let n = ty.row_bytes(cols) * rows;
        let region = &mut buf[off..off + n];
        match ty {
            GgmlType::F32 => region.as_chunks_mut::<4>().0.iter_mut().for_each(|c| {
                c.copy_from_slice(&((rnd() % 2000) as f32 / 1000.0 - 1.0).to_le_bytes())
            }),
            GgmlType::F16 => region.as_chunks_mut::<2>().0.iter_mut().for_each(|c| {
                c.copy_from_slice(&f32_to_f16((rnd() % 2000) as f32 / 1000.0 - 1.0).to_le_bytes())
            }),
            _ => {
                region.iter_mut().for_each(|b| *b = rnd() as u8);
                let scale_offs: &[usize] = match ty {
                    GgmlType::Q6_K => &[208],
                    GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
                    _ => &[0],
                };
                for blk in region.chunks_exact_mut(ty.block_bytes()) {
                    for &o in scale_offs {
                        blk[o..o + 2].copy_from_slice(
                            &f32_to_f16((rnd() % 1000) as f32 * 1e-5).to_le_bytes(),
                        );
                    }
                }
            }
        }
    }
    (mem, offsets)
}

fn input(n: usize) -> Vec<f32> {
    (0..n * COLS)
        .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
        .collect()
}

fn cpu_matvec(buf: &[u8], ty: GgmlType, off: usize, x: &[f32]) -> Vec<f32> {
    let m = QMatrix::new(ty, ROWS, COLS, &buf[off..off + ty.row_bytes(COLS) * ROWS]);
    let mut y = vec![0.0; ROWS];
    m.matvec_with(x, &mut y, true);
    y
}

fn assert_close(ty: GgmlType, what: &str, got: &[f32], want: &[f32]) {
    for (r, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
            "{ty:?} {what} row {r}: gpu {g} vs cpu {w}"
        );
    }
}

#[test]
fn metal_matvec_matches_cpu_for_every_supported_type() {
    let (mut mem, offsets) = fixture();
    let buf = mem.bytes();
    let x = input(1);
    let gpu =
        unsafe { Gpu::new(buf, std::slice::from_ref(&(0..buf.len()))) }.expect("Metal device");
    for (&ty, &off) in TYPES.iter().zip(&offsets) {
        assert!(gpu.supports(ty));
        let mut got = vec![0.0; ROWS];
        gpu.matvec(ty, ROWS, COLS, off, &x, &mut got);
        assert_close(ty, "matvec", &got, &cpu_matvec(buf, ty, off, &x));
    }
}

#[test]
fn metal_matmul_matches_cpu_matvec_per_token_for_every_supported_type() {
    let (mut mem, offsets) = fixture();
    let buf = mem.bytes();
    let gpu =
        unsafe { Gpu::new(buf, std::slice::from_ref(&(0..buf.len()))) }.expect("Metal device");
    // 2: one partial token tile; 17: two full tiles plus a partial one.
    for n in [2, 17] {
        let x = input(n);
        for (&ty, &off) in TYPES.iter().zip(&offsets) {
            let mut got = vec![0.0; n * ROWS];
            gpu.matmul(ty, ROWS, COLS, off, &x, &mut got, n);
            for t in 0..n {
                let want = cpu_matvec(buf, ty, off, &x[t * COLS..(t + 1) * COLS]);
                assert_close(
                    ty,
                    &format!("n={n} token {t}"),
                    &got[t * ROWS..(t + 1) * ROWS],
                    &want,
                );
            }
        }
    }
}

/// Each matrix its own unit, buffers capped at six pages: the matrices land in several
/// buffers (neighbours sharing a page), and each matvec must bind the right one.
#[test]
fn metal_matvec_across_several_buffers_matches_cpu() {
    let (mut mem, offsets) = fixture();
    let buf = mem.bytes();
    let units: Vec<_> = TYPES
        .iter()
        .zip(&offsets)
        .map(|(ty, &off)| off..off + ty.row_bytes(COLS) * ROWS)
        .collect();
    let spans = rli_core::metal::buffer_spans(&units, 6 * PAGE).unwrap();
    assert!(spans.len() >= 3, "only {} buffers", spans.len());
    let gpu = unsafe { Gpu::new_capped(buf, &units, 6 * PAGE) }.expect("Metal device");
    let x = input(1);
    for (&ty, &off) in TYPES.iter().zip(&offsets) {
        let mut got = vec![0.0; ROWS];
        gpu.matvec(ty, ROWS, COLS, off, &x, &mut got);
        assert_close(
            ty,
            "multi-buffer matvec",
            &got,
            &cpu_matvec(buf, ty, off, &x),
        );
    }
}

/// Wrapping only part of the mapping, starting mid-page: offsets stay file-relative.
#[test]
fn metal_matvec_on_a_partial_span_matches_cpu() {
    let (mut mem, offsets) = fixture();
    let buf = mem.bytes();
    let start = offsets[2];
    assert!(
        start % PAGE != 0 && start > PAGE,
        "span must start mid-page, past page 0"
    );
    let end = offsets[TYPES.len() - 1] + TYPES[TYPES.len() - 1].row_bytes(COLS) * ROWS;
    let gpu = unsafe { Gpu::new(buf, std::slice::from_ref(&(start..end))) }.expect("Metal device");
    let x = input(1);
    for (&ty, &off) in TYPES.iter().zip(&offsets).skip(2) {
        let mut got = vec![0.0; ROWS];
        gpu.matvec(ty, ROWS, COLS, off, &x, &mut got);
        assert_close(
            ty,
            "partial-span matvec",
            &got,
            &cpu_matvec(buf, ty, off, &x),
        );
    }
}

#[test]
#[should_panic(expected = "weights out of range")]
fn metal_matvec_outside_the_span_panics() {
    let (mut mem, offsets) = fixture();
    let buf = mem.bytes();
    let gpu = unsafe { Gpu::new(buf, std::slice::from_ref(&(offsets[2]..buf.len()))) }
        .expect("Metal device");
    let mut got = vec![0.0; ROWS];
    gpu.matvec(TYPES[1], ROWS, COLS, offsets[1], &input(1), &mut got);
}

/// GPU MoE routing against a CPU reference with 128 experts (four per lane), a tie (the lower
/// index wins), with and without renormalization, and the shared expert's sigmoid gate.
#[test]
fn moe_topk_matches_cpu_routing_with_many_experts() {
    use rli_core::metal::At;
    let mut mem = Pages::new(PAGE);
    let gpu = unsafe { Gpu::new(mem.bytes(), std::slice::from_ref(&(0..PAGE))) }.unwrap();
    let (ne, k) = (128, 8);
    let mut seed = 0x9e37_79b9u32;
    let mut logits: Vec<f32> = (0..ne)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed % 10_000) as f32 / 1000.0 - 5.0
        })
        .collect();
    // A tie for a top place between experts 70 and 101: 70 must win.
    logits[70] = 9.0;
    logits[101] = 9.0;
    let gate = 0.3f32;
    for norm in [false, true] {
        let (lx, sg) = (gpu.upload(&logits), gpu.upload(&[gate]));
        let (ids, w) = (gpu.alloc(k), gpu.alloc(k + 1));
        let mut b = gpu.batch();
        b.moe_topk(
            At(&lx, 0),
            At(&sg, 0),
            At(&ids, 0),
            At(&w, 0),
            ne,
            k,
            norm,
            true,
        );
        b.finish();
        let (got_ids, got_w): (Vec<u32>, Vec<f32>) = unsafe {
            (
                ids.as_slice().iter().map(|v| v.to_bits()).collect(),
                w.as_slice().to_vec(),
            )
        };
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        let want_ids: Vec<u32> = order[..k].iter().map(|&e| e as u32).collect();
        let probs: Vec<f32> = order[..k]
            .iter()
            .map(|&e| (logits[e] - mx).exp() / sum)
            .collect();
        let total: f32 = if norm { probs.iter().sum() } else { 1.0 };
        assert_eq!(got_ids, want_ids, "norm={norm}");
        assert_eq!(&got_ids[..2], &[70, 101]);
        for (g, p) in got_w.iter().zip(&probs) {
            assert!(
                (g - p / total).abs() < 1e-6,
                "norm={norm}: {g} vs {}",
                p / total
            );
        }
        let sigmoid = 1.0 / (1.0 + (-gate).exp());
        assert!((got_w[k] - sigmoid).abs() < 1e-6);
    }
}
