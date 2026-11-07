//! Metal kernels against the CPU reference (f32 activations) for every supported weight type.
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

#[test]
fn metal_matvec_matches_cpu_for_every_supported_type() {
    let (rows, cols) = (37, 512); // rows not a multiple of the 4-row threadgroup
    let types = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ];
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
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
        .collect();
    let gpu = unsafe { Gpu::new(buf) }.expect("Metal device");
    for (&ty, &off) in types.iter().zip(&offsets) {
        assert!(gpu.supports(ty));
        let m = QMatrix::new(ty, rows, cols, &buf[off..off + ty.row_bytes(cols) * rows]);
        let mut want = vec![0.0; rows];
        m.matvec_with(&x, &mut want, true);
        let mut got = vec![0.0; rows];
        gpu.matvec(ty, rows, cols, off, &x, &mut got);
        for (r, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
                "{ty:?} row {r}: gpu {g} vs cpu {w}"
            );
        }
    }
}
