//! Dequantizers checked against values produced by the reference `gguf` Python package
//! (see scripts/gen_quant_fixture.py).

use rli_core::GgmlType;
use rli_core::quant::dequantize_row;

#[test]
fn dequantize_matches_gguf_python_reference() {
    let data = include_bytes!("fixtures/dequant_ref.bin");
    let u32_at = |p: usize| u32::from_le_bytes(data[p..p + 4].try_into().unwrap()) as usize;
    let mut p = 0;
    let mut seen = Vec::new();
    while p < data.len() {
        let ty = GgmlType::from_id(u32_at(p) as u32).expect("known type");
        let nb = u32_at(p + 4);
        let raw = &data[p + 8..p + 8 + nb];
        p += 8 + nb;
        let nf = u32_at(p);
        let want: Vec<f32> = data[p + 4..p + 4 + 4 * nf]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        p += 4 + 4 * nf;

        let mut got = vec![0.0; nf];
        dequantize_row(ty, raw, &mut got);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 * (1.0 + w.abs()),
                "{ty:?}[{i}]: {g} vs {w}"
            );
        }
        seen.push(ty);
    }
    assert_eq!(
        seen,
        [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K
        ]
    );
}
