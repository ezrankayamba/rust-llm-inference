//! IEEE half (f16) and bfloat16 conversions. `std`'s `f16` is still unstable, so these are bit-level.

/// Hardware conversion: `fcvt Sd, Hn` is in the base AArch64 ISA (no FP16 extension needed).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let out: f32;
    // SAFETY: register-only conversion; no memory access.
    unsafe {
        std::arch::asm!(
            "fmov {t:s}, {h:w}",
            "fcvt {o:s}, {t:h}",
            h = in(reg) h as u32,
            t = out(vreg) _,
            o = lateout(vreg) out,
            options(pure, nomem, nostack),
        );
    }
    out
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    f16_to_f32_soft(h)
}

/// Portable bit-level conversion; the reference for the hardware path.
#[inline]
pub fn f16_to_f32_soft(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    match exp {
        0 => {
            // Zero or subnormal: value = mant * 2^-24.
            let v = mant as f32 * (1.0 / 16_777_216.0);
            if sign != 0 { -v } else { v }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (mant << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13)),
    }
}

/// Round-to-nearest-even conversion.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    // A carry out of the mantissa correctly bumps the exponent (up to infinity).
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exact_for_all_finite_halves() {
        for h in 0u16..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16(f), h, "h={h:#06x} f={f}");
            }
        }
    }

    #[test]
    fn hardware_and_software_f16_conversions_agree_bitwise() {
        for h in 0u16..=u16::MAX {
            let (hw, sw) = (f16_to_f32(h), f16_to_f32_soft(h));
            assert!(
                hw.to_bits() == sw.to_bits() || (hw.is_nan() && sw.is_nan()),
                "h={h:#06x}: {hw} vs {sw}"
            );
        }
    }

    #[test]
    fn f32_to_f16_rounds_to_nearest_even() {
        // 1 + 2^-11 is exactly halfway between 1.0 and the next half; ties go to even (1.0).
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(1e9), 0x7c00);
        assert_eq!(f32_to_f16(1e-9), 0);
    }

    #[test]
    fn bf16_is_upper_half_of_f32() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc000), -2.0);
    }
}
