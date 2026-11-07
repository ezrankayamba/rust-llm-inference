//! Tensor storage types and CPU kernels for rust-llm-inference.

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod half;
pub mod matrix;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod ops;
pub mod qdot;
pub mod quant;
pub mod simd;

pub use matrix::QMatrix;
pub use quant::GgmlType;

#[cfg(test)]
pub(crate) mod testutil {
    /// Deterministic xorshift generator so kernel tests need no `rand` dependency.
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }
        pub fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }
        /// Uniform in [-1, 1).
        pub fn f32(&mut self) -> f32 {
            self.next_u32() as f32 / (1u64 << 31) as f32 - 1.0
        }
        pub fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f32()).collect()
        }
    }
}
