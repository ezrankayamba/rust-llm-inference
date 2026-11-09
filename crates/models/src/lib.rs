//! Model architectures, tokenizer, sampling and the generation loop.

pub mod chat;
pub mod config;
pub mod engine;
pub mod model;
pub mod sampler;
mod stop;
pub mod tokenizer;

pub use chat::{ChatDefaults, ChatTemplate, Message, Rendered};
pub use engine::{Engine, Finish, GenOutput, GenParams};
pub use model::{Accelerator, Model, Session, Weight};
pub use sampler::SamplerConfig;
pub mod backend;

/// Sets the number of CPU threads used for inference. Call it before anything runs on the CPU:
/// the pool can only be set up once, and defaults to one thread per logical core.
pub fn set_threads(n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("thread count must be at least 1".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| format!("cannot set up {n} threads: {e}"))
}

#[cfg(all(feature = "cuda", not(all(feature = "metal", target_os = "macos"))))]
mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
