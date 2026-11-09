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

static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

/// Sets the number of CPU threads used for inference. Call it before anything runs on the CPU:
/// the pool can only be set up once, and defaults to one thread per logical core.
pub fn set_threads(n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("thread count must be at least 1".into());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .map_err(|e| format!("cannot set up {n} threads: {e}"))?;
    POOL.set(pool)
        .map_err(|_| "the CPU threads are already set up".to_string())
}

/// How many CPU threads inference uses.
pub fn threads() -> usize {
    pool().current_num_threads()
}

/// The pool every forward pass runs in, so that the thread doing a pass is one of its workers:
/// a caller outside it would hand each parallel step to the pool and wait, about 200 times per
/// token (measured as 4-10% of single-thread decode on x86).
pub(crate) fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .build()
            .expect("start the CPU thread pool")
    })
}

#[cfg(all(feature = "cuda", not(all(feature = "metal", target_os = "macos"))))]
mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
