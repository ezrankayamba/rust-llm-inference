//! Model architectures, tokenizer, sampling and the generation loop.

pub mod chat;
pub mod config;
pub mod engine;
pub mod model;
pub mod sampler;
pub mod tokenizer;

pub use chat::{ChatTemplate, Message};
pub use engine::{Engine, Finish, GenOutput, GenParams};
pub use model::{Accelerator, Model, Session, Weight};
pub use sampler::SamplerConfig;
pub mod backend;
#[cfg(all(feature = "cuda", not(all(feature = "metal", target_os = "macos"))))]
mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
