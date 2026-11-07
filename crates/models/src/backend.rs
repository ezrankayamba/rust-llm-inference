//! Selects the accelerator compiled into this build (`metal` or `cuda` feature).

use crate::model::Model;

#[derive(Debug)]
pub struct BackendError(pub String);

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BackendError {}

/// Offloads the first `layers` layers' matvecs to the GPU backend of this build.
#[allow(unused_variables)]
pub fn attach(model: &mut Model, layers: usize) -> Result<(), BackendError> {
    if layers == 0 {
        return Ok(());
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        return crate::metal::attach(model, layers);
    }
    #[cfg(all(feature = "cuda", not(all(feature = "metal", target_os = "macos"))))]
    {
        return crate::cuda::attach(model, layers);
    }
    #[allow(unreachable_code)]
    Err(BackendError(
        "this build has no GPU backend; rebuild with --features metal or --features cuda".into(),
    ))
}
