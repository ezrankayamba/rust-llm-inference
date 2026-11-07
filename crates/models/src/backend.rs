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

/// Whether this build has a GPU backend for the current target.
pub const HAS_GPU: bool = cfg!(any(
    all(feature = "metal", target_os = "macos"),
    feature = "cuda"
));

/// Applies a `--gpu-layers` choice. `Some(n)` offloads exactly `n` layers and fails loudly.
/// `None` offloads every layer when this build has a GPU backend; if the GPU can't take the
/// model, it stays on the CPU and the reason is returned as a warning to show the user.
pub fn attach_default(
    model: &mut Model,
    requested: Option<usize>,
) -> Result<Option<String>, BackendError> {
    match requested {
        Some(n) => attach(model, n).map(|()| None),
        None if HAS_GPU => Ok(attach(model, model.cfg.n_layer)
            .err()
            .map(|e| format!("GPU offload unavailable, running on the CPU: {e}"))),
        None => Ok(None),
    }
}

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
