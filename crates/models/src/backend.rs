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
/// `None` offloads as many layers as the GPU backend can take (every layer, unless the model
/// is larger than one Metal buffer); if the GPU can't take any, the model stays on the CPU.
/// Either shortfall is returned as a warning to show the user.
pub fn attach_default(
    model: &mut Model,
    requested: Option<usize>,
) -> Result<Option<String>, BackendError> {
    if let Some(n) = requested {
        return attach(model, n).map(|()| None);
    }
    if !HAS_GPU {
        return Ok(None);
    }
    let total = model.cfg.n_layer;
    let n = default_layers(model);
    if n == 0 {
        return Ok(Some(
            "GPU offload unavailable, running on the CPU: no layer's weights fit in one Metal buffer"
                .into(),
        ));
    }
    Ok(match attach(model, n) {
        Err(e) => Some(format!("GPU offload unavailable, running on the CPU: {e}")),
        Ok(()) if n < total => Some(format!(
            "offloading {n} of {total} layers to the GPU (the rest exceed Metal's buffer limit); \
             the other layers run on the CPU"
        )),
        Ok(()) => None,
    })
}

/// How many layers `attach_default` offloads without an explicit count.
#[allow(unused_variables)]
fn default_layers(model: &Model) -> usize {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        return crate::metal::fitting_layers(model).unwrap_or(model.cfg.n_layer);
    }
    #[allow(unreachable_code)]
    model.cfg.n_layer
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
