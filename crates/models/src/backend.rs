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
/// exceeds Metal's recommended working set); if the GPU can't take any, the model stays on
/// the CPU.
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
            "GPU offload unavailable, running on the CPU: no layer's weights fit the GPU's memory"
                .into(),
        ));
    }
    Ok(match attach(model, n) {
        Err(e) => Some(format!("GPU offload unavailable, running on the CPU: {e}")),
        Ok(()) if n < total => Some(format!(
            "offloading {n} of {total} layers to the GPU (the rest don't fit its memory); \
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

/// Metal [`attach`] with the weights wrapped in buffers of at most `max_buffer` bytes, so
/// tests can split a small model across several buffers.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[doc(hidden)]
pub fn attach_capped(
    model: &mut Model,
    layers: usize,
    max_buffer: usize,
) -> Result<(), BackendError> {
    crate::metal::attach_capped(model, layers, max_buffer)
}

/// Offloads the first `layers` layers' matvecs to the GPU backend of this build.
#[allow(unused_variables)]
pub fn attach(model: &mut Model, layers: usize) -> Result<(), BackendError> {
    if layers == 0 {
        return Ok(());
    }
    if model.cfg.mla.is_some() {
        return Err(BackendError(
            "DeepSeek's latent attention has no GPU kernels yet; use --gpu-layers 0".into(),
        ));
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
