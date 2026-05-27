//! Specular IBL reflection-probe baking, binding, and CPU-side selection.

mod atlas;
mod captures;
mod selection;
mod source;
mod system;

pub(crate) use captures::{
    RuntimeReflectionProbeCapture, RuntimeReflectionProbeCaptureKey,
    RuntimeReflectionProbeCaptureStore,
};
pub(crate) use selection::MAX_LOCAL_PROBES;
pub use selection::{ReflectionProbeDrawSelection, ReflectionProbeFrameSelection};
pub(crate) use system::ReflectionProbeSpecularMaintainParams;
pub use system::ReflectionProbeSpecularSystem;
