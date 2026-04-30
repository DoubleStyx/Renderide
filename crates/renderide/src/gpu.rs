//! GPU device, adapter, swapchain, frame uniforms, and VR mirror blit.
//!
//! Layout: [`context`] ([`GpuContext`]), [`instance_limits`] ([`instance_flags_for_gpu_init`]),
//! [`frame_globals`] ([`FrameGpuUniforms`]), [`frame_cpu_gpu_timing`] (debug HUD CPU/GPU intervals),
//! [`present`] (surface acquire / clear helpers), [`vr_mirror`] (HMD eye → staging → window).

mod context;
pub mod driver_thread;
mod frame_cpu_gpu_timing;
mod instance_limits;
pub mod limits;
pub mod msaa_depth_resolve;
pub mod present;
mod queue_access_gate;
mod vr_mirror;

pub mod frame_globals;

pub use context::{GpuContext, GpuError};
pub use driver_thread::{DriverError, DriverErrorKind, DriverThread, SubmitBatch, SubmitWait};
pub use frame_globals::{ClusteredFrameGlobalsParams, FrameGpuUniforms};
pub use instance_limits::instance_flags_for_gpu_init;
pub use limits::{
    GpuLimits, GpuLimitsError, CUBEMAP_ARRAY_LAYERS, REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE,
};
pub use msaa_depth_resolve::{
    MsaaDepthResolveMonoTargets, MsaaDepthResolveResources, MsaaDepthResolveStereoTargets,
};
pub use queue_access_gate::GpuQueueAccessGate;
pub use vr_mirror::{VrMirrorBlitResources, VR_MIRROR_EYE_LAYER};
