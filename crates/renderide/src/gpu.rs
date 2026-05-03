//! GPU device, adapter, swapchain, frame uniforms, and VR mirror blit.
//!
//! Layout: [`context`] ([`GpuContext`]), [`instance_limits`] ([`instance_flags_for_gpu_init`]),
//! [`frame_globals`] ([`FrameGpuUniforms`]), [`frame_cpu_gpu_timing`] (debug HUD CPU/GPU intervals),
//! [`present`] (surface acquire / clear helpers), [`vr_mirror`] (HMD eye -> staging -> window).

mod adapter;
pub mod bind_layout;
mod context;
pub mod depth;
pub mod driver_thread;
pub(crate) mod frame_bracket;
pub(crate) mod frame_cpu_gpu_timing;
mod instance_limits;
pub mod limits;
pub mod msaa_depth_resolve;
pub mod output_depth_mode;
pub mod present;
mod queue_access_gate;
mod submission_state;
mod vr_mirror;

pub mod frame_globals;

pub use context::{GpuContext, GpuError};
pub use depth::{
    MAIN_FORWARD_DEPTH_CLEAR, MAIN_FORWARD_DEPTH_COMPARE, main_forward_depth_stencil_format,
};
pub use driver_thread::{DriverError, DriverErrorKind, DriverThread, SubmitBatch, SubmitWait};
pub use frame_globals::{ClusteredFrameGlobalsParams, FrameGpuUniforms};
pub use instance_limits::instance_flags_for_gpu_init;
pub use limits::{
    CUBEMAP_ARRAY_LAYERS, GpuLimits, GpuLimitsError, REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE,
};
pub use msaa_depth_resolve::{
    MsaaDepthResolveMonoTargets, MsaaDepthResolveResources, MsaaDepthResolveStereoTargets,
};
pub use output_depth_mode::{OutputDepthMode, OutputDepthModeError};
pub use queue_access_gate::GpuQueueAccessGate;
pub use vr_mirror::{VR_MIRROR_EYE_LAYER, VrMirrorBlitResources};
