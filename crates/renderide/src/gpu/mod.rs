//! GPU device, adapter, swapchain, frame uniforms, and VR mirror blit.
//!
//! Layout: [`context`] ([`GpuContext`]), [`instance_limits`] ([`instance_flags_for_gpu_init`]),
//! [`frame_globals`] ([`FrameGpuUniforms`]), [`frame_cpu_gpu_timing`] (debug HUD CPU/GPU intervals),
//! [`present`] (surface acquire / clear helpers), [`vr_mirror`] (HMD eye → staging → window).

mod context;
mod frame_cpu_gpu_timing;
mod instance_limits;
pub mod limits;
pub mod msaa_depth_resolve;
pub mod present;
mod vr_mirror;

pub mod frame_globals;

pub use context::{GpuContext, MsaaTargets};
pub use frame_globals::{ClusteredFrameGlobalsParams, FrameGpuUniforms};
pub use instance_limits::instance_flags_for_gpu_init;
pub use limits::{GpuLimits, GpuLimitsError};
pub use msaa_depth_resolve::MsaaDepthResolveResources;
pub use vr_mirror::{VrMirrorBlitResources, VR_MIRROR_EYE_LAYER};
