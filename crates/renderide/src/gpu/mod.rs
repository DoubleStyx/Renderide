//! GPU device, adapter, and swapchain configuration.

mod context;
#[cfg(feature = "debug-hud")]
mod frame_cpu_gpu_timing;
pub mod vr_mirror_blit;

pub mod frame_globals;

pub use context::{instance_flags_for_gpu_init, GpuContext};
pub use frame_globals::FrameGpuUniforms;
pub use vr_mirror_blit::{VrMirrorBlitResources, VR_MIRROR_EYE_LAYER};
