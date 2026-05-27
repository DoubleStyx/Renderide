//! GPU profiling state types used by the renderer's HUD and Tracy GPU timeline.
//!
//! - [`frame_bracket`] -- GPU timestamp sessions that bracket tracked command buffers and
//!   feed the debug HUD's `gpu_frame_ms`.
//! - [`frame_cpu_gpu_timing`] -- CPU/GPU wall-clock accumulator that pairs each frame's
//!   active main-thread CPU work with the GPU completion callback.
//!
//! The standalone state-machine types live here. Per-method facades on [`super::GpuContext`]
//! live alongside the context struct.

pub mod frame_bracket;
pub mod frame_cpu_gpu_timing;
