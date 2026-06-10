//! Host-requested offscreen rendering with CPU readback.
//!
//! - [`camera`] -- host camera capture queue, offscreen render, GPU readback, IPC writeback.
//! - [`reflection_probe`] -- host reflection-probe cubemap bake tasks, IBL convolution, readback.
//! - [`readback`] -- GPU buffer-mapping plumbing shared by both task drains.

pub(in crate::runtime) mod camera;
pub(in crate::runtime) mod cube_capture;
pub(in crate::runtime) mod readback;
pub(in crate::runtime) mod reflection_probe;
pub(in crate::runtime) mod submit_completion;
