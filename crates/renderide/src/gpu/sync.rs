//! Cross-thread synchronisation primitives for the GPU stack.
//!
//! Both submodules support the renderer's main-frame submission path:
//! - [`queue_access_gate`] -- [`queue_access_gate::GpuQueueAccessGate`] serialises operations that
//!   may access the Vulkan queue shared by `wgpu` and OpenXR.
//! - [`mapped_buffer_health`] -- generation counter the renderer reads to detect events that
//!   invalidate CPU-mapped staging/readback buffers (e.g. device loss).
//! - [`device_health`] -- generation counter the renderer reads to detect fatal device loss.

pub mod device_health;
pub mod mapped_buffer_health;
pub mod queue_access_gate;
