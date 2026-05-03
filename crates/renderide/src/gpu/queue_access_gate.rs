//! Mutex that serialises access to the Vulkan queue shared by wgpu and OpenXR.
//!
//! # wgpu-core lock ordering
//!
//! `Queue::write_texture` acquires the destination texture's `initialization_status`
//! `RwLock` (write) at `wgpu-core-29 queue.rs:821` and then, with that guard still live,
//! acquires `device.trackers` `Mutex` at `queue.rs:939`. `Queue::submit` does the
//! opposite: `trackers` first at `queue.rs:1304`, then via `initialize_texture_memory`
//! at `memory_init.rs:271` takes `initialization_status` write for every texture
//! referenced by the baked command buffers. With `write_texture` on the main thread and
//! `submit` on the [`super::driver_thread::DriverThread`], the two inner locks form an
//! ABBA cycle -- observed as a futex hang with the main thread parked in
//! `Queue::write_texture` and the driver parked in
//! `BakedCommands::initialize_texture_memory`.
//!
//! `Queue::write_buffer` (the asymmetric cousin) takes `trackers` first and
//! `initialization_status` second with no nesting (see `queue.rs:689-731`), so it is not
//! part of this cycle and is left ungated.
//!
//! # OpenXR queue ownership
//!
//! OpenXR's Vulkan binding requires external synchronization for calls that may access the bound
//! `VkQueue`. Renderide binds OpenXR to the same Vulkan queue that backs [`wgpu::Queue`], so the
//! gate is also held around `xrBeginFrame`, `xrAcquireSwapchainImage`,
//! `xrReleaseSwapchainImage`, and `xrEndFrame`.
//!
//! # Scope
//!
//! The gate is held around main-thread `Queue::write_texture` call sites in the asset
//! texture upload path, around the driver thread's `Queue::submit`, and around the narrow
//! OpenXR calls listed above. Long waits such as `xrWaitFrame`, `xrWaitSwapchainImage`,
//! view location, and input sync stay outside the gate so compositor stalls do not block
//! unrelated GPU submissions.

use std::sync::Arc;

use parking_lot::Mutex;

/// Shared mutex acquired before operations that may access the renderer's Vulkan queue.
///
/// Instantiated once by [`super::GpuContext`] and cloned into the
/// [`super::driver_thread::DriverThread`], the texture asset upload path, and OpenXR
/// frame submission.
#[derive(Clone, Default)]
pub struct GpuQueueAccessGate {
    inner: Arc<Mutex<()>>,
}

impl GpuQueueAccessGate {
    /// Creates an uncontended gate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Locks the gate for the duration of the returned guard. Call immediately before
    /// [`wgpu::Queue::write_texture`], [`wgpu::Queue::submit`], or an OpenXR queue-access
    /// call and drop the guard as soon as that call returns.
    pub fn lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.inner.lock()
    }
}
