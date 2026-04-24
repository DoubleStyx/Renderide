//! Mutex that serialises [`wgpu::Queue::write_texture`] against [`wgpu::Queue::submit`]
//! to work around an ABBA lock-ordering bug in wgpu-core 29.
//!
//! # The bug
//!
//! `Queue::write_texture` acquires the destination texture's `initialization_status`
//! `RwLock` (write) at `wgpu-core-29 queue.rs:821` and then, with that guard still live,
//! acquires `device.trackers` `Mutex` at `queue.rs:939`. `Queue::submit` does the
//! opposite: `trackers` first at `queue.rs:1304`, then via `initialize_texture_memory`
//! at `memory_init.rs:271` takes `initialization_status` write for every texture
//! referenced by the baked command buffers. With `write_texture` on the main thread and
//! `submit` on the [`super::driver_thread::DriverThread`], the two inner locks form an
//! ABBA cycle — observed as a futex hang with the main thread parked in
//! `Queue::write_texture` and the driver parked in
//! `BakedCommands::initialize_texture_memory`.
//!
//! `Queue::write_buffer` (the asymmetric cousin) takes `trackers` first and
//! `initialization_status` second with no nesting (see `queue.rs:689-731`), so it is not
//! part of this cycle and is left ungated.
//!
//! # Scope
//!
//! The gate is held around main-thread `Queue::write_texture` call sites in the asset
//! texture upload path, and around the driver thread's `Queue::submit`. Nothing else
//! needs it; the serialisation window is tight because each `write_texture` and each
//! `submit` are short, and the mutex is uncontended outside the narrow race window.

use std::sync::Arc;

use parking_lot::Mutex;

/// Shared mutex acquired before every main-thread [`wgpu::Queue::write_texture`] and
/// every driver-thread [`wgpu::Queue::submit`]. See module docs for the deadlock it
/// prevents.
///
/// Instantiated once by [`super::GpuContext`] and cloned into the
/// [`super::driver_thread::DriverThread`] and the texture asset upload path.
#[derive(Clone, Default)]
pub struct WriteTextureSubmitGate {
    inner: Arc<Mutex<()>>,
}

impl WriteTextureSubmitGate {
    /// Creates an uncontended gate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Locks the gate for the duration of the returned guard. Call immediately
    /// before [`wgpu::Queue::write_texture`] or [`wgpu::Queue::submit`] and drop the
    /// guard as soon as that call returns — no other wgpu work should happen under
    /// the guard.
    pub fn lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.inner.lock()
    }
}
