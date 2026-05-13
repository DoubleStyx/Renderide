//! GPU attachment state used by asset upload apply steps.

use std::sync::Arc;

use crate::gpu::{GpuLimits, GpuMappedBufferHealth};

/// Device, queue, and limits captured after backend attach.
#[derive(Default)]
pub(crate) struct AssetGpuRuntime {
    /// Bound wgpu device after backend attach.
    pub(crate) gpu_device: Option<Arc<wgpu::Device>>,
    /// Submission queue paired with [`Self::gpu_device`].
    pub(crate) gpu_queue: Option<Arc<wgpu::Queue>>,
    /// Shared gate held around `Queue::write_texture` to avoid submit/write lock inversion.
    pub(crate) gpu_queue_access_gate: Option<crate::gpu::GpuQueueAccessGate>,
    /// Effective device limits snapshot.
    pub(crate) gpu_limits: Option<Arc<GpuLimits>>,
    /// Shared mapped-buffer invalidation generation from the active GPU context.
    pub(crate) mapped_buffer_health: Option<Arc<GpuMappedBufferHealth>>,
}

impl AssetGpuRuntime {
    /// Returns whether both device and queue are available for asset work.
    pub(crate) fn is_attached(&self) -> bool {
        self.gpu_device.is_some() && self.gpu_queue.is_some()
    }

    /// Stores the GPU handles needed by subsequent asset uploads.
    pub(crate) fn attach(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        gate: crate::gpu::GpuQueueAccessGate,
        limits: Arc<GpuLimits>,
        mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    ) {
        self.gpu_device = Some(device);
        self.gpu_queue = Some(queue);
        self.gpu_queue_access_gate = Some(gate);
        self.gpu_limits = Some(limits);
        self.mapped_buffer_health = Some(mapped_buffer_health);
    }
}
