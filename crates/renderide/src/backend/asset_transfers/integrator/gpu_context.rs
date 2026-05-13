//! GPU handles shared by every asset upload step within one cooperative drain.

use std::sync::Arc;

use crate::gpu::{GpuLimits, GpuMappedBufferHealth, GpuQueueAccessGate, GpuQueueAccessMode};

use super::super::AssetTransferQueue;

/// Owned GPU handles collected from [`crate::backend::AssetTransferQueue::gpu`]; outlives the
/// borrowed [`AssetUploadGpuContext`] passed into step functions.
pub(super) struct GpuHandles {
    device: Arc<wgpu::Device>,
    gpu_limits: Arc<GpuLimits>,
    queue: Arc<wgpu::Queue>,
    gate: GpuQueueAccessGate,
    mapped_buffer_health: Arc<GpuMappedBufferHealth>,
}

impl GpuHandles {
    /// Borrows the handles as a step-time context.
    pub(super) fn as_context(
        &self,
        queue_access_mode: GpuQueueAccessMode,
    ) -> AssetUploadGpuContext<'_> {
        AssetUploadGpuContext {
            device: &self.device,
            gpu_limits: &self.gpu_limits,
            queue: &self.queue,
            gpu_queue_access_gate: &self.gate,
            queue_access_mode,
            mapped_buffer_health: &self.mapped_buffer_health,
        }
    }
}

/// Returns the GPU handles captured at backend attach when every required handle is present, or
/// `None` if any handle is missing (e.g. before the renderer has a device).
pub(super) fn collect_gpu_handles(asset: &AssetTransferQueue) -> Option<GpuHandles> {
    match (
        asset.gpu.gpu_device.clone(),
        asset.gpu.gpu_limits.clone(),
        asset.gpu.gpu_queue.clone(),
        asset.gpu.gpu_queue_access_gate.clone(),
        asset.gpu.mapped_buffer_health.clone(),
    ) {
        (Some(device), Some(gpu_limits), Some(queue), Some(gate), Some(mapped_buffer_health)) => {
            Some(GpuHandles {
                device,
                gpu_limits,
                queue,
                gate,
                mapped_buffer_health,
            })
        }
        _ => None,
    }
}

/// GPU handles shared across all asset task step calls in one drain.
pub(super) struct AssetUploadGpuContext<'a> {
    /// Device for resource creation and format capability queries.
    pub(super) device: &'a Arc<wgpu::Device>,
    /// GPU adapter limits shared with mesh upload paths.
    pub(super) gpu_limits: &'a Arc<GpuLimits>,
    /// Queue for [`wgpu::Queue::write_texture`] / [`wgpu::Queue::write_buffer`] uploads.
    pub(super) queue: &'a Arc<wgpu::Queue>,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    pub(super) gpu_queue_access_gate: &'a GpuQueueAccessGate,
    /// Queue-gate acquisition policy for texture writes in this drain.
    pub(super) queue_access_mode: GpuQueueAccessMode,
    /// Shared mapped-buffer invalidation generation from the active GPU context.
    pub(super) mapped_buffer_health: &'a Arc<GpuMappedBufferHealth>,
}
