//! GPU handles shared by every asset upload step within one cooperative drain.

use std::sync::Arc;

use crate::gpu::{GpuLimits, GpuMappedBufferHealth, GpuQueueAccessGate, GpuQueueAccessMode};

use super::super::AssetTransferQueue;
use super::super::MeshUploadStagingBatch;
use super::super::mesh_task::MeshTaskGpu;
use super::super::particle_task::ParticleTaskGpu;
use super::super::texture_task_common::TextureTaskGpu;

/// Owned GPU handles collected from [`crate::backend::AssetTransferQueue::gpu`]; outlives the
/// borrowed [`AssetUploadGpuContext`] passed into step functions.
pub(super) struct GpuHandles {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) gpu_limits: Arc<GpuLimits>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) driver_submitter: crate::gpu::driver_thread::DriverSubmitter,
    pub(super) gate: GpuQueueAccessGate,
    pub(super) mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    pub(super) mesh_upload_batch: Arc<MeshUploadStagingBatch>,
    pub(super) mesh_validation_scopes_enabled: bool,
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
            mesh_upload_batch: &self.mesh_upload_batch,
            mesh_validation_scopes_enabled: self.mesh_validation_scopes_enabled,
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
        asset.gpu.driver_submitter.clone(),
        asset.gpu.gpu_queue_access_gate.clone(),
        asset.gpu.mapped_buffer_health.clone(),
    ) {
        (
            Some(device),
            Some(gpu_limits),
            Some(queue),
            Some(driver_submitter),
            Some(gate),
            Some(mapped_buffer_health),
        ) => Some(GpuHandles {
            device,
            gpu_limits,
            queue,
            driver_submitter,
            gate,
            mapped_buffer_health,
            mesh_upload_batch: Arc::clone(&asset.gpu.mesh_upload_batch),
            mesh_validation_scopes_enabled: asset.gpu.mesh_validation_scopes_enabled,
        }),
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
    /// Mesh upload batch for deferred buffer writes.
    pub(super) mesh_upload_batch: &'a Arc<MeshUploadStagingBatch>,
    /// Whether mesh uploads should use wgpu validation scopes.
    pub(super) mesh_validation_scopes_enabled: bool,
}

impl AssetUploadGpuContext<'_> {
    /// Returns the subset of GPU handles needed by mesh upload tasks.
    pub(super) fn mesh_task_gpu(&self) -> MeshTaskGpu<'_> {
        MeshTaskGpu {
            device: self.device,
            gpu_limits: self.gpu_limits,
            mapped_buffer_health: self.mapped_buffer_health,
            mesh_upload_batch: self.mesh_upload_batch,
            mesh_validation_scopes_enabled: self.mesh_validation_scopes_enabled,
        }
    }

    /// Returns the subset of GPU handles needed by texture-family upload tasks.
    pub(super) fn texture_task_gpu(&self) -> TextureTaskGpu<'_> {
        TextureTaskGpu {
            device: self.device,
            queue: self.queue.as_ref(),
            queue_access_gate: self.gpu_queue_access_gate,
            queue_access_mode: self.queue_access_mode,
        }
    }

    /// Returns the subset of GPU handles needed by particle mesh build tasks.
    pub(super) fn particle_task_gpu(&self) -> ParticleTaskGpu<'_> {
        ParticleTaskGpu {
            device: self.device,
            gpu_limits: self.gpu_limits,
            mapped_buffer_health: self.mapped_buffer_health,
            mesh_upload_batch: self.mesh_upload_batch,
            mesh_validation_scopes_enabled: self.mesh_validation_scopes_enabled,
        }
    }
}
