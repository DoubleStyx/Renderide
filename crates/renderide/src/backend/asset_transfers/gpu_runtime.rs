//! GPU attachment state used by asset upload apply steps.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::gpu::{GpuLimits, GpuMappedBufferHealth};
use crate::upload_arena::PersistentUploadArena;

use super::mesh_upload_batch::MeshUploadStagingBatch;

/// GPU handles and upload settings captured during backend attach.
pub(crate) struct AssetGpuRuntimeAttach {
    /// Bound wgpu device after backend attach.
    pub(crate) device: Arc<wgpu::Device>,
    /// Submission queue paired with [`Self::device`].
    pub(crate) queue: Arc<wgpu::Queue>,
    /// Driver-thread producer used for staged upload command-buffer submits.
    pub(crate) driver_submitter: crate::gpu::driver_thread::DriverSubmitter,
    /// Shared gate held around texture writes to avoid submit/write lock inversion.
    pub(crate) gate: crate::gpu::GpuQueueAccessGate,
    /// Effective device limits snapshot.
    pub(crate) limits: Arc<GpuLimits>,
    /// Shared mapped-buffer invalidation generation from the active GPU context.
    pub(crate) mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    /// Whether mesh uploads should use per-mesh wgpu validation scopes.
    pub(crate) mesh_validation_scopes_enabled: bool,
}

/// Device, queue, and limits captured after backend attach.
#[derive(Default)]
pub(crate) struct AssetGpuRuntime {
    /// Bound wgpu device after backend attach.
    pub(crate) gpu_device: Option<Arc<wgpu::Device>>,
    /// Submission queue paired with [`Self::gpu_device`].
    pub(crate) gpu_queue: Option<Arc<wgpu::Queue>>,
    /// Driver-thread producer used for staged upload command-buffer submits.
    pub(crate) driver_submitter: Option<crate::gpu::driver_thread::DriverSubmitter>,
    /// Shared gate held around texture writes to avoid submit/write lock inversion.
    pub(crate) gpu_queue_access_gate: Option<crate::gpu::GpuQueueAccessGate>,
    /// Effective device limits snapshot.
    pub(crate) gpu_limits: Option<Arc<GpuLimits>>,
    /// Shared mapped-buffer invalidation generation from the active GPU context.
    pub(crate) mapped_buffer_health: Option<Arc<GpuMappedBufferHealth>>,
    /// Mesh buffer upload batch reused across cooperative drains.
    pub(crate) mesh_upload_batch: Arc<MeshUploadStagingBatch>,
    /// Persistent staging arena for mesh upload batch copies.
    pub(crate) mesh_upload_arena: Mutex<PersistentUploadArena>,
    /// Whether mesh uploads should use per-mesh wgpu validation scopes.
    pub(crate) mesh_validation_scopes_enabled: bool,
}

impl AssetGpuRuntime {
    /// Returns whether device, queue, and driver-submit ownership are available for asset work.
    pub(crate) fn is_attached(&self) -> bool {
        self.gpu_device.is_some() && self.gpu_queue.is_some() && self.driver_submitter.is_some()
    }

    /// Stores the GPU handles needed by subsequent asset uploads.
    pub(crate) fn attach(&mut self, desc: AssetGpuRuntimeAttach) {
        let AssetGpuRuntimeAttach {
            device,
            queue,
            driver_submitter,
            gate,
            limits,
            mapped_buffer_health,
            mesh_validation_scopes_enabled,
        } = desc;
        self.gpu_device = Some(device);
        self.gpu_queue = Some(queue);
        self.driver_submitter = Some(driver_submitter);
        self.gpu_queue_access_gate = Some(gate);
        self.gpu_limits = Some(limits);
        self.mapped_buffer_health = Some(mapped_buffer_health);
        self.mesh_validation_scopes_enabled = mesh_validation_scopes_enabled;
    }
}
