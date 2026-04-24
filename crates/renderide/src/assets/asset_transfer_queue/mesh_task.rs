//! Cooperative [`MeshUploadData`] integration: layout validation then GPU upload from shared memory.

use std::sync::Arc;

use crate::assets::mesh::{
    compute_and_validate_mesh_layout, mesh_upload_input_fingerprint, try_upload_mesh_from_raw,
};
use crate::gpu::GpuLimits;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::MeshUploadData;

use super::integrator::StepResult;
use super::AssetTransferQueue;

/// Stage for a single mesh upload task ([`Renderite.Unity.MeshAsset.Upload`]–style splitting).
#[derive(Debug)]
enum MeshStage {
    /// Compute and cache [`MeshBufferLayout`] (CPU only).
    PendingLayout,
    /// Background thread extraction and GPU upload.
    Decoding {
        rx: crossbeam_channel::Receiver<Option<crate::assets::mesh::GpuMesh>>,
    },
}

/// One in-flight mesh upload driven by [`super::integrator::drain_asset_tasks`].
#[derive(Debug)]
pub struct MeshUploadTask {
    data: MeshUploadData,
    stage: MeshStage,
}

impl MeshUploadTask {
    /// Builds a task starting at layout validation.
    pub fn new(data: MeshUploadData) -> Self {
        Self {
            data,
            stage: MeshStage::PendingLayout,
        }
    }

    /// [`MeshUploadData::high_priority`].
    pub fn high_priority(&self) -> bool {
        self.data.high_priority
    }

    /// Runs at most one stage (layout, then GPU upload).
    pub fn step(
        &mut self,
        queue: &mut AssetTransferQueue,
        device: &Arc<wgpu::Device>,
        gpu_limits: &Arc<GpuLimits>,
        gpu_queue: &Arc<wgpu::Queue>,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let asset_id = self.data.asset_id;
        match &mut self.stage {
            MeshStage::PendingLayout => {
                if self.data.buffer.length <= 0 {
                    return StepResult::Done;
                }
                let input_fp = mesh_upload_input_fingerprint(&self.data);
                let layout = if let Some(l) =
                    queue.mesh_pool.get_cached_mesh_layout(asset_id, input_fp)
                {
                    l
                } else {
                    let Some(l) = compute_and_validate_mesh_layout(&self.data) else {
                        logger::error!("mesh {asset_id}: invalid mesh layout or buffer descriptor");
                        return StepResult::Done;
                    };
                    queue
                        .mesh_pool
                        .set_cached_mesh_layout(asset_id, input_fp, l);
                    l
                };

                let data = self.data.clone();
                let existing = queue.mesh_pool.get_mesh(asset_id).cloned();
                let raw_len = data.buffer.length.max(0) as usize;

                let raw_arc = shm.with_read_bytes(&data.buffer, |raw| {
                    if raw.len() < raw_len {
                        logger::error!(
                            "mesh {asset_id}: raw too short (need {}, got {})",
                            raw_len,
                            raw.len()
                        );
                        return None;
                    }
                    Some(Arc::from(&raw[..raw_len]))
                });

                let Some(raw) = raw_arc else {
                    return StepResult::Done;
                };

                let (tx, rx) = crossbeam_channel::bounded(1);

                let device_clone = Arc::clone(device);
                let gpu_limits_clone = Arc::clone(gpu_limits);
                let gpu_queue_clone = Arc::clone(gpu_queue);

                rayon::spawn(move || {
                    let mesh = try_upload_mesh_from_raw(
                        device_clone.as_ref(),
                        gpu_limits_clone.as_ref(),
                        Some(gpu_queue_clone.as_ref()),
                        &raw,
                        &data,
                        existing,
                        &layout,
                    );
                    let _ = tx.send(mesh);
                });

                self.stage = MeshStage::Decoding { rx };
                StepResult::YieldBackground
            }
            MeshStage::Decoding { rx } => match rx.try_recv() {
                Ok(upload_result) => {
                    let Some(mesh) = upload_result else {
                        logger::error!(
                            "mesh {asset_id}: upload failed or rejected — host callback not completed (no MeshUploadResult sent)"
                        );
                        return StepResult::Done;
                    };
                    let existed_before = queue.mesh_pool.insert_mesh(mesh);
                    if let Some(ipc) = ipc.as_mut() {
                        use crate::shared::{MeshUploadResult, RendererCommand};
                        let _ = ipc.send_background(RendererCommand::MeshUploadResult(
                            MeshUploadResult {
                                asset_id,
                                instance_changed: !existed_before,
                            },
                        ));
                    }
                    logger::trace!(
                        "mesh {} uploaded via integrator (replaced={} resident_bytes≈{})",
                        asset_id,
                        existed_before,
                        queue.mesh_pool.accounting().total_resident_bytes()
                    );
                    StepResult::Done
                }
                Err(crossbeam_channel::TryRecvError::Empty) => StepResult::YieldBackground,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    logger::error!("mesh {asset_id}: background decode thread panicked");
                    StepResult::Done
                }
            },
        }
    }
}
