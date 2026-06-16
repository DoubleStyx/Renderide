//! Cooperative [`SetCubemapData`] integration: one face x mip per step.

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetCubemapData, SetCubemapFormat, SetCubemapResult, TextureFormat,
    TextureUpdateResultType,
};

use super::AssetTransferQueue;
use super::cubemap_upload_plan::{
    CubemapUploadCompletion, CubemapUploadPlan, CubemapUploadStepper,
};
use super::integrator::StepResult;
use super::texture_task_common::{
    TextureTaskGpu, failed_upload, missing_payload, resident_texture_arc, send_background_result,
    storage_orientation_allows_mark, storage_orientation_allows_upload,
};

/// One in-flight cubemap data upload.
#[derive(Debug)]
pub struct CubemapUploadTask {
    data: SetCubemapData,
    format: SetCubemapFormat,
    wgpu_format: wgpu::TextureFormat,
    generation: u64,
    stepper: CubemapUploadStepper,
}

impl CubemapUploadTask {
    /// Builds a task; `fmt` and `wgpu_format` must match the resident [`crate::gpu_pools::GpuCubemap`].
    pub fn new(
        data: SetCubemapData,
        format: SetCubemapFormat,
        wgpu_format: wgpu::TextureFormat,
        generation: u64,
    ) -> Self {
        Self {
            data,
            format,
            wgpu_format,
            generation,
            stepper: CubemapUploadStepper::default(),
        }
    }

    /// Returns whether this upload came from a high-priority host command.
    #[cfg(test)]
    pub fn high_priority(&self) -> bool {
        self.data.high_priority
    }

    /// Runs at most one integration sub-step.
    pub(super) fn step(
        &mut self,
        queue: &mut AssetTransferQueue,
        gpu: TextureTaskGpu<'_>,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let id = self.data.asset_id;
        if !queue.cubemap_upload_generation_is_current(id, self.generation) {
            logger::trace!(
                "cubemap {id}: dropped stale data upload generation {}",
                self.generation
            );
            self.finalize_failure(ipc);
            return StepResult::Done;
        }
        let storage_v_inverted = host_cubemap_upload_uses_storage_v_inversion(
            self.format.format,
            self.wgpu_format,
            self.data.flip_y,
        );
        if !self.storage_orientation_allows_upload(queue, storage_v_inverted) {
            self.finalize_failure(ipc);
            return StepResult::Done;
        }
        let Some(tex_arc) = resident_texture_arc(
            "cubemap",
            id,
            queue
                .pools
                .cubemap_pool
                .get(id)
                .map(|texture| texture.texture.clone()),
        ) else {
            self.finalize_failure(ipc);
            return StepResult::Done;
        };
        let texture = tex_arc.as_ref();

        let completion = self.stepper.step(
            shm,
            CubemapUploadPlan {
                device: gpu.device.as_ref(),
                queue: gpu.queue,
                gpu_queue_access_gate: gpu.queue_access_gate,
                queue_access_mode: gpu.queue_access_mode,
                texture,
                format: &self.format,
                wgpu_format: self.wgpu_format,
                upload: &self.data,
            },
        );
        match completion {
            Ok(CubemapUploadCompletion::MissingPayload) => {
                missing_payload("cubemap", id);
                self.finalize_failure(ipc);
                StepResult::Done
            }
            Ok(CubemapUploadCompletion::Continue) => StepResult::Continue,
            Ok(CubemapUploadCompletion::UploadedOne { storage_v_inverted }) => {
                self.mark_storage_orientation(queue, storage_v_inverted);
                StepResult::Continue
            }
            Ok(CubemapUploadCompletion::YieldBackground) => StepResult::YieldBackground,
            Ok(CubemapUploadCompletion::Complete {
                uploaded_face_mips,
                storage_v_inverted,
            }) => {
                self.finalize_success(queue, ipc, uploaded_face_mips, storage_v_inverted);
                StepResult::Done
            }
            Err(e) if e.is_queue_access_busy() => StepResult::YieldBackground,
            Err(e) => {
                failed_upload("cubemap", id, &e);
                self.finalize_failure(ipc);
                StepResult::Done
            }
        }
    }

    /// Returns `false` when this upload would mix storage orientations in one resident cubemap.
    fn storage_orientation_allows_upload(
        &self,
        queue: &AssetTransferQueue,
        storage_v_inverted: bool,
    ) -> bool {
        let Some(t) = queue.pools.cubemap_pool.get(self.data.asset_id) else {
            return true;
        };
        storage_orientation_allows_upload(
            "cubemap",
            t.asset_id,
            t.mip_levels_resident,
            t.storage_v_inverted,
            storage_v_inverted,
            "face mips",
        )
    }

    /// Records the storage orientation after a successful face-mip write.
    fn mark_storage_orientation(&self, queue: &mut AssetTransferQueue, storage_v_inverted: bool) {
        if let Some(t) = queue.pools.cubemap_pool.get_mut(self.data.asset_id) {
            if !storage_orientation_allows_mark(
                "cubemap",
                t.asset_id,
                t.mip_levels_resident,
                t.storage_v_inverted,
                storage_v_inverted,
                "after write",
            ) {
                return;
            }
            t.storage_v_inverted = storage_v_inverted;
        }
    }

    fn finalize_success(
        &self,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
        uploaded_face_mips: u32,
        storage_v_inverted: bool,
    ) {
        let id = self.data.asset_id;
        if !queue.cubemap_upload_generation_is_current(id, self.generation) {
            self.finalize_failure(ipc);
            return;
        }
        if uploaded_face_mips > 0
            && let Some(t) = queue.pools.cubemap_pool.get_mut(id)
        {
            if !storage_orientation_allows_mark(
                "cubemap",
                t.asset_id,
                t.mip_levels_resident,
                t.storage_v_inverted,
                storage_v_inverted,
                "at finalize",
            ) {
                self.finalize_failure(ipc);
                return;
            }
            t.storage_v_inverted = storage_v_inverted;
            t.mip_levels_resident = t.mip_levels_total;
            t.mark_content_uploaded();
        }
        send_background_result(
            ipc,
            RendererCommand::SetCubemapResult(SetCubemapResult {
                asset_id: id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }),
        );
        logger::trace!("cubemap {id}: data upload ok ({uploaded_face_mips} face-mips, integrator)");
    }

    fn finalize_failure(&self, ipc: &mut Option<&mut DualQueueIpc>) {
        send_background_result(
            ipc,
            RendererCommand::SetCubemapResult(SetCubemapResult {
                asset_id: self.data.asset_id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }),
        );
    }
}

fn host_cubemap_upload_uses_storage_v_inversion(
    _host_format: TextureFormat,
    _wgpu_format: wgpu::TextureFormat,
    _flip_y: bool,
) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use crate::shared::{SetCubemapData, SetCubemapFormat, TextureFormat};

    use super::*;

    fn task(high_priority: bool, flip_y: bool, host_format: TextureFormat) -> CubemapUploadTask {
        CubemapUploadTask::new(
            SetCubemapData {
                high_priority,
                flip_y,
                ..Default::default()
            },
            SetCubemapFormat {
                format: host_format,
                ..Default::default()
            },
            wgpu::TextureFormat::Bc7RgbaUnorm,
            1,
        )
    }

    #[test]
    fn high_priority_reflects_upload_command() {
        assert!(task(true, false, TextureFormat::RGBA32).high_priority());
        assert!(!task(false, false, TextureFormat::RGBA32).high_priority());
    }

    #[test]
    fn host_cubemap_uploads_use_native_cube_orientation() {
        for task in [
            task(false, true, TextureFormat::BC7),
            task(false, false, TextureFormat::BC7),
            task(false, true, TextureFormat::BC1),
            task(false, false, TextureFormat::RGBA32),
        ] {
            assert!(!host_cubemap_upload_uses_storage_v_inversion(
                task.format.format,
                task.wgpu_format,
                task.data.flip_y
            ));
        }
    }
}
