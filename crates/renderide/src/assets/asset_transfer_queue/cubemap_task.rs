//! Cooperative [`SetCubemapData`] integration: one face x mip per step.

use std::sync::Arc;

use crate::assets::texture::upload_uses_storage_v_inversion;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetCubemapData, SetCubemapFormat, SetCubemapResult, TextureUpdateResultType,
};

use super::AssetTransferQueue;
use super::cubemap_upload_plan::{
    CubemapUploadCompletion, CubemapUploadPlan, CubemapUploadStepper,
};
use super::integrator::StepResult;
use super::texture_task_common::{
    failed_upload, missing_payload, resident_texture_arc, send_background_result,
    storage_orientation_allows_mark, storage_orientation_allows_upload,
};

/// One in-flight cubemap data upload.
#[derive(Debug)]
pub struct CubemapUploadTask {
    data: SetCubemapData,
    format: SetCubemapFormat,
    wgpu_format: wgpu::TextureFormat,
    stepper: CubemapUploadStepper,
}

impl CubemapUploadTask {
    /// Builds a task; `fmt` and `wgpu_format` must match the resident [`crate::gpu_pools::GpuCubemap`].
    pub fn new(
        data: SetCubemapData,
        format: SetCubemapFormat,
        wgpu_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            data,
            format,
            wgpu_format,
            stepper: CubemapUploadStepper::default(),
        }
    }

    /// [`SetCubemapData::high_priority`].
    pub fn high_priority(&self) -> bool {
        self.data.high_priority
    }

    /// Runs at most one integration sub-step.
    pub fn step(
        &mut self,
        queue: &mut AssetTransferQueue,
        device: &Arc<wgpu::Device>,
        gpu_queue: &wgpu::Queue,
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let id = self.data.asset_id;
        let storage_v_inverted = self.upload_uses_storage_v_inversion();
        if !self.storage_orientation_allows_upload(queue, storage_v_inverted) {
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
            return StepResult::Done;
        };
        let texture = tex_arc.as_ref();

        let completion = self.stepper.step(
            shm,
            CubemapUploadPlan {
                device: device.as_ref(),
                queue: gpu_queue,
                gpu_queue_access_gate,
                texture,
                format: &self.format,
                wgpu_format: self.wgpu_format,
                upload: &self.data,
            },
        );
        match completion {
            Ok(CubemapUploadCompletion::MissingPayload) => missing_payload("cubemap", id),
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
            Err(e) => failed_upload("cubemap", id, &e),
        }
    }

    /// Whether this upload will leave native compressed face bytes in host V orientation.
    fn upload_uses_storage_v_inversion(&self) -> bool {
        upload_uses_storage_v_inversion(self.format.format, self.wgpu_format, self.data.flip_y)
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
                return;
            }
            t.storage_v_inverted = storage_v_inverted;
            t.mip_levels_resident = t.mip_levels_total;
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
        )
    }

    #[test]
    fn high_priority_reflects_upload_command() {
        assert!(task(true, false, TextureFormat::RGBA32).high_priority());
        assert!(!task(false, false, TextureFormat::RGBA32).high_priority());
    }

    #[test]
    fn host_cubemap_uploads_always_use_unity_orientation() {
        // After the unified-orientation refactor every host cubemap upload is treated as
        // Unity V=0 bottom regardless of host format or `flip_y`.
        assert!(task(false, true, TextureFormat::BC7).upload_uses_storage_v_inversion());
        assert!(task(false, false, TextureFormat::BC7).upload_uses_storage_v_inversion());
        assert!(task(false, true, TextureFormat::BC1).upload_uses_storage_v_inversion());
        assert!(task(false, false, TextureFormat::RGBA32).upload_uses_storage_v_inversion());
    }
}
