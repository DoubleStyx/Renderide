//! Cooperative [`SetTexture2DData`] integration: sub-region or one mip per step.

use std::sync::Arc;

use crate::assets::texture::upload_uses_storage_v_inversion;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture2DData, SetTexture2DFormat, SetTexture2DResult,
    TextureUpdateResultType,
};

use super::AssetTransferQueue;
use super::integrator::StepResult;
use super::texture_task_common::{
    failed_upload, missing_payload, resident_texture_arc, send_background_result,
    storage_orientation_allows_mark, storage_orientation_allows_upload,
};
use super::texture_upload_plan::{TextureUploadPlan, TextureUploadStepper, UploadCompletion};

/// One in-flight Texture2D data upload.
#[derive(Debug)]
pub struct TextureUploadTask {
    data: SetTexture2DData,
    /// Cached from [`AssetTransferQueue::texture_formats`] at enqueue time.
    format: SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    stepper: TextureUploadStepper,
}

impl TextureUploadTask {
    /// Builds a task; `fmt` and `wgpu_format` must match the resident [`crate::gpu_pools::GpuTexture2d`].
    pub fn new(
        data: SetTexture2DData,
        format: SetTexture2DFormat,
        wgpu_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            data,
            format,
            wgpu_format,
            stepper: TextureUploadStepper::default(),
        }
    }

    /// [`SetTexture2DData::high_priority`].
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
            "texture",
            id,
            queue
                .texture_pool
                .get_texture(id)
                .map(|texture| texture.texture.clone()),
        ) else {
            return StepResult::Done;
        };
        let texture = tex_arc.as_ref();

        match self.stepper.step(
            shm,
            TextureUploadPlan {
                device: device.as_ref(),
                queue: gpu_queue,
                gpu_queue_access_gate,
                texture,
                format: &self.format,
                wgpu_format: self.wgpu_format,
                upload: &self.data,
                storage_v_inverted,
            },
        ) {
            Ok(UploadCompletion::MissingPayload) => missing_payload("texture", id),
            Ok(UploadCompletion::Continue) => StepResult::Continue,
            Ok(UploadCompletion::UploadedOne {
                uploaded_mips,
                storage_v_inverted,
            }) => {
                self.mark_uploaded_mips(queue, uploaded_mips, storage_v_inverted);
                StepResult::Continue
            }
            Ok(UploadCompletion::YieldBackground) => StepResult::YieldBackground,
            Ok(UploadCompletion::Complete {
                uploaded_mips,
                storage_v_inverted,
            }) => {
                self.finalize_success(queue, ipc, uploaded_mips, storage_v_inverted);
                StepResult::Done
            }
            Err(e) => failed_upload("texture", id, &e),
        }
    }

    /// Whether this upload will leave native compressed bytes in host V orientation.
    fn upload_uses_storage_v_inversion(&self) -> bool {
        upload_uses_storage_v_inversion(self.format.format, self.wgpu_format, self.data.flip_y)
    }

    /// Returns `false` when this upload would mix storage orientations in one resident texture.
    fn storage_orientation_allows_upload(
        &self,
        queue: &AssetTransferQueue,
        storage_v_inverted: bool,
    ) -> bool {
        let Some(t) = queue.texture_pool.get_texture(self.data.asset_id) else {
            return true;
        };
        storage_orientation_allows_upload(
            "texture",
            t.asset_id,
            t.mip_levels_resident,
            t.storage_v_inverted,
            storage_v_inverted,
            "mips",
        )
    }

    /// Marks resident mips and records the upload's storage orientation.
    fn mark_uploaded_mips(
        &self,
        queue: &mut AssetTransferQueue,
        uploaded_mips: u32,
        storage_v_inverted: bool,
    ) {
        if uploaded_mips == 0 {
            return;
        }
        if let Some(t) = queue.texture_pool.get_texture_mut(self.data.asset_id) {
            if !storage_orientation_allows_mark(
                "texture",
                t.asset_id,
                t.mip_levels_resident,
                t.storage_v_inverted,
                storage_v_inverted,
                "after write",
            ) {
                return;
            }
            t.storage_v_inverted = storage_v_inverted;
            let start = self.data.start_mip_level.max(0) as u32;
            t.mark_mips_resident(start, uploaded_mips);
            if t.mip_levels_total > 1 && t.mip_levels_resident < t.mip_levels_total {
                logger::trace!(
                    "texture {}: {} of {} mips resident; sampling clamped to LOD {} until remaining mips stream in",
                    t.asset_id,
                    t.mip_levels_resident,
                    t.mip_levels_total,
                    t.mip_levels_resident.saturating_sub(1)
                );
            }
        }
    }

    fn finalize_success(
        &self,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
        uploaded_mips: u32,
        storage_v_inverted: bool,
    ) {
        let id = self.data.asset_id;
        self.mark_uploaded_mips(queue, uploaded_mips, storage_v_inverted);
        send_background_result(
            ipc,
            RendererCommand::SetTexture2DResult(SetTexture2DResult {
                asset_id: id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }),
        );
        logger::trace!("texture {id}: data upload ok ({uploaded_mips} mips, integrator)");
    }
}
