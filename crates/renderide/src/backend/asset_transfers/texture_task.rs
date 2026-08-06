//! Cooperative [`SetTexture2DData`] integration: sub-region or one mip per step.

use crate::assets::texture::upload_uses_storage_v_inversion;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture2DData, SetTexture2DFormat, SetTexture2DResult,
    TextureUpdateResultType,
};

use super::AssetTransferQueue;
use super::integrator::StepResult;
use super::texture_task_common::{
    TextureTaskGpu, failed_upload, missing_payload, resident_texture_arc, send_background_result,
    storage_orientation_allows_mark, storage_orientation_allows_upload,
};
use super::texture_upload_plan::{
    TextureResidencyUpdate, TextureUploadPlan, TextureUploadStepper, UploadCompletion,
};

/// One in-flight Texture2D data upload.
#[derive(Debug)]
pub struct TextureUploadTask {
    data: SetTexture2DData,
    /// Cached from [`AssetTransferQueue::texture_formats`] at enqueue time.
    format: SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    generation: u64,
    stepper: TextureUploadStepper,
    /// Whether this full-chain upload has replaced the previous upload's resident mip mask.
    mip_stream_started: bool,
}

impl TextureUploadTask {
    /// Builds a task; `fmt` and `wgpu_format` must match the resident [`crate::gpu_pools::GpuTexture2d`].
    pub fn new(
        data: SetTexture2DData,
        format: SetTexture2DFormat,
        wgpu_format: wgpu::TextureFormat,
        generation: u64,
    ) -> Self {
        Self {
            data,
            format,
            wgpu_format,
            generation,
            stepper: TextureUploadStepper::default(),
            mip_stream_started: false,
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
        if !queue.texture_upload_generation_is_current(id, self.generation) {
            logger::trace!(
                "texture {id}: dropped stale data upload generation {}",
                self.generation
            );
            self.finalize_failure(ipc);
            return StepResult::Done;
        }
        let storage_v_inverted = self.upload_uses_storage_v_inversion();
        if !self.storage_orientation_allows_upload(queue, storage_v_inverted) {
            self.finalize_failure(ipc);
            return StepResult::Done;
        }
        let Some(tex_arc) = resident_texture_arc(
            "texture",
            id,
            queue
                .pools
                .texture_pool
                .get(id)
                .map(|texture| texture.texture.clone()),
        ) else {
            self.finalize_failure(ipc);
            return StepResult::Done;
        };
        let texture = tex_arc.as_ref();

        match self.stepper.step(
            shm,
            TextureUploadPlan {
                device: gpu.device.as_ref(),
                queue: gpu.queue,
                gpu_queue_access_gate: gpu.queue_access_gate,
                queue_access_mode: gpu.queue_access_mode,
                texture,
                format: &self.format,
                wgpu_format: self.wgpu_format,
                upload: &self.data,
                storage_v_inverted,
            },
        ) {
            Ok(UploadCompletion::MissingPayload) => {
                missing_payload("texture", id);
                self.finalize_failure(ipc);
                StepResult::Done
            }
            Ok(UploadCompletion::Continue) => StepResult::Continue,
            Ok(UploadCompletion::UploadedOne {
                mip_level,
                replaces_full_texture,
                storage_v_inverted,
            }) => {
                self.mark_uploaded_chain_mip(
                    queue,
                    mip_level,
                    storage_v_inverted,
                    replaces_full_texture,
                );
                StepResult::Continue
            }
            Ok(UploadCompletion::YieldBackground) => StepResult::YieldBackground,
            Ok(UploadCompletion::Complete {
                uploaded_mips,
                residency_update,
                storage_v_inverted,
            }) => {
                self.finalize_success(
                    queue,
                    ipc,
                    uploaded_mips,
                    residency_update,
                    storage_v_inverted,
                );
                StepResult::Done
            }
            Err(e) if e.is_queue_access_busy() => StepResult::YieldBackground,
            Err(e) => {
                failed_upload("texture", id, &e);
                self.finalize_failure(ipc);
                StepResult::Done
            }
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
        let Some(t) = queue.pools.texture_pool.get(self.data.asset_id) else {
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
        start_mip: u32,
        uploaded_mips: u32,
        storage_v_inverted: bool,
        begin_stream: bool,
    ) -> bool {
        if uploaded_mips == 0 {
            return false;
        }
        if let Some(t) = queue.pools.texture_pool.get_mut(self.data.asset_id) {
            if !storage_orientation_allows_mark(
                "texture",
                t.asset_id,
                t.mip_levels_resident,
                t.storage_v_inverted,
                storage_v_inverted,
                "after write",
            ) {
                return false;
            }
            t.storage_v_inverted = storage_v_inverted;
            if begin_stream {
                t.begin_mip_stream(start_mip, uploaded_mips);
            } else {
                t.mark_mips_resident(start_mip, uploaded_mips);
            }
            if t.mip_levels_total > 1 && t.mip_levels_resident < t.mip_levels_total {
                logger::trace!(
                    "texture {}: {} of {} contiguous mips exposed; sampling clamped to view LOD {} until remaining mips stream in",
                    t.asset_id,
                    t.mip_levels_resident,
                    t.mip_levels_total,
                    t.mip_levels_resident.saturating_sub(1)
                );
            }
            return true;
        }
        false
    }

    /// Publishes one full-chain mip, resetting residency only at this upload's first completed write.
    fn mark_uploaded_chain_mip(
        &mut self,
        queue: &mut AssetTransferQueue,
        mip_level: u32,
        storage_v_inverted: bool,
        replaces_full_texture: bool,
    ) -> bool {
        let begin_stream = replaces_full_texture && !self.mip_stream_started;
        let marked = self.mark_uploaded_mips(queue, mip_level, 1, storage_v_inverted, begin_stream);
        if marked && replaces_full_texture {
            self.mip_stream_started = true;
        }
        marked
    }

    fn finalize_success(
        &mut self,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
        uploaded_mips: u32,
        residency_update: TextureResidencyUpdate,
        storage_v_inverted: bool,
    ) {
        let id = self.data.asset_id;
        if !queue.texture_upload_generation_is_current(id, self.generation) {
            self.finalize_failure(ipc);
            return;
        }
        let residency_ok = match residency_update {
            TextureResidencyUpdate::None => true,
            TextureResidencyUpdate::Range {
                start_mip,
                mip_count,
            } => self.mark_uploaded_mips(queue, start_mip, mip_count, storage_v_inverted, false),
            TextureResidencyUpdate::Mip {
                mip_level,
                replaces_full_texture,
            } => self.mark_uploaded_chain_mip(
                queue,
                mip_level,
                storage_v_inverted,
                replaces_full_texture,
            ),
        };
        if residency_ok
            && uploaded_mips > 0
            && let Some(t) = queue.pools.texture_pool.get_mut(id)
        {
            t.mark_content_uploaded();
        }
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

    fn finalize_failure(&self, ipc: &mut Option<&mut DualQueueIpc>) {
        send_background_result(
            ipc,
            RendererCommand::SetTexture2DResult(SetTexture2DResult {
                asset_id: self.data.asset_id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::{SetTexture2DData, SetTexture2DFormat, TextureFormat};

    use super::*;

    fn task(high_priority: bool, flip_y: bool, host_format: TextureFormat) -> TextureUploadTask {
        TextureUploadTask::new(
            SetTexture2DData {
                high_priority,
                flip_y,
                ..Default::default()
            },
            SetTexture2DFormat {
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
    fn host_uploads_always_use_unity_orientation() {
        // After the unified-orientation refactor every host upload is treated as Unity
        // V=0 bottom regardless of host format or `flip_y`.
        assert!(task(false, true, TextureFormat::BC7).upload_uses_storage_v_inversion());
        assert!(task(false, false, TextureFormat::BC7).upload_uses_storage_v_inversion());
        assert!(task(false, true, TextureFormat::BC1).upload_uses_storage_v_inversion());
        assert!(task(false, false, TextureFormat::RGBA32).upload_uses_storage_v_inversion());
    }
}
