//! Cooperative [`SetTexture2DData`] integration: sub-region or one mip per step.

use std::sync::Arc;

use crate::assets::texture::{
    texture_upload_start, upload_uses_storage_v_inversion, MipChainAdvance, Texture2dUploadContext,
    TextureDataStart, TextureMipChainUploader, TextureMipUploadStep,
};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture2DData, SetTexture2DFormat, SetTexture2DResult,
    TextureUpdateResultType,
};

use super::integrator::StepResult;
use super::shared_memory_payload::build_with_optional_owned_payload;
use super::AssetTransferQueue;

/// Stage for one texture data upload.
#[derive(Debug)]
enum TextureStage {
    /// First step: sub-region or create mip-chain uploader.
    Start,
    /// Upload one mip per [`super::integrator::drain_asset_tasks`] step from an owned payload copy.
    MipChain {
        uploader: TextureMipChainUploader,
        payload: Arc<[u8]>,
    },
}

/// One in-flight Texture2D data upload.
#[derive(Debug)]
pub struct TextureUploadTask {
    data: SetTexture2DData,
    /// Cached from [`AssetTransferQueue::texture_formats`] at enqueue time.
    format: SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    stage: TextureStage,
}

impl TextureUploadTask {
    /// Builds a task; `fmt` and `wgpu_format` must match the resident [`crate::resources::GpuTexture2d`].
    pub fn new(
        data: SetTexture2DData,
        format: SetTexture2DFormat,
        wgpu_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            data,
            format,
            wgpu_format,
            stage: TextureStage::Start,
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
        write_texture_submit_gate: &crate::gpu::WriteTextureSubmitGate,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let id = self.data.asset_id;
        let storage_v_inverted = self.upload_uses_storage_v_inversion();
        if !self.storage_orientation_allows_upload(queue, storage_v_inverted) {
            return StepResult::Done;
        }
        let Some(tex_arc) = self.resident_texture_arc(queue) else {
            return StepResult::Done;
        };
        let texture = tex_arc.as_ref();

        match &mut self.stage {
            TextureStage::Start => {
                let fmt = &self.format;
                let wgpu_format = self.wgpu_format;
                let upload = &self.data;
                let start = build_with_optional_owned_payload(
                    shm,
                    &upload.data,
                    |raw| {
                        texture_upload_start(&Texture2dUploadContext {
                            device: device.as_ref(),
                            queue: gpu_queue,
                            write_texture_submit_gate,
                            texture,
                            fmt,
                            wgpu_format,
                            upload,
                            raw,
                        })
                    },
                    |start| matches!(start, TextureDataStart::MipChain(_)),
                );
                let Some(start) = start else {
                    logger::warn!("texture {id}: shared memory slice missing");
                    return StepResult::Done;
                };
                let payload = start.payload;
                match start.result {
                    Ok(TextureDataStart::SubregionComplete(uploaded_mips)) => {
                        self.finalize_success(queue, ipc, uploaded_mips, storage_v_inverted);
                        StepResult::Done
                    }
                    Ok(TextureDataStart::MipChain(uploader)) => {
                        self.stage = TextureStage::MipChain { uploader, payload };
                        StepResult::Continue
                    }
                    Err(e) => {
                        logger::warn!("texture {id}: upload failed: {e}");
                        StepResult::Done
                    }
                }
            }
            TextureStage::MipChain { uploader, payload } => {
                let fmt = &self.format;
                let wgpu_format = self.wgpu_format;
                let upload = &self.data;
                let mip_result = uploader.upload_next_mip(TextureMipUploadStep {
                    device: device.as_ref(),
                    queue: gpu_queue,
                    write_texture_submit_gate,
                    texture,
                    fmt,
                    wgpu_format,
                    upload,
                    payload,
                });
                match mip_result {
                    Ok(MipChainAdvance::UploadedOne {
                        total_uploaded,
                        storage_v_inverted,
                    }) => {
                        self.mark_uploaded_mips(queue, total_uploaded, storage_v_inverted);
                        StepResult::Continue
                    }
                    Ok(MipChainAdvance::Finished {
                        total_uploaded,
                        storage_v_inverted,
                    }) => {
                        self.finalize_success(queue, ipc, total_uploaded, storage_v_inverted);
                        StepResult::Done
                    }
                    Ok(MipChainAdvance::YieldBackground) => StepResult::YieldBackground,
                    Err(e) => {
                        logger::warn!("texture {id}: upload failed: {e}");
                        StepResult::Done
                    }
                }
            }
        }
    }

    /// Clones the resident GPU texture handle for this upload step.
    fn resident_texture_arc(&self, queue: &AssetTransferQueue) -> Option<Arc<wgpu::Texture>> {
        queue
            .texture_pool
            .get_texture(self.data.asset_id)
            .map(|t| t.texture.clone())
            .or_else(|| {
                logger::warn!(
                    "texture {}: missing GPU texture during integration step",
                    self.data.asset_id
                );
                None
            })
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
        if t.mip_levels_resident > 0 && t.storage_v_inverted != storage_v_inverted {
            logger::warn!(
                "texture {}: upload storage orientation mismatch (resident inverted={}, upload inverted={}); aborting to avoid mixed-orientation mips",
                t.asset_id,
                t.storage_v_inverted,
                storage_v_inverted
            );
            return false;
        }
        true
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
            if t.mip_levels_resident > 0 && t.storage_v_inverted != storage_v_inverted {
                logger::warn!(
                    "texture {}: upload storage orientation mismatch after write (resident inverted={}, upload inverted={})",
                    t.asset_id,
                    t.storage_v_inverted,
                    storage_v_inverted
                );
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
        &mut self,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
        uploaded_mips: u32,
        storage_v_inverted: bool,
    ) {
        let id = self.data.asset_id;
        self.mark_uploaded_mips(queue, uploaded_mips, storage_v_inverted);
        if let Some(ipc) = ipc.as_mut() {
            let _ = ipc.send_background(RendererCommand::SetTexture2DResult(SetTexture2DResult {
                asset_id: id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }));
        }
        logger::trace!("texture {id}: data upload ok ({uploaded_mips} mips, integrator)");
    }
}
