//! Cooperative [`SetTexture2DData`] integration: sub-region or one mip per step.

use std::sync::Arc;

use crate::assets::texture::{
    texture_upload_start, MipChainAdvance, TextureDataStart, TextureMipChainUploader,
    TextureMipUploadStep, TextureUploadError,
};
use crate::gpu::GpuLimits;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture2DData, SetTexture2DFormat, SetTexture2DResult,
    TextureUpdateResultType,
};

use super::integrator::StepResult;
use super::AssetTransferQueue;

/// Stage for one texture data upload.
#[derive(Debug)]
enum TextureStage {
    /// First step: sub-region or create mip-chain uploader.
    Start,
    /// Upload one mip per [`super::integrator::drain_asset_tasks`] step.
    MipChain { uploader: TextureMipChainUploader },
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
        _gpu_limits: &Arc<GpuLimits>,
        gpu_queue: &wgpu::Queue,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let id = self.data.asset_id;
        let (tex_arc, _) = match queue.texture_pool.get_texture(id) {
            Some(t) => (t.texture.clone(), t.wgpu_format),
            None => {
                logger::warn!("texture {id}: missing GPU texture during integration step");
                return StepResult::Done;
            }
        };
        let texture = tex_arc.as_ref();

        match &mut self.stage {
            TextureStage::Start => {
                let fmt = &self.format;
                let wgpu_format = self.wgpu_format;
                let upload = &self.data;
                let start = shm.with_read_bytes(&upload.data, |raw| {
                    Some(texture_upload_start(
                        device.as_ref(),
                        gpu_queue,
                        texture,
                        fmt,
                        wgpu_format,
                        upload,
                        raw,
                    ))
                });
                let Some(start) = start else {
                    logger::warn!("texture {id}: shared memory slice missing");
                    return StepResult::Done;
                };
                match start {
                    Ok(TextureDataStart::SubregionComplete(uploaded_mips)) => {
                        self.finalize_success(queue, ipc, uploaded_mips);
                        StepResult::Done
                    }
                    Ok(TextureDataStart::MipChain(uploader)) => {
                        self.stage = TextureStage::MipChain { uploader };
                        StepResult::Continue
                    }
                    Err(e) => {
                        logger::warn!("texture {id}: upload failed: {e}");
                        StepResult::Done
                    }
                }
            }
            TextureStage::MipChain { uploader } => {
                let fmt = &self.format;
                let wgpu_format = self.wgpu_format;
                let upload = &self.data;
                let want = upload.data.length.max(0) as usize;
                let mip_out = shm.with_read_bytes(&upload.data, |raw| {
                    if raw.len() < want {
                        return Some(Err(TextureUploadError::from(format!(
                            "raw shorter than descriptor (need {want}, got {})",
                            raw.len()
                        ))));
                    }
                    let payload = &raw[..want];
                    Some(uploader.upload_next_mip(TextureMipUploadStep {
                        device: device.as_ref(),
                        queue: gpu_queue,
                        texture,
                        fmt,
                        wgpu_format,
                        upload,
                        payload,
                    }))
                });
                let Some(mip_result) = mip_out else {
                    logger::warn!("texture {id}: shared memory slice missing");
                    return StepResult::Done;
                };
                match mip_result {
                    Ok(MipChainAdvance::UploadedOne) => StepResult::Continue,
                    Ok(MipChainAdvance::Finished { total_uploaded }) => {
                        self.finalize_success(queue, ipc, total_uploaded);
                        StepResult::Done
                    }
                    Err(e) => {
                        logger::warn!("texture {id}: upload failed: {e}");
                        StepResult::Done
                    }
                }
            }
        }
    }

    fn finalize_success(
        &mut self,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
        uploaded_mips: u32,
    ) {
        let id = self.data.asset_id;
        if uploaded_mips > 0 {
            if let Some(t) = queue.texture_pool.get_texture_mut(id) {
                let start = self.data.start_mip_level.max(0) as u32;
                let end_exclusive = start.saturating_add(uploaded_mips).min(t.mip_levels_total);
                t.mip_levels_resident = t.mip_levels_resident.max(end_exclusive);
            }
        }
        if let Some(ipc) = ipc.as_mut() {
            ipc.send_background(RendererCommand::SetTexture2DResult(SetTexture2DResult {
                asset_id: id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }));
        }
        logger::trace!("texture {id}: data upload ok ({uploaded_mips} mips, integrator)");
    }
}
