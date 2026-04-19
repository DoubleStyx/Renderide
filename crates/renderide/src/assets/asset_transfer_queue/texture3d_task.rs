//! Cooperative [`SetTexture3DData`] integration: one mip per step.

use std::sync::Arc;

use crate::assets::texture::{
    Texture3dMipAdvance, Texture3dMipChainUploader, Texture3dMipUploadStep, TextureUploadError,
};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture3DData, SetTexture3DFormat, SetTexture3DResult,
    TextureUpdateResultType,
};

use super::integrator::StepResult;
use super::AssetTransferQueue;

/// Stage for one Texture3D data upload.
#[derive(Debug)]
enum Texture3dStage {
    /// First step: create mip-chain uploader.
    Start,
    /// Upload one mip per drain step.
    MipChain { uploader: Texture3dMipChainUploader },
}

/// One in-flight Texture3D data upload.
#[derive(Debug)]
pub struct Texture3dUploadTask {
    data: SetTexture3DData,
    /// Cached from [`AssetTransferQueue::texture3d_formats`] at enqueue time.
    format: SetTexture3DFormat,
    wgpu_format: wgpu::TextureFormat,
    stage: Texture3dStage,
}

impl Texture3dUploadTask {
    /// Builds a task; `fmt` and `wgpu_format` must match the resident [`crate::resources::GpuTexture3d`].
    pub fn new(
        data: SetTexture3DData,
        format: SetTexture3DFormat,
        wgpu_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            data,
            format,
            wgpu_format,
            stage: Texture3dStage::Start,
        }
    }

    /// [`SetTexture3DData::high_priority`].
    pub fn high_priority(&self) -> bool {
        self.data.high_priority
    }

    /// Runs at most one integration sub-step.
    pub fn step(
        &mut self,
        queue: &mut AssetTransferQueue,
        device: &Arc<wgpu::Device>,
        gpu_queue: &wgpu::Queue,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let id = self.data.asset_id;
        let tex_arc = match queue.texture3d_pool.get_texture(id) {
            Some(t) => t.texture.clone(),
            None => {
                logger::warn!("texture3d {id}: missing GPU texture during integration step");
                return StepResult::Done;
            }
        };
        let texture = tex_arc.as_ref();

        match &mut self.stage {
            Texture3dStage::Start => {
                let fmt = &self.format;
                let upload = &self.data;
                let start = shm.with_read_bytes(&upload.data, |raw| {
                    Some(Texture3dMipChainUploader::new(texture, fmt, upload, raw))
                });
                let Some(uploader_result) = start else {
                    logger::warn!("texture3d {id}: shared memory slice missing");
                    return StepResult::Done;
                };
                match uploader_result {
                    Ok(uploader) => {
                        self.stage = Texture3dStage::MipChain { uploader };
                        StepResult::Continue
                    }
                    Err(e) => {
                        logger::warn!("texture3d {id}: upload init failed: {e}");
                        StepResult::Done
                    }
                }
            }
            Texture3dStage::MipChain { uploader } => {
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
                    Some(uploader.upload_next_mip(Texture3dMipUploadStep {
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
                    logger::warn!("texture3d {id}: shared memory slice missing");
                    return StepResult::Done;
                };
                match mip_result {
                    Ok(Texture3dMipAdvance::UploadedOne) => StepResult::Continue,
                    Ok(Texture3dMipAdvance::Finished { total_uploaded }) => {
                        self.finalize_success(queue, ipc, total_uploaded);
                        StepResult::Done
                    }
                    Err(e) => {
                        logger::warn!("texture3d {id}: upload failed: {e}");
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
            if let Some(t) = queue.texture3d_pool.get_texture_mut(id) {
                t.mip_levels_resident = t
                    .mip_levels_resident
                    .max(uploaded_mips.min(t.mip_levels_total));
            }
        }
        if let Some(ipc) = ipc.as_mut() {
            let _ = ipc.send_background(RendererCommand::SetTexture3DResult(SetTexture3DResult {
                asset_id: id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }));
        }
        logger::trace!("texture3d {id}: data upload ok ({uploaded_mips} mips, integrator)");
    }
}
