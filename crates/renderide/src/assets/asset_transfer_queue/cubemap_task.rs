//! Cooperative [`SetCubemapData`] integration: one face × mip per step.

use std::sync::Arc;

use crate::assets::texture::{
    CubemapFaceMipUploadStep, CubemapMipChainUploader, MipChainAdvance, TextureUploadError,
};
use crate::gpu::GpuLimits;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetCubemapData, SetCubemapFormat, SetCubemapResult, TextureUpdateResultType,
};

use super::integrator::StepResult;
use super::AssetTransferQueue;

/// Stage for one cubemap data upload.
#[derive(Debug)]
enum CubemapStage {
    Start,
    Chain { uploader: CubemapMipChainUploader },
}

/// One in-flight cubemap data upload.
#[derive(Debug)]
pub struct CubemapUploadTask {
    data: SetCubemapData,
    format: SetCubemapFormat,
    wgpu_format: wgpu::TextureFormat,
    stage: CubemapStage,
}

impl CubemapUploadTask {
    /// Builds a task; `fmt` and `wgpu_format` must match the resident [`crate::resources::GpuCubemap`].
    pub fn new(
        data: SetCubemapData,
        format: SetCubemapFormat,
        wgpu_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            data,
            format,
            wgpu_format,
            stage: CubemapStage::Start,
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
        _gpu_limits: &Arc<GpuLimits>,
        gpu_queue: &wgpu::Queue,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let id = self.data.asset_id;
        let tex_arc = match queue.cubemap_pool.get_texture(id) {
            Some(t) => t.texture.clone(),
            None => {
                logger::warn!("cubemap {id}: missing GPU texture during integration step");
                return StepResult::Done;
            }
        };
        let texture = tex_arc.as_ref();

        match &mut self.stage {
            CubemapStage::Start => {
                let fmt = &self.format;
                let upload = &self.data;
                let start = shm.with_read_bytes(&upload.data, |raw| {
                    Some(CubemapMipChainUploader::new(texture, fmt, upload, raw))
                });
                let Some(uploader_result) = start else {
                    logger::warn!("cubemap {id}: shared memory slice missing");
                    return StepResult::Done;
                };
                match uploader_result {
                    Ok(uploader) => {
                        self.stage = CubemapStage::Chain { uploader };
                        StepResult::Continue
                    }
                    Err(e) => {
                        logger::warn!("cubemap {id}: upload init failed: {e}");
                        StepResult::Done
                    }
                }
            }
            CubemapStage::Chain { uploader } => {
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
                    Some(uploader.upload_next_face_mip(CubemapFaceMipUploadStep {
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
                    logger::warn!("cubemap {id}: shared memory slice missing");
                    return StepResult::Done;
                };
                match mip_result {
                    Ok(MipChainAdvance::UploadedOne) => StepResult::Continue,
                    Ok(MipChainAdvance::Finished { total_uploaded }) => {
                        self.finalize_success(queue, ipc, total_uploaded);
                        StepResult::Done
                    }
                    Err(e) => {
                        logger::warn!("cubemap {id}: upload failed: {e}");
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
        uploaded_face_mips: u32,
    ) {
        let id = self.data.asset_id;
        if uploaded_face_mips > 0 {
            if let Some(t) = queue.cubemap_pool.get_texture_mut(id) {
                t.mip_levels_resident = t.mip_levels_total;
            }
        }
        if let Some(ipc) = ipc.as_mut() {
            ipc.send_background(RendererCommand::SetCubemapResult(SetCubemapResult {
                asset_id: id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }));
        }
        logger::trace!("cubemap {id}: data upload ok ({uploaded_face_mips} face-mips, integrator)");
    }
}
