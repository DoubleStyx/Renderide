//! Cooperative [`SetCubemapData`] integration: one face × mip per step.

use std::sync::Arc;

use crate::assets::texture::{
    upload_uses_storage_v_inversion, CubemapFaceMipUploadStep, CubemapMipChainUploader,
    MipChainAdvance,
};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetCubemapData, SetCubemapFormat, SetCubemapResult, TextureUpdateResultType,
};

use super::integrator::StepResult;
use super::shared_memory_payload::build_with_optional_owned_payload;
use super::AssetTransferQueue;

/// Stage for one cubemap data upload.
#[derive(Debug)]
enum CubemapStage {
    Start,
    Chain {
        uploader: CubemapMipChainUploader,
        payload: Arc<[u8]>,
    },
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
                let start = build_with_optional_owned_payload(
                    shm,
                    &upload.data,
                    |raw| CubemapMipChainUploader::new(texture, fmt, upload, raw),
                    |_| true,
                );
                let Some(uploader_result) = start else {
                    logger::warn!("cubemap {id}: shared memory slice missing");
                    return StepResult::Done;
                };
                let payload = uploader_result.payload;
                match uploader_result.result {
                    Ok(uploader) => {
                        self.stage = CubemapStage::Chain { uploader, payload };
                        StepResult::Continue
                    }
                    Err(e) => {
                        logger::warn!("cubemap {id}: upload init failed: {e}");
                        StepResult::Done
                    }
                }
            }
            CubemapStage::Chain { uploader, payload } => {
                let fmt = &self.format;
                let wgpu_format = self.wgpu_format;
                let upload = &self.data;
                let mip_result = uploader.upload_next_face_mip(CubemapFaceMipUploadStep {
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
                        storage_v_inverted, ..
                    }) => {
                        self.mark_storage_orientation(queue, storage_v_inverted);
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
                        logger::warn!("cubemap {id}: upload failed: {e}");
                        StepResult::Done
                    }
                }
            }
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
        let Some(t) = queue.cubemap_pool.get_texture(self.data.asset_id) else {
            return true;
        };
        if t.mip_levels_resident > 0 && t.storage_v_inverted != storage_v_inverted {
            logger::warn!(
                "cubemap {}: upload storage orientation mismatch (resident inverted={}, upload inverted={}); aborting to avoid mixed-orientation face mips",
                t.asset_id,
                t.storage_v_inverted,
                storage_v_inverted
            );
            return false;
        }
        true
    }

    /// Records the storage orientation after a successful face-mip write.
    fn mark_storage_orientation(&self, queue: &mut AssetTransferQueue, storage_v_inverted: bool) {
        if let Some(t) = queue.cubemap_pool.get_texture_mut(self.data.asset_id) {
            if t.mip_levels_resident > 0 && t.storage_v_inverted != storage_v_inverted {
                logger::warn!(
                    "cubemap {}: upload storage orientation mismatch after write (resident inverted={}, upload inverted={})",
                    t.asset_id,
                    t.storage_v_inverted,
                    storage_v_inverted
                );
                return;
            }
            t.storage_v_inverted = storage_v_inverted;
        }
    }

    fn finalize_success(
        &mut self,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
        uploaded_face_mips: u32,
        storage_v_inverted: bool,
    ) {
        let id = self.data.asset_id;
        if uploaded_face_mips > 0 {
            if let Some(t) = queue.cubemap_pool.get_texture_mut(id) {
                if t.mip_levels_resident > 0 && t.storage_v_inverted != storage_v_inverted {
                    logger::warn!(
                        "cubemap {}: upload storage orientation mismatch at finalize (resident inverted={}, upload inverted={})",
                        t.asset_id,
                        t.storage_v_inverted,
                        storage_v_inverted
                    );
                    return;
                }
                t.storage_v_inverted = storage_v_inverted;
                t.mip_levels_resident = t.mip_levels_total;
            }
        }
        if let Some(ipc) = ipc.as_mut() {
            let _ = ipc.send_background(RendererCommand::SetCubemapResult(SetCubemapResult {
                asset_id: id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }));
        }
        logger::trace!("cubemap {id}: data upload ok ({uploaded_face_mips} face-mips, integrator)");
    }
}
