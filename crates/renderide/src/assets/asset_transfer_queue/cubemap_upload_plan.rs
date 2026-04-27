//! Data-oriented cubemap upload planning and cooperative stepping.

use std::sync::Arc;

use crate::assets::texture::{
    CubemapFaceMipUploadStep, CubemapMipChainUploader, MipChainAdvance, TextureUploadError,
};
use crate::ipc::SharedMemoryAccessor;
use crate::shared::{SetCubemapData, SetCubemapFormat};

use super::shared_memory_payload::build_with_optional_owned_payload;

/// Immutable inputs needed to execute one cubemap upload step.
pub(crate) struct CubemapUploadPlan<'a> {
    /// Device used by decode paths.
    pub(crate) device: &'a wgpu::Device,
    /// Queue used for `write_texture` calls.
    pub(crate) queue: &'a wgpu::Queue,
    /// Shared gate held around GPU queue access to avoid write/submit lock inversion.
    pub(crate) gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Destination GPU cubemap texture.
    pub(crate) texture: &'a wgpu::Texture,
    /// Host-side format record for the cubemap.
    pub(crate) format: &'a SetCubemapFormat,
    /// Resolved GPU texture format.
    pub(crate) wgpu_format: wgpu::TextureFormat,
    /// Host upload command.
    pub(crate) upload: &'a SetCubemapData,
}

/// Cooperative cubemap upload state.
#[derive(Debug)]
pub(crate) struct CubemapUploadStepper {
    /// Current step in the cubemap upload.
    stage: CubemapUploadStage,
}

/// One state in the cooperative cubemap upload.
#[derive(Debug)]
enum CubemapUploadStage {
    /// First step: read the shared-memory descriptor and create the face/mip uploader.
    Start,
    /// Full face/mip-chain path with an owned descriptor payload.
    Chain {
        /// Incremental face/mip uploader.
        uploader: CubemapMipChainUploader,
        /// Owned shared-memory descriptor bytes used across integration ticks.
        payload: Arc<[u8]>,
    },
}

/// Result of one cubemap upload step.
#[expect(
    variant_size_differences,
    reason = "completion variants carry the exact upload metadata needed by the task finalizer"
)]
pub(crate) enum CubemapUploadCompletion {
    /// The shared-memory descriptor was not available.
    MissingPayload,
    /// The task initialized a face/mip uploader and should run again later.
    Continue,
    /// One face/mip was uploaded.
    UploadedOne {
        /// Whether the written face/mip used host-V-inverted storage.
        storage_v_inverted: bool,
    },
    /// The task is waiting on background decode/downsample work.
    YieldBackground,
    /// The upload finished successfully.
    Complete {
        /// Number of face/mip writes completed by this upload.
        uploaded_face_mips: u32,
        /// Whether any written face/mip used host-V-inverted storage.
        storage_v_inverted: bool,
    },
}

impl Default for CubemapUploadStepper {
    fn default() -> Self {
        Self {
            stage: CubemapUploadStage::Start,
        }
    }
}

impl CubemapUploadStepper {
    /// Executes at most one cubemap upload unit.
    pub(crate) fn step(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        plan: CubemapUploadPlan<'_>,
    ) -> Result<CubemapUploadCompletion, TextureUploadError> {
        match &mut self.stage {
            CubemapUploadStage::Start => self.start(shm, plan),
            CubemapUploadStage::Chain { uploader, payload } => {
                Self::upload_next_face_mip(uploader, payload, plan)
            }
        }
    }

    /// Starts the upload by reading the descriptor payload and creating the upload state.
    fn start(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        plan: CubemapUploadPlan<'_>,
    ) -> Result<CubemapUploadCompletion, TextureUploadError> {
        let start = build_with_optional_owned_payload(
            shm,
            &plan.upload.data,
            |raw| CubemapMipChainUploader::new(plan.texture, plan.format, plan.upload, raw),
            |_| true,
        );
        let Some(start) = start else {
            return Ok(CubemapUploadCompletion::MissingPayload);
        };

        self.stage = CubemapUploadStage::Chain {
            uploader: start.result?,
            payload: start.payload,
        };
        Ok(CubemapUploadCompletion::Continue)
    }

    /// Uploads or polls one face/mip-chain step.
    fn upload_next_face_mip(
        uploader: &mut CubemapMipChainUploader,
        payload: &Arc<[u8]>,
        plan: CubemapUploadPlan<'_>,
    ) -> Result<CubemapUploadCompletion, TextureUploadError> {
        match uploader.upload_next_face_mip(CubemapFaceMipUploadStep {
            device: plan.device,
            queue: plan.queue,
            gpu_queue_access_gate: plan.gpu_queue_access_gate,
            texture: plan.texture,
            fmt: plan.format,
            wgpu_format: plan.wgpu_format,
            upload: plan.upload,
            payload,
        })? {
            MipChainAdvance::UploadedOne {
                storage_v_inverted, ..
            } => Ok(CubemapUploadCompletion::UploadedOne { storage_v_inverted }),
            MipChainAdvance::Finished {
                total_uploaded,
                storage_v_inverted,
            } => Ok(CubemapUploadCompletion::Complete {
                uploaded_face_mips: total_uploaded,
                storage_v_inverted,
            }),
            MipChainAdvance::YieldBackground => Ok(CubemapUploadCompletion::YieldBackground),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CubemapUploadStage, CubemapUploadStepper};

    #[test]
    fn default_stepper_starts_at_payload_read() {
        let stepper = CubemapUploadStepper::default();

        assert!(matches!(stepper.stage, CubemapUploadStage::Start));
    }
}
