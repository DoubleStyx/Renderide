//! Data-oriented Texture3D upload planning and cooperative stepping.

use std::sync::Arc;

use crate::assets::texture::{
    Texture3dMipAdvance, Texture3dMipChainUploader, Texture3dMipUploadStep, TextureUploadError,
};
use crate::ipc::SharedMemoryAccessor;
use crate::shared::{SetTexture3DData, SetTexture3DFormat};

use super::shared_memory_payload::build_with_optional_owned_payload;

/// Immutable inputs needed to execute one Texture3D upload step.
pub(crate) struct Texture3dUploadPlan<'a> {
    /// Device used by decode paths.
    pub(crate) device: &'a wgpu::Device,
    /// Queue used for `write_texture` calls.
    pub(crate) queue: &'a wgpu::Queue,
    /// Shared gate held around GPU queue access to avoid write/submit lock inversion.
    pub(crate) gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Destination GPU texture.
    pub(crate) texture: &'a wgpu::Texture,
    /// Host-side format record for the texture.
    pub(crate) format: &'a SetTexture3DFormat,
    /// Resolved GPU texture format.
    pub(crate) wgpu_format: wgpu::TextureFormat,
    /// Host upload command.
    pub(crate) upload: &'a SetTexture3DData,
}

/// Cooperative Texture3D upload state.
#[derive(Debug)]
pub(crate) struct Texture3dUploadStepper {
    /// Current step in the texture upload.
    stage: Texture3dUploadStage,
}

/// One state in the cooperative Texture3D upload.
#[derive(Debug)]
enum Texture3dUploadStage {
    /// First step: read the shared-memory descriptor and create the mip-chain uploader.
    Start,
    /// Full mip-chain path with an owned descriptor payload.
    MipChain {
        /// Incremental mip-chain uploader.
        uploader: Texture3dMipChainUploader,
        /// Owned shared-memory descriptor bytes used across integration ticks.
        payload: Arc<[u8]>,
    },
}

/// Result of one Texture3D upload step.
pub(crate) enum Texture3dUploadCompletion {
    /// The shared-memory descriptor was not available.
    MissingPayload,
    /// The task initialized a mip-chain uploader and should run again later.
    Continue,
    /// One mip was uploaded and the task should run again later.
    UploadedOne,
    /// The task is waiting on background decode/downsample work.
    YieldBackground,
    /// The upload finished successfully.
    Complete {
        /// Number of mip levels made resident by this upload.
        uploaded_mips: u32,
    },
}

impl Default for Texture3dUploadStepper {
    fn default() -> Self {
        Self {
            stage: Texture3dUploadStage::Start,
        }
    }
}

impl Texture3dUploadStepper {
    /// Executes at most one Texture3D upload unit.
    pub(crate) fn step(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        plan: Texture3dUploadPlan<'_>,
    ) -> Result<Texture3dUploadCompletion, TextureUploadError> {
        match &mut self.stage {
            Texture3dUploadStage::Start => self.start(shm, plan),
            Texture3dUploadStage::MipChain { uploader, payload } => {
                Self::upload_next_mip(uploader, payload, plan)
            }
        }
    }

    /// Starts the upload by reading the descriptor payload and creating the upload state.
    fn start(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        plan: Texture3dUploadPlan<'_>,
    ) -> Result<Texture3dUploadCompletion, TextureUploadError> {
        let start = build_with_optional_owned_payload(
            shm,
            &plan.upload.data,
            |raw| Texture3dMipChainUploader::new(plan.texture, plan.format, plan.upload, raw),
            |_| true,
        );
        let Some(start) = start else {
            return Ok(Texture3dUploadCompletion::MissingPayload);
        };

        self.stage = Texture3dUploadStage::MipChain {
            uploader: start.result?,
            payload: start.payload,
        };
        Ok(Texture3dUploadCompletion::Continue)
    }

    /// Uploads or polls one mip-chain step.
    fn upload_next_mip(
        uploader: &mut Texture3dMipChainUploader,
        payload: &Arc<[u8]>,
        plan: Texture3dUploadPlan<'_>,
    ) -> Result<Texture3dUploadCompletion, TextureUploadError> {
        match uploader.upload_next_mip(Texture3dMipUploadStep {
            device: plan.device,
            queue: plan.queue,
            gpu_queue_access_gate: plan.gpu_queue_access_gate,
            texture: plan.texture,
            fmt: plan.format,
            wgpu_format: plan.wgpu_format,
            upload: plan.upload,
            payload,
        })? {
            Texture3dMipAdvance::UploadedOne => Ok(Texture3dUploadCompletion::UploadedOne),
            Texture3dMipAdvance::Finished { total_uploaded } => {
                Ok(Texture3dUploadCompletion::Complete {
                    uploaded_mips: total_uploaded,
                })
            }
            Texture3dMipAdvance::YieldBackground => Ok(Texture3dUploadCompletion::YieldBackground),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Texture3dUploadStage, Texture3dUploadStepper};

    #[test]
    fn default_stepper_starts_at_payload_read() {
        let stepper = Texture3dUploadStepper::default();

        assert!(matches!(stepper.stage, Texture3dUploadStage::Start));
    }
}
