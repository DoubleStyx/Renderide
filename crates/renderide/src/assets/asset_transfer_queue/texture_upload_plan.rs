//! Data-oriented Texture2D upload planning and cooperative stepping.

use std::sync::Arc;

use crate::assets::texture::{
    texture_upload_start, MipChainAdvance, Texture2dUploadContext, TextureDataStart,
    TextureMipChainUploader, TextureMipUploadStep, TextureUploadError,
};
use crate::ipc::SharedMemoryAccessor;
use crate::shared::{SetTexture2DData, SetTexture2DFormat};

use super::shared_memory_payload::build_with_optional_owned_payload;

/// Immutable inputs needed to execute one Texture2D upload step.
pub(crate) struct TextureUploadPlan<'a> {
    /// Device used by decode paths and texture-upload start validation.
    pub(crate) device: &'a wgpu::Device,
    /// Queue used for `write_texture` calls.
    pub(crate) queue: &'a wgpu::Queue,
    /// Shared gate held around GPU queue access to avoid write/submit lock inversion.
    pub(crate) gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Destination GPU texture.
    pub(crate) texture: &'a wgpu::Texture,
    /// Host-side format record for the texture.
    pub(crate) format: &'a SetTexture2DFormat,
    /// Resolved GPU texture format.
    pub(crate) wgpu_format: wgpu::TextureFormat,
    /// Host upload command.
    pub(crate) upload: &'a SetTexture2DData,
    /// Whether this upload stores compressed bytes in host-V orientation.
    pub(crate) storage_v_inverted: bool,
}

/// Cooperative Texture2D upload state.
#[derive(Debug)]
pub(crate) struct TextureUploadStepper {
    /// Current step in the texture upload.
    stage: TextureUploadStage,
}

/// One state in the cooperative Texture2D upload.
#[derive(Debug)]
enum TextureUploadStage {
    /// First step: read the shared-memory descriptor and decide subregion vs mip chain.
    Start,
    /// Full mip-chain path with an owned descriptor payload.
    MipChain {
        /// Incremental mip-chain uploader.
        uploader: TextureMipChainUploader,
        /// Owned shared-memory descriptor bytes used across integration ticks.
        payload: Arc<[u8]>,
    },
}

/// Result of one Texture2D upload step.
pub(crate) enum UploadCompletion {
    /// The shared-memory descriptor was not available.
    MissingPayload,
    /// One step completed and the task should run again later.
    Continue,
    /// One mip was uploaded and the task should update residency before continuing.
    UploadedOne {
        /// Total mips uploaded by this chain so far.
        uploaded_mips: u32,
        /// Whether any written mip used host-V-inverted storage.
        storage_v_inverted: bool,
    },
    /// The task is waiting on background decode/downsample work.
    YieldBackground,
    /// The upload finished successfully.
    Complete {
        /// Number of mip levels made resident by this upload.
        uploaded_mips: u32,
        /// Whether any written mip used host-V-inverted storage.
        storage_v_inverted: bool,
    },
}

impl Default for TextureUploadStepper {
    fn default() -> Self {
        Self {
            stage: TextureUploadStage::Start,
        }
    }
}

impl TextureUploadStepper {
    /// Executes at most one Texture2D upload unit.
    pub(crate) fn step(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        plan: TextureUploadPlan<'_>,
    ) -> Result<UploadCompletion, TextureUploadError> {
        profiling::scope!("asset::texture2d_upload_step");
        match &mut self.stage {
            TextureUploadStage::Start => self.start(shm, plan),
            TextureUploadStage::MipChain { uploader, payload } => {
                Self::upload_next_mip(uploader, payload, plan)
            }
        }
    }

    /// Starts the upload by reading the descriptor payload and selecting the upload path.
    fn start(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        plan: TextureUploadPlan<'_>,
    ) -> Result<UploadCompletion, TextureUploadError> {
        profiling::scope!("asset::texture2d_upload_start");
        let start = build_with_optional_owned_payload(
            shm,
            &plan.upload.data,
            |raw| {
                texture_upload_start(&Texture2dUploadContext {
                    device: plan.device,
                    queue: plan.queue,
                    gpu_queue_access_gate: plan.gpu_queue_access_gate,
                    texture: plan.texture,
                    fmt: plan.format,
                    wgpu_format: plan.wgpu_format,
                    upload: plan.upload,
                    raw,
                })
            },
            |start| matches!(start, TextureDataStart::MipChain(_)),
        );
        let Some(start) = start else {
            return Ok(UploadCompletion::MissingPayload);
        };

        match start.result? {
            TextureDataStart::SubregionComplete(uploaded_mips) => Ok(UploadCompletion::Complete {
                uploaded_mips,
                storage_v_inverted: plan.storage_v_inverted,
            }),
            TextureDataStart::MipChain(uploader) => {
                self.stage = TextureUploadStage::MipChain {
                    uploader,
                    payload: start.payload,
                };
                Ok(UploadCompletion::Continue)
            }
        }
    }

    /// Uploads or polls one mip-chain step.
    fn upload_next_mip(
        uploader: &mut TextureMipChainUploader,
        payload: &Arc<[u8]>,
        plan: TextureUploadPlan<'_>,
    ) -> Result<UploadCompletion, TextureUploadError> {
        profiling::scope!("asset::texture2d_upload_next_mip");
        match uploader.upload_next_mip(TextureMipUploadStep {
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
                total_uploaded,
                storage_v_inverted,
            } => Ok(UploadCompletion::UploadedOne {
                uploaded_mips: total_uploaded,
                storage_v_inverted,
            }),
            MipChainAdvance::Finished {
                total_uploaded,
                storage_v_inverted,
            } => Ok(UploadCompletion::Complete {
                uploaded_mips: total_uploaded,
                storage_v_inverted,
            }),
            MipChainAdvance::YieldBackground => Ok(UploadCompletion::YieldBackground),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TextureUploadStage, TextureUploadStepper};

    #[test]
    fn default_stepper_starts_at_payload_read() {
        let stepper = TextureUploadStepper::default();

        assert!(matches!(stepper.stage, TextureUploadStage::Start));
    }
}
