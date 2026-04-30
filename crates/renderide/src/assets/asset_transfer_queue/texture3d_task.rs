//! Cooperative [`SetTexture3DData`] integration: one mip per step.

use std::sync::Arc;

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture3DData, SetTexture3DFormat, SetTexture3DResult,
    TextureUpdateResultType,
};

use super::AssetTransferQueue;
use super::integrator::StepResult;
use super::texture_task_common::{
    failed_upload, missing_payload, resident_texture_arc, send_background_result,
};
use super::texture3d_upload_plan::{
    Texture3dUploadCompletion, Texture3dUploadPlan, Texture3dUploadStepper,
};

/// One in-flight Texture3D data upload.
#[derive(Debug)]
pub struct Texture3dUploadTask {
    data: SetTexture3DData,
    /// Cached from [`AssetTransferQueue::texture3d_formats`] at enqueue time.
    format: SetTexture3DFormat,
    wgpu_format: wgpu::TextureFormat,
    stepper: Texture3dUploadStepper,
}

impl Texture3dUploadTask {
    /// Builds a task; `fmt` and `wgpu_format` must match the resident [`crate::gpu_pools::GpuTexture3d`].
    pub fn new(
        data: SetTexture3DData,
        format: SetTexture3DFormat,
        wgpu_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            data,
            format,
            wgpu_format,
            stepper: Texture3dUploadStepper::default(),
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
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let id = self.data.asset_id;
        let Some(tex_arc) = resident_texture_arc(
            "texture3d",
            id,
            queue
                .texture3d_pool
                .get_texture(id)
                .map(|texture| texture.texture.clone()),
        ) else {
            return StepResult::Done;
        };
        let texture = tex_arc.as_ref();

        let completion = self.stepper.step(
            shm,
            Texture3dUploadPlan {
                device: device.as_ref(),
                queue: gpu_queue,
                gpu_queue_access_gate,
                texture,
                format: &self.format,
                wgpu_format: self.wgpu_format,
                upload: &self.data,
            },
        );
        match completion {
            Ok(Texture3dUploadCompletion::MissingPayload) => missing_payload("texture3d", id),
            Ok(Texture3dUploadCompletion::Continue | Texture3dUploadCompletion::UploadedOne) => {
                StepResult::Continue
            }
            Ok(Texture3dUploadCompletion::YieldBackground) => StepResult::YieldBackground,
            Ok(Texture3dUploadCompletion::Complete { uploaded_mips }) => {
                self.finalize_success(queue, ipc, uploaded_mips);
                StepResult::Done
            }
            Err(e) => failed_upload("texture3d", id, &e),
        }
    }

    fn finalize_success(
        &self,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
        uploaded_mips: u32,
    ) {
        let id = self.data.asset_id;
        if uploaded_mips > 0
            && let Some(t) = queue.texture3d_pool.get_texture_mut(id)
        {
            t.mip_levels_resident = t
                .mip_levels_resident
                .max(uploaded_mips.min(t.mip_levels_total));
        }
        send_background_result(
            ipc,
            RendererCommand::SetTexture3DResult(SetTexture3DResult {
                asset_id: id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }),
        );
        logger::trace!("texture3d {id}: data upload ok ({uploaded_mips} mips, integrator)");
    }
}
