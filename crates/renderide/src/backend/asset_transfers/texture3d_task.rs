//! Cooperative [`SetTexture3DData`] integration: one mip per step.

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture3DData, SetTexture3DFormat, SetTexture3DResult,
    TextureUpdateResultType,
};

use super::AssetTransferQueue;
use super::integrator::StepResult;
use super::texture_task_common::{
    TextureTaskGpu, failed_upload, missing_payload, resident_texture_arc, send_background_result,
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
    generation: u64,
    stepper: Texture3dUploadStepper,
}

impl Texture3dUploadTask {
    /// Builds a task; `fmt` and `wgpu_format` must match the resident [`crate::gpu_pools::GpuTexture3d`].
    pub fn new(
        data: SetTexture3DData,
        format: SetTexture3DFormat,
        wgpu_format: wgpu::TextureFormat,
        generation: u64,
    ) -> Self {
        Self {
            data,
            format,
            wgpu_format,
            generation,
            stepper: Texture3dUploadStepper::default(),
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
        if !queue.texture3d_upload_generation_is_current(id, self.generation) {
            logger::trace!(
                "texture3d {id}: dropped stale data upload generation {}",
                self.generation
            );
            self.finalize_failure(ipc);
            return StepResult::Done;
        }
        let Some(tex_arc) = resident_texture_arc(
            "texture3d",
            id,
            queue
                .pools
                .texture3d_pool
                .get(id)
                .map(|texture| texture.texture.clone()),
        ) else {
            self.finalize_failure(ipc);
            return StepResult::Done;
        };
        let texture = tex_arc.as_ref();

        let completion = self.stepper.step(
            shm,
            Texture3dUploadPlan {
                device: gpu.device.as_ref(),
                queue: gpu.queue,
                gpu_queue_access_gate: gpu.queue_access_gate,
                queue_access_mode: gpu.queue_access_mode,
                texture,
                format: &self.format,
                wgpu_format: self.wgpu_format,
                upload: &self.data,
            },
        );
        match completion {
            Ok(Texture3dUploadCompletion::MissingPayload) => {
                missing_payload("texture3d", id);
                self.finalize_failure(ipc);
                StepResult::Done
            }
            Ok(Texture3dUploadCompletion::Continue | Texture3dUploadCompletion::UploadedOne) => {
                StepResult::Continue
            }
            Ok(Texture3dUploadCompletion::YieldBackground) => StepResult::YieldBackground,
            Ok(Texture3dUploadCompletion::Complete { uploaded_mips }) => {
                self.finalize_success(queue, ipc, uploaded_mips);
                StepResult::Done
            }
            Err(e) if e.is_queue_access_busy() => StepResult::YieldBackground,
            Err(e) => {
                failed_upload("texture3d", id, &e);
                self.finalize_failure(ipc);
                StepResult::Done
            }
        }
    }

    fn finalize_success(
        &self,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
        uploaded_mips: u32,
    ) {
        let id = self.data.asset_id;
        if !queue.texture3d_upload_generation_is_current(id, self.generation) {
            self.finalize_failure(ipc);
            return;
        }
        if uploaded_mips > 0
            && let Some(t) = queue.pools.texture3d_pool.get_mut(id)
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

    fn finalize_failure(&self, ipc: &mut Option<&mut DualQueueIpc>) {
        send_background_result(
            ipc,
            RendererCommand::SetTexture3DResult(SetTexture3DResult {
                asset_id: self.data.asset_id,
                r#type: TextureUpdateResultType(TextureUpdateResultType::DATA_UPLOAD),
                instance_changed: false,
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::{SetTexture3DData, SetTexture3DFormat};

    use super::*;

    fn task(high_priority: bool) -> Texture3dUploadTask {
        Texture3dUploadTask::new(
            SetTexture3DData {
                high_priority,
                ..Default::default()
            },
            SetTexture3DFormat::default(),
            wgpu::TextureFormat::Rgba8Unorm,
            1,
        )
    }

    #[test]
    fn high_priority_reflects_upload_command() {
        assert!(task(true).high_priority());
        assert!(!task(false).high_priority());
    }
}
