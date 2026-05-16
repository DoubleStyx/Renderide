//! Host lights buffer submission over shared memory into [`crate::scene::LightCache`].

use crate::ipc::DualQueueIpc;
use crate::ipc::SharedMemoryAccessor;
use crate::scene::SceneCoordinator;
use crate::shared::{
    LIGHT_DATA_HOST_ROW_BYTES, LightData, LightsBufferRendererConsumed,
    LightsBufferRendererSubmission, RendererCommand,
};

use super::super::RendererRuntime;

impl RendererRuntime {
    /// Resolves transport handles and forwards the host light submission to
    /// [`apply_lights_buffer_submission`].
    pub(in crate::runtime) fn apply_lights_buffer_renderer_submission(
        &mut self,
        sub: LightsBufferRendererSubmission,
    ) {
        let buffer_id = sub.lights_buffer_unique_id;
        let (shm, ipc) = self.frontend.transport_pair_mut();
        let Some(shm) = shm else {
            logger::warn!("lights_buffer_renderer_submission: no shared memory (id={buffer_id})");
            return;
        };
        apply_lights_buffer_submission(&mut self.scene, shm, ipc, sub);
    }
}

/// Copies packed light rows from SHM, stores them in the scene cache, and ACKs the host.
fn apply_lights_buffer_submission(
    scene: &mut SceneCoordinator,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
    sub: LightsBufferRendererSubmission,
) {
    let buffer_id = sub.lights_buffer_unique_id;
    let ctx = format!("lights_buffer_renderer_submission id={buffer_id}");
    let vec = match shm.access_copy_memory_packable_rows::<LightData>(
        &sub.lights,
        LIGHT_DATA_HOST_ROW_BYTES,
        Some(&ctx),
    ) {
        Ok(v) => v,
        Err(_e) => {
            return;
        }
    };
    let count = sub.lights_count.max(0) as usize;
    let take = count.min(vec.len());
    if count != vec.len() && !vec.is_empty() {
        logger::debug!(
            "lights_buffer_renderer_submission id={buffer_id}: host count {} SHM elems {} (using {})",
            sub.lights_count,
            vec.len(),
            take
        );
    }
    let payload: Vec<LightData> = vec.into_iter().take(take).collect();
    scene.light_cache_mut().store_full(buffer_id, payload);
    if let Some(ipc) = ipc {
        let _ = ipc.send_background_reliable(RendererCommand::LightsBufferRendererConsumed(
            LightsBufferRendererConsumed {
                global_unique_id: buffer_id,
            },
        ));
    }
}
