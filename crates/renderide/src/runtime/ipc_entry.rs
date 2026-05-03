//! IPC-facing entry points on [`super::RendererRuntime`].
//!
//! Owns the per-tick command drain ([`Self::poll_ipc`]) and the `pub(crate)` shims invoked by
//! [`crate::frontend::dispatch::ipc_init::dispatch_ipc_command`] when classifying commands by
//! init phase. Keeping every IPC ingress on one file makes the boundary between the runtime
//! facade and the dispatch routers explicit.

use crate::frontend::InitState;
use crate::frontend::dispatch::renderer_command_kind::renderer_command_variant_tag;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    FrameSubmitData, LightsBufferRendererSubmission, MaterialsUpdateBatch, RendererCommand,
    RendererInitData, ShaderUnload, ShaderUpload,
};

use super::{RendererRuntime, lockstep};

impl RendererRuntime {
    /// Total number of post-handshake IPC commands logged as unhandled (sum of per-variant counters).
    pub fn unhandled_ipc_command_event_total(&self) -> u64 {
        self.unhandled_ipc_command_counts.values().copied().sum()
    }

    pub(crate) fn record_unhandled_renderer_command(&mut self, tag: &'static str) {
        *self.unhandled_ipc_command_counts.entry(tag).or_insert(0) += 1;
    }

    /// Drains IPC and dispatches commands. Each poll batch is ordered so `renderer_init_data` runs
    /// first, then frame submits, then the rest (see [`crate::frontend::RendererFrontend::poll_commands`]).
    pub fn poll_ipc(&mut self) {
        profiling::scope!("ipc::poll_batch");
        crate::frontend::dispatch::shader_material_ipc::drain_pending_shader_resolutions(
            &mut self.pending_shader_resolutions,
            &mut self.backend,
            &mut self.frontend,
        );
        let mut batch = self.frontend.poll_commands();
        trace_ipc_batch(
            &batch,
            self.frontend.init_state(),
            self.pending_shader_resolutions.len(),
        );
        for cmd in batch.drain(..) {
            let _tag = renderer_command_variant_tag(&cmd);
            profiling::scope!("ipc::dispatch", _tag);
            crate::frontend::dispatch::ipc_init::dispatch_ipc_command(self, cmd);
        }
        self.frontend.recycle_command_batch(batch);
    }

    pub(crate) fn on_init_data(&mut self, d: RendererInitData) {
        logger::info!(
            "IPC init data received: output_device={:?} shared_memory_prefix_present={}",
            d.output_device,
            d.shared_memory_prefix.is_some(),
        );
        self.host_camera.output_device = d.output_device;
        if let Some(ref prefix) = d.shared_memory_prefix {
            self.frontend
                .set_shared_memory(SharedMemoryAccessor::new(prefix.clone()));
            logger::info!("Shared memory prefix: {}", prefix);
            let (shm, ipc) = self.frontend.transport_pair_mut();
            if let (Some(shm), Some(ipc)) = (shm, ipc) {
                self.backend.flush_pending_material_batches(shm, ipc);
            }
        }
        self.frontend.set_pending_init(d.clone());
        if let Some(ref mut ipc) = self.frontend.ipc_mut() {
            let settings = self.settings.read().map(|g| g.clone()).unwrap_or_default();
            if !crate::frontend::dispatch::ipc_init::send_renderer_init_result(
                ipc,
                d.output_device,
                &settings,
                None,
            ) {
                logger::error!(
                    "IPC: RendererInitResult was not sent (primary queue full); stopping init handshake"
                );
                self.frontend.set_fatal_error(true);
                return;
            }
        }
        self.frontend.on_init_received();
    }

    pub(crate) fn handle_running_command(&mut self, cmd: RendererCommand) {
        crate::frontend::dispatch::commands::handle_running_command(self, cmd);
    }

    pub(crate) fn on_shader_upload(&mut self, upload: ShaderUpload) {
        crate::frontend::dispatch::shader_material_ipc::on_shader_upload(
            &mut self.pending_shader_resolutions,
            upload,
        );
    }

    pub(crate) fn on_shader_unload(&mut self, unload: ShaderUnload) {
        crate::frontend::dispatch::shader_material_ipc::on_shader_unload(&mut self.backend, unload);
    }

    pub(crate) fn on_materials_update_batch(&mut self, batch: MaterialsUpdateBatch) {
        crate::frontend::dispatch::shader_material_ipc::on_materials_update_batch(
            &mut self.frontend,
            &mut self.backend,
            batch,
        );
    }

    pub(crate) fn on_lights_buffer_renderer_submission(
        &mut self,
        sub: LightsBufferRendererSubmission,
    ) {
        let buffer_id = sub.lights_buffer_unique_id;
        let (shm, ipc) = self.frontend.transport_pair_mut();
        let Some(shm) = shm else {
            logger::warn!("lights_buffer_renderer_submission: no shared memory (id={buffer_id})");
            return;
        };
        crate::frontend::dispatch::lights_ipc::apply_lights_buffer_submission(
            &mut self.scene,
            shm,
            ipc,
            sub,
        );
    }

    pub(crate) fn on_frame_submit(&mut self, data: FrameSubmitData) {
        let prev_frame_index = self.host_camera.frame_index;
        lockstep::trace_duplicate_frame_index_if_interesting(data.frame_index, prev_frame_index);
        crate::frontend::dispatch::frame_submit::process_frame_submit(self, data);
    }
}

fn trace_ipc_batch(batch: &[RendererCommand], init_state: InitState, pending_shaders: usize) {
    if batch.is_empty() || !logger::enabled(logger::LogLevel::Trace) {
        return;
    }
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    for cmd in batch {
        let tag = renderer_command_variant_tag(cmd);
        if let Some((_, count)) = counts.iter_mut().find(|(existing, _)| *existing == tag) {
            *count += 1;
        } else {
            counts.push((tag, 1));
        }
    }
    let mut kinds = String::new();
    for (idx, (tag, count)) in counts.iter().enumerate() {
        if idx > 0 {
            kinds.push_str(", ");
        }
        kinds.push_str(tag);
        kinds.push('=');
        kinds.push_str(&count.to_string());
    }
    logger::trace!(
        "IPC poll batch: commands={} init_state={:?} pending_shader_resolutions={} kinds=[{}]",
        batch.len(),
        init_state,
        pending_shaders,
        kinds,
    );
}
