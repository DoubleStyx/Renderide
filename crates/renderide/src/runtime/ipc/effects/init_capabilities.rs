//! Renderer init data, init capabilities, decoupling config, desktop config, and host
//! `FrameStartData` trace logging.

use crate::frontend::dispatch::ipc_init::{self, RendererInitCapabilities};
use crate::frontend::output_device::head_output_device_wants_openxr;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    DesktopConfig, FrameStartData, RenderDecouplingConfig, RendererCommand, RendererInitData,
};

use super::super::super::RendererRuntime;

impl RendererRuntime {
    pub(in crate::runtime) fn apply_renderer_init_data(&mut self, d: RendererInitData) {
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
        let init_result = ipc_init::build_renderer_init_result(
            d.output_device,
            renderer_init_capabilities(d.output_device),
        );
        if let Some(ipc) = self.frontend.ipc_mut()
            && !ipc.send_primary(RendererCommand::RendererInitResult(init_result))
        {
            logger::error!(
                "IPC: RendererInitResult was not sent (primary queue full); stopping init handshake"
            );
            self.frontend.set_fatal_error(true);
            return;
        }
        self.frontend.on_init_received();
    }

    pub(in crate::runtime) fn apply_render_decoupling_config(
        &mut self,
        cfg: RenderDecouplingConfig,
    ) {
        logger::info!(
            "runtime: render_decoupling_config activate_interval_s={:.4} decoupled_max_asset_processing_s={:.4} recouple_frame_count={}",
            cfg.decouple_activate_interval,
            cfg.decoupled_max_asset_processing_time,
            cfg.recouple_frame_count
        );
        self.frontend.set_decoupling_config(cfg);
    }

    pub(in crate::runtime) fn apply_desktop_config(&self, _cfg: DesktopConfig) {
        logger::trace!(
            "runtime: desktop_config ignored; renderer config owns desktop frame pacing"
        );
    }
}

fn renderer_init_capabilities(
    output_device: crate::shared::HeadOutputDevice,
) -> RendererInitCapabilities {
    let stereo_rendering_mode = if head_output_device_wants_openxr(output_device) {
        "OpenXR(multiview)"
    } else {
        "None"
    };
    RendererInitCapabilities {
        stereo_rendering_mode: stereo_rendering_mode.into(),
        max_texture_size: crate::gpu::RENDERER_MAX_TEXTURE_DIMENSION_2D as i32,
        supported_texture_formats: crate::assets::texture::supported_host_formats_for_init(),
    }
}

/// Logs structured fields from a host [`FrameStartData`] payload (lock-step / diagnostics only).
pub(in crate::runtime) fn log_frame_start_data_trace(fs: &FrameStartData) {
    logger::trace!(
        "host frame_start_data: last_frame_index={} has_performance={} has_inputs={} reflection_probes={} video_clock_errors={}",
        fs.last_frame_index,
        fs.performance.is_some(),
        fs.inputs.is_some(),
        fs.rendered_reflection_probes.len(),
        fs.video_clock_errors.len(),
    );
}
