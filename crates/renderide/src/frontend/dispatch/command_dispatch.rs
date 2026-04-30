//! Domain-grouped dispatch for [`super::commands::handle_running_command`].
//!
//! Keeps the runtime entrypoint as a thin wrapper while grouping related [`RendererCommand`] arms.

use crate::shared::{
    DesktopConfig, FrameStartData, MaterialPropertyIdRequest, MaterialPropertyIdResult,
    MeshUploadData, RenderDecouplingConfig, RendererCommand, SetCubemapData, SetCubemapFormat,
    SetCubemapProperties, SetTexture2DData, SetTexture2DFormat, SetTexture2DProperties,
    SetTexture3DData, SetTexture3DFormat, SetTexture3DProperties,
};

use super::renderer_command_kind::renderer_command_variant_tag;
use crate::runtime::RendererRuntime;

/// Logs structured fields from a host [`FrameStartData`] payload (lock-step / diagnostics only).
fn log_frame_start_data_trace(fs: &FrameStartData) {
    logger::trace!(
        "host frame_start_data: last_frame_index={} has_performance={} has_inputs={} reflection_probes={} video_clock_errors={}",
        fs.last_frame_index,
        fs.performance.is_some(),
        fs.inputs.is_some(),
        fs.rendered_reflection_probes.len(),
        fs.video_clock_errors.len(),
    );
}

/// Routes a post-handshake [`RendererCommand`] to the appropriate runtime / backend handler.
pub(crate) fn dispatch_running_command(runtime: &mut RendererRuntime, cmd: RendererCommand) {
    match cmd {
        RendererCommand::KeepAlive(_) => {}
        RendererCommand::RendererShutdown(_) | RendererCommand::RendererShutdownRequest(_) => {
            request_shutdown(runtime);
        }
        RendererCommand::FrameSubmitData(data) => runtime.on_frame_submit(data),
        RendererCommand::MeshUploadData(d) => process_mesh_upload(runtime, d),
        RendererCommand::MeshUnload(u) => runtime.backend.on_mesh_unload(u),
        RendererCommand::SetTexture2DFormat(f) => dispatch_texture_2d_format(runtime, f),
        RendererCommand::SetTexture2DProperties(p) => dispatch_texture_2d_properties(runtime, p),
        RendererCommand::SetTexture2DData(d) => dispatch_texture_2d_data(runtime, d),
        RendererCommand::UnloadTexture2D(u) => runtime.backend.on_unload_texture_2d(u),
        RendererCommand::SetTexture3DFormat(f) => dispatch_texture_3d_format(runtime, f),
        RendererCommand::SetTexture3DProperties(p) => dispatch_texture_3d_properties(runtime, p),
        RendererCommand::SetTexture3DData(d) => dispatch_texture_3d_data(runtime, d),
        RendererCommand::UnloadTexture3D(u) => runtime.backend.on_unload_texture_3d(u),
        RendererCommand::SetCubemapFormat(f) => dispatch_cubemap_format(runtime, f),
        RendererCommand::SetCubemapProperties(p) => dispatch_cubemap_properties(runtime, p),
        RendererCommand::SetCubemapData(d) => dispatch_cubemap_data(runtime, d),
        RendererCommand::UnloadCubemap(u) => runtime.backend.on_unload_cubemap(u),
        RendererCommand::SetRenderTextureFormat(f) => {
            runtime
                .backend
                .on_set_render_texture_format(f, runtime.frontend.ipc_mut());
        }
        RendererCommand::UnloadRenderTexture(u) => runtime.backend.on_unload_render_texture(u),
        RendererCommand::VideoTextureLoad(l) => runtime.backend.on_video_texture_load(l),
        RendererCommand::VideoTextureUpdate(u) => runtime.backend.on_video_texture_update(u),
        RendererCommand::VideoTextureProperties(p) => {
            runtime.backend.on_video_texture_properties(p);
        }
        RendererCommand::VideoTextureStartAudioTrack(s) => {
            runtime.backend.on_video_texture_start_audio_track(s);
        }
        RendererCommand::UnloadVideoTexture(u) => runtime.backend.on_unload_video_texture(u),
        RendererCommand::FreeSharedMemoryView(f) => {
            release_shared_memory_view(runtime, f.buffer_id);
        }
        RendererCommand::MaterialPropertyIdRequest(req) => {
            material_property_id_request(runtime, req);
        }
        RendererCommand::MaterialsUpdateBatch(batch) => {
            runtime.on_materials_update_batch(batch);
        }
        RendererCommand::UnloadMaterial(u) => runtime.backend.on_unload_material(u.asset_id),
        RendererCommand::UnloadMaterialPropertyBlock(u) => {
            runtime
                .backend
                .on_unload_material_property_block(u.asset_id);
        }
        RendererCommand::ShaderUpload(u) => runtime.on_shader_upload(u),
        RendererCommand::ShaderUnload(u) => runtime.on_shader_unload(u),
        RendererCommand::FrameStartData(ref fs) => log_frame_start_data_trace(fs),
        RendererCommand::LightsBufferRendererSubmission(sub) => {
            runtime.on_lights_buffer_renderer_submission(sub);
        }
        RendererCommand::LightsBufferRendererConsumed(_) => {
            logger::trace!("runtime: lights_buffer_renderer_consumed from host (ignored)");
        }
        RendererCommand::RenderTextureResult(_) => {
            logger::trace!(
                "runtime: render_texture_result from host (ignored; renderer is source)"
            );
        }
        RendererCommand::RendererEngineReady(_) => {
            logger::trace!(
                "runtime: renderer_engine_ready from host (post-init lifecycle ack; no action)"
            );
        }
        RendererCommand::DesktopConfig(cfg) => {
            apply_desktop_config(runtime, cfg);
        }
        RendererCommand::RenderDecouplingConfig(cfg) => {
            apply_render_decoupling_config(runtime, cfg);
        }
        ref cmd => {
            let tag = renderer_command_variant_tag(cmd);
            runtime.record_unhandled_renderer_command(tag);
            logger::warn!(
                "runtime: no handler for RendererCommand::{tag} (host sent unexpected command)"
            );
        }
    }
}

fn request_shutdown(runtime: &mut RendererRuntime) {
    runtime.frontend.set_shutdown_requested(true);
}

fn process_mesh_upload(runtime: &mut RendererRuntime, d: MeshUploadData) {
    let (shm, ipc) = runtime.frontend.transport_pair_mut();
    if let Some(shm) = shm {
        runtime.backend.try_process_mesh_upload(d, shm, ipc);
    } else {
        logger::warn!("mesh upload: no shared memory (standalone?)");
    }
}

fn release_shared_memory_view(runtime: &mut RendererRuntime, buffer_id: i32) {
    if let Some(shm) = runtime.frontend.shared_memory_mut() {
        shm.release_view(buffer_id);
    }
}

fn dispatch_texture_2d_format(runtime: &mut RendererRuntime, f: SetTexture2DFormat) {
    runtime
        .backend
        .on_set_texture_2d_format(f, runtime.frontend.ipc_mut());
}

fn dispatch_texture_2d_properties(runtime: &mut RendererRuntime, p: SetTexture2DProperties) {
    runtime
        .backend
        .on_set_texture_2d_properties(p, runtime.frontend.ipc_mut());
}

fn dispatch_texture_2d_data(runtime: &mut RendererRuntime, d: SetTexture2DData) {
    let (shm, ipc) = runtime.frontend.transport_pair_mut();
    runtime.backend.on_set_texture_2d_data(d, shm, ipc);
}

fn dispatch_texture_3d_format(runtime: &mut RendererRuntime, f: SetTexture3DFormat) {
    runtime
        .backend
        .on_set_texture_3d_format(f, runtime.frontend.ipc_mut());
}

fn dispatch_texture_3d_properties(runtime: &mut RendererRuntime, p: SetTexture3DProperties) {
    runtime
        .backend
        .on_set_texture_3d_properties(p, runtime.frontend.ipc_mut());
}

fn dispatch_texture_3d_data(runtime: &mut RendererRuntime, d: SetTexture3DData) {
    let (shm, ipc) = runtime.frontend.transport_pair_mut();
    runtime.backend.on_set_texture_3d_data(d, shm, ipc);
}

fn dispatch_cubemap_format(runtime: &mut RendererRuntime, f: SetCubemapFormat) {
    runtime
        .backend
        .on_set_cubemap_format(f, runtime.frontend.ipc_mut());
}

fn dispatch_cubemap_properties(runtime: &mut RendererRuntime, p: SetCubemapProperties) {
    runtime
        .backend
        .on_set_cubemap_properties(p, runtime.frontend.ipc_mut());
}

fn dispatch_cubemap_data(runtime: &mut RendererRuntime, d: SetCubemapData) {
    let (shm, ipc) = runtime.frontend.transport_pair_mut();
    runtime.backend.on_set_cubemap_data(d, shm, ipc);
}

fn apply_render_decoupling_config(runtime: &mut RendererRuntime, cfg: RenderDecouplingConfig) {
    logger::info!(
        "runtime: render_decoupling_config activate_interval_s={:.4} decoupled_max_asset_processing_s={:.4} recouple_frame_count={}",
        cfg.decouple_activate_interval,
        cfg.decoupled_max_asset_processing_time,
        cfg.recouple_frame_count
    );
    runtime.frontend.set_decoupling_config(cfg);
}

/// Applies host desktop FPS caps to the live renderer settings store.
///
/// The host command also carries a `v_sync` boolean, but the renderer exposes a richer
/// [`crate::config::VsyncMode`] policy (`Off` / `On` / `Auto`). The renderer config remains
/// authoritative for vsync so host IPC cannot collapse `Auto` back to a boolean mode.
fn apply_desktop_config(runtime: &RendererRuntime, cfg: DesktopConfig) {
    let focused_fps_cap = desktop_config_fps_cap(cfg.maximum_foreground_framerate);
    let unfocused_fps_cap = desktop_config_fps_cap(cfg.maximum_background_framerate);

    match runtime.settings().write() {
        Ok(mut settings) => {
            settings.display.focused_fps_cap = focused_fps_cap;
            settings.display.unfocused_fps_cap = unfocused_fps_cap;
            logger::info!(
                "runtime: desktop_config applied focused_fps_cap={} unfocused_fps_cap={} host_vsync_ignored={} renderer_vsync={:?}",
                focused_fps_cap,
                unfocused_fps_cap,
                cfg.v_sync,
                settings.rendering.vsync
            );
        }
        Err(e) => {
            logger::warn!("runtime: desktop_config ignored because settings lock is poisoned: {e}");
        }
    }
}

/// Converts host desktop framerate caps into renderer display settings.
///
/// [`None`] means uncapped. [`Some`] mirrors the host renderer's behavior by enforcing a minimum
/// capped rate of 5 fps; normal host UI ranges should already be above that, but the wire command
/// is still sanitized here.
fn desktop_config_fps_cap(host_cap: Option<i32>) -> u32 {
    match host_cap {
        Some(cap) => cap.max(5) as u32,
        None => 0,
    }
}

fn material_property_id_request(runtime: &mut RendererRuntime, req: MaterialPropertyIdRequest) {
    profiling::scope!("command::material_property_id_request");
    let property_ids: Vec<i32> = {
        let reg = runtime.backend.property_id_registry();
        req.property_names
            .iter()
            .map(|n| reg.intern_for_host_request(n.as_deref().unwrap_or("")))
            .collect()
    };
    if let Some(ref mut ipc) = runtime.frontend.ipc_mut() {
        let _ = ipc.send_background(RendererCommand::MaterialPropertyIdResult(
            MaterialPropertyIdResult {
                request_id: req.request_id,
                property_ids,
            },
        ));
    }
}
