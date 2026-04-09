//! Dispatches [`RendererCommand`] values after the host init handshake is finalized.

use crate::shared::{MaterialPropertyIdResult, RendererCommand};

use super::RendererRuntime;

/// Handles IPC commands in the normal running state ([`crate::frontend::InitState::Finalized`]).
pub(super) fn handle_running_command(runtime: &mut RendererRuntime, cmd: RendererCommand) {
    match cmd {
        RendererCommand::keep_alive(_) => {}
        RendererCommand::renderer_shutdown(_) | RendererCommand::renderer_shutdown_request(_) => {
            runtime.frontend.shutdown_requested = true;
        }
        RendererCommand::frame_submit_data(data) => runtime.on_frame_submit(data),
        RendererCommand::mesh_upload_data(d) => {
            let (shm, ipc) = runtime.frontend.transport_pair_mut();
            if let Some(shm) = shm {
                runtime.backend.try_process_mesh_upload(d, shm, ipc);
            } else {
                logger::warn!("mesh upload: no shared memory (standalone?)");
            }
        }
        RendererCommand::mesh_unload(u) => runtime.backend.on_mesh_unload(u),
        RendererCommand::set_texture_2d_format(f) => {
            runtime
                .backend
                .on_set_texture_2d_format(f, runtime.frontend.ipc_mut());
        }
        RendererCommand::set_texture_2d_properties(p) => {
            runtime
                .backend
                .on_set_texture_2d_properties(p, runtime.frontend.ipc_mut());
        }
        RendererCommand::set_texture_2d_data(d) => {
            let (shm, ipc) = runtime.frontend.transport_pair_mut();
            runtime.backend.on_set_texture_2d_data(d, shm, ipc);
        }
        RendererCommand::unload_texture_2d(u) => runtime.backend.on_unload_texture_2d(u),
        RendererCommand::free_shared_memory_view(f) => {
            if let Some(shm) = runtime.frontend.shared_memory_mut() {
                shm.release_view(f.buffer_id);
            }
        }
        RendererCommand::material_property_id_request(req) => {
            let property_ids: Vec<i32> = {
                let reg = runtime.backend.property_id_registry();
                req.property_names
                    .iter()
                    .map(|n| reg.intern_for_host_request(n.as_deref().unwrap_or("")))
                    .collect()
            };
            if let Some(ref mut ipc) = runtime.frontend.ipc_mut() {
                ipc.send_background(RendererCommand::material_property_id_result(
                    MaterialPropertyIdResult {
                        request_id: req.request_id,
                        property_ids,
                    },
                ));
            }
        }
        RendererCommand::materials_update_batch(batch) => {
            runtime.on_materials_update_batch(batch);
        }
        RendererCommand::unload_material(u) => runtime.backend.on_unload_material(u.asset_id),
        RendererCommand::unload_material_property_block(u) => {
            runtime
                .backend
                .on_unload_material_property_block(u.asset_id);
        }
        RendererCommand::shader_upload(u) => runtime.on_shader_upload(u),
        RendererCommand::shader_unload(u) => runtime.on_shader_unload(u),
        RendererCommand::frame_start_data(fs) => {
            logger::trace!(
                "host frame_start_data: last_frame_index={} has_performance={} has_inputs={} reflection_probes={} video_clock_errors={}",
                fs.last_frame_index,
                fs.performance.is_some(),
                fs.inputs.is_some(),
                fs.rendered_reflection_probes.len(),
                fs.video_clock_errors.len(),
            );
        }
        RendererCommand::lights_buffer_renderer_submission(sub) => {
            runtime.on_lights_buffer_renderer_submission(sub);
        }
        RendererCommand::lights_buffer_renderer_consumed(_) => {
            logger::trace!("runtime: lights_buffer_renderer_consumed from host (ignored)");
        }
        _ => {
            logger::trace!("runtime: unhandled RendererCommand (expand handlers here)");
        }
    }
}
