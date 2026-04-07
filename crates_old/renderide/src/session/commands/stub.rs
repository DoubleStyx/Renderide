//! Stub command handler for unimplemented RendererCommand variants.
//!
//! Must be last in the dispatcher. Exhaustive match ensures new variants cause a compile error.
//! Add real handlers above to implement features.

use std::collections::HashSet;

use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Stub handler for all unimplemented RendererCommand variants. Documents the IPC surface and ensures
/// the dispatcher never silently falls through. Add real handlers above to implement features.
/// Logs each unhandled command type only once per session to avoid trace spam.
pub struct StubCommandHandler {
    /// Command types we have already logged as unhandled.
    logged_unhandled: HashSet<&'static str>,
}

impl StubCommandHandler {
    /// Creates a new stub handler.
    pub fn new() -> Self {
        Self {
            logged_unhandled: HashSet::new(),
        }
    }

    fn log_once(&mut self, name: &'static str) {
        if self.logged_unhandled.insert(name) {
            logger::debug!("Unhandled command: {} (logged once per session)", name);
        }
    }
}

impl Default for StubCommandHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandHandler for StubCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, _ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            // --- Handled by earlier handlers; should not reach here ---
            RendererCommand::renderer_init_data(_)
            | RendererCommand::renderer_init_finalize_data(_)
            | RendererCommand::renderer_shutdown(_)
            | RendererCommand::renderer_shutdown_request(_)
            | RendererCommand::frame_submit_data(_)
            | RendererCommand::mesh_upload_data(_)
            | RendererCommand::mesh_unload(_)
            | RendererCommand::shader_upload(_)
            | RendererCommand::shader_unload(_)
            | RendererCommand::set_texture_2d_format(_)
            | RendererCommand::set_texture_2d_properties(_)
            | RendererCommand::set_texture_2d_data(_)
            | RendererCommand::set_texture_2d_result(_)
            | RendererCommand::unload_texture_2d(_)
            | RendererCommand::desktop_config(_)
            | RendererCommand::free_shared_memory_view(_)
            | RendererCommand::keep_alive(_)
            | RendererCommand::renderer_init_progress_update(_)
            | RendererCommand::renderer_engine_ready(_)
            | RendererCommand::renderer_init_result(_)
            | RendererCommand::frame_start_data(_)
            | RendererCommand::unload_material(_)
            | RendererCommand::unload_material_property_block(_)
            | RendererCommand::material_property_id_request(_)
            | RendererCommand::materials_update_batch(_) => {
                unreachable!("command handled by earlier handler")
            }

            // --- Window ---
            RendererCommand::renderer_parent_window(_) => {
                self.log_once("renderer_parent_window");
                CommandResult::Handled
            }
            RendererCommand::set_window_icon(_) => {
                self.log_once("set_window_icon");
                CommandResult::Handled
            }
            RendererCommand::set_window_icon_result(_) => CommandResult::Handled,
            RendererCommand::set_taskbar_progress(_) => {
                self.log_once("set_taskbar_progress");
                CommandResult::Handled
            }

            // --- Config ---
            RendererCommand::post_processing_config(_) => {
                self.log_once("post_processing_config");
                CommandResult::Handled
            }
            RendererCommand::quality_config(_) => {
                self.log_once("quality_config");
                CommandResult::Handled
            }
            RendererCommand::resolution_config(_) => {
                self.log_once("resolution_config");
                CommandResult::Handled
            }
            RendererCommand::render_decoupling_config(_) => {
                self.log_once("render_decoupling_config");
                CommandResult::Handled
            }
            RendererCommand::gaussian_splat_config(_) => {
                self.log_once("gaussian_splat_config");
                CommandResult::Handled
            }

            // --- Response / no-op ---
            RendererCommand::mesh_upload_result(_) => CommandResult::Handled,

            // --- Shaders ---
            RendererCommand::shader_upload_result(_) => CommandResult::Handled,

            // --- Materials ---
            RendererCommand::material_property_id_result(_) => CommandResult::Handled,
            RendererCommand::materials_update_batch_result(_) => CommandResult::Handled,

            // --- Textures 2D --- (handled by TextureCommandHandler)

            // --- Textures 3D ---
            RendererCommand::set_texture_3d_format(_) => {
                self.log_once("set_texture_3d_format");
                CommandResult::Handled
            }
            RendererCommand::set_texture_3d_properties(_) => {
                self.log_once("set_texture_3d_properties");
                CommandResult::Handled
            }
            RendererCommand::set_texture_3d_data(_) => {
                self.log_once("set_texture_3d_data");
                CommandResult::Handled
            }
            RendererCommand::set_texture_3d_result(_) => CommandResult::Handled,
            RendererCommand::unload_texture_3d(_) => {
                self.log_once("unload_texture_3d");
                CommandResult::Handled
            }

            // --- Cubemaps ---
            RendererCommand::set_cubemap_format(_) => {
                self.log_once("set_cubemap_format");
                CommandResult::Handled
            }
            RendererCommand::set_cubemap_properties(_) => {
                self.log_once("set_cubemap_properties");
                CommandResult::Handled
            }
            RendererCommand::set_cubemap_data(_) => {
                self.log_once("set_cubemap_data");
                CommandResult::Handled
            }
            RendererCommand::set_cubemap_result(_) => CommandResult::Handled,
            RendererCommand::unload_cubemap(_) => {
                self.log_once("unload_cubemap");
                CommandResult::Handled
            }

            // --- Render textures ---
            RendererCommand::set_render_texture_format(_) => {
                self.log_once("set_render_texture_format");
                CommandResult::Handled
            }
            RendererCommand::render_texture_result(_) => CommandResult::Handled,
            RendererCommand::unload_render_texture(_) => {
                self.log_once("unload_render_texture");
                CommandResult::Handled
            }

            // --- Desktop textures ---
            RendererCommand::set_desktop_texture_properties(_) => {
                self.log_once("set_desktop_texture_properties");
                CommandResult::Handled
            }
            RendererCommand::desktop_texture_properties_update(_) => {
                self.log_once("desktop_texture_properties_update");
                CommandResult::Handled
            }
            RendererCommand::unload_desktop_texture(_) => {
                self.log_once("unload_desktop_texture");
                CommandResult::Handled
            }

            // --- Point / trail buffers ---
            RendererCommand::point_render_buffer_upload(_) => {
                self.log_once("point_render_buffer_upload");
                CommandResult::Handled
            }
            RendererCommand::point_render_buffer_consumed(_) => CommandResult::Handled,
            RendererCommand::point_render_buffer_unload(_) => {
                self.log_once("point_render_buffer_unload");
                CommandResult::Handled
            }
            RendererCommand::trail_render_buffer_upload(_) => {
                self.log_once("trail_render_buffer_upload");
                CommandResult::Handled
            }
            RendererCommand::trail_render_buffer_consumed(_) => CommandResult::Handled,
            RendererCommand::trail_render_buffer_unload(_) => {
                self.log_once("trail_render_buffer_unload");
                CommandResult::Handled
            }

            // --- Gaussian splats ---
            RendererCommand::gaussian_splat_upload_raw(_) => {
                self.log_once("gaussian_splat_upload_raw");
                CommandResult::Handled
            }
            RendererCommand::gaussian_splat_upload_encoded(_) => {
                self.log_once("gaussian_splat_upload_encoded");
                CommandResult::Handled
            }
            RendererCommand::gaussian_splat_result(_) => CommandResult::Handled,
            RendererCommand::unload_gaussian_splat(_) => {
                self.log_once("unload_gaussian_splat");
                CommandResult::Handled
            }

            // --- Lights buffer ---
            RendererCommand::lights_buffer_renderer_submission(_) => {
                unreachable!("command handled by earlier handler")
            }
            RendererCommand::lights_buffer_renderer_consumed(_) => CommandResult::Handled,

            // --- Reflection probes ---
            RendererCommand::reflection_probe_render_result(_) => {
                self.log_once("reflection_probe_render_result");
                CommandResult::Handled
            }

            // --- Video ---
            RendererCommand::video_texture_load(_) => {
                self.log_once("video_texture_load");
                CommandResult::Handled
            }
            RendererCommand::video_texture_update(_) => {
                self.log_once("video_texture_update");
                CommandResult::Handled
            }
            RendererCommand::video_texture_ready(_) => CommandResult::Handled,
            RendererCommand::video_texture_changed(_) => {
                self.log_once("video_texture_changed");
                CommandResult::Handled
            }
            RendererCommand::video_texture_properties(_) => {
                self.log_once("video_texture_properties");
                CommandResult::Handled
            }
            RendererCommand::video_texture_start_audio_track(_) => {
                self.log_once("video_texture_start_audio_track");
                CommandResult::Handled
            }
            RendererCommand::unload_video_texture(_) => {
                self.log_once("unload_video_texture");
                CommandResult::Handled
            }
        }
    }
}
