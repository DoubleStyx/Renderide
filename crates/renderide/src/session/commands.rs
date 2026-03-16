//! Command handler pattern for renderer commands.
//!
//! Replaces the monolithic match in Session with a registry of handlers.
//! New commands can be added by implementing CommandHandler without editing Session.

use crate::assets::AssetRegistry;
use crate::config::RenderConfig;
use crate::ipc::receiver::CommandReceiver;
use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::SceneGraph;
use crate::session::init::send_renderer_init_result;
use crate::session::state::ViewState;
use crate::shared::{
    FrameSubmitData, MeshUploadResult, RendererCommand, ShaderUploadResult,
};

/// Result of handling a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandResult {
    /// Handler processed the command; dispatch stops.
    Handled,
    /// Handler did not handle; try next handler.
    Ignored,
    /// Fatal error; dispatch stops and Session sets fatal_error.
    FatalError,
}

/// Context passed to command handlers with mutable access to Session state.
pub struct CommandContext<'a> {
    /// Shared memory accessor for reading asset data.
    pub shared_memory: &'a mut Option<SharedMemoryAccessor>,
    /// Asset registry for mesh uploads/unloads.
    pub asset_registry: &'a mut AssetRegistry,
    /// Scene graph for frame updates.
    pub scene_graph: &'a mut SceneGraph,
    /// View state (clip planes, FOV).
    pub view_state: &'a mut ViewState,
    /// Command receiver for sending responses.
    pub receiver: &'a mut CommandReceiver,
    /// Whether renderer_init_data has been received.
    pub init_received: &'a mut bool,
    /// Whether renderer_init_finalize_data has been received.
    pub init_finalized: &'a mut bool,
    /// Whether shutdown was requested.
    pub shutdown: &'a mut bool,
    /// Whether a fatal error occurred.
    pub fatal_error: &'a mut bool,
    /// Whether the last frame was processed (for FrameStartData timing).
    pub last_frame_data_processed: &'a mut bool,
    /// Asset IDs unloaded this frame (drained by Session).
    pub pending_mesh_unloads: &'a mut Vec<i32>,
    /// Render configuration (clip planes, vsync).
    pub render_config: &'a mut RenderConfig,
    /// Whether cursor lock was requested.
    pub lock_cursor: &'a mut bool,
    /// Frame data to process; set by FrameSubmitCommandHandler, drained by Session.
    pub pending_frame_data: Option<FrameSubmitData>,
}

/// Trait for command handlers. Handlers are tried in order until one returns Handled or FatalError.
pub trait CommandHandler {
    /// Handles a command. Returns Handled to stop dispatch, Ignored to try next handler, FatalError to abort.
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult;
}

/// Dispatches commands to a list of handlers. Stops on Handled or FatalError.
pub struct CommandDispatcher {
    handlers: Vec<Box<dyn CommandHandler>>,
}

impl CommandDispatcher {
    /// Creates a new dispatcher with the default handler set.
    pub fn new() -> Self {
        Self {
            handlers: vec![
                Box::new(InitCommandHandler),
                Box::new(InitFinalizeCommandHandler),
                Box::new(ShutdownCommandHandler),
                Box::new(FrameSubmitCommandHandler),
                Box::new(MeshCommandHandler),
                Box::new(ShaderCommandHandler),
                Box::new(DesktopConfigCommandHandler),
                Box::new(FreeSharedMemoryCommandHandler),
                Box::new(NoopCommandHandler),
                Box::new(ExhaustiveStubCommandHandler),
            ],
        }
    }

    /// Dispatches a command to handlers. Returns when a handler returns Handled or FatalError.
    pub fn dispatch(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        for handler in &mut self.handlers {
            let result = handler.handle(cmd.clone(), ctx);
            match result {
                CommandResult::Handled | CommandResult::FatalError => return result,
                CommandResult::Ignored => continue,
            }
        }
        CommandResult::Handled
    }
}

impl Default for CommandDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Handles `renderer_init_data`. Must be first; before `init_received`, only this command is accepted.
struct InitCommandHandler;

impl CommandHandler for InitCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        if *ctx.init_received {
            return CommandResult::Ignored;
        }
        match cmd {
            RendererCommand::renderer_init_data(x) => {
                if let Some(prefix) = x.shared_memory_prefix {
                    *ctx.shared_memory = Some(SharedMemoryAccessor::new(prefix));
                }
                send_renderer_init_result(ctx.receiver);
                *ctx.init_received = true;
                CommandResult::Handled
            }
            _ => CommandResult::FatalError,
        }
    }
}

/// Handles `renderer_init_finalize_data`. Marks init as finalized.
struct InitFinalizeCommandHandler;

impl CommandHandler for InitFinalizeCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::renderer_init_finalize_data(_) => {
                *ctx.init_finalized = true;
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}

/// Handles `renderer_shutdown` and `renderer_shutdown_request`. Post-finalize only.
struct ShutdownCommandHandler;

impl CommandHandler for ShutdownCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        if !*ctx.init_finalized {
            return CommandResult::Ignored;
        }
        match cmd {
            RendererCommand::renderer_shutdown(_) | RendererCommand::renderer_shutdown_request(_) => {
                *ctx.shutdown = true;
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}

/// Handles `frame_submit_data`. Stores data in context for Session to process after dispatch.
struct FrameSubmitCommandHandler;

impl CommandHandler for FrameSubmitCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::frame_submit_data(data) => {
                ctx.pending_frame_data = Some(data);
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}

/// Handles `mesh_upload_data` and `mesh_unload`. Sends mesh upload result on success.
struct MeshCommandHandler;

impl CommandHandler for MeshCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::mesh_upload_data(data) => {
                let asset_id = data.asset_id;
                let (success, existed_before) = match ctx.shared_memory {
                    Some(shm) => ctx.asset_registry.handle_mesh_upload(shm, data),
                    None => (false, false),
                };
                if success {
                    ctx.receiver
                        .send_background(RendererCommand::mesh_upload_result(MeshUploadResult {
                            asset_id,
                            instance_changed: !existed_before,
                        }));
                }
                CommandResult::Handled
            }
            RendererCommand::mesh_unload(x) => {
                ctx.asset_registry.handle_mesh_unload(x.asset_id);
                ctx.pending_mesh_unloads.push(x.asset_id);
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}

/// Handles `shader_upload`. Stores shader in asset registry and sends result on success.
struct ShaderCommandHandler;

impl CommandHandler for ShaderCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::shader_upload(data) => {
                let asset_id = data.asset_id;
                let (success, existed_before) = ctx.asset_registry.handle_shader_upload(data);
                if success {
                    ctx.receiver
                        .send_background(RendererCommand::shader_upload_result(
                            ShaderUploadResult {
                                asset_id,
                                instance_changed: !existed_before,
                            },
                        ));
                }
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}

/// Handles `free_shared_memory_view`. Releases cached mmap views to avoid leaking shared memory.
/// Mirrors SharedMemoryAccessor.ReleaseView in the C# host.
struct FreeSharedMemoryCommandHandler;

impl CommandHandler for FreeSharedMemoryCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::free_shared_memory_view(x) => {
                if let Some(shm) = ctx.shared_memory.as_mut() {
                    shm.release_view(x.buffer_id);
                }
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}

/// Handles `desktop_config`. Updates view state and render config. Post-finalize only.
struct DesktopConfigCommandHandler;

impl CommandHandler for DesktopConfigCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        if !*ctx.init_finalized {
            return CommandResult::Ignored;
        }
        match cmd {
            RendererCommand::desktop_config(x) => {
                ctx.view_state.near_clip = 0.01;
                ctx.view_state.far_clip = 1024.0;
                ctx.view_state.desktop_fov = 75.0;
                *ctx.render_config = RenderConfig {
                    near_clip: 0.01,
                    far_clip: 1024.0,
                    desktop_fov: 75.0,
                    vsync: x.v_sync,
                };
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}

/// Handles no-op commands: `keep_alive`, progress updates, engine ready, init result, frame start.
struct NoopCommandHandler;

impl CommandHandler for NoopCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, _ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::keep_alive(_)
            | RendererCommand::renderer_init_progress_update(_)
            | RendererCommand::renderer_engine_ready(_)
            | RendererCommand::renderer_init_result(_)
            | RendererCommand::frame_start_data(_) => CommandResult::Handled,
            _ => CommandResult::Ignored,
        }
    }
}

/// Exhaustive stub handler for all RendererCommand variants. Documents the IPC surface and ensures
/// the dispatcher never silently falls through. Each arm returns Handled; add real handlers above
/// to implement features. Compiler will error when new variants are added.
struct ExhaustiveStubCommandHandler;

impl CommandHandler for ExhaustiveStubCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, _ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            // --- Already handled by earlier handlers; should not reach here ---
            RendererCommand::renderer_init_data(_)
            | RendererCommand::renderer_init_finalize_data(_)
            | RendererCommand::renderer_shutdown(_)
            | RendererCommand::renderer_shutdown_request(_)
            | RendererCommand::frame_submit_data(_)
            | RendererCommand::mesh_upload_data(_)
            | RendererCommand::mesh_unload(_)
            | RendererCommand::shader_upload(_)
            | RendererCommand::desktop_config(_)
            | RendererCommand::free_shared_memory_view(_)
            | RendererCommand::keep_alive(_)
            | RendererCommand::renderer_init_progress_update(_)
            | RendererCommand::renderer_engine_ready(_)
            | RendererCommand::renderer_init_result(_)
            | RendererCommand::frame_start_data(_) => CommandResult::Handled,

            // --- Window ---
            RendererCommand::renderer_parent_window(_) => {
                // TODO: implement parent window embedding
                CommandResult::Handled
            }
            RendererCommand::set_window_icon(_) => {
                // TODO: implement window icon
                CommandResult::Handled
            }
            RendererCommand::set_window_icon_result(_) => {
                // Stub for: window icon result (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::set_taskbar_progress(_) => {
                // TODO: implement taskbar progress
                CommandResult::Handled
            }

            // --- Config ---
            RendererCommand::post_processing_config(_) => {
                // TODO: implement post-processing config
                CommandResult::Handled
            }
            RendererCommand::quality_config(_) => {
                // TODO: implement quality config
                CommandResult::Handled
            }
            RendererCommand::resolution_config(_) => {
                // TODO: implement resolution config
                CommandResult::Handled
            }
            RendererCommand::render_decoupling_config(_) => {
                // TODO: implement render decoupling config
                CommandResult::Handled
            }
            RendererCommand::gaussian_splat_config(_) => {
                // TODO: implement gaussian splat config
                CommandResult::Handled
            }

            // --- Response / no-op ---
            RendererCommand::mesh_upload_result(_) => {
                // Stub for: mesh upload result (renderer sends; host may echo)
                CommandResult::Handled
            }

            // --- Shaders ---
            RendererCommand::shader_unload(_) => {
                // TODO: implement shader unload
                CommandResult::Handled
            }
            RendererCommand::shader_upload_result(_) => {
                // Stub for: shader upload result (renderer sends; host may echo)
                CommandResult::Handled
            }

            // --- Materials ---
            RendererCommand::material_property_id_request(_) => {
                // TODO: implement material property ID request
                CommandResult::Handled
            }
            RendererCommand::material_property_id_result(_) => {
                // Stub for: material property ID result (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::materials_update_batch(_) => {
                // TODO: implement materials update batch
                CommandResult::Handled
            }
            RendererCommand::materials_update_batch_result(_) => {
                // Stub for: materials update batch result (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::unload_material(_) => {
                // TODO: implement unload material
                CommandResult::Handled
            }
            RendererCommand::unload_material_property_block(_) => {
                // TODO: implement unload material property block
                CommandResult::Handled
            }

            // --- Textures 2D ---
            RendererCommand::set_texture_2d_format(_) => {
                // TODO: implement texture 2D format
                CommandResult::Handled
            }
            RendererCommand::set_texture_2d_properties(_) => {
                // TODO: implement texture 2D properties
                CommandResult::Handled
            }
            RendererCommand::set_texture_2d_data(_) => {
                // TODO: implement texture 2D data
                CommandResult::Handled
            }
            RendererCommand::set_texture_2d_result(_) => {
                // Stub for: texture 2D result (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::unload_texture_2d(_) => {
                // TODO: implement unload texture 2D
                CommandResult::Handled
            }

            // --- Textures 3D ---
            RendererCommand::set_texture_3d_format(_) => {
                // TODO: implement texture 3D format
                CommandResult::Handled
            }
            RendererCommand::set_texture_3d_properties(_) => {
                // TODO: implement texture 3D properties
                CommandResult::Handled
            }
            RendererCommand::set_texture_3d_data(_) => {
                // TODO: implement texture 3D data
                CommandResult::Handled
            }
            RendererCommand::set_texture_3d_result(_) => {
                // Stub for: texture 3D result (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::unload_texture_3d(_) => {
                // TODO: implement unload texture 3D
                CommandResult::Handled
            }

            // --- Cubemaps ---
            RendererCommand::set_cubemap_format(_) => {
                // TODO: implement cubemap format
                CommandResult::Handled
            }
            RendererCommand::set_cubemap_properties(_) => {
                // TODO: implement cubemap properties
                CommandResult::Handled
            }
            RendererCommand::set_cubemap_data(_) => {
                // TODO: implement cubemap data
                CommandResult::Handled
            }
            RendererCommand::set_cubemap_result(_) => {
                // Stub for: cubemap result (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::unload_cubemap(_) => {
                // TODO: implement unload cubemap
                CommandResult::Handled
            }

            // --- Render textures ---
            RendererCommand::set_render_texture_format(_) => {
                // TODO: implement render texture format
                CommandResult::Handled
            }
            RendererCommand::render_texture_result(_) => {
                // Stub for: render texture result (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::unload_render_texture(_) => {
                // TODO: implement unload render texture
                CommandResult::Handled
            }

            // --- Desktop textures ---
            RendererCommand::set_desktop_texture_properties(_) => {
                // TODO: implement desktop texture properties
                CommandResult::Handled
            }
            RendererCommand::desktop_texture_properties_update(_) => {
                // TODO: implement desktop texture properties update
                CommandResult::Handled
            }
            RendererCommand::unload_desktop_texture(_) => {
                // TODO: implement unload desktop texture
                CommandResult::Handled
            }

            // --- Point / trail buffers ---
            RendererCommand::point_render_buffer_upload(_) => {
                // TODO: implement point render buffer upload
                CommandResult::Handled
            }
            RendererCommand::point_render_buffer_consumed(_) => {
                // Stub for: point render buffer consumed (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::point_render_buffer_unload(_) => {
                // TODO: implement point render buffer unload
                CommandResult::Handled
            }
            RendererCommand::trail_render_buffer_upload(_) => {
                // TODO: implement trail render buffer upload
                CommandResult::Handled
            }
            RendererCommand::trail_render_buffer_consumed(_) => {
                // Stub for: trail render buffer consumed (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::trail_render_buffer_unload(_) => {
                // TODO: implement trail render buffer unload
                CommandResult::Handled
            }

            // --- Gaussian splats ---
            RendererCommand::gaussian_splat_upload_raw(_) => {
                // TODO: implement gaussian splat upload raw
                CommandResult::Handled
            }
            RendererCommand::gaussian_splat_upload_encoded(_) => {
                // TODO: implement gaussian splat upload encoded
                CommandResult::Handled
            }
            RendererCommand::gaussian_splat_result(_) => {
                // Stub for: gaussian splat result (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::unload_gaussian_splat(_) => {
                // TODO: implement unload gaussian splat
                CommandResult::Handled
            }

            // --- Lights buffer ---
            RendererCommand::lights_buffer_renderer_submission(_) => {
                // TODO: implement lights buffer renderer submission
                CommandResult::Handled
            }
            RendererCommand::lights_buffer_renderer_consumed(_) => {
                // Stub for: lights buffer consumed (renderer sends; host may echo)
                CommandResult::Handled
            }

            // --- Reflection probes ---
            RendererCommand::reflection_probe_render_result(_) => {
                // TODO: implement reflection probe render result
                CommandResult::Handled
            }

            // --- Video ---
            RendererCommand::video_texture_load(_) => {
                // TODO: implement video texture load
                CommandResult::Handled
            }
            RendererCommand::video_texture_update(_) => {
                // TODO: implement video texture update
                CommandResult::Handled
            }
            RendererCommand::video_texture_ready(_) => {
                // Stub for: video texture ready (renderer sends; host may echo)
                CommandResult::Handled
            }
            RendererCommand::video_texture_changed(_) => {
                // TODO: implement video texture changed
                CommandResult::Handled
            }
            RendererCommand::video_texture_properties(_) => {
                // TODO: implement video texture properties
                CommandResult::Handled
            }
            RendererCommand::video_texture_start_audio_track(_) => {
                // TODO: implement video texture start audio track
                CommandResult::Handled
            }
            RendererCommand::unload_video_texture(_) => {
                // TODO: implement unload video texture
                CommandResult::Handled
            }
        }
    }
}
