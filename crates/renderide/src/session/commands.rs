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
    FrameSubmitData, MeshUploadResult, RendererCommand,
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
                Box::new(DesktopConfigCommandHandler),
                Box::new(NoopCommandHandler),
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
