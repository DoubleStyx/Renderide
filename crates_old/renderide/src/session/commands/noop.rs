//! No-op command handlers: keep_alive, progress updates, engine ready, init result, frame start.

use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles no-op commands: `keep_alive`, progress updates, engine ready, init result, frame start.
pub struct NoopCommandHandler;

impl CommandHandler for NoopCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, _ctx: &mut CommandContext<'_>) -> CommandResult {
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
