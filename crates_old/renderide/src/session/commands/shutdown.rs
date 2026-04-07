//! Shutdown command handlers: renderer_shutdown, renderer_shutdown_request.

use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `renderer_shutdown` and `renderer_shutdown_request`. Post-finalize only.
pub struct ShutdownCommandHandler;

impl CommandHandler for ShutdownCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        if !ctx.session_flags.init_state.is_finalized() {
            return CommandResult::Ignored;
        }
        match cmd {
            RendererCommand::renderer_shutdown(_)
            | RendererCommand::renderer_shutdown_request(_) => {
                *ctx.session_flags.shutdown = true;
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
