//! Init command handlers: renderer_init_data, renderer_init_finalize_data.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::session::init::send_renderer_init_result;
use crate::shared::RendererCommand;

use crate::session::state::InitState;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `renderer_init_data`. Must be first; before `init_received`, only this command is accepted.
pub struct InitCommandHandler;

impl CommandHandler for InitCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        if !ctx.session_flags.init_state.is_uninitialized() {
            return CommandResult::Ignored;
        }
        match cmd {
            RendererCommand::renderer_init_data(x) => {
                if let Some(ref prefix) = x.shared_memory_prefix {
                    *ctx.assets.shared_memory = Some(SharedMemoryAccessor::new(prefix.clone()));
                }
                send_renderer_init_result(ctx.receiver);
                *ctx.session_flags.init_state = InitState::InitReceived;
                CommandResult::Handled
            }
            _ => CommandResult::FatalError,
        }
    }
}

/// Handles `renderer_init_finalize_data`. Marks init as finalized.
pub struct InitFinalizeCommandHandler;

impl CommandHandler for InitFinalizeCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::renderer_init_finalize_data(_) => {
                *ctx.session_flags.init_state = InitState::Finalized;
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
