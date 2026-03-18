//! Shared memory command handlers: free_shared_memory_view.

use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `free_shared_memory_view`. Releases cached mmap views to avoid leaking shared memory.
pub struct FreeSharedMemoryCommandHandler;

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
