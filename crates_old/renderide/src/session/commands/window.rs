//! Window command handlers: renderer_parent_window, set_window_icon, set_taskbar_progress.
//!
//! Placeholder; returns Ignored for all window commands. StubCommandHandler handles them until implemented.

use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles window commands. Placeholder; returns Ignored until window support is implemented.
pub struct WindowCommandHandler;

impl CommandHandler for WindowCommandHandler {
    fn handle(&mut self, _cmd: &RendererCommand, _ctx: &mut CommandContext<'_>) -> CommandResult {
        CommandResult::Ignored
    }
}
