//! Dispatches [`RendererCommand`] values after the host init handshake is finalized.

use crate::shared::RendererCommand;

use super::command_dispatch;
use crate::runtime::RendererRuntime;

/// Handles IPC commands in the normal running state ([`crate::frontend::InitState::Finalized`]).
pub(crate) fn handle_running_command(runtime: &mut RendererRuntime, cmd: RendererCommand) {
    command_dispatch::dispatch_running_command(runtime, cmd);
}
