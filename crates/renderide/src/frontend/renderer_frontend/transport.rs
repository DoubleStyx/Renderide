//! IPC queue, shared-memory, and command-batch methods on [`RendererFrontend`].

use std::time::Duration;

use crate::connection::InitError;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::RendererCommand;

use super::RendererFrontend;

impl RendererFrontend {
    /// Large-payload shared-memory accessor when the host mapped views are available.
    pub fn shared_memory(&self) -> Option<&SharedMemoryAccessor> {
        self.transport.shared_memory()
    }

    /// Mutable shared-memory accessor for mesh/texture uploads.
    pub fn shared_memory_mut(&mut self) -> Option<&mut SharedMemoryAccessor> {
        self.transport.shared_memory_mut()
    }

    /// Installs the SHM accessor produced after init handshake mapping.
    pub fn set_shared_memory(&mut self, shm: SharedMemoryAccessor) {
        self.transport.set_shared_memory(shm);
    }

    /// Mutable reference to the dual-queue IPC when connected.
    pub fn ipc_mut(&mut self) -> Option<&mut DualQueueIpc> {
        self.transport.ipc_mut()
    }

    /// Disjoint mutable handles for backends that need both shared memory and IPC in one call.
    pub fn transport_pair_mut(
        &mut self,
    ) -> (Option<&mut SharedMemoryAccessor>, Option<&mut DualQueueIpc>) {
        self.transport.pair_mut()
    }

    /// Opens Primary/Background queues when connection parameters were provided at construction.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        self.transport.connect_ipc()
    }

    /// Whether [`Self::connect_ipc`] successfully opened the host queues.
    pub fn is_ipc_connected(&self) -> bool {
        self.transport.is_ipc_connected()
    }

    /// Clears per-tick outbound IPC drop flags on the dual queue.
    pub fn reset_ipc_outbound_drop_tick_flags(&mut self) {
        self.transport.reset_outbound_drop_tick_flags();
    }

    /// Whether any primary outbound send failed since the last drop-flag reset.
    pub fn ipc_outbound_primary_drop_this_tick(&self) -> bool {
        self.transport.outbound_primary_drop_this_tick()
    }

    /// Whether any background outbound send failed since the last drop-flag reset.
    pub fn ipc_outbound_background_drop_this_tick(&self) -> bool {
        self.transport.outbound_background_drop_this_tick()
    }

    /// Current consecutive outbound drop streaks per channel.
    pub fn ipc_consecutive_outbound_drop_streaks(&self) -> (u32, u32) {
        self.transport.consecutive_outbound_drop_streaks()
    }

    /// Poll and sort commands by lifecycle priority.
    pub fn poll_commands(&mut self) -> Vec<RendererCommand> {
        self.transport.poll_commands()
    }

    /// Waits briefly for primary-queue work, then polls and sorts commands by lifecycle priority.
    pub fn poll_commands_after_primary_wait(
        &mut self,
        timeout: Duration,
    ) -> (Vec<RendererCommand>, Duration) {
        self.transport.poll_commands_after_primary_wait(timeout)
    }

    /// Returns an empty command batch so its allocation is retained for the next poll.
    pub fn recycle_command_batch(&mut self, batch: Vec<RendererCommand>) {
        self.transport.recycle_command_batch(batch);
    }
}
