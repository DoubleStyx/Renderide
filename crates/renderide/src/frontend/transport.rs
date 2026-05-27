//! IPC and shared-memory ownership for the frontend facade.

use crate::connection::{ConnectionParams, InitError};
use crate::frontend::dispatch::command_kind::classify_renderer_command;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::RendererCommand;

/// Owns host transport handles and reusable IPC polling scratch.
pub(crate) struct FrontendTransport {
    ipc: Option<DualQueueIpc>,
    params: Option<ConnectionParams>,
    command_batch: Vec<RendererCommand>,
    shared_memory: Option<SharedMemoryAccessor>,
}

impl FrontendTransport {
    /// Builds an unopened frontend transport from optional connection parameters.
    pub(crate) fn new(params: Option<ConnectionParams>) -> Self {
        Self {
            ipc: None,
            params,
            command_batch: Vec::new(),
            shared_memory: None,
        }
    }

    /// Opens the primary/background host queues when connection parameters were supplied.
    pub(crate) fn connect_ipc(&mut self) -> Result<(), InitError> {
        let Some(ref p) = self.params.clone() else {
            return Ok(());
        };
        logger::info!(
            "Opening renderer IPC queues: base={} capacity={} primary_sub={} primary_pub={} background_sub={} background_pub={}",
            p.queue_name,
            p.queue_capacity,
            crate::connection::subscriber_queue_name(&p.queue_name, "Primary"),
            crate::connection::publisher_queue_name(&p.queue_name, "Primary"),
            crate::connection::subscriber_queue_name(&p.queue_name, "Background"),
            crate::connection::publisher_queue_name(&p.queue_name, "Background"),
        );
        self.ipc = Some(DualQueueIpc::connect(p)?);
        Ok(())
    }

    /// Whether the host IPC queues are connected.
    pub(crate) fn is_ipc_connected(&self) -> bool {
        self.ipc.is_some()
    }

    /// Whether this renderer was constructed without host IPC parameters.
    pub(crate) fn is_standalone(&self) -> bool {
        self.params.is_none()
    }

    /// Mutable reference to the connected dual-queue IPC, if present.
    pub(crate) fn ipc_mut(&mut self) -> Option<&mut DualQueueIpc> {
        self.ipc.as_mut()
    }

    /// Shared-memory accessor when host mapped views are available.
    pub(crate) fn shared_memory(&self) -> Option<&SharedMemoryAccessor> {
        self.shared_memory.as_ref()
    }

    /// Mutable shared-memory accessor for asset and scene payloads.
    pub(crate) fn shared_memory_mut(&mut self) -> Option<&mut SharedMemoryAccessor> {
        self.shared_memory.as_mut()
    }

    /// Installs the shared-memory accessor produced by init handshake mapping.
    pub(crate) fn set_shared_memory(&mut self, shm: SharedMemoryAccessor) {
        self.shared_memory = Some(shm);
    }

    /// Disjoint mutable handles for consumers that need both SHM and IPC.
    pub(crate) fn pair_mut(
        &mut self,
    ) -> (Option<&mut SharedMemoryAccessor>, Option<&mut DualQueueIpc>) {
        (self.shared_memory.as_mut(), self.ipc.as_mut())
    }

    /// Clears per-tick outbound drop flags.
    pub(crate) fn reset_outbound_drop_tick_flags(&mut self) {
        if let Some(ipc) = self.ipc.as_mut() {
            ipc.reset_outbound_drop_tick_flags();
        }
    }

    /// Whether the primary outbound queue dropped a message this tick.
    pub(crate) fn outbound_primary_drop_this_tick(&self) -> bool {
        self.ipc
            .as_ref()
            .is_some_and(DualQueueIpc::had_outbound_primary_drop_this_tick)
    }

    /// Whether the background outbound queue dropped a message this tick.
    pub(crate) fn outbound_background_drop_this_tick(&self) -> bool {
        self.ipc
            .as_ref()
            .is_some_and(DualQueueIpc::had_outbound_background_drop_this_tick)
    }

    /// Current consecutive outbound drop streaks per channel.
    pub(crate) fn consecutive_outbound_drop_streaks(&self) -> (u32, u32) {
        self.ipc.as_ref().map_or((0, 0), |ipc| {
            (
                ipc.consecutive_primary_drop_streak(),
                ipc.consecutive_background_drop_streak(),
            )
        })
    }

    /// Polls host queues into the reusable batch and sorts by lifecycle priority.
    pub(crate) fn poll_commands(&mut self) -> Vec<RendererCommand> {
        profiling::scope!("frontend::poll_commands");
        let mut batch = std::mem::take(&mut self.command_batch);
        if let Some(ipc) = self.ipc.as_mut() {
            ipc.poll_into(&mut batch);
            batch.sort_by_key(|cmd| classify_renderer_command(cmd).poll_priority());
        } else {
            batch.clear();
        }
        batch
    }

    /// Returns the drained command batch so the allocation is retained for the next poll.
    pub(crate) fn recycle_command_batch(&mut self, batch: Vec<RendererCommand>) {
        self.command_batch = batch;
    }
}
