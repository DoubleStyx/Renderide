//! IPC and shared-memory ownership for the frontend facade.

use std::time::Duration;

use crate::connection::{ConnectionParams, InitError};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor, TimedRendererCommand};
use crate::profiling::{IpcPollProfileSample, plot_ipc_poll};
use crate::shared::RendererCommand;

/// Owns host transport handles and reusable IPC polling scratch.
pub(crate) struct FrontendTransport {
    ipc: Option<DualQueueIpc>,
    params: Option<ConnectionParams>,
    command_batch: Vec<TimedRendererCommand>,
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

    /// Whether a reliable background IPC payload failed before it could be retained.
    pub(crate) fn reliable_background_failed(&self) -> bool {
        self.ipc
            .as_ref()
            .is_some_and(DualQueueIpc::reliable_background_failed)
    }

    /// Polls host queues into the reusable batch and prioritizes init data.
    pub(crate) fn poll_commands(&mut self) -> Vec<TimedRendererCommand> {
        profiling::scope!("frontend::poll_commands");
        let mut batch = std::mem::take(&mut self.command_batch);
        if let Some(ipc) = self.ipc.as_mut() {
            ipc.poll_timed_into(&mut batch);
            prioritize_renderer_init_data(&mut batch);
        } else {
            batch.clear();
        }
        batch
    }

    /// Waits briefly for primary-queue work, then polls and prioritizes init data.
    pub(crate) fn poll_commands_after_primary_wait(
        &mut self,
        timeout: Duration,
    ) -> (Vec<TimedRendererCommand>, Duration) {
        profiling::scope!("frontend::poll_commands_after_primary_wait");
        let mut batch = std::mem::take(&mut self.command_batch);
        let mut waited = Duration::ZERO;
        if let Some(ipc) = self.ipc.as_mut() {
            let stats = ipc.poll_timed_into_after_primary_wait_profiled(&mut batch, timeout);
            waited = stats.waited;
            {
                profiling::scope!("ipc::batch_prioritize");
                prioritize_renderer_init_data(&mut batch);
            }
            let total = stats.total_drain();
            plot_ipc_poll(&IpcPollProfileSample {
                waited: stats.waited,
                messages: total.messages,
                bytes: total.bytes,
                decode_duration: total.decode_duration,
                timed_out: stats.timed_out,
            });
        } else {
            batch.clear();
        }
        (batch, waited)
    }

    /// Returns the drained command batch so the allocation is retained for the next poll.
    pub(crate) fn recycle_command_batch(&mut self, batch: Vec<TimedRendererCommand>) {
        self.command_batch = batch;
    }
}

/// Moves host init data to the front while preserving all other command arrival order.
fn prioritize_renderer_init_data(batch: &mut [TimedRendererCommand]) {
    let mut insert_at = 0;
    for idx in 0..batch.len() {
        if matches!(&batch[idx].command, RendererCommand::RendererInitData(_)) {
            batch[insert_at..=idx].rotate_right(1);
            insert_at += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::prioritize_renderer_init_data;
    use crate::frontend::dispatch::renderer_command_kind::renderer_command_variant_tag;
    use crate::ipc::TimedRendererCommand;
    use crate::shared::{
        FrameSubmitData, QualityConfig, RendererCommand, RendererInitData, SetTexture2DFormat,
    };

    fn timed(cmd: RendererCommand) -> TimedRendererCommand {
        TimedRendererCommand::received_now(cmd)
    }

    fn command_tags(commands: &[TimedRendererCommand]) -> Vec<&'static str> {
        commands
            .iter()
            .map(|timed| renderer_command_variant_tag(&timed.command))
            .collect()
    }

    #[test]
    fn init_data_priority_preserves_non_init_order() {
        let mut batch = vec![
            timed(RendererCommand::SetTexture2DFormat(
                SetTexture2DFormat::default(),
            )),
            timed(RendererCommand::FrameSubmitData(FrameSubmitData::default())),
            timed(RendererCommand::QualityConfig(QualityConfig::default())),
        ];

        prioritize_renderer_init_data(&mut batch);

        assert_eq!(
            command_tags(&batch),
            vec!["SetTexture2DFormat", "FrameSubmitData", "QualityConfig"]
        );
    }

    #[test]
    fn init_data_priority_stably_moves_init_data_to_front() {
        let mut batch = vec![
            timed(RendererCommand::QualityConfig(QualityConfig::default())),
            timed(RendererCommand::RendererInitData(
                RendererInitData::default(),
            )),
            timed(RendererCommand::SetTexture2DFormat(
                SetTexture2DFormat::default(),
            )),
            timed(RendererCommand::RendererInitData(
                RendererInitData::default(),
            )),
            timed(RendererCommand::FrameSubmitData(FrameSubmitData::default())),
        ];

        prioritize_renderer_init_data(&mut batch);

        assert_eq!(
            command_tags(&batch),
            vec![
                "RendererInitData",
                "RendererInitData",
                "QualityConfig",
                "SetTexture2DFormat",
                "FrameSubmitData"
            ]
        );
    }

    #[test]
    fn init_data_priority_keeps_timestamps_attached() {
        let base = Instant::now();
        let init_received_at = base + Duration::from_millis(7);
        let mut batch = vec![
            TimedRendererCommand::new(
                RendererCommand::QualityConfig(QualityConfig::default()),
                base,
            ),
            TimedRendererCommand::new(
                RendererCommand::RendererInitData(RendererInitData::default()),
                init_received_at,
            ),
        ];

        prioritize_renderer_init_data(&mut batch);

        assert!(matches!(
            batch[0].command,
            RendererCommand::RendererInitData(_)
        ));
        assert_eq!(batch[0].received_at, init_received_at);
    }
}
