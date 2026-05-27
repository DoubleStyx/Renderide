//! IPC-facing entry points on [`RendererRuntime`].
//!
//! Owns the per-tick command drain ([`RendererRuntime::poll_ipc`]). Incoming commands are decoded
//! by [`crate::frontend::dispatch`] and applied by [`super::effects`], keeping frontend dispatch
//! independent of the runtime facade.

use std::time::Duration;

use crate::frontend::InitState;
use crate::frontend::dispatch::renderer_command_kind::renderer_command_variant_tag;
use crate::shared::RendererCommand;

use crate::diagnostics::log_throttle::LogThrottle;

use super::super::RendererRuntime;
use super::shader_material;

/// IPC command count that starts emitting throttled large-batch diagnostics.
const LARGE_IPC_BATCH_LOG_THRESHOLD: usize = 256;

/// Throttle for large IPC batches so burst diagnostics remain bounded.
static LARGE_IPC_BATCH_LOG: LogThrottle = LogThrottle::new();

impl RendererRuntime {
    /// Total number of post-handshake IPC commands logged as unhandled (sum of per-variant counters).
    pub fn unhandled_ipc_command_event_total(&self) -> u64 {
        self.ipc_state.unhandled_command_event_total()
    }

    /// Records one unhandled post-handshake renderer command for diagnostics.
    pub(crate) fn record_unhandled_renderer_command(&mut self, tag: &'static str) -> u64 {
        self.ipc_state.record_unhandled_renderer_command(tag)
    }

    /// Drains IPC and dispatches commands. Each poll batch is ordered so `renderer_init_data` runs
    /// first, then frame submits, then the rest (see [`crate::frontend::RendererFrontend::poll_commands`]).
    pub fn poll_ipc(&mut self) {
        let _ = self.poll_ipc_inner(None);
    }

    /// Waits up to `timeout` for primary-queue work before draining and dispatching IPC commands.
    pub(crate) fn poll_ipc_after_primary_wait(&mut self, timeout: Duration) -> Duration {
        self.poll_ipc_inner(Some(timeout))
    }

    #[expect(
        clippy::iter_with_drain,
        reason = "the drained batch is recycled to preserve IPC Vec allocation across frames"
    )]
    fn poll_ipc_inner(&mut self, primary_wait: Option<Duration>) -> Duration {
        profiling::scope!("ipc::poll_batch");
        shader_material::drain_pending_shader_resolutions(
            &mut self.ipc_state.pending_shader_resolutions,
            &mut self.backend,
            &mut self.frontend,
        );
        let (mut batch, waited) = match primary_wait {
            Some(timeout) => self.frontend.poll_commands_after_primary_wait(timeout),
            None => (self.frontend.poll_commands(), Duration::ZERO),
        };
        trace_ipc_batch(
            &batch,
            self.frontend.init_state(),
            self.ipc_state.pending_shader_resolutions.len(),
        );
        log_large_ipc_batch_if_needed(
            &batch,
            self.frontend.init_state(),
            self.ipc_state.pending_shader_resolutions.len(),
        );
        for cmd in batch.drain(..) {
            let _tag = renderer_command_variant_tag(&cmd);
            profiling::scope!("ipc::dispatch", _tag);
            self.handle_ipc_command(cmd);
        }
        self.frontend.recycle_command_batch(batch);
        waited
    }
}

fn trace_ipc_batch(batch: &[RendererCommand], init_state: InitState, pending_shaders: usize) {
    if batch.is_empty() || !logger::enabled(logger::LogLevel::Trace) {
        return;
    }
    let kinds = crate::runtime::state::ipc::summarize_renderer_command_mix(batch.iter());
    logger::trace!(
        "IPC poll batch: commands={} init_state={:?} pending_shader_resolutions={} kinds=[{}]",
        batch.len(),
        init_state,
        pending_shaders,
        kinds,
    );
}

/// Emits a throttled debug summary for unusually large IPC command batches.
fn log_large_ipc_batch_if_needed(
    batch: &[RendererCommand],
    init_state: InitState,
    pending_shaders: usize,
) {
    if !ipc_batch_is_large(batch.len()) {
        return;
    }
    if !logger::enabled(logger::LogLevel::Debug) {
        return;
    }
    let Some(observation) = LARGE_IPC_BATCH_LOG.should_log(4, 64) else {
        return;
    };
    let kinds = crate::runtime::state::ipc::summarize_renderer_command_mix(batch.iter());
    logger::debug!(
        "IPC large poll batch: commands={} init_state={:?} pending_shader_resolutions={} occurrence={} kinds=[{}]",
        batch.len(),
        init_state,
        pending_shaders,
        observation,
        kinds,
    );
}

/// Returns whether a polled IPC batch is large enough for summary diagnostics.
fn ipc_batch_is_large(command_count: usize) -> bool {
    command_count >= LARGE_IPC_BATCH_LOG_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::{LARGE_IPC_BATCH_LOG_THRESHOLD, ipc_batch_is_large};

    #[test]
    fn ipc_batch_large_threshold_is_inclusive() {
        assert!(!ipc_batch_is_large(LARGE_IPC_BATCH_LOG_THRESHOLD - 1));
        assert!(ipc_batch_is_large(LARGE_IPC_BATCH_LOG_THRESHOLD));
    }
}
