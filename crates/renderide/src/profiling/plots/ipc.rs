//! Tracy plots for host IPC wait, drain, and decode behavior.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them. These counters separate intentional lockstep frame pacing from command decoding
//! work so IPC wait time does not mask real CPU hotspots.

use std::time::Duration;

use super::tracy_plot::tracy_plot;

/// Per-poll IPC diagnostic counters emitted after a primary-wait poll.
pub struct IpcPollProfileSample {
    /// Time spent waiting for primary queue readiness.
    pub waited: Duration,
    /// Successfully decoded renderer commands across both inbound queues.
    pub messages: usize,
    /// Encoded inbound payload bytes consumed across both inbound queues.
    pub bytes: usize,
    /// Wall-clock time spent decoding renderer commands.
    pub decode_duration: Duration,
    /// Whether the primary wait consumed the caller's timeout without a ready message.
    pub timed_out: bool,
}

/// Records IPC poll counters on Tracy plots.
#[inline]
pub fn plot_ipc_poll(sample: &IpcPollProfileSample) {
    tracy_plot!("ipc::primary_wait_ms", sample.waited.as_secs_f64() * 1000.0);
    tracy_plot!("ipc::messages", sample.messages as f64);
    tracy_plot!("ipc::bytes", sample.bytes as f64);
    tracy_plot!(
        "ipc::decode_ms",
        sample.decode_duration.as_secs_f64() * 1000.0
    );
    tracy_plot!("ipc::primary_wait_timed_out", timed_out_value(sample));
}

/// Converts timeout state to a numeric Tracy plot sample.
#[inline]
fn timed_out_value(sample: &IpcPollProfileSample) -> f64 {
    if sample.timed_out { 1.0 } else { 0.0 }
}
