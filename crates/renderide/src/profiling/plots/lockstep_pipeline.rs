//! Tracy plots for host/renderer lock-step pipelining decisions.

use super::tracy_plot::tracy_plot;

/// Per-frame lock-step pipeline diagnostics.
pub struct LockstepPipelineProfileSample {
    /// Stable code for the selected pipeline action.
    pub action: f64,
    /// Stable code for why one-credit early begin-frame was blocked.
    pub one_credit_block: f64,
    /// Stable code for why a host-submit wait fallback ran.
    pub wait_reason: f64,
}

/// Records lock-step pipeline diagnostics on Tracy plots.
#[inline]
pub fn plot_lockstep_pipeline(sample: &LockstepPipelineProfileSample) {
    tracy_plot!("lockstep::pipeline_action", sample.action);
    tracy_plot!("lockstep::one_credit_block_reason", sample.one_credit_block);
    tracy_plot!("lockstep::host_wait_reason", sample.wait_reason);
}
