//! Tracy plots for asset-integration backlog and budget pressure.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;

/// Asset-integration backlog and budget-exhaustion counters for one drain.
#[derive(Clone, Copy, Debug, Default)]
pub struct AssetIntegrationProfileSample {
    /// Main-lane tasks still queued after the drain.
    pub main_queued: usize,
    /// High-priority tasks still queued after the drain.
    pub high_priority_queued: usize,
    /// Render-lane tasks still queued after the drain.
    pub render_queued: usize,
    /// Normal-priority tasks still queued after the drain.
    pub normal_priority_queued: usize,
    /// Particle-lane tasks still queued after the drain.
    pub particle_queued: usize,
    /// Asset-worker jobs waiting in the bounded queue.
    pub worker_queued: usize,
    /// Asset-worker jobs currently executing.
    pub worker_running: usize,
    /// Highest asset-worker queue depth observed since startup.
    pub worker_max_queued: usize,
    /// Asset-worker jobs executed inline because the bounded queue was saturated or unavailable.
    pub worker_inline_executed: u64,
    /// Asset-worker queue-full events.
    pub worker_saturated: u64,
    /// Main-lane steps processed by the drain.
    pub main_processed: u32,
    /// High-priority steps processed by the drain.
    pub high_priority_processed: u32,
    /// Render-lane steps processed by the drain.
    pub render_processed: u32,
    /// Normal-priority steps processed by the drain.
    pub normal_priority_processed: u32,
    /// Particle-lane steps processed by the drain.
    pub particle_processed: u32,
    /// Whether the high-priority emergency ceiling stopped the drain.
    pub high_priority_budget_exhausted: bool,
    /// Whether the normal-priority frame budget stopped the drain.
    pub normal_priority_budget_exhausted: bool,
}

/// Records asset-integration backlog and budget pressure for the current frame.
#[inline]
pub fn plot_asset_integration(sample: AssetIntegrationProfileSample) {
    tracy_plot!("asset_integration::main_queued", sample.main_queued as f64);
    tracy_plot!(
        "asset_integration::high_priority_queued",
        sample.high_priority_queued as f64
    );
    tracy_plot!(
        "asset_integration::render_queued",
        sample.render_queued as f64
    );
    tracy_plot!(
        "asset_integration::normal_priority_queued",
        sample.normal_priority_queued as f64
    );
    tracy_plot!(
        "asset_integration::particle_queued",
        sample.particle_queued as f64
    );
    tracy_plot!("asset_worker::queued", sample.worker_queued as f64);
    tracy_plot!("asset_worker::running", sample.worker_running as f64);
    tracy_plot!("asset_worker::max_queued", sample.worker_max_queued as f64);
    tracy_plot!(
        "asset_worker::inline_executed",
        sample.worker_inline_executed as f64
    );
    tracy_plot!("asset_worker::saturated", sample.worker_saturated as f64);
    tracy_plot!(
        "asset_integration::main_processed",
        sample.main_processed as f64
    );
    tracy_plot!(
        "asset_integration::high_priority_processed",
        sample.high_priority_processed as f64
    );
    tracy_plot!(
        "asset_integration::render_processed",
        sample.render_processed as f64
    );
    tracy_plot!(
        "asset_integration::normal_priority_processed",
        sample.normal_priority_processed as f64
    );
    tracy_plot!(
        "asset_integration::particle_processed",
        sample.particle_processed as f64
    );
    tracy_plot!(
        "asset_integration::high_priority_budget_exhausted",
        if sample.high_priority_budget_exhausted {
            1.0
        } else {
            0.0
        }
    );
    tracy_plot!(
        "asset_integration::normal_priority_budget_exhausted",
        if sample.normal_priority_budget_exhausted {
            1.0
        } else {
            0.0
        }
    );
}
