//! Tracy plots for Rayon admission decisions.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;

/// Rayon admission counters for one runtime work-site decision.
#[derive(Clone, Copy, Debug, Default)]
pub struct RayonAdmissionProfileSample {
    /// Stable numeric id for the work site.
    pub site_id: u64,
    /// Domain work units considered for this admission decision.
    pub work_units: u64,
    /// Independent items available to the Rayon split.
    pub independent_items: u64,
    /// Domain items assigned to one Rayon task packet, or zero for serial execution.
    pub chunk_size: u64,
    /// Number of task packets implied by `work_units` and `chunk_size`, or zero for serial.
    pub chunk_count: u64,
    /// Reference-capped worker count available to the decision.
    pub worker_count: u64,
    /// Non-zero when this decision selected Rayon.
    pub parallel: u64,
}

/// Records one Rayon admission decision.
#[inline]
pub fn plot_rayon_admission(sample: RayonAdmissionProfileSample) {
    tracy_plot!("rayon_admission::site_id", sample.site_id as f64);
    tracy_plot!("rayon_admission::work_units", sample.work_units as f64);
    tracy_plot!(
        "rayon_admission::independent_items",
        sample.independent_items as f64
    );
    tracy_plot!("rayon_admission::chunk_size", sample.chunk_size as f64);
    tracy_plot!("rayon_admission::chunk_count", sample.chunk_count as f64);
    tracy_plot!("rayon_admission::worker_count", sample.worker_count as f64);
    tracy_plot!("rayon_admission::parallel", sample.parallel as f64);
}
