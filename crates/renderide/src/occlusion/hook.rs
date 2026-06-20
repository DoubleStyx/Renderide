//! Render-graph-facing trait implemented by the occlusion subsystem.
//!
//! Render-graph code and graph-driven passes interact with occlusion through this trait so they
//! depend on a thin contract rather than the concrete [`crate::occlusion::OcclusionSystem`]. The
//! trait lives next to its implementer to keep the wide method surface (which references
//! occlusion-internal types like [`HiZBuildRecord`] and [`HiZBuildInput`]) in one place.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::camera::ViewId;
use crate::cull_contract::HiZTemporalState;
use crate::occlusion::gpu::{HiZBuildRecord, HiZGpuState};
use crate::occlusion::system::HiZBuildInput;

/// Contract for the occlusion system as seen by render-graph code and graph-driven passes.
///
/// Implemented by [`crate::occlusion::OcclusionSystem`]; render-graph code and pass code see
/// occlusion only through this trait. Methods preserve the inherent-method semantics so the
/// boundary is a pure decoupling step, not a behavior change.
///
/// `Send + Sync` so per-view parallel recording can share a single trait object across rayon
/// workers without additional synchronization.
pub trait OcclusionGraphHook: Send + Sync {
    /// Returns the mutex-wrapped Hi-Z slot for `view`, creating it if needed.
    fn ensure_hi_z_state(&self, view: ViewId) -> Arc<Mutex<HiZGpuState>>;

    /// Records the Hi-Z pyramid build into the supplied encoder.
    fn encode_hi_z_build_pass(
        &self,
        record: HiZBuildRecord<'_>,
        state_slot: &Mutex<HiZGpuState>,
        input: HiZBuildInput<'_>,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    );

    /// Stores the current view/projection snapshot for next-frame Hi-Z occlusion tests.
    fn store_hi_z_temporal_for_next_frame(
        &self,
        temporal: HiZTemporalState,
        state_slot: &Mutex<HiZGpuState>,
    );
}
