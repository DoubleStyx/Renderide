//! Profiling and trace diagnostics for retained GPU-cull dispatches.

use std::mem::size_of;

use crate::camera::ViewId;
use crate::gpu::cull_compact::{GpuCullCandidate, GpuCullDispatch, GpuCullMatrix, GpuCullRun};

/// Which component of the structural cache key failed on the last probe.
#[derive(Clone, Copy, Default)]
pub(super) struct StructuralMissReason {
    pub(super) plan: bool,
    pub(super) arena: bool,
    pub(super) packets: bool,
}

/// Inputs needed to report one retained GPU-cull dispatch without retaining its resources.
#[derive(Clone, Copy)]
pub(super) struct GpuCullDispatchProfile {
    pub(super) view_id: ViewId,
    pub(super) structural_hit: bool,
    pub(super) input_hit: bool,
    pub(super) structural_miss: StructuralMissReason,
    pub(super) input_candidate_count: usize,
    pub(super) input_run_count: usize,
    pub(super) matrix_count: usize,
    pub(super) dispatch: GpuCullDispatch,
    pub(super) structural_hits: u64,
    pub(super) structural_misses: u64,
    pub(super) materialized_hits: u64,
    pub(super) materialized_misses: u64,
}

/// Records cache-hit and upload-volume plots for one compute dispatch.
pub(super) fn plot_dispatch(profile: &GpuCullDispatchProfile) {
    let static_input_bytes = profile
        .input_candidate_count
        .saturating_mul(size_of::<GpuCullCandidate>())
        .saturating_add(
            profile
                .input_run_count
                .saturating_mul(size_of::<GpuCullRun>()),
        );
    crate::profiling::plot_world_mesh_gpu_cull_cache(
        crate::profiling::WorldMeshGpuCullCacheProfileSample {
            structural_hit: profile.structural_hit,
            input_hit: profile.input_hit,
            structural_miss_plan: profile.structural_miss.plan,
            structural_miss_arena: profile.structural_miss.arena,
            structural_miss_packets: profile.structural_miss.packets,
            static_upload_bytes: if profile.dispatch.static_inputs_uploaded {
                static_input_bytes
            } else {
                0
            },
            avoided_static_upload_bytes: if profile.dispatch.static_inputs_uploaded {
                0
            } else {
                static_input_bytes
            },
            matrix_upload_bytes: profile
                .matrix_count
                .saturating_mul(size_of::<GpuCullMatrix>()),
        },
    );
}

/// Emits the human-readable cache summary for one compute dispatch.
pub(super) fn trace_dispatch(profile: &GpuCullDispatchProfile) {
    logger::trace!(
        "GPU world-mesh cull {:?}: {} candidates across {} runs ({:?}), plan_cache={}, input_cache={}, static_upload={}, totals=plan({}/{}) inputs({}/{})",
        profile.view_id,
        profile.dispatch.candidate_count,
        profile.input_run_count,
        profile.dispatch.output_mode,
        if profile.structural_hit {
            "hit"
        } else {
            "miss"
        },
        if profile.input_hit { "hit" } else { "miss" },
        profile.dispatch.static_inputs_uploaded,
        profile.structural_hits,
        profile.structural_misses,
        profile.materialized_hits,
        profile.materialized_misses,
    );
}
