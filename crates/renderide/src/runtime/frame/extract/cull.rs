//! Per-view frustum and Hi-Z cull snapshot capture for frame extraction.

use crate::backend::ExtractedFrameShared;
use crate::hi_z_cpu::HiZCullData;
use crate::render_graph::OffscreenWriteTarget;
use crate::world_mesh::{
    HiZTemporalState, WorldMeshCullProjParams, build_world_mesh_cull_proj_params,
};

use super::super::view_plan::FrameViewPlan;

/// Frustum + Hi-Z cull inputs for one planned view.
pub(super) struct ViewCullSnapshot {
    /// Projection parameters matching the view's camera/viewport.
    pub(super) proj: WorldMeshCullProjParams,
    /// CPU-side Hi-Z snapshot for this view's occlusion slot.
    pub(super) hi_z: Option<HiZCullData>,
    /// Temporal Hi-Z state captured after the prior frame's depth pyramid author pass.
    pub(super) hi_z_temporal: Option<HiZTemporalState>,
}

/// Builds frustum + Hi-Z cull inputs for one prepared view.
///
/// Suppressed temporal occlusion still builds frustum inputs, but skips Hi-Z snapshots. Safe to
/// call in parallel across views:
/// [`OcclusionSystem`](crate::occlusion::OcclusionSystem) is `Sync` because its internal readback
/// channel uses `crossbeam_channel`.
pub(super) fn cull_snapshot_for_view(
    setup: &ExtractedFrameShared<'_>,
    prep: &FrameViewPlan<'_>,
) -> Option<ViewCullSnapshot> {
    build_cull_snapshot_for_view(setup.scene, setup.occlusion, prep)
}

pub(super) fn build_cull_snapshot_for_view(
    scene: &crate::scene::SceneCoordinator,
    occlusion: &crate::occlusion::OcclusionSystem,
    prep: &FrameViewPlan<'_>,
) -> Option<ViewCullSnapshot> {
    let camera_proj = build_world_mesh_cull_proj_params(scene, prep.viewport_px, &prep.host_camera);
    let proj = cull_projection_for_write_target(&camera_proj, prep.write_target());
    let depth_mode = prep.output_depth_mode();
    let (hi_z, hi_z_temporal) = if prep.host_camera.suppress_occlusion_temporal {
        (None, None)
    } else {
        (
            occlusion.hi_z_cull_data(depth_mode, prep.view_id),
            occlusion.hi_z_temporal_snapshot(prep.view_id),
        )
    };
    Some(ViewCullSnapshot {
        proj,
        hi_z,
        hi_z_temporal,
    })
}

pub(super) fn cull_projection_for_write_target(
    proj: &WorldMeshCullProjParams,
    write_target: OffscreenWriteTarget,
) -> WorldMeshCullProjParams {
    proj.map_projection_matrices(|projection| write_target.render_projection(projection))
}
