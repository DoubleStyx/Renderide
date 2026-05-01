//! CPU frustum and Hi-Z culling helpers for [`crate::world_mesh::draw_prep::collect_and_sort_draws`].
//!
//! Shares one bounds evaluation per draw slot using the same view–projection rules as the forward pass
//! ([`super::build_world_mesh_cull_proj_params`]), including
//! [`crate::camera::HostCameraFrame::explicit_world_to_view`] when an explicit camera view is
//! present (e.g. for secondary render-texture cameras).

use glam::{Mat4, Vec3};

use crate::scene::RenderSpaceId;
use crate::shared::RenderingContext;

use super::frustum::world_aabb_visible_in_homogeneous_clip;
use super::geometry::{MeshCullGeometry, MeshCullTarget, mesh_world_geometry_for_cull};
use super::{HiZTemporalState, WorldMeshCullInput, WorldMeshCullProjParams};
use crate::camera::view_matrix_for_world_mesh_render_space;
use crate::occlusion::HiZCullData;
use crate::occlusion::hi_z_view_proj_matrices;
use crate::occlusion::mesh_fully_occluded_in_hiz;
use crate::occlusion::stereo_hiz_keeps_draw;
use crate::scene::SceneCoordinator;

/// Frustum acceptance for one world AABB using the same stereo / overlay rules as the forward pass.
fn cpu_cull_frustum_visible(
    proj: &WorldMeshCullProjParams,
    is_overlay: bool,
    view: Mat4,
    wmin: Vec3,
    wmax: Vec3,
) -> bool {
    if let Some((sl, sr)) = proj.vr_stereo {
        if is_overlay {
            let vp = proj.overlay_proj * view;
            world_aabb_visible_in_homogeneous_clip(vp, wmin, wmax)
        } else {
            world_aabb_visible_in_homogeneous_clip(sl, wmin, wmax)
                || world_aabb_visible_in_homogeneous_clip(sr, wmin, wmax)
        }
    } else {
        let base_proj = if is_overlay {
            proj.overlay_proj
        } else {
            proj.world_proj
        };
        let vp = base_proj * view;
        world_aabb_visible_in_homogeneous_clip(vp, wmin, wmax)
    }
}

/// Returns `true` when the draw should be **culled** by Hi-Z (fully occluded).
fn cpu_cull_hi_z_should_cull(
    space_id: RenderSpaceId,
    wmin: Vec3,
    wmax: Vec3,
    culling: &WorldMeshCullInput<'_>,
) -> bool {
    let Some(hi) = &culling.hi_z else {
        return false;
    };
    let Some(temporal) = &culling.hi_z_temporal else {
        return false;
    };
    if !hi_z_snapshot_matches_temporal(hi, temporal) {
        return false;
    }
    let Some(prev_view) = temporal.prev_view_by_space.get(&space_id).copied() else {
        return false;
    };

    let passes_hiz = match hi {
        HiZCullData::Desktop(snap) => {
            if temporal.prev_cull.vr_stereo.is_some() {
                true
            } else {
                let vps = hi_z_view_proj_matrices(&temporal.prev_cull, prev_view, false);
                match vps.first().copied() {
                    None => true,
                    Some(vp) => !mesh_fully_occluded_in_hiz(snap, vp, wmin, wmax),
                }
            }
        }
        HiZCullData::Stereo { left, right } => match temporal.prev_cull.vr_stereo {
            None => true,
            Some((sl, sr)) => {
                let oc_l = mesh_fully_occluded_in_hiz(left, sl, wmin, wmax);
                let oc_r = mesh_fully_occluded_in_hiz(right, sr, wmin, wmax);
                stereo_hiz_keeps_draw(oc_l, oc_r)
            }
        },
    };

    !passes_hiz
}

/// Which CPU cull stage rejected the draw (for diagnostics counters).
pub(crate) enum CpuCullFailure {
    Frustum,
    HiZ,
}

/// Frustum + optional Hi-Z culling using a single [`mesh_world_geometry_for_cull`] evaluation.
///
/// On success, returns the rigid world matrix when the draw is non-skinned and the matrix was
/// computed while building bounds (reuse in the forward pass).
pub(crate) fn mesh_draw_passes_cpu_cull(
    target: &MeshCullTarget<'_>,
    is_overlay: bool,
    culling: &WorldMeshCullInput<'_>,
    render_context: RenderingContext,
) -> Result<Option<Mat4>, CpuCullFailure> {
    let geom = mesh_world_geometry_for_cull(target, culling, render_context);
    mesh_cpu_cull_with_geometry(geom, target.scene, target.space_id, is_overlay, culling)
}

/// Like [`mesh_draw_passes_cpu_cull`] but skips the [`mesh_world_geometry_for_cull`] call when
/// the caller already has a frame-time precomputed [`MeshCullGeometry`] (typical for non-overlay
/// draws cached on [`crate::world_mesh::draw_prep::FramePreparedRenderables`]).
///
/// Frustum + Hi-Z tests stay per-view; only the matrix and AABB derivation is amortized. Returns
/// the same `Result<Option<Mat4>, CpuCullFailure>` as [`mesh_draw_passes_cpu_cull`].
pub(crate) fn mesh_cpu_cull_with_geometry(
    geom: MeshCullGeometry,
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    is_overlay: bool,
    culling: &WorldMeshCullInput<'_>,
) -> Result<Option<Mat4>, CpuCullFailure> {
    let Some((wmin, wmax)) = geom.world_aabb else {
        return Ok(geom.rigid_world_matrix);
    };

    let Some(space) = scene.space(space_id) else {
        return Ok(geom.rigid_world_matrix);
    };
    let view = culling
        .host_camera
        .explicit_world_to_view()
        .unwrap_or_else(|| view_matrix_for_world_mesh_render_space(scene, space));
    let proj = &culling.proj;

    if !cpu_cull_frustum_visible(proj, is_overlay, view, wmin, wmax) {
        return Err(CpuCullFailure::Frustum);
    }

    if is_overlay {
        return Ok(geom.rigid_world_matrix);
    }

    if cpu_cull_hi_z_should_cull(space_id, wmin, wmax, culling) {
        return Err(CpuCullFailure::HiZ);
    }
    Ok(geom.rigid_world_matrix)
}

/// Ensures CPU Hi-Z dimensions match the temporal viewport used when the pyramid was built.
fn hi_z_snapshot_matches_temporal(hi: &HiZCullData, t: &HiZTemporalState) -> bool {
    let (w, h) = t.depth_viewport_px;
    match hi {
        HiZCullData::Desktop(s) => s.base_width == w && s.base_height == h,
        HiZCullData::Stereo { left, .. } => left.base_width == w && left.base_height == h,
    }
}

#[cfg(test)]
mod hi_z_temporal_match_tests {
    //! [`super::hi_z_snapshot_matches_temporal`] dimension checks (stale-pyramid guard).

    use std::sync::Arc;

    use glam::Mat4;
    use hashbrown::HashMap;

    use super::hi_z_snapshot_matches_temporal;
    use crate::occlusion::cpu::pyramid::total_float_count;
    use crate::occlusion::{HiZCpuSnapshot, HiZCullData};
    use crate::world_mesh::culling::{HiZTemporalState, WorldMeshCullProjParams};

    fn dummy_temporal(depth_viewport_px: (u32, u32)) -> HiZTemporalState {
        HiZTemporalState {
            prev_cull: WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj: Mat4::IDENTITY,
                vr_stereo: None,
            },
            prev_view_by_space: Arc::new(HashMap::new()),
            depth_viewport_px,
        }
    }

    fn snapshot(wx: u32, hy: u32) -> HiZCpuSnapshot {
        let mip_levels = 1u32;
        let n = total_float_count(wx, hy, mip_levels);
        HiZCpuSnapshot {
            base_width: wx,
            base_height: hy,
            mip_levels,
            mips: Arc::from(vec![0.0; n]),
        }
    }

    #[test]
    fn desktop_matches_when_mip0_matches_temporal_viewport() {
        let t = dummy_temporal((128, 96));
        let hi = HiZCullData::Desktop(snapshot(128, 96));
        assert!(hi_z_snapshot_matches_temporal(&hi, &t));
    }

    #[test]
    fn desktop_mismatches_when_pyramid_resolution_differs() {
        let t = dummy_temporal((128, 96));
        let hi = HiZCullData::Desktop(snapshot(64, 96));
        assert!(!hi_z_snapshot_matches_temporal(&hi, &t));
    }

    #[test]
    fn stereo_matches_left_eye_mip0_against_temporal_viewport() {
        let t = dummy_temporal((256, 144));
        let left = snapshot(256, 144);
        let right = snapshot(1, 1);
        let hi = HiZCullData::Stereo { left, right };
        assert!(hi_z_snapshot_matches_temporal(&hi, &t));
    }

    #[test]
    fn stereo_mismatches_when_left_eye_size_differs() {
        let t = dummy_temporal((256, 144));
        let left = snapshot(128, 144);
        let right = snapshot(256, 144);
        let hi = HiZCullData::Stereo { left, right };
        assert!(!hi_z_snapshot_matches_temporal(&hi, &t));
    }
}
