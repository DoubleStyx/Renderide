//! CPU frustum and Hi-Z culling helpers for [`crate::world_mesh::draw_prep::queue_draws_with_parallelism`].
//!
//! Shares one bounds evaluation per draw slot using the same view-projection rules as the forward pass
//! ([`super::build_world_mesh_cull_proj_params`]), including
//! [`crate::camera::HostCameraFrame::explicit_world_to_view`] when an explicit camera view is
//! present (e.g. for secondary render-texture cameras).

use glam::{Mat4, Vec3, Vec4};

use crate::scene::RenderSpaceId;
use crate::shared::RenderingContext;

use super::frustum::world_aabb_visible_in_homogeneous_clip;
use super::geometry::{MeshCullGeometry, MeshCullTarget, mesh_world_geometry_for_cull};
use super::{HiZTemporalState, WorldMeshCullInput, WorldMeshCullProjParams};
use crate::camera::{overlay_camera_view_matrix, view_matrix_for_world_mesh_render_space};
use crate::hi_z_cpu::HiZCullData;
use crate::hi_z_cpu::hi_z_view_proj_matrices;
use crate::hi_z_cpu::mesh_fully_occluded_in_hiz;
use crate::hi_z_cpu::stereo_hiz_keeps_draw;
use crate::scene::SceneCoordinator;

/// Frustum acceptance for one world AABB using the same stereo / overlay rules as the forward pass.
fn cpu_cull_frustum_visible(
    proj: &WorldMeshCullProjParams,
    is_overlay: bool,
    view: Mat4,
    wmin: Vec3,
    wmax: Vec3,
) -> bool {
    if is_overlay {
        let vp = proj.overlay_proj * overlay_camera_view_matrix();
        return world_aabb_visible_in_homogeneous_clip(vp, wmin, wmax);
    }
    if let Some((sl, sr)) = proj.vr_stereo {
        world_aabb_visible_in_homogeneous_clip(sl, wmin, wmax)
            || world_aabb_visible_in_homogeneous_clip(sr, wmin, wmax)
    } else {
        let vp = proj.world_proj * view;
        world_aabb_visible_in_homogeneous_clip(vp, wmin, wmax)
    }
}

/// Returns whether a world-space AABB is visible to the culling frustum for one render space.
///
/// This is the frustum-only portion of [`mesh_cpu_cull_with_geometry`], exposed so CPU spatial
/// indices can reject whole renderer candidate groups before per-renderer material slots are
/// expanded.
pub(crate) fn world_aabb_visible_for_cull(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    is_overlay: bool,
    culling: &WorldMeshCullInput<'_>,
    wmin: Vec3,
    wmax: Vec3,
) -> bool {
    if is_overlay {
        return cpu_cull_frustum_visible(&culling.proj, true, Mat4::IDENTITY, wmin, wmax);
    }
    let Some(space) = scene.space(space_id) else {
        return true;
    };
    let view = culling
        .host_camera
        .explicit_world_to_view()
        .unwrap_or_else(|| view_matrix_for_world_mesh_render_space(scene, space));
    cpu_cull_frustum_visible(&culling.proj, is_overlay, view, wmin, wmax)
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
    /// Overlay UI draw rejected by the object-local rect-clip mask projected to screen space.
    UiRectMask,
}

/// Frustum + optional Hi-Z culling using a single [`mesh_world_geometry_for_cull`] evaluation.
///
/// On success, returns the rigid world matrix when the draw is non-skinned and the matrix was
/// computed while building bounds (reuse in the forward pass).
///
/// `ui_rect_clip_local` is the object-local UI rect (`xMin, yMin, xMax, yMax`) for overlay draws
/// that opt in to `_RectClip`. When `Some`, the overlay path projects the rect's four corners
/// through the same desktop overlay camera view-projection used by the draw pass and rejects the
/// draw when the projected screen-space AABB doesn't intersect the viewport.
pub(crate) fn mesh_draw_passes_cpu_cull(
    target: &MeshCullTarget<'_>,
    is_overlay: bool,
    culling: &WorldMeshCullInput<'_>,
    render_context: RenderingContext,
    ui_rect_clip_local: Option<Vec4>,
) -> Result<Option<Mat4>, CpuCullFailure> {
    let geom = mesh_world_geometry_for_cull(target, culling, render_context);
    mesh_cpu_cull_with_geometry(
        geom,
        target.scene,
        target.space_id,
        is_overlay,
        culling,
        ui_rect_clip_local,
    )
}

/// Like [`mesh_draw_passes_cpu_cull`] but skips the [`mesh_world_geometry_for_cull`] call when
/// the caller already has a frame-time precomputed [`MeshCullGeometry`] (typical for non-overlay
/// draws cached on [`crate::world_mesh::draw_prep::FramePreparedRenderables`]).
///
/// Frustum + Hi-Z tests stay per-view; only the matrix and AABB derivation is amortized. Returns
/// the same `Result<Option<Mat4>, CpuCullFailure>` as [`mesh_draw_passes_cpu_cull`]. The new
/// `ui_rect_clip_local` arg behaves the same as in [`mesh_draw_passes_cpu_cull`].
pub(crate) fn mesh_cpu_cull_with_geometry(
    geom: MeshCullGeometry,
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    is_overlay: bool,
    culling: &WorldMeshCullInput<'_>,
    ui_rect_clip_local: Option<Vec4>,
) -> Result<Option<Mat4>, CpuCullFailure> {
    if is_overlay {
        return cull_overlay_draw(scene, space_id, culling, ui_rect_clip_local, &geom);
    }

    let Some((wmin, wmax)) = geom.world_aabb else {
        return Ok(geom.rigid_world_matrix);
    };
    if !world_aabb_visible_for_cull(scene, space_id, is_overlay, culling, wmin, wmax) {
        return Err(CpuCullFailure::Frustum);
    }
    if cpu_cull_hi_z_should_cull(space_id, wmin, wmax, culling) {
        return Err(CpuCullFailure::HiZ);
    }
    Ok(geom.rigid_world_matrix)
}

/// Cull decision for overlay-layer draws: optional `_RectClip` projection check, then accept.
///
/// Overlay draws bypass the world-space frustum and Hi-Z stages by design -- their model matrix
/// already encodes screen-space position via
/// [`crate::scene::SceneCoordinator::overlay_layer_model_matrix_for_context`].
fn cull_overlay_draw(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    culling: &WorldMeshCullInput<'_>,
    ui_rect_clip_local: Option<Vec4>,
    geom: &MeshCullGeometry,
) -> Result<Option<Mat4>, CpuCullFailure> {
    if let (Some(rect), Some(model)) = (ui_rect_clip_local, geom.rigid_world_matrix)
        && !overlay_rect_clip_visible(scene, space_id, culling, rect, model)
    {
        return Err(CpuCullFailure::UiRectMask);
    }
    Ok(geom.rigid_world_matrix)
}

/// Projects the object-local UI rect through `model` then through the overlay projection and
/// returns `true` when its screen-space AABB intersects the viewport.
///
/// `_Rect` is in **object-local** space (matches `obj_xy` in `ui_unlit.wgsl`); the four corners
/// are `(rect.x|z, rect.y|w, 0)`. Overlay models already encode screen-space-relative position via
/// [`crate::scene::SceneCoordinator::overlay_layer_model_matrix_for_context`], so this uses the
/// fixed desktop overlay camera view matrix rather than the world camera view.
///
pub(crate) fn overlay_rect_clip_visible(
    _scene: &SceneCoordinator,
    _space_id: RenderSpaceId,
    culling: &WorldMeshCullInput<'_>,
    rect: Vec4,
    model: Mat4,
) -> bool {
    let corners = [
        model.transform_point3(Vec3::new(rect.x, rect.y, 0.0)),
        model.transform_point3(Vec3::new(rect.z, rect.y, 0.0)),
        model.transform_point3(Vec3::new(rect.z, rect.w, 0.0)),
        model.transform_point3(Vec3::new(rect.x, rect.w, 0.0)),
    ];
    let (wmin, wmax) = aabb_from_points(&corners);
    let view_proj = culling.proj.overlay_proj * overlay_camera_view_matrix();
    world_aabb_visible_in_homogeneous_clip(view_proj, wmin, wmax)
}

#[inline]
fn aabb_from_points(points: &[Vec3]) -> (Vec3, Vec3) {
    let mut min = points[0];
    let mut max = points[0];
    for &p in &points[1..] {
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
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
    use crate::hi_z_cpu::pyramid::total_float_count;
    use crate::hi_z_cpu::{HiZCpuSnapshot, HiZCullData};
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

#[cfg(test)]
mod overlay_cull_tests {
    use glam::{Mat4, Vec3, Vec4};

    use super::{CpuCullFailure, mesh_cpu_cull_with_geometry};
    use crate::camera::{CameraClipPlanes, HostCameraFrame, Viewport};
    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::world_mesh::culling::{
        MeshCullGeometry, WorldMeshCullInput, WorldMeshCullProjParams,
    };

    fn culling_with_overlay_proj(host_camera: &HostCameraFrame) -> WorldMeshCullInput<'_> {
        let overlay_proj = HostCameraFrame::overlay_projection(
            Viewport::from_tuple((100, 100)),
            CameraClipPlanes::new(3.0, 4.0),
        );
        WorldMeshCullInput {
            proj: WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj,
                vr_stereo: None,
            },
            host_camera,
            hi_z: None,
            hi_z_temporal: None,
        }
    }

    #[test]
    fn overlay_draws_bypass_world_space_frustum_checks() {
        let scene = SceneCoordinator::new();
        let host_camera = HostCameraFrame::default();
        let culling = WorldMeshCullInput {
            proj: WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj: Mat4::IDENTITY,
                vr_stereo: None,
            },
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let model = Mat4::from_translation(Vec3::new(1234.0, 5678.0, 0.0));
        let geom = MeshCullGeometry {
            world_aabb: Some((Vec3::splat(10_000.0), Vec3::splat(10_001.0))),
            rigid_world_matrix: Some(model),
            front_face_world_matrix: Some(model),
        };

        let Ok(accepted) =
            mesh_cpu_cull_with_geometry(geom, &scene, RenderSpaceId(999), true, &culling, None)
        else {
            panic!("overlay items should skip frustum/Hi-Z rejection");
        };

        assert_eq!(accepted, Some(model));
    }

    #[test]
    fn overlay_rect_outside_viewport_is_culled() {
        let scene = SceneCoordinator::new();
        let host_camera = HostCameraFrame::default();
        let culling = culling_with_overlay_proj(&host_camera);
        // Translate the rect 10 units to the right -- well outside the desktop overlay frustum.
        let model = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let rect = Vec4::new(0.0, 0.0, 0.5, 0.5);
        let geom = MeshCullGeometry {
            world_aabb: None,
            rigid_world_matrix: Some(model),
            front_face_world_matrix: Some(model),
        };

        let res = mesh_cpu_cull_with_geometry(
            geom,
            &scene,
            RenderSpaceId(999),
            true,
            &culling,
            Some(rect),
        );
        assert!(matches!(res, Err(CpuCullFailure::UiRectMask)));
    }

    #[test]
    fn overlay_rect_inside_viewport_passes() {
        let scene = SceneCoordinator::new();
        let host_camera = HostCameraFrame::default();
        let culling = culling_with_overlay_proj(&host_camera);
        let model = Mat4::IDENTITY;
        let rect = Vec4::new(-0.5, -0.5, 0.5, 0.5);
        let geom = MeshCullGeometry {
            world_aabb: None,
            rigid_world_matrix: Some(model),
            front_face_world_matrix: Some(model),
        };

        let res = mesh_cpu_cull_with_geometry(
            geom,
            &scene,
            RenderSpaceId(999),
            true,
            &culling,
            Some(rect),
        );
        assert!(matches!(res, Ok(Some(m)) if m == model));
    }

    #[test]
    fn overlay_without_rect_clip_still_passes() {
        let scene = SceneCoordinator::new();
        let host_camera = HostCameraFrame::default();
        let culling = culling_with_overlay_proj(&host_camera);
        // Same off-screen model as the culled case -- without a rect the overlay fast path must
        // still accept the draw, otherwise non-`_RectClip` overlay UI would regress.
        let model = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let geom = MeshCullGeometry {
            world_aabb: None,
            rigid_world_matrix: Some(model),
            front_face_world_matrix: Some(model),
        };

        let res =
            mesh_cpu_cull_with_geometry(geom, &scene, RenderSpaceId(999), true, &culling, None);
        assert!(matches!(res, Ok(Some(m)) if m == model));
    }
}
