//! CPU frustum and Hi-Z culling helpers for [`super::draw_prep::collect_and_sort_world_mesh_draws`].
//!
//! Shares one bounds evaluation per draw slot using the same view–projection rules as the forward pass
//! ([`super::cull::build_world_mesh_cull_proj_params`]), including
//! [`crate::camera::HostCameraFrame::explicit_world_to_view`] when an explicit camera view is
//! present (e.g. for secondary render-texture cameras).

use glam::{Mat4, Vec3};

use crate::assets::mesh::GpuMesh;
use crate::scene::{RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer};
use crate::shared::RenderingContext;

use super::cull::{HiZTemporalState, WorldMeshCullInput, WorldMeshCullProjParams};
use super::frustum::{
    mesh_bounds_degenerate_for_cull, world_aabb_from_local_bounds,
    world_aabb_visible_in_homogeneous_clip,
};
use crate::camera::view_matrix_for_world_mesh_render_space;
use crate::occlusion::HiZCullData;
use crate::occlusion::hi_z_view_proj_matrices;
use crate::occlusion::mesh_fully_occluded_in_hiz;
use crate::occlusion::stereo_hiz_keeps_draw;

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

/// Identity of a mesh renderable being evaluated for CPU frustum / Hi-Z culling.
pub(crate) struct MeshCullTarget<'a> {
    /// Scene graph and spaces.
    pub scene: &'a SceneCoordinator,
    /// Render space containing the mesh.
    pub space_id: RenderSpaceId,
    /// Resident GPU mesh (bounds, skinning buffers).
    pub mesh: &'a GpuMesh,
    /// Whether this path uses skinned bone bounds.
    pub skinned: bool,
    /// Skinned renderer when `skinned` is true.
    pub skinned_renderer: Option<&'a SkinnedMeshRenderer>,
    /// Scene node index for rigid transform lookup.
    pub node_id: i32,
}

/// World-space bounds and rigid transform for a single CPU cull evaluation.
#[derive(Clone, Copy)]
struct MeshCullGeometry {
    /// When `None`, culling treats the draw as visible (conservative).
    world_aabb: Option<(Vec3, Vec3)>,
    /// World matrix for rigid meshes when [`MeshCullGeometry::world_aabb`] was built from local bounds.
    rigid_world_matrix: Option<Mat4>,
}

/// World-space AABB (and rigid matrix when applicable) for culling, evaluated once per draw slot.
fn mesh_world_geometry_for_cull(
    target: &MeshCullTarget<'_>,
    culling: &WorldMeshCullInput<'_>,
    render_context: RenderingContext,
) -> MeshCullGeometry {
    if mesh_bounds_degenerate_for_cull(&target.mesh.bounds) {
        return MeshCullGeometry {
            world_aabb: None,
            rigid_world_matrix: None,
        };
    }
    if target.scene.space(target.space_id).is_none() {
        return MeshCullGeometry {
            world_aabb: None,
            rigid_world_matrix: None,
        };
    }
    let hc = culling.host_camera;
    if target.skinned {
        let Some(sk) = target.skinned_renderer else {
            return MeshCullGeometry {
                world_aabb: None,
                rigid_world_matrix: None,
            };
        };
        // Posed bound from the host (`SkinnedMeshBoundsUpdate.local_bounds`) lives in the
        // renderer-root local frame (Unity-parity: `SkinnedMeshRenderer.localBounds` is in
        // `rootBone` local space when a root bone is set). Transform it by that bone's world
        // matrix. First-frame / no-root fallback is the mesh bind-pose AABB transformed by the
        // renderable's own world matrix — strictly tighter than the old bone-origin pad.
        let root_node = sk
            .root_bone_transform_id
            .filter(|&id| id >= 0)
            .map_or(target.node_id as usize, |id| id as usize);
        let Some(root_world) = target.scene.world_matrix_for_render_context(
            target.space_id,
            root_node,
            render_context,
            hc.head_output_transform,
        ) else {
            return MeshCullGeometry {
                world_aabb: None,
                rigid_world_matrix: None,
            };
        };
        let object_bounds = sk
            .posed_object_bounds
            .as_ref()
            .unwrap_or(&target.mesh.bounds);
        MeshCullGeometry {
            world_aabb: world_aabb_from_local_bounds(object_bounds, root_world),
            rigid_world_matrix: None,
        }
    } else {
        let Some(model) = target.scene.world_matrix_for_render_context(
            target.space_id,
            target.node_id as usize,
            render_context,
            hc.head_output_transform,
        ) else {
            return MeshCullGeometry {
                world_aabb: None,
                rigid_world_matrix: None,
            };
        };
        MeshCullGeometry {
            world_aabb: world_aabb_from_local_bounds(&target.mesh.bounds, model),
            rigid_world_matrix: Some(model),
        }
    }
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

    let Some((wmin, wmax)) = geom.world_aabb else {
        return Ok(geom.rigid_world_matrix);
    };

    let Some(space) = target.scene.space(target.space_id) else {
        return Ok(geom.rigid_world_matrix);
    };
    let view = culling
        .host_camera
        .explicit_world_to_view()
        .unwrap_or_else(|| view_matrix_for_world_mesh_render_space(target.scene, space));
    let proj = &culling.proj;

    if !cpu_cull_frustum_visible(proj, is_overlay, view, wmin, wmax) {
        return Err(CpuCullFailure::Frustum);
    }

    if is_overlay {
        return Ok(geom.rigid_world_matrix);
    }

    if cpu_cull_hi_z_should_cull(target.space_id, wmin, wmax, culling) {
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
    use crate::world_mesh::cull::{HiZTemporalState, WorldMeshCullProjParams};

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
