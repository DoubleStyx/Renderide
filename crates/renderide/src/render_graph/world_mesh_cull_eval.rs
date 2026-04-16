//! CPU frustum and Hi-Z culling helpers for [`super::world_mesh_draw_prep::collect_and_sort_world_mesh_draws`].
//!
//! Shares one bounds evaluation per draw slot using the same view–projection rules as the forward pass
//! ([`super::world_mesh_cull::build_world_mesh_cull_proj_params`]), including
//! [`super::frame_params::HostCameraFrame::secondary_camera_world_to_view`] when set for secondary
//! render-texture cameras.

use glam::{Mat4, Vec3};

use crate::assets::mesh::GpuMesh;
use crate::scene::{RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer};
use crate::shared::RenderingContext;

use super::camera::view_matrix_for_world_mesh_render_space;
use super::frustum::{
    mesh_bounds_degenerate_for_cull, world_aabb_from_local_bounds,
    world_aabb_from_skinned_bone_origins, world_aabb_visible_in_homogeneous_clip,
};
use super::hi_z_view_proj_matrices;
use super::mesh_fully_occluded_in_hiz;
use super::skinning_palette::{build_skinning_palette, SkinningPaletteParams};
use super::stereo_hiz_keeps_draw;
use super::world_mesh_cull::{HiZTemporalState, WorldMeshCullInput, WorldMeshCullProjParams};
use super::HiZCullData;

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
        HiZCullData::Desktop(ref snap) => {
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
        HiZCullData::Stereo {
            ref left,
            ref right,
        } => match temporal.prev_cull.vr_stereo {
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
        let Some(pal) = build_skinning_palette(SkinningPaletteParams {
            scene: target.scene,
            space_id: target.space_id,
            skinning_bind_matrices: &target.mesh.skinning_bind_matrices,
            has_skeleton: target.mesh.has_skeleton,
            bone_transform_indices: &sk.bone_transform_indices,
            smr_node_id: sk.base.node_id,
            render_context,
            head_output_transform: hc.head_output_transform,
        }) else {
            return MeshCullGeometry {
                world_aabb: None,
                rigid_world_matrix: None,
            };
        };
        MeshCullGeometry {
            world_aabb: world_aabb_from_skinned_bone_origins(&target.mesh.bounds, &pal),
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
        .secondary_camera_world_to_view
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
