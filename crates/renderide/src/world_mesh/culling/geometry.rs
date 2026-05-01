//! Mesh target identity and world-space geometry construction for CPU culling.

use glam::{Mat4, Vec3};

use crate::assets::mesh::GpuMesh;
use crate::scene::{RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer};
use crate::shared::RenderingContext;

use super::WorldMeshCullInput;
use super::frustum::{mesh_bounds_degenerate_for_cull, world_aabb_from_local_bounds};

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

/// World-space AABB and rigid transform for a single CPU cull evaluation.
///
/// View-invariant for non-overlay spaces (the matrix and bounds are functions of the scene,
/// mesh, and `render_context` only); overlay spaces re-root against the view's
/// `head_output_transform`, so a precomputed value is invalid for them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeshCullGeometry {
    /// When `None`, culling treats the draw as visible (conservative).
    pub world_aabb: Option<(Vec3, Vec3)>,
    /// World matrix for rigid meshes when [`Self::world_aabb`] was built from local bounds.
    pub rigid_world_matrix: Option<Mat4>,
}

/// World-space AABB (and rigid matrix when applicable) for culling, evaluated once per draw slot.
pub(crate) fn mesh_world_geometry_for_cull(
    target: &MeshCullTarget<'_>,
    culling: &WorldMeshCullInput<'_>,
    render_context: RenderingContext,
) -> MeshCullGeometry {
    mesh_world_geometry_for_cull_with_head(
        target,
        culling.host_camera.head_output_transform,
        render_context,
    )
}

/// Same as [`mesh_world_geometry_for_cull`] but takes the per-view `head_output_transform`
/// directly so non-overlay frame-time precompute (which has no view yet) can pass `Mat4::IDENTITY`.
///
/// Caller is responsible for ensuring overlay spaces use the live per-view transform; the result
/// is only view-invariant when `target.scene.space(target.space_id).is_overlay == false`.
pub(crate) fn mesh_world_geometry_for_cull_with_head(
    target: &MeshCullTarget<'_>,
    head_output_transform: Mat4,
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
    if target.skinned {
        let Some(sk) = target.skinned_renderer else {
            return MeshCullGeometry {
                world_aabb: None,
                rigid_world_matrix: None,
            };
        };
        // Posed bound from the host lives in the renderer-root local frame. Transform it by the
        // root bone world matrix; when absent, fall back to the renderable node.
        let root_node = sk
            .root_bone_transform_id
            .filter(|&id| id >= 0)
            .map_or(target.node_id as usize, |id| id as usize);
        let Some(root_world) = target.scene.world_matrix_for_render_context(
            target.space_id,
            root_node,
            render_context,
            head_output_transform,
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
            head_output_transform,
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
