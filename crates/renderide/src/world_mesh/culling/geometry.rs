//! Mesh target identity and world-space geometry construction for CPU culling.

use glam::{Mat4, Vec3};

use crate::assets::mesh::GpuMesh;
use crate::bounds::world_aabb_from_local_bounds;
use crate::scene::{RenderSpaceId, SceneCoordinator, SceneTransformRead, SkinnedMeshRenderer};
use crate::shared::{RenderBoundingBox, RenderingContext};

#[cfg(test)]
use super::WorldMeshCullInput;
use super::frustum::mesh_bounds_degenerate_for_cull;

/// Identity of a mesh renderable being evaluated for CPU frustum / Hi-Z culling.
pub(crate) struct MeshCullTarget<'a, S: SceneTransformRead + ?Sized = SceneCoordinator> {
    /// Scene graph and spaces.
    pub scene: &'a S,
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MeshCullGeometry {
    /// When `None`, culling treats the draw as visible (conservative).
    pub world_aabb: Option<(Vec3, Vec3)>,
    /// World matrix for rigid meshes when [`Self::world_aabb`] was built from local bounds.
    pub rigid_world_matrix: Option<Mat4>,
    /// World transform whose upper-3x3 determinant selects front-face winding.
    ///
    /// This is separate from [`Self::rigid_world_matrix`] because skinned meshes can provide
    /// world-space deformed vertex streams while still needing root-transform parity for culling and
    /// `front_facing`-driven shading.
    pub front_face_world_matrix: Option<Mat4>,
}

/// Builds conservative cull geometry for a rigid draw that already carries its world matrix.
///
/// Generated mesh-particle rows use an explicit matrix instead of a scene transform. They still
/// need mesh-derived world bounds for CPU frustum culling, reflection-probe selection, and shadow
/// visibility. Untrusted bounds remain uncullable while the supplied matrices are retained for
/// raster winding and placement.
pub(crate) fn mesh_world_geometry_for_rigid_override(
    bounds: &RenderBoundingBox,
    model: Mat4,
    orientation_invariant: bool,
) -> MeshCullGeometry {
    let world_aabb = if mesh_bounds_degenerate_for_cull(bounds) {
        None
    } else if orientation_invariant {
        // View/Facing mesh particles replace model rotation with a camera-derived orthonormal
        // basis in WGSL. Bound every possible basis by a sphere around the model translation. The
        // shader preserves only model column lengths, so use the same component scale here.
        let scale = Vec3::new(
            model.x_axis.truncate().length(),
            model.y_axis.truncate().length(),
            model.z_axis.truncate().length(),
        );
        let farthest_local = (bounds.center.abs() + bounds.extents.abs()) * scale;
        let radius = farthest_local.length();
        let center = model.w_axis.truncate();
        (center.is_finite() && radius.is_finite()).then(|| {
            let extent = Vec3::splat(radius);
            (center - extent, center + extent)
        })
    } else {
        world_aabb_from_local_bounds(bounds, model)
    };
    MeshCullGeometry {
        world_aabb,
        rigid_world_matrix: Some(model),
        front_face_world_matrix: Some(model),
    }
}

/// World-space AABB (and rigid matrix when applicable) for culling, evaluated once per draw slot.
#[cfg(test)]
pub(crate) fn mesh_world_geometry_for_cull<S>(
    target: &MeshCullTarget<'_, S>,
    culling: &WorldMeshCullInput<'_>,
    render_context: RenderingContext,
) -> MeshCullGeometry
where
    S: SceneTransformRead + ?Sized,
{
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
/// is only view-invariant when `target.scene.space(target.space_id).is_overlay() == false`.
pub(crate) fn mesh_world_geometry_for_cull_with_head<S>(
    target: &MeshCullTarget<'_, S>,
    head_output_transform: Mat4,
    render_context: RenderingContext,
) -> MeshCullGeometry
where
    S: SceneTransformRead + ?Sized,
{
    if mesh_bounds_degenerate_for_cull(&target.mesh.bounds) {
        return MeshCullGeometry {
            world_aabb: None,
            rigid_world_matrix: None,
            front_face_world_matrix: None,
        };
    }
    if target.scene.space(target.space_id).is_none() {
        return MeshCullGeometry {
            world_aabb: None,
            rigid_world_matrix: None,
            front_face_world_matrix: None,
        };
    }
    if target.skinned {
        let Some(sk) = target.skinned_renderer else {
            return MeshCullGeometry {
                world_aabb: None,
                rigid_world_matrix: None,
                front_face_world_matrix: None,
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
                front_face_world_matrix: None,
            };
        };
        let object_bounds = sk
            .posed_object_bounds
            .as_ref()
            .unwrap_or(&target.mesh.bounds);
        MeshCullGeometry {
            world_aabb: world_aabb_from_local_bounds(object_bounds, root_world),
            rigid_world_matrix: None,
            front_face_world_matrix: Some(root_world),
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
                front_face_world_matrix: None,
            };
        };
        mesh_world_geometry_for_rigid_override(&target.mesh.bounds, model, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_override_geometry_transforms_mesh_bounds() {
        let bounds = RenderBoundingBox {
            center: Vec3::ZERO,
            extents: Vec3::ONE,
        };
        let model = Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            glam::Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 0.0),
        );

        let geometry = mesh_world_geometry_for_rigid_override(&bounds, model, false);

        assert_eq!(
            geometry.world_aabb,
            Some((Vec3::new(8.0, -2.0, -2.0), Vec3::new(12.0, 2.0, 2.0)))
        );
        assert_eq!(geometry.rigid_world_matrix, Some(model));
        assert_eq!(geometry.front_face_world_matrix, Some(model));
    }

    #[test]
    fn rigid_override_geometry_keeps_matrix_when_bounds_are_untrusted() {
        let model = Mat4::from_translation(Vec3::X);

        let geometry =
            mesh_world_geometry_for_rigid_override(&RenderBoundingBox::default(), model, false);

        assert_eq!(geometry.world_aabb, None);
        assert_eq!(geometry.rigid_world_matrix, Some(model));
        assert_eq!(geometry.front_face_world_matrix, Some(model));
    }

    #[test]
    fn rigid_override_geometry_can_bound_every_camera_aligned_orientation() {
        let bounds = RenderBoundingBox {
            center: Vec3::new(1.0, 0.0, 0.0),
            extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 1.0, 0.5),
            glam::Quat::from_rotation_y(0.7),
            Vec3::new(10.0, 20.0, 30.0),
        );

        let geometry = mesh_world_geometry_for_rigid_override(&bounds, model, true);
        let expected_radius = Vec3::new(4.0, 2.0, 1.5).length();
        assert!(geometry.world_aabb.is_some());
        let (min, max) = geometry.world_aabb.unwrap_or((Vec3::ZERO, Vec3::ZERO));

        assert!((min - (model.w_axis.truncate() - Vec3::splat(expected_radius))).length() < 1e-5);
        assert!((max - (model.w_axis.truncate() + Vec3::splat(expected_radius))).length() < 1e-5);
    }
}
