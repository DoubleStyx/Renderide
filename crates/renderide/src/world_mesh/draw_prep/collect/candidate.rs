//! Shared draw-candidate evaluation for world-mesh collection.

use glam::{Mat4, Vec3};

use crate::materials::host_data::MaterialPropertyLookupIds;
use crate::materials::{
    MaterialDepthCompareOverride, RasterFrontFace, RasterPrimitiveTopology,
    render_queue_is_transparent,
};
use crate::reflection_probes::specular::ReflectionProbeDrawSelection;
use crate::scene::{MeshRendererInstanceId, RenderSpaceId};
use crate::world_mesh::culling::overlay_rect_clip_visible;
use crate::world_mesh::materials::{
    FrameMaterialBatchCache, MaterialResolveCtx, batch_key_for_slot_cached, compute_batch_key_hash,
};

use super::super::item::WorldMeshDrawItem;
use super::DrawCollectionContext;

/// View-local material-slot draw candidate shared by scene-walk and prepared collection.
pub(super) struct DrawCandidate {
    /// Render space containing the source renderer.
    pub(super) space_id: RenderSpaceId,
    /// Scene node id used for transform and filter decisions.
    pub(super) node_id: i32,
    /// Dense renderer index inside the static or skinned renderer table selected by [`Self::skinned`].
    pub(super) renderable_index: usize,
    /// Renderer-local identity that survives dense table reindexing.
    pub(super) instance_id: MeshRendererInstanceId,
    /// Mesh asset id referenced by the source renderer.
    pub(super) mesh_asset_id: i32,
    /// Material slot index within the source renderer.
    pub(super) slot_index: usize,
    /// First index in the mesh index buffer.
    pub(super) first_index: u32,
    /// Number of indices emitted by the draw.
    pub(super) index_count: u32,
    /// Overlay layer flag copied into cull and draw metadata.
    pub(super) is_overlay: bool,
    /// Renderer sorting order copied into transparent ordering.
    pub(super) sorting_order: i32,
    /// Whether this draw uses skinned vertex streams.
    pub(super) skinned: bool,
    /// Whether skinning writes world-space positions.
    pub(super) world_space_deformed: bool,
    /// Whether blendshape scatter writes cache-backed positions for this draw.
    pub(super) blendshape_deformed: bool,
    /// Whether active blendshape tangent deltas should deform this draw if the material reads tangents.
    pub(super) tangent_blendshape_deform_active: bool,
    /// Material asset after render-context override resolution.
    pub(super) material_asset_id: i32,
    /// Property block associated with material slot zero.
    pub(super) property_block_id: Option<i32>,
    /// World-space object AABB used for transparent sorting and CPU reflection-probe selection.
    pub(super) world_aabb: Option<(Vec3, Vec3)>,
}

/// Builds a draw item from a cull-surviving material-slot candidate without allocating.
pub(super) fn evaluate_draw_candidate(
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    candidate: DrawCandidate,
    front_face: RasterFrontFace,
    primitive_topology: RasterPrimitiveTopology,
    rigid_world_matrix: Option<Mat4>,
    alpha_distance_sq: f32,
) -> Option<WorldMeshDrawItem> {
    if candidate.index_count == 0 || candidate.material_asset_id < 0 {
        return None;
    }
    let lookup_ids = MaterialPropertyLookupIds {
        material_asset_id: candidate.material_asset_id,
        mesh_property_block_slot0: candidate.property_block_id,
        mesh_renderer_property_block_id: None,
    };
    let (batch_key, ui_rect_clip_local) = batch_key_for_slot_cached(
        candidate.material_asset_id,
        candidate.property_block_id,
        candidate.skinned,
        front_face,
        primitive_topology,
        cache,
        MaterialResolveCtx {
            dict: ctx.material_dict,
            router: ctx.material_router,
            pipeline_property_ids: ctx.pipeline_property_ids,
            shader_perm: ctx.shader_perm,
        },
    );
    // Per-slot UI rect-mask CPU cull: the per-renderer cull above runs once per renderer (and
    // bypasses overlay anyway), so the actual rect-vs-viewport check has to live here, where
    // `_Rect` / `_RectClip` are finally known per material slot. Without this, every off-screen
    // masked UI element still hits the GPU and only `discard`s in the fragment shader -- which
    // is what tanks the friend list FPS.
    if candidate.is_overlay
        && let (Some(rect), Some(model), Some(culling)) =
            (ui_rect_clip_local, rigid_world_matrix, ctx.culling)
        && !overlay_rect_clip_visible(ctx.scene, candidate.space_id, culling, rect, model)
    {
        return None;
    }
    let batch_key = apply_overlay_layer_depth_policy(batch_key, candidate.is_overlay);
    let blendshape_deformed = candidate.blendshape_deformed
        || (candidate.tangent_blendshape_deform_active && batch_key.embedded_needs_tangent);
    let camera_distance_sq = if render_queue_is_transparent(batch_key.render_queue) {
        transparent_sort_distance_sq(
            ctx.view_origin_world,
            candidate.world_aabb,
            alpha_distance_sq,
        )
    } else {
        0.0
    };
    let batch_key_hash = compute_batch_key_hash(&batch_key);
    // Precompute the opaque depth bucket here so the sort comparator does not redo `sqrt + log2`
    // on every pairwise compare. The argument matches the previous comparator-side computation
    // (`opaque_depth_bucket(item.camera_distance_sq)`) so the resulting order is stable:
    // transparent-queue draws use the class-compatible sort metric preserved on
    // `camera_distance_sq`, while opaque-queue draws feed `0.0` and bucket to `0`,
    // leaving batch-key tiebreaking intact.
    let opaque_depth_bucket =
        crate::world_mesh::draw_prep::sort::opaque_depth_bucket(camera_distance_sq);
    let sort_prefix = crate::world_mesh::draw_prep::sort::pack_sort_prefix(
        candidate.is_overlay,
        batch_key.render_queue,
        opaque_depth_bucket,
        batch_key_hash,
    );
    let reflection_probes = match (ctx.reflection_probes, candidate.world_aabb) {
        (Some(selection), Some(aabb)) => selection.select(candidate.space_id, aabb),
        _ => ReflectionProbeDrawSelection::default(),
    };
    Some(WorldMeshDrawItem {
        space_id: candidate.space_id,
        node_id: candidate.node_id,
        renderable_index: candidate.renderable_index,
        instance_id: candidate.instance_id,
        mesh_asset_id: candidate.mesh_asset_id,
        slot_index: candidate.slot_index,
        first_index: candidate.first_index,
        index_count: candidate.index_count,
        is_overlay: candidate.is_overlay,
        sorting_order: candidate.sorting_order,
        skinned: candidate.skinned,
        world_space_deformed: candidate.world_space_deformed,
        blendshape_deformed,
        collect_order: 0,
        camera_distance_sq,
        lookup_ids,
        batch_key,
        batch_key_hash,
        _opaque_depth_bucket: opaque_depth_bucket,
        sort_prefix,
        rigid_world_matrix,
        reflection_probes,
        ui_rect_clip_local,
    })
}

/// Returns the transparent back-to-front sort metric for a draw candidate.
fn transparent_sort_distance_sq(
    view_origin_world: Vec3,
    world_aabb: Option<(Vec3, Vec3)>,
    fallback_distance_sq: f32,
) -> f32 {
    let fallback = finite_nonnegative_distance_sq(fallback_distance_sq).unwrap_or(0.0);
    let Some((min, max)) = world_aabb else {
        return fallback;
    };
    if !view_origin_world.is_finite() || !min.is_finite() || !max.is_finite() {
        return fallback;
    }
    let lo = min.min(max);
    let hi = min.max(max);
    let dx = (lo.x - view_origin_world.x)
        .abs()
        .max((hi.x - view_origin_world.x).abs());
    let dy = (lo.y - view_origin_world.y)
        .abs()
        .max((hi.y - view_origin_world.y).abs());
    let dz = (lo.z - view_origin_world.z)
        .abs()
        .max((hi.z - view_origin_world.z).abs());
    finite_nonnegative_distance_sq(dx.mul_add(dx, dy.mul_add(dy, dz * dz))).unwrap_or(fallback)
}

/// Returns finite non-negative distance values unchanged.
fn finite_nonnegative_distance_sq(value: f32) -> Option<f32> {
    (value.is_finite() && value >= 0.0).then_some(value)
}

/// Overlay-layer meshes are drawn through a separate camera stack after the world.
///
/// Renderide currently folds those meshes into the main forward pass, so they must bypass the
/// world depth buffer explicitly or ordinary scene geometry still occludes them even after their
/// transforms are projected into screen space.
fn apply_overlay_layer_depth_policy(
    mut batch_key: crate::world_mesh::MaterialDrawBatchKey,
    is_overlay: bool,
) -> crate::world_mesh::MaterialDrawBatchKey {
    if is_overlay {
        batch_key.render_state.depth_write = Some(false);
        batch_key.render_state.depth_compare = Some(MaterialDepthCompareOverride::Always);
    }
    batch_key
}

#[cfg(test)]
mod tests {
    //! CPU-only draw-candidate identity tests.

    use glam::{Mat4, Vec3};

    use super::*;
    use crate::gpu_pools::MeshPool;
    use crate::materials::host_data::{
        MaterialDictionary, MaterialPropertyStore, PropertyIdRegistry,
    };
    use crate::materials::{
        MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind, ShaderPermutation,
    };
    use crate::scene::SceneCoordinator;
    use crate::shared::RenderingContext;

    #[test]
    fn evaluate_draw_candidate_preserves_renderer_identity_separate_from_node_id() {
        let scene = SceneCoordinator::new();
        let mesh_pool = MeshPool::default_pool();
        let store = MaterialPropertyStore::new();
        let material_dict = MaterialDictionary::new(&store);
        let router = MaterialRouter::new(RasterPipelineKind::Null);
        let registry = PropertyIdRegistry::new();
        let property_ids = MaterialPipelinePropertyIds::new(&registry);
        let cache = FrameMaterialBatchCache::new();
        let ctx = DrawCollectionContext {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            material_router: &router,
            pipeline_property_ids: &property_ids,
            shader_perm: ShaderPermutation::default(),
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
            view_origin_world: Vec3::ZERO,
            culling: None,
            transform_filter: None,
            render_space_filter: None,
            material_cache: None,
            reflection_probes: None,
            prepared: None,
        };
        let candidate = DrawCandidate {
            space_id: RenderSpaceId(3),
            node_id: 9,
            renderable_index: 42,
            instance_id: MeshRendererInstanceId(99),
            mesh_asset_id: 7,
            slot_index: 0,
            first_index: 0,
            index_count: 3,
            is_overlay: false,
            sorting_order: 0,
            skinned: true,
            world_space_deformed: true,
            blendshape_deformed: true,
            tangent_blendshape_deform_active: false,
            material_asset_id: 11,
            property_block_id: None,
            world_aabb: None,
        };

        let item = evaluate_draw_candidate(
            &ctx,
            &cache,
            candidate,
            RasterFrontFace::Clockwise,
            RasterPrimitiveTopology::TriangleList,
            None,
            0.0,
        )
        .expect("draw item");

        assert_eq!(item.node_id, 9);
        assert_eq!(item.renderable_index, 42);
        assert_eq!(item.instance_id, MeshRendererInstanceId(99));
    }

    #[test]
    fn evaluate_draw_candidate_forces_overlay_depth_policy() {
        let scene = SceneCoordinator::new();
        let mesh_pool = MeshPool::default_pool();
        let store = MaterialPropertyStore::new();
        let material_dict = MaterialDictionary::new(&store);
        let router = MaterialRouter::new(RasterPipelineKind::Null);
        let registry = PropertyIdRegistry::new();
        let property_ids = MaterialPipelinePropertyIds::new(&registry);
        let cache = FrameMaterialBatchCache::new();
        let ctx = DrawCollectionContext {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            material_router: &router,
            pipeline_property_ids: &property_ids,
            shader_perm: ShaderPermutation::default(),
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
            view_origin_world: Vec3::ZERO,
            culling: None,
            transform_filter: None,
            render_space_filter: None,
            material_cache: None,
            reflection_probes: None,
            prepared: None,
        };
        let candidate = DrawCandidate {
            space_id: RenderSpaceId(3),
            node_id: 9,
            renderable_index: 42,
            instance_id: MeshRendererInstanceId(99),
            mesh_asset_id: 7,
            slot_index: 0,
            first_index: 0,
            index_count: 3,
            is_overlay: true,
            sorting_order: 0,
            skinned: false,
            world_space_deformed: false,
            blendshape_deformed: false,
            tangent_blendshape_deform_active: false,
            material_asset_id: 11,
            property_block_id: None,
            world_aabb: None,
        };

        let item = evaluate_draw_candidate(
            &ctx,
            &cache,
            candidate,
            RasterFrontFace::Clockwise,
            RasterPrimitiveTopology::TriangleList,
            None,
            0.0,
        )
        .expect("overlay draw item");

        assert_eq!(item.batch_key.render_state.depth_write, Some(false));
        assert_eq!(
            item.batch_key.render_state.depth_compare,
            Some(MaterialDepthCompareOverride::Always)
        );
    }

    #[test]
    fn transparent_sort_distance_uses_bounds_farthest_corner() {
        let distance = transparent_sort_distance_sq(
            Vec3::ZERO,
            Some((Vec3::new(9.0, -2.0, 0.0), Vec3::new(11.0, 2.0, 0.0))),
            100.0,
        );

        assert_eq!(distance, 125.0);
    }

    #[test]
    fn transparent_sort_distance_falls_back_without_valid_bounds() {
        assert_eq!(transparent_sort_distance_sq(Vec3::ZERO, None, 16.0), 16.0);
        assert_eq!(
            transparent_sort_distance_sq(Vec3::ZERO, Some((Vec3::NAN, Vec3::ONE)), 16.0),
            16.0
        );
        assert_eq!(
            transparent_sort_distance_sq(Vec3::ZERO, None, f32::NAN),
            0.0
        );
    }

    #[test]
    fn evaluate_draw_candidate_preserves_zero_scale_rigid_world_matrix() {
        let scene = SceneCoordinator::new();
        let mesh_pool = MeshPool::default_pool();
        let store = MaterialPropertyStore::new();
        let material_dict = MaterialDictionary::new(&store);
        let router = MaterialRouter::new(RasterPipelineKind::Null);
        let registry = PropertyIdRegistry::new();
        let property_ids = MaterialPipelinePropertyIds::new(&registry);
        let cache = FrameMaterialBatchCache::new();
        let ctx = DrawCollectionContext {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            material_router: &router,
            pipeline_property_ids: &property_ids,
            shader_perm: ShaderPermutation::default(),
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
            view_origin_world: Vec3::ZERO,
            culling: None,
            transform_filter: None,
            render_space_filter: None,
            material_cache: None,
            reflection_probes: None,
            prepared: None,
        };
        let candidate = DrawCandidate {
            space_id: RenderSpaceId(3),
            node_id: 9,
            renderable_index: 42,
            instance_id: MeshRendererInstanceId(99),
            mesh_asset_id: 7,
            slot_index: 0,
            first_index: 0,
            index_count: 3,
            is_overlay: false,
            sorting_order: 0,
            skinned: false,
            world_space_deformed: false,
            blendshape_deformed: false,
            tangent_blendshape_deform_active: false,
            material_asset_id: 11,
            property_block_id: None,
            world_aabb: None,
        };

        let item = evaluate_draw_candidate(
            &ctx,
            &cache,
            candidate,
            RasterFrontFace::Clockwise,
            RasterPrimitiveTopology::TriangleList,
            Some(Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0))),
            0.0,
        )
        .expect("draw item");
        let matrix = item.rigid_world_matrix.expect("rigid world matrix");

        assert_eq!(matrix.col(0).x, 0.0);
        assert_eq!(matrix.col(1).y, 1.0);
        assert_eq!(matrix.col(2).z, 1.0);
    }
}
