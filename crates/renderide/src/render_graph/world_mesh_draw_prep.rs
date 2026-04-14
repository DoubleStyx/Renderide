//! Flatten scene mesh renderables into sorted draw items for [`super::passes::WorldMeshForwardPass`].
//!
//! Batches are keyed by raster pipeline kind (from host shader → [`crate::materials::resolve_raster_pipeline`]),
//! material asset id, property block slot0, and skinned—aligned with legacy `SpaceDrawBatch` ordering in
//! `crates_old/renderide` so pipeline and future per-material bind groups change only on boundaries.
//!
//! Optional CPU frustum and Hi-Z culling share one bounds evaluation per draw slot
//! ([`super::world_mesh_cull_eval::mesh_draw_passes_cpu_cull`]) using the same view–projection rules as the forward pass
//! ([`super::world_mesh_cull::build_world_mesh_cull_proj_params`]).
//!
//! A future parallel per-space split is described in `docs/parallel_draw_prep_plan.md` (design only).

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashSet;

use glam::{Mat4, Vec3};

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::assets::mesh::GpuMesh;
use crate::materials::{
    embedded_stem_needs_color_stream, embedded_stem_needs_uv0_stream,
    embedded_stem_requires_intersection_pass, embedded_stem_uses_alpha_blending,
    resolve_raster_pipeline, MaterialRouter, RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;
use crate::resources::MeshPool;
use crate::scene::{
    MeshMaterialSlot, RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer,
};
use crate::shared::RenderingContext;

use super::world_mesh_cull_eval::{mesh_draw_passes_cpu_cull, CpuCullFailure};

/// Selective / exclude transform lists for secondary cameras (Unity `CameraRenderer.Render` semantics).
#[derive(Clone, Debug, Default)]
pub struct CameraTransformDrawFilter {
    /// When `Some`, only these transform node ids are drawn.
    pub only: Option<HashSet<i32>>,
    /// When [`Self::only`] is `None`, transforms in this set are skipped.
    pub exclude: HashSet<i32>,
}

impl CameraTransformDrawFilter {
    /// Returns `true` if `node_id` should be rendered under this filter.
    pub fn passes(&self, node_id: i32) -> bool {
        if let Some(only) = &self.only {
            only.contains(&node_id)
        } else {
            !self.exclude.contains(&node_id)
        }
    }
}

/// Builds a filter from a host [`crate::scene::CameraRenderableEntry`].
pub fn draw_filter_from_camera_entry(
    entry: &crate::scene::CameraRenderableEntry,
) -> CameraTransformDrawFilter {
    if !entry.selective_transform_ids.is_empty() {
        CameraTransformDrawFilter {
            only: Some(entry.selective_transform_ids.iter().copied().collect()),
            exclude: HashSet::new(),
        }
    } else {
        CameraTransformDrawFilter {
            only: None,
            exclude: entry.exclude_transform_ids.iter().copied().collect(),
        }
    }
}

/// Groups draws that can share the same raster pipeline and material bind data (Unity material +
/// [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)-style slot0).
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MaterialDrawBatchKey {
    /// Resolved from host `set_shader` → [`resolve_raster_pipeline`].
    pub pipeline: RasterPipelineKind,
    /// Host shader asset id from material `set_shader` (or `-1` when unknown).
    pub shader_asset_id: i32,
    /// Material asset id for this submesh slot (or `-1` when missing).
    pub material_asset_id: i32,
    /// Per-slot property block id when present; `None` is distinct from `Some` for batching.
    pub property_block_slot0: Option<i32>,
    /// Skinned deform path uses different vertex buffers.
    pub skinned: bool,
    /// When [`Self::pipeline`] is [`RasterPipelineKind::EmbeddedStem`], whether the active [`ShaderPermutation`]
    /// requires a UV0 vertex stream (computed once per draw item, not per frame in the raster pass).
    pub embedded_needs_uv0: bool,
    /// When [`Self::pipeline`] is [`RasterPipelineKind::EmbeddedStem`], whether the active [`ShaderPermutation`]
    /// requires a color vertex stream at `@location(3)`.
    pub embedded_needs_color: bool,
    /// When [`Self::pipeline`] is [`RasterPipelineKind::EmbeddedStem`], whether reflection reports `_IntersectColor`
    /// in the material uniform (second forward subpass with depth snapshot).
    pub embedded_requires_intersection_pass: bool,
    /// Transparent alpha-blended UI/text stems should preserve stable canvas order.
    pub alpha_blended: bool,
}

/// Result of [`collect_and_sort_world_mesh_draws`] including optional frustum cull counts.
#[derive(Clone, Debug)]
pub struct WorldMeshDrawCollection {
    /// Draw items after culling and sorting.
    pub items: Vec<WorldMeshDrawItem>,
    /// Draw slots considered for culling (one per material slot × submesh that passed earlier filters).
    pub draws_pre_cull: usize,
    /// Draws removed by frustum culling.
    pub draws_culled: usize,
    /// Draws removed by hierarchical depth occlusion (after frustum), when Hi-Z data was available.
    pub draws_hi_z_culled: usize,
}

/// One indexed draw after pairing a material slot with a mesh submesh range.
#[derive(Clone, Debug)]
pub struct WorldMeshDrawItem {
    /// Host render space.
    pub space_id: RenderSpaceId,
    /// Scene graph node id for this drawable.
    pub node_id: i32,
    /// Resident mesh asset id in [`crate::resources::MeshPool`].
    pub mesh_asset_id: i32,
    /// Index into [`crate::resources::GpuMesh::submeshes`].
    pub slot_index: usize,
    /// First index in the mesh index buffer for this submesh draw.
    pub first_index: u32,
    /// Number of indices for this submesh draw.
    pub index_count: u32,
    /// `true` if [`LayerType::Overlay`](crate::shared::LayerType).
    pub is_overlay: bool,
    /// Host sorting order for transparent draw ordering.
    pub sorting_order: i32,
    /// Whether the mesh uses skinning / deform paths.
    pub skinned: bool,
    /// Stable insertion order before sorting; used for transparent UI/text.
    pub collect_order: usize,
    /// Approximate camera distance used for transparent back-to-front sorting.
    pub camera_distance_sq: f32,
    /// Merge key for host material + property block lookups (e.g. [`MaterialDictionary::get_merged`]).
    pub lookup_ids: MaterialPropertyLookupIds,
    /// Cached batch key for the forward pass.
    pub batch_key: MaterialDrawBatchKey,
    /// Rigid-body world matrix for non-skinned draws, filled during draw collection to avoid
    /// recomputing [`SceneCoordinator::world_matrix_for_render_context`] in the forward pass.
    pub rigid_world_matrix: Option<Mat4>,
}

/// Resolves [`MeshMaterialSlot`] list like legacy `crates_old` `resolved_material_slots`.
///
/// Returns a borrow of [`StaticMeshRenderer::material_slots`] when non-empty; otherwise a single
/// owned slot from the primary material, or an empty slice.
pub fn resolved_material_slots<'a>(
    renderer: &'a StaticMeshRenderer,
) -> Cow<'a, [MeshMaterialSlot]> {
    if !renderer.material_slots.is_empty() {
        Cow::Borrowed(renderer.material_slots.as_slice())
    } else {
        match renderer.primary_material_asset_id {
            Some(material_asset_id) => Cow::Owned(vec![MeshMaterialSlot {
                material_asset_id,
                property_block_id: renderer.primary_property_block_id,
            }]),
            None => Cow::Borrowed(&[]),
        }
    }
}

fn batch_key_for_slot(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    shader_perm: ShaderPermutation,
) -> MaterialDrawBatchKey {
    let shader_asset_id = dict
        .shader_asset_for_material(material_asset_id)
        .unwrap_or(-1);
    let pipeline = resolve_raster_pipeline(shader_asset_id, router);
    let embedded_needs_uv0 = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_uv0_stream(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let embedded_needs_color = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_color_stream(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let embedded_requires_intersection_pass = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_requires_intersection_pass(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let alpha_blended = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => embedded_stem_uses_alpha_blending(stem.as_ref()),
        RasterPipelineKind::DebugWorldNormals => false,
    };
    MaterialDrawBatchKey {
        pipeline,
        shader_asset_id,
        material_asset_id,
        property_block_slot0: property_block_id,
        skinned,
        embedded_needs_uv0,
        embedded_needs_color,
        embedded_requires_intersection_pass,
        alpha_blended,
    }
}

/// Expands one static mesh renderer into draw items (material slots × submeshes).
#[allow(clippy::too_many_arguments)] // Single fan-out site; grouping would obscure the mesh pass.
fn push_draws_for_renderer(
    out: &mut Vec<WorldMeshDrawItem>,
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    renderer: &StaticMeshRenderer,
    renderable_index: usize,
    skinned: bool,
    skinned_renderer: Option<&SkinnedMeshRenderer>,
    mesh: &GpuMesh,
    submeshes: &[(u32, u32)],
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    shader_perm: ShaderPermutation,
    context: RenderingContext,
    head_output_transform: Mat4,
    mismatch_warned: &mut HashSet<i32>,
    culling: Option<&super::world_mesh_cull::WorldMeshCullInput<'_>>,
    cull_stats: &mut (usize, usize, usize),
    transform_filter: Option<&CameraTransformDrawFilter>,
) {
    if let Some(f) = transform_filter {
        if !f.passes(renderer.node_id) {
            return;
        }
    }
    let slots = resolved_material_slots(renderer);
    if slots.is_empty() {
        return;
    }
    let n_sub = submeshes.len();
    let n_slot = slots.len();
    if n_sub != n_slot && mismatch_warned.insert(renderer.mesh_asset_id) {
        logger::trace!(
            "mesh_asset_id={}: material slot count {} != submesh count {} (using first {} pairings only)",
            renderer.mesh_asset_id,
            n_slot,
            n_sub,
            n_sub.min(n_slot),
        );
    }
    let n = n_sub.min(n_slot);
    if n == 0 {
        return;
    }

    let is_overlay = renderer.layer == crate::shared::LayerType::Overlay;

    for slot_index in 0..n {
        let slot = &slots[slot_index];
        let material_asset_id = scene
            .overridden_material_asset_id(space_id, context, skinned, renderable_index, slot_index)
            .unwrap_or(slot.material_asset_id);
        let (first_index, index_count) = submeshes[slot_index];
        if index_count == 0 {
            continue;
        }
        if material_asset_id < 0 {
            continue;
        }
        let rigid_world_matrix = if skinned {
            None
        } else if let Some(c) = culling {
            cull_stats.0 += 1;
            match mesh_draw_passes_cpu_cull(
                scene,
                space_id,
                mesh,
                is_overlay,
                skinned,
                skinned_renderer,
                renderer.node_id,
                c,
                context,
            ) {
                Err(CpuCullFailure::Frustum) => {
                    cull_stats.1 += 1;
                    continue;
                }
                Err(CpuCullFailure::HiZ) => {
                    cull_stats.2 += 1;
                    continue;
                }
                Ok(m) => m,
            }
        } else {
            scene.world_matrix_for_render_context(
                space_id,
                renderer.node_id as usize,
                context,
                head_output_transform,
            )
        };
        let lookup_ids = MaterialPropertyLookupIds {
            material_asset_id,
            mesh_property_block_slot0: slot.property_block_id,
        };
        let batch_key = batch_key_for_slot(
            material_asset_id,
            slot.property_block_id,
            skinned,
            dict,
            router,
            shader_perm,
        );
        out.push(WorldMeshDrawItem {
            space_id,
            node_id: renderer.node_id,
            mesh_asset_id: renderer.mesh_asset_id,
            slot_index,
            first_index,
            index_count,
            is_overlay,
            sorting_order: renderer.sorting_order,
            skinned,
            collect_order: out.len(),
            camera_distance_sq: 0.0,
            lookup_ids,
            batch_key,
            rigid_world_matrix,
        });
    }
}

/// Sorts opaque draws for batching and alpha UI/text draws in stable canvas order.
pub fn sort_world_mesh_draws(items: &mut [WorldMeshDrawItem]) {
    items.sort_unstable_by(|a, b| {
        a.is_overlay
            .cmp(&b.is_overlay)
            .then(a.batch_key.alpha_blended.cmp(&b.batch_key.alpha_blended))
            .then_with(
                || match (a.batch_key.alpha_blended, b.batch_key.alpha_blended) {
                    (false, false) => a
                        .batch_key
                        .cmp(&b.batch_key)
                        .then(b.sorting_order.cmp(&a.sorting_order))
                        .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
                        .then(a.node_id.cmp(&b.node_id))
                        .then(a.slot_index.cmp(&b.slot_index)),
                    (true, true) => a
                        .sorting_order
                        .cmp(&b.sorting_order)
                        .then_with(|| b.camera_distance_sq.total_cmp(&a.camera_distance_sq))
                        .then(a.collect_order.cmp(&b.collect_order)),
                    _ => Ordering::Equal,
                },
            )
    });
}

/// Updates alpha-blended draw distance keys from the active camera, then re-sorts the full draw list.
///
/// Reserved for frame-graph paths that move the camera without rebuilding the full draw collection.
#[allow(dead_code)]
pub fn resort_world_mesh_draws_for_camera(
    items: &mut [WorldMeshDrawItem],
    scene: &SceneCoordinator,
    render_context: RenderingContext,
    head_output_transform: glam::Mat4,
    camera_world: Vec3,
) {
    for item in items.iter_mut() {
        item.camera_distance_sq = if item.batch_key.alpha_blended {
            scene
                .world_matrix_for_render_context(
                    item.space_id,
                    item.node_id as usize,
                    render_context,
                    head_output_transform,
                )
                .map(|m| m.col(3).truncate().distance_squared(camera_world))
                .unwrap_or(0.0)
        } else {
            0.0
        };
    }
    sort_world_mesh_draws(items);
}

/// Collects draws from active spaces, then sorts for batching (material / pipeline boundaries).
///
/// When `culling` is [`Some`], instances outside the frustum (and optional Hi-Z) are dropped (see
/// [`mesh_draw_passes_cpu_cull`](super::world_mesh_cull_eval::mesh_draw_passes_cpu_cull)).
#[allow(clippy::too_many_arguments)] // Frame-graph entry mirrors host camera + cull snapshot inputs.
pub fn collect_and_sort_world_mesh_draws(
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    shader_perm: ShaderPermutation,
    context: RenderingContext,
    head_output_transform: Mat4,
    culling: Option<&super::world_mesh_cull::WorldMeshCullInput<'_>>,
    transform_filter: Option<&CameraTransformDrawFilter>,
) -> WorldMeshDrawCollection {
    let mut mismatch_warned = HashSet::new();
    let mut out = Vec::new();
    let mut cull_stats = (0usize, 0usize, 0usize);

    let mut cap_hint = 0usize;
    for space_id in scene.render_space_ids() {
        let Some(space) = scene.space(space_id) else {
            continue;
        };
        if space.is_active {
            cap_hint = cap_hint
                .saturating_add(space.static_mesh_renderers.len())
                .saturating_add(space.skinned_mesh_renderers.len());
        }
    }
    out.reserve(cap_hint.saturating_mul(8));

    for space_id in scene.render_space_ids() {
        let Some(space) = scene.space(space_id) else {
            continue;
        };
        if !space.is_active {
            continue;
        }

        for (renderable_index, r) in space.static_mesh_renderers.iter().enumerate() {
            if r.mesh_asset_id < 0 || r.node_id < 0 {
                continue;
            }
            let Some(mesh) = mesh_pool.get_mesh(r.mesh_asset_id) else {
                continue;
            };
            if mesh.submeshes.is_empty() {
                continue;
            }
            push_draws_for_renderer(
                &mut out,
                scene,
                space_id,
                r,
                renderable_index,
                false,
                None,
                mesh,
                &mesh.submeshes,
                dict,
                router,
                shader_perm,
                context,
                head_output_transform,
                &mut mismatch_warned,
                culling,
                &mut cull_stats,
                transform_filter,
            );
        }
        for (renderable_index, skinned) in space.skinned_mesh_renderers.iter().enumerate() {
            let r = &skinned.base;
            if r.mesh_asset_id < 0 || r.node_id < 0 {
                continue;
            }
            let Some(mesh) = mesh_pool.get_mesh(r.mesh_asset_id) else {
                continue;
            };
            if mesh.submeshes.is_empty() {
                continue;
            }
            push_draws_for_renderer(
                &mut out,
                scene,
                space_id,
                r,
                renderable_index,
                true,
                Some(skinned),
                mesh,
                &mesh.submeshes,
                dict,
                router,
                shader_perm,
                context,
                head_output_transform,
                &mut mismatch_warned,
                culling,
                &mut cull_stats,
                transform_filter,
            );
        }
    }

    sort_world_mesh_draws(&mut out);
    WorldMeshDrawCollection {
        items: out,
        draws_pre_cull: cull_stats.0,
        draws_culled: cull_stats.1,
        draws_hi_z_culled: cull_stats.2,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        resolved_material_slots, sort_world_mesh_draws, MaterialDrawBatchKey, WorldMeshDrawItem,
    };
    use crate::assets::material::MaterialPropertyLookupIds;
    use crate::materials::RasterPipelineKind;
    use crate::scene::{MeshMaterialSlot, RenderSpaceId, StaticMeshRenderer};

    #[test]
    fn resolved_material_slots_prefers_explicit_vec() {
        let r = StaticMeshRenderer {
            material_slots: vec![
                MeshMaterialSlot {
                    material_asset_id: 1,
                    property_block_id: Some(10),
                },
                MeshMaterialSlot {
                    material_asset_id: 2,
                    property_block_id: None,
                },
            ],
            primary_material_asset_id: Some(99),
            ..Default::default()
        };
        let slots = resolved_material_slots(&r);
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].material_asset_id, 1);
    }

    #[test]
    fn resolved_material_slots_falls_back_to_primary() {
        let r = StaticMeshRenderer {
            primary_material_asset_id: Some(7),
            primary_property_block_id: Some(42),
            ..Default::default()
        };
        let slots = resolved_material_slots(&r);
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].material_asset_id, 7);
        assert_eq!(slots[0].property_block_id, Some(42));
    }

    #[allow(clippy::too_many_arguments)]
    fn dummy_item(
        mid: i32,
        pb: Option<i32>,
        skinned: bool,
        sort: i32,
        mesh: i32,
        node: i32,
        slot: usize,
        collect_order: usize,
        alpha_blended: bool,
    ) -> WorldMeshDrawItem {
        WorldMeshDrawItem {
            space_id: RenderSpaceId(0),
            node_id: node,
            mesh_asset_id: mesh,
            slot_index: slot,
            first_index: 0,
            index_count: 3,
            is_overlay: false,
            sorting_order: sort,
            skinned,
            collect_order,
            camera_distance_sq: 0.0,
            lookup_ids: MaterialPropertyLookupIds {
                material_asset_id: mid,
                mesh_property_block_slot0: pb,
            },
            batch_key: MaterialDrawBatchKey {
                pipeline: RasterPipelineKind::DebugWorldNormals,
                shader_asset_id: -1,
                material_asset_id: mid,
                property_block_slot0: pb,
                skinned,
                embedded_needs_uv0: false,
                embedded_needs_color: false,
                embedded_requires_intersection_pass: false,
                alpha_blended,
            },
            rigid_world_matrix: None,
        }
    }

    #[test]
    fn sort_orders_by_material_then_higher_sorting_order() {
        let mut v = vec![
            dummy_item(2, None, false, 0, 1, 0, 0, 0, false),
            dummy_item(1, None, false, 0, 1, 0, 0, 1, false),
            dummy_item(1, None, false, 5, 2, 0, 0, 2, false),
            dummy_item(1, None, false, 10, 1, 0, 1, 3, false),
        ];
        sort_world_mesh_draws(&mut v);
        assert_eq!(v[0].lookup_ids.material_asset_id, 1);
        assert_eq!(v[0].sorting_order, 10);
        assert_eq!(v[1].sorting_order, 5);
        assert_eq!(v[2].sorting_order, 0);
        assert_eq!(v[3].lookup_ids.material_asset_id, 2);
    }

    #[test]
    fn property_block_splits_batch_keys() {
        let a = MaterialDrawBatchKey {
            pipeline: RasterPipelineKind::DebugWorldNormals,
            shader_asset_id: -1,
            material_asset_id: 1,
            property_block_slot0: None,
            skinned: false,
            embedded_needs_uv0: false,
            embedded_needs_color: false,
            embedded_requires_intersection_pass: false,
            alpha_blended: false,
        };
        let b = MaterialDrawBatchKey {
            pipeline: RasterPipelineKind::DebugWorldNormals,
            shader_asset_id: -1,
            material_asset_id: 1,
            property_block_slot0: Some(99),
            skinned: false,
            embedded_needs_uv0: false,
            embedded_needs_color: false,
            embedded_requires_intersection_pass: false,
            alpha_blended: false,
        };
        assert_ne!(a, b);
        assert!(a < b || b < a);
    }

    #[test]
    fn transparent_ui_preserves_collection_order_within_sorting_order() {
        let mut v = vec![
            dummy_item(10, None, false, 0, 1, 0, 0, 2, true),
            dummy_item(11, None, false, 0, 1, 0, 1, 0, true),
            dummy_item(12, None, false, 1, 1, 0, 2, 1, true),
        ];
        sort_world_mesh_draws(&mut v);
        assert_eq!(v[0].collect_order, 0);
        assert_eq!(v[1].collect_order, 2);
        assert_eq!(v[2].collect_order, 1);
    }

    #[test]
    fn transparent_ui_sorts_farther_items_first() {
        let mut far = dummy_item(10, None, false, 0, 1, 0, 0, 0, true);
        far.camera_distance_sq = 9.0;
        let mut near = dummy_item(11, None, false, 0, 1, 0, 1, 1, true);
        near.camera_distance_sq = 1.0;
        let mut v = vec![near, far];
        sort_world_mesh_draws(&mut v);
        assert_eq!(v[0].camera_distance_sq, 9.0);
        assert_eq!(v[1].camera_distance_sq, 1.0);
    }
}
