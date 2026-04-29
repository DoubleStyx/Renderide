//! Batch keys and draw list ordering for world mesh forward.

use std::cmp::Ordering;

use rayon::slice::ParallelSliceMut;

use crate::materials::{
    embedded_stem_needs_color_stream, embedded_stem_needs_extended_vertex_streams,
    embedded_stem_needs_uv0_stream, embedded_stem_requires_grab_pass,
    embedded_stem_requires_intersection_pass, embedded_stem_uses_alpha_blending,
    material_blend_mode_for_lookup, material_render_state_for_lookup, resolve_raster_pipeline,
    RasterFrontFace, RasterPipelineKind,
};

use super::material_batch_cache::{
    FrameMaterialBatchCache, MaterialResolveCtx, ResolvedMaterialBatch,
};
use super::types::{MaterialDrawBatchKey, WorldMeshDrawItem};

/// Compact ordering prefix for the hot draw-sort comparator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DrawSortKey {
    /// Overlay draws sort after main-world draws.
    pub overlay: bool,
    /// Alpha-blended draws sort after opaque draws and then use transparent ordering.
    pub alpha_blended: bool,
    /// Coarse front-to-back bucket for opaque draws.
    pub opaque_depth_bucket: u16,
}

impl DrawSortKey {
    /// Builds the sort-key prefix for a draw item.
    fn from_draw(item: &WorldMeshDrawItem) -> Self {
        Self {
            overlay: item.is_overlay,
            alpha_blended: item.batch_key.alpha_blended,
            opaque_depth_bucket: opaque_depth_bucket(item.camera_distance_sq),
        }
    }
}

/// Maps camera-distance squared into a coarse logarithmic front-to-back bucket.
fn opaque_depth_bucket(distance_sq: f32) -> u16 {
    if !distance_sq.is_finite() || distance_sq <= 0.0 {
        return 0;
    }
    let distance = distance_sq.sqrt().max(1e-4);
    ((distance.log2() + 16.0).floor().clamp(0.0, 255.0)) as u16
}

/// Builds a [`MaterialDrawBatchKey`] for one material slot from dictionary + router state.
///
/// This is the full per-draw computation path. Used for cache warm-up and as a fallback for
/// materials not present in [`FrameMaterialBatchCache`] (e.g. render-context override materials).
pub(super) fn batch_key_for_slot(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    front_face: RasterFrontFace,
    ctx: MaterialResolveCtx<'_>,
) -> MaterialDrawBatchKey {
    let shader_asset_id = ctx
        .dict
        .shader_asset_for_material(material_asset_id)
        .unwrap_or(-1);
    let pipeline = resolve_raster_pipeline(shader_asset_id, ctx.router);
    let embedded_needs_uv0 = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_uv0_stream(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_needs_color = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_color_stream(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_needs_extended_vertex_streams = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_extended_vertex_streams(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_requires_intersection_pass = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_requires_intersection_pass(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_requires_grab_pass = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_requires_grab_pass(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let lookup_ids = crate::assets::material::MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: property_block_id,
    };
    let material_blend_mode =
        material_blend_mode_for_lookup(ctx.dict, lookup_ids, ctx.pipeline_property_ids);
    let render_state =
        material_render_state_for_lookup(ctx.dict, lookup_ids, ctx.pipeline_property_ids);
    let alpha_blended = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => embedded_stem_uses_alpha_blending(stem.as_ref()),
        RasterPipelineKind::Null => false,
    } || material_blend_mode.is_transparent()
        || embedded_requires_grab_pass;
    MaterialDrawBatchKey {
        pipeline,
        shader_asset_id,
        material_asset_id,
        property_block_slot0: property_block_id,
        skinned,
        front_face,
        embedded_needs_uv0,
        embedded_needs_color,
        embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass,
        embedded_requires_grab_pass,
        render_state,
        blend_mode: material_blend_mode,
        alpha_blended,
    }
}

/// Assembles a [`MaterialDrawBatchKey`] from a pre-resolved [`ResolvedMaterialBatch`] entry.
#[inline]
fn batch_key_from_resolved(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    front_face: RasterFrontFace,
    r: &ResolvedMaterialBatch,
) -> MaterialDrawBatchKey {
    MaterialDrawBatchKey {
        pipeline: r.pipeline.clone(),
        shader_asset_id: r.shader_asset_id,
        material_asset_id,
        property_block_slot0: property_block_id,
        skinned,
        front_face,
        embedded_needs_uv0: r.embedded_needs_uv0,
        embedded_needs_color: r.embedded_needs_color,
        embedded_needs_extended_vertex_streams: r.embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass: r.embedded_requires_intersection_pass,
        embedded_requires_grab_pass: r.embedded_requires_grab_pass,
        render_state: r.render_state,
        blend_mode: r.blend_mode,
        alpha_blended: r.alpha_blended,
    }
}

/// Builds a [`MaterialDrawBatchKey`] using a pre-built [`FrameMaterialBatchCache`].
///
/// Falls back to the full dictionary / router lookup path when the material is not cached (e.g.
/// render-context override materials not encountered during the eager pre-build pass).
pub(super) fn batch_key_for_slot_cached(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    front_face: RasterFrontFace,
    cache: &FrameMaterialBatchCache,
    ctx: MaterialResolveCtx<'_>,
) -> MaterialDrawBatchKey {
    if let Some(resolved) = cache.get(material_asset_id, property_block_id) {
        batch_key_from_resolved(
            material_asset_id,
            property_block_id,
            skinned,
            front_face,
            resolved,
        )
    } else {
        batch_key_for_slot(
            material_asset_id,
            property_block_id,
            skinned,
            front_face,
            ctx,
        )
    }
}

/// Ordering for world mesh draws (opaque batching vs alpha distance sort).
///
/// Shared by [`sort_world_mesh_draws`] (parallel) and [`sort_world_mesh_draws_serial`].
#[inline]
pub(super) fn cmp_world_mesh_draw_items(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    let a_sort = DrawSortKey::from_draw(a);
    let b_sort = DrawSortKey::from_draw(b);
    a_sort
        .overlay
        .cmp(&b_sort.overlay)
        .then(a_sort.alpha_blended.cmp(&b_sort.alpha_blended))
        .then_with(
            || match (a.batch_key.alpha_blended, b.batch_key.alpha_blended) {
                (false, false) => a_sort
                    .opaque_depth_bucket
                    .cmp(&b_sort.opaque_depth_bucket)
                    .then_with(|| a.batch_key.cmp(&b.batch_key))
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
}

/// Pre-depth-bucket ordering retained for regression tests that need to isolate batch-key order.
#[cfg(test)]
fn cmp_world_mesh_draw_items_without_depth_bucket(
    a: &WorldMeshDrawItem,
    b: &WorldMeshDrawItem,
) -> Ordering {
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
}

/// Sorts opaque draws for batching and alpha UI/text draws in stable canvas order.
pub fn sort_world_mesh_draws(items: &mut [WorldMeshDrawItem]) {
    profiling::scope!("mesh::sort_world_mesh_draws");
    items.par_sort_unstable_by(cmp_world_mesh_draw_items);
}

/// Same ordering as [`sort_world_mesh_draws`] without rayon (for nested parallel batches).
pub(super) fn sort_world_mesh_draws_serial(items: &mut [WorldMeshDrawItem]) {
    profiling::scope!("mesh::sort_world_mesh_draws_serial");
    items.sort_unstable_by(cmp_world_mesh_draw_items);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::test_fixtures::{dummy_world_mesh_draw_item, DummyDrawItemSpec};

    #[test]
    fn opaque_sort_prefers_nearer_depth_bucket_before_batch_key() {
        let mut near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 2,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 2,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        near.camera_distance_sq = 1.0;
        let mut far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: false,
        });
        far.camera_distance_sq = 4096.0;

        assert_eq!(
            cmp_world_mesh_draw_items(&near, &far),
            Ordering::Less,
            "near opaque draws should sort before lower material ids when depth buckets differ"
        );
        assert_eq!(
            cmp_world_mesh_draw_items_without_depth_bucket(&near, &far),
            Ordering::Greater,
            "the regression setup must differ from pure batch-key ordering"
        );
    }

    #[test]
    fn transparent_sort_remains_back_to_front() {
        let mut near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        near.camera_distance_sq = 1.0;
        let mut far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 2,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: true,
        });
        far.camera_distance_sq = 4096.0;

        assert_eq!(cmp_world_mesh_draw_items(&far, &near), Ordering::Less);
    }
}
