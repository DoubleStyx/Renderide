//! Batch keys and draw list ordering for world mesh forward.

use std::cmp::Ordering;

use rayon::slice::ParallelSliceMut;

use crate::assets::material::MaterialDictionary;
use crate::materials::{
    embedded_stem_needs_color_stream, embedded_stem_needs_extended_vertex_streams,
    embedded_stem_needs_uv0_stream, embedded_stem_requires_grab_pass,
    embedded_stem_requires_intersection_pass, embedded_stem_uses_alpha_blending,
    material_blend_mode_for_lookup, material_render_state_for_lookup, resolve_raster_pipeline,
    MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;

use super::material_batch_cache::{
    FrameMaterialBatchCache, MaterialResolveCtx, ResolvedMaterialBatch,
};
use super::types::{MaterialDrawBatchKey, WorldMeshDrawItem};

/// Builds a [`MaterialDrawBatchKey`] for one material slot from dictionary + router state.
///
/// This is the full per-draw computation path. Used for cache warm-up and as a fallback for
/// materials not present in [`FrameMaterialBatchCache`] (e.g. render-context override materials).
pub(super) fn batch_key_for_slot(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
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
        RasterPipelineKind::Null => false,
    };
    let embedded_needs_color = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_color_stream(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_needs_extended_vertex_streams = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_extended_vertex_streams(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_requires_intersection_pass = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_requires_intersection_pass(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_requires_grab_pass = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_requires_grab_pass(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let lookup_ids = crate::assets::material::MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: property_block_id,
    };
    let material_blend_mode =
        material_blend_mode_for_lookup(dict, lookup_ids, pipeline_property_ids);
    let render_state = material_render_state_for_lookup(dict, lookup_ids, pipeline_property_ids);
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
    r: &ResolvedMaterialBatch,
) -> MaterialDrawBatchKey {
    MaterialDrawBatchKey {
        pipeline: r.pipeline.clone(),
        shader_asset_id: r.shader_asset_id,
        material_asset_id,
        property_block_slot0: property_block_id,
        skinned,
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
    cache: &FrameMaterialBatchCache,
    ctx: MaterialResolveCtx<'_>,
) -> MaterialDrawBatchKey {
    if let Some(resolved) = cache.get(material_asset_id, property_block_id) {
        batch_key_from_resolved(material_asset_id, property_block_id, skinned, resolved)
    } else {
        batch_key_for_slot(
            material_asset_id,
            property_block_id,
            skinned,
            ctx.dict,
            ctx.router,
            ctx.pipeline_property_ids,
            ctx.shader_perm,
        )
    }
}

/// Ordering for world mesh draws (opaque batching vs alpha distance sort).
///
/// Shared by [`sort_world_mesh_draws`] (parallel) and [`sort_world_mesh_draws_serial`].
#[inline]
pub(super) fn cmp_world_mesh_draw_items(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
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
    items.par_sort_unstable_by(cmp_world_mesh_draw_items);
}

/// Same ordering as [`sort_world_mesh_draws`] without rayon (for nested parallel batches).
pub(super) fn sort_world_mesh_draws_serial(items: &mut [WorldMeshDrawItem]) {
    items.sort_unstable_by(cmp_world_mesh_draw_items);
}
