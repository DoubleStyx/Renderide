//! Batch keys and draw list ordering for world mesh forward.

use std::cmp::Ordering;

use glam::Vec3;
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;

use crate::assets::material::MaterialDictionary;
use crate::materials::{
    embedded_stem_needs_color_stream, embedded_stem_needs_uv0_stream,
    embedded_stem_requires_intersection_pass, embedded_stem_uses_alpha_blending,
    resolve_raster_pipeline, MaterialRouter, RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;

use super::types::{MaterialDrawBatchKey, WorldMeshDrawItem};

/// Builds a [`MaterialDrawBatchKey`] for one material slot from dictionary + router state.
pub(super) fn batch_key_for_slot(
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

/// Ordering for world mesh draws (opaque batching vs alpha distance sort).
///
/// Shared by [`sort_world_mesh_draws`] (parallel) and [`sort_world_mesh_draws_serial`].
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
    items.par_iter_mut().for_each(|item| {
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
    });
    sort_world_mesh_draws(items);
}
