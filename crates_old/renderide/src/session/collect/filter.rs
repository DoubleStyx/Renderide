//! Filters scene drawables by layer and render lists, resolves world matrices, and emits [`super::pipeline::FilteredDrawable`] rows.

use std::collections::HashSet;

use crate::assets::AssetRegistry;
use crate::assets::mesh::{
    cpu_submesh_count_for_material_pairing, cpu_submesh_index_range_for_pairing,
};
use crate::config::RenderConfig;
use crate::scene::{Scene, SceneGraph, render_transform_to_matrix};
use crate::shared::LayerType;

use super::pipeline::{
    FilteredDrawable, compute_pipeline_variant_for_drawable, resolve_overlay_stencil_state,
    resolve_pipeline_for_material_draw, resolved_material_slots,
};

/// Filters drawables by layer, render lists, and skinned validity; collects world matrices.
///
/// Skips Hidden layer, applies only/exclude lists, validates bone_transform_ids and bind_poses
/// for skinned draws. Returns [`FilteredDrawable`] for each valid draw (including [`ShaderKey`](crate::gpu::ShaderKey)
/// and resolved [`PipelineVariant`](crate::gpu::PipelineVariant)).
#[allow(clippy::too_many_arguments)]
pub(in crate::session) fn filter_and_collect_drawables(
    scene: &Scene,
    only_render_list: &[i32],
    exclude_render_list: &[i32],
    scene_graph: &SceneGraph,
    space_id: i32,
    asset_registry: &AssetRegistry,
    render_config: &RenderConfig,
    use_debug_uv: bool,
    use_pbr: bool,
) -> Vec<FilteredDrawable> {
    let only_set: HashSet<i32> = only_render_list.iter().copied().collect();
    let exclude_set: HashSet<i32> = exclude_render_list.iter().copied().collect();
    let use_only = !only_set.is_empty();
    let use_exclude = !exclude_set.is_empty();

    let mut out = Vec::new();
    let combined = scene
        .drawables
        .iter()
        .map(|d| (d, false))
        .chain(scene.skinned_drawables.iter().map(|d| (d, true)));

    for (entry, is_skinned) in combined {
        if entry.node_id < 0 {
            continue;
        }
        if entry.layer == LayerType::hidden {
            continue;
        }
        if use_only && !only_set.contains(&entry.node_id) {
            continue;
        }
        if use_exclude && exclude_set.contains(&entry.node_id) {
            continue;
        }
        if is_skinned {
            if entry
                .bone_transform_ids
                .as_ref()
                .is_none_or(|b| b.is_empty())
            {
                logger::trace!(
                    "Skinned draw skipped: bone_transform_ids missing or empty (node_id={})",
                    entry.node_id
                );
                continue;
            }
            if let Some(mesh) = asset_registry.get_mesh(entry.mesh_handle)
                && mesh.bind_poses.as_ref().is_none_or(|b| b.is_empty())
            {
                logger::trace!(
                    "Skinned draw skipped: mesh missing bind_poses (mesh={}, node_id={})",
                    entry.mesh_handle,
                    entry.node_id
                );
                continue;
            }
        }
        let idx = entry.node_id as usize;
        let world_matrix = match scene_graph.get_world_matrix(space_id, idx) {
            Some(m) => m,
            None => {
                if idx >= scene.nodes.len() {
                    continue;
                }
                render_transform_to_matrix(&scene.nodes[idx])
            }
        };

        let stencil_state = resolve_overlay_stencil_state(scene.is_overlay, entry, asset_registry);
        let mut drawable = entry.clone();
        drawable.stencil_state = stencil_state;

        let fallback_variant = compute_pipeline_variant_for_drawable(
            scene.is_overlay,
            is_skinned,
            &drawable,
            entry.mesh_handle,
            use_debug_uv,
            use_pbr,
            asset_registry,
        );

        let slots = resolved_material_slots(&drawable);
        let mesh = asset_registry.get_mesh(drawable.mesh_handle);
        let submesh_count = mesh
            .map(cpu_submesh_count_for_material_pairing)
            .unwrap_or(1)
            .max(1);

        let use_split =
            render_config.multi_material_submeshes && submesh_count > 1 && slots.len() > 1;

        if render_config.multi_material_submeshes
            && render_config.log_multi_material_submesh_mismatch
            && mesh.is_some()
            && !slots.is_empty()
            && slots.len() != submesh_count
        {
            logger::trace!(
                "multi_material: material_slots_len={} submesh_count={} mesh_asset_id={} node_id={}",
                slots.len(),
                submesh_count,
                drawable.mesh_handle,
                drawable.node_id
            );
        }

        if !use_split {
            let material_block_id = drawable.material_handle.unwrap_or(-1);
            let (pipeline_variant, shader_key) = resolve_pipeline_for_material_draw(
                scene,
                render_config,
                &drawable,
                use_pbr,
                is_skinned,
                asset_registry,
                material_block_id,
                fallback_variant,
            );
            out.push(FilteredDrawable {
                drawable,
                world_matrix,
                pipeline_variant,
                shader_key,
                submesh_index_range: None,
            });
            continue;
        }

        for i in 0..submesh_count {
            let Some(slot) = slots.get(i).or_else(|| slots.last()) else {
                break;
            };
            let Some(range) = mesh.and_then(|m| cpu_submesh_index_range_for_pairing(m, i)) else {
                continue;
            };
            let mut d_slot = drawable.clone();
            d_slot.material_handle = Some(slot.material_asset_id);
            d_slot.mesh_renderer_property_block_slot0_id = slot.property_block_id;
            d_slot.material_slots = vec![*slot];
            let material_block_id = slot.material_asset_id;
            let (pipeline_variant, shader_key) = resolve_pipeline_for_material_draw(
                scene,
                render_config,
                &d_slot,
                use_pbr,
                is_skinned,
                asset_registry,
                material_block_id,
                fallback_variant,
            );
            out.push(FilteredDrawable {
                drawable: d_slot,
                world_matrix,
                pipeline_variant,
                shader_key,
                submesh_index_range: Some(range),
            });
        }
    }

    out
}
