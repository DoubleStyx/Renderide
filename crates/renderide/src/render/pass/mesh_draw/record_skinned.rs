//! GPU recording of skinned mesh draws.

use super::pbr_bind::{
    get_or_create_pbr_scene_bind_group, pipeline_uses_standalone_mrt_gbuffer_origin_bind_group,
};
use super::pipeline::resolve_pipeline_for_group;
use super::types::{MeshDrawParams, SkinnedBatchedDraw};
use crate::gpu::PipelineKey;

/// Records skinned mesh draws into the render pass.
pub fn record_skinned_draws(
    pass: &mut wgpu::RenderPass,
    params: &mut MeshDrawParams,
    draws: &[SkinnedBatchedDraw],
    debug_blendshapes: bool,
) {
    if draws.is_empty() {
        return;
    }
    let mut i = 0;
    while i < draws.len() {
        let variant = draws[i].pipeline_variant;
        let group_end = draws[i..]
            .iter()
            .take_while(|d| d.pipeline_variant == variant)
            .count();
        let group = &draws[i..i + group_end];

        let pipeline_variant =
            resolve_pipeline_for_group(&variant, params, group.iter().any(|d| d.is_overlay));
        let Some(skinned) = params.pipeline_manager.get_pipeline(
            PipelineKey(None, pipeline_variant),
            params.device,
            params.config,
            Some(params.material_property_store),
            params.render_config,
            Some(params.asset_registry),
        ) else {
            i += group_end;
            continue;
        };
        let items: Vec<_> = group
            .iter()
            .map(|d| {
                (
                    d.mvp,
                    d.bone_matrices.as_slice(),
                    d.blendshape_weights.as_deref(),
                    d.num_vertices,
                )
            })
            .collect();
        if debug_blendshapes {
            let count = group.len();
            let first_with_weights = group
                .iter()
                .find(|d| d.blendshape_weights.as_ref().is_some_and(|w| !w.is_empty()));
            if let Some(d) = first_with_weights {
                if let Some(w) = d.blendshape_weights.as_ref() {
                    let preview: Vec<_> = w.iter().take(8).copied().collect();
                    logger::trace!(
                        "blendshape batch_count={} first_draw_weights_len={} preview={:?}",
                        count,
                        w.len(),
                        preview
                    );
                }
            } else {
                logger::trace!("blendshape batch_count={} first_draw_weights_len=0", count);
            }
        }
        skinned.upload_skinned_batch(params.queue, &items, params.frame_index);
        let is_stencil_pipeline = matches!(
            pipeline_variant,
            crate::gpu::PipelineVariant::OverlayStencilSkinned
                | crate::gpu::PipelineVariant::OverlayStencilMaskWriteSkinned
                | crate::gpu::PipelineVariant::OverlayStencilMaskClearSkinned
        );
        skinned.bind_pipeline(pass);
        if params.use_mrt
            && pipeline_uses_standalone_mrt_gbuffer_origin_bind_group(&pipeline_variant)
            && let Some(bg) = params.mrt_gbuffer_origin_bind_group
        {
            pass.set_bind_group(1, bg, &[]);
        }
        // Nested if required: pbr must be destructured before passing params mutably to avoid borrow conflict.
        #[allow(clippy::collapsible_if)]
        if matches!(
            pipeline_variant,
            crate::gpu::PipelineVariant::SkinnedPbr
                | crate::gpu::PipelineVariant::SkinnedPbrMRT
                | crate::gpu::PipelineVariant::SkinnedPbrRayQuery
                | crate::gpu::PipelineVariant::SkinnedPbrMRTRayQuery
        ) && let Some(ref pbr) = params.pbr_scene
        {
            if let Some(scene_bg) = get_or_create_pbr_scene_bind_group(
                params,
                skinned.as_ref(),
                pipeline_variant,
                pbr.view_position,
                pbr.view_space_z_coeffs,
                pbr.cluster_count_x,
                pbr.cluster_count_y,
                pbr.cluster_count_z,
                pbr.near_clip,
                pbr.far_clip,
                pbr.light_count,
                pbr.viewport_width,
                pbr.viewport_height,
                pbr.light_buffer,
                pbr.cluster_light_counts,
                pbr.cluster_light_indices,
                params.pbr_tlas_ptr,
            ) {
                skinned.bind_scene(pass, Some(scene_bg));
            }
        }
        let mut order: Vec<usize> = (0..group.len()).collect();
        order.sort_by_key(|&idx| group[idx].mesh_asset_id);
        let mut last_mesh_asset_id: Option<i32> = None;
        for j in order {
            let d = &group[j];
            let Some(buffers) = params.mesh_buffer_cache.get(&d.mesh_asset_id) else {
                continue;
            };
            let draw_bind_group = params
                .skinned_bind_group_cache
                .entry((pipeline_variant, d.mesh_asset_id))
                .or_insert_with(|| {
                    skinned
                        .create_skinned_draw_bind_group(params.device, buffers)
                        .expect("skinned pipeline must create draw bind groups")
                });
            skinned.bind_draw(
                pass,
                Some(j as u32),
                params.frame_index,
                Some(draw_bind_group),
            );
            if let Some(ref stencil) = d.stencil_state {
                pass.set_stencil_reference(stencil.reference as u32);
            } else if is_stencil_pipeline {
                debug_assert!(
                    d.stencil_state.is_some(),
                    "OverlayStencilSkinned draws must have stencil_state"
                );
            }
            if last_mesh_asset_id != Some(d.mesh_asset_id) {
                skinned.set_skinned_buffers(pass, buffers);
                last_mesh_asset_id = Some(d.mesh_asset_id);
            }
            skinned.draw_skinned_indexed(pass, buffers, d.submesh_index_range);
        }
        i += group_end;
    }
}
