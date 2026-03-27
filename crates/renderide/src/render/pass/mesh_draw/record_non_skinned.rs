//! GPU recording of non-skinned mesh draws.

use super::pbr_bind::{
    fill_pbr_host_uniform_extras, get_or_create_pbr_scene_bind_group,
    pipeline_uses_standalone_mrt_gbuffer_origin_bind_group,
};
use super::pipeline::resolve_pipeline_for_group;
use super::types::{BatchedDraw, MeshDrawParams};
use crate::assets::{MaterialPropertyStore, MaterialPropertyValue, texture2d_asset_id_from_packed};
use crate::config::RenderConfig;
use crate::gpu::pipeline::{
    PbrHostAlbedoPipeline, UiTextUnlitNativePipeline, UiUnlitNativePipeline, WorldUnlitPipeline,
};
use crate::gpu::state::ensure_texture2d_gpu_view;
use crate::gpu::{
    NativeUiMaterialBindCache, NonSkinnedUniformUpload, PipelineKey, PipelineVariant,
    RenderPipeline,
};
use crate::render::pass::material_draw_context::MaterialDrawContext;
use std::collections::HashMap;

/// Resolves per-draw bind group 0 for [`PipelineVariant::PbrHostAlbedo`] (host `_MainTex`).
///
/// Takes disjoint store/registry/cache handles instead of [`MeshDrawParams`] so the borrow
/// checker can overlap with other `params` uses in [`record_non_skinned_draws`].
#[allow(clippy::too_many_arguments)]
fn pbr_host_albedo_draw_bind<'a>(
    variant: PipelineVariant,
    pipeline: &'a dyn RenderPipeline,
    d: &BatchedDraw,
    material_property_store: &'a MaterialPropertyStore,
    render_config: &'a RenderConfig,
    asset_registry: &'a crate::assets::AssetRegistry,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    texture2d_gpu: &'a mut HashMap<i32, (wgpu::Texture, wgpu::TextureView)>,
    texture2d_last_uploaded_version: &'a mut HashMap<i32, u64>,
    native_ui_material_bind_cache: &'a mut NativeUiMaterialBindCache,
    pbr_host_albedo_bind_cache: &'a mut HashMap<i32, wgpu::BindGroup>,
) -> Option<&'a wgpu::BindGroup> {
    if !matches!(variant, PipelineVariant::PbrHostAlbedo) {
        return None;
    }
    let pbr = pipeline.as_any().downcast_ref::<PbrHostAlbedoPipeline>()?;
    let lookup = MaterialDrawContext::for_non_skinned_draw(
        d.material_asset_id,
        d.mesh_renderer_property_block_slot0_id,
    )
    .property_lookup;
    let pid = render_config.pbr_host_main_tex_property_id;
    if pid < 0 {
        return None;
    }
    let packed = match material_property_store.get_merged(lookup, pid)? {
        MaterialPropertyValue::Texture(p) => *p,
        _ => return None,
    };
    let tid = texture2d_asset_id_from_packed(packed)?;
    let tex = asset_registry.get_texture(tid)?;
    let _ = ensure_texture2d_gpu_view(
        device,
        queue,
        texture2d_gpu,
        texture2d_last_uploaded_version,
        native_ui_material_bind_cache,
        pbr_host_albedo_bind_cache,
        tid,
        tex,
    );
    let view = texture2d_gpu.get(&tid).map(|(_, v)| v)?;
    let bg = pbr_host_albedo_bind_cache
        .entry(tid)
        .or_insert_with(|| pbr.create_albedo_bind_group(device, view));
    Some(&*bg)
}

/// Records non-skinned mesh draws into the render pass.
pub fn record_non_skinned_draws(
    pass: &mut wgpu::RenderPass<'_>,
    params: &mut MeshDrawParams<'_>,
    draws: &[BatchedDraw],
) {
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
        let pipeline_key = PipelineKey(None, pipeline_variant);
        let Some(pipeline) = params.pipeline_manager.get_pipeline(
            pipeline_key,
            params.device,
            params.config,
            Some(params.material_property_store),
            params.render_config,
            Some(params.asset_registry),
        ) else {
            i += group_end;
            continue;
        };

        let use_overlay_upload = matches!(
            variant,
            crate::gpu::PipelineVariant::OverlayStencilContent
                | crate::gpu::PipelineVariant::OverlayStencilMaskWrite
                | crate::gpu::PipelineVariant::OverlayStencilMaskClear
        );
        if use_overlay_upload {
            let items: Vec<_> = group
                .iter()
                .map(|d| {
                    let clip = d
                        .stencil_state
                        .and_then(|s| s.clip_rect)
                        .map(|r| [r.x, r.y, r.width, r.height]);
                    (d.mvp, d.model, clip)
                })
                .collect();
            pipeline.upload_batch_overlay(params.queue, &items, params.frame_index);
        } else {
            let draws_upload: Vec<NonSkinnedUniformUpload> = group
                .iter()
                .map(|d| {
                    let mut u = NonSkinnedUniformUpload::new(d.mvp, d.model);
                    if matches!(
                        pipeline_variant,
                        PipelineVariant::Pbr
                            | PipelineVariant::PbrHostAlbedo
                            | PipelineVariant::PbrMRT
                            | PipelineVariant::PbrRayQuery
                            | PipelineVariant::PbrMRTRayQuery
                    ) {
                        fill_pbr_host_uniform_extras(
                            &mut u,
                            params.material_property_store,
                            params.render_config,
                            d,
                        );
                    }
                    u
                })
                .collect();
            pipeline.upload_batch(params.queue, &draws_upload, params.frame_index);
        }

        let is_stencil_pipeline = matches!(
            pipeline_variant,
            crate::gpu::PipelineVariant::OverlayStencilContent
                | crate::gpu::PipelineVariant::OverlayStencilMaskWrite
                | crate::gpu::PipelineVariant::OverlayStencilMaskClear
                | crate::gpu::PipelineVariant::OverlayStencilSkinned
                | crate::gpu::PipelineVariant::OverlayStencilMaskWriteSkinned
                | crate::gpu::PipelineVariant::OverlayStencilMaskClearSkinned
        );
        pipeline.bind_pipeline(pass);
        let is_native_ui = matches!(
            pipeline_variant,
            PipelineVariant::NativeUiUnlit { .. }
                | PipelineVariant::NativeUiTextUnlit { .. }
                | PipelineVariant::NativeUiUnlitStencil { .. }
                | PipelineVariant::NativeUiTextUnlitStencil { .. }
        );
        if is_native_ui && let Some(bg) = params.native_ui_scene_depth_bind {
            pass.set_bind_group(1, bg, &[]);
        }
        if params.use_mrt
            && !is_native_ui
            && pipeline_uses_standalone_mrt_gbuffer_origin_bind_group(&pipeline_variant)
            && let Some(bg) = params.mrt_gbuffer_origin_bind_group
        {
            pass.set_bind_group(1, bg, &[]);
        }
        // Nested if required: pbr must be destructured before passing params mutably to avoid borrow conflict.
        #[allow(clippy::collapsible_if)]
        if matches!(
            pipeline_variant,
            crate::gpu::PipelineVariant::Pbr
                | crate::gpu::PipelineVariant::PbrHostAlbedo
                | crate::gpu::PipelineVariant::PbrMRT
                | crate::gpu::PipelineVariant::PbrRayQuery
                | crate::gpu::PipelineVariant::PbrMRTRayQuery
        ) && let Some(ref pbr) = params.pbr_scene
        {
            if let Some(scene_bg) = get_or_create_pbr_scene_bind_group(
                params,
                pipeline.as_ref(),
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
                pipeline.bind_scene(pass, Some(scene_bg));
            }
        }
        let native_ui_unlit_ref = if matches!(
            pipeline_variant,
            PipelineVariant::NativeUiUnlit { .. } | PipelineVariant::NativeUiUnlitStencil { .. }
        ) {
            pipeline
                .as_ref()
                .as_any()
                .downcast_ref::<UiUnlitNativePipeline>()
        } else {
            None
        };
        let native_ui_text_ref = if matches!(
            pipeline_variant,
            PipelineVariant::NativeUiTextUnlit { .. }
                | PipelineVariant::NativeUiTextUnlitStencil { .. }
        ) {
            pipeline
                .as_ref()
                .as_any()
                .downcast_ref::<UiTextUnlitNativePipeline>()
        } else {
            None
        };
        let world_unlit_ref = if matches!(pipeline_variant, PipelineVariant::Material { .. }) {
            pipeline
                .as_ref()
                .as_any()
                .downcast_ref::<WorldUnlitPipeline>()
        } else {
            None
        };

        let mut order: Vec<usize> = (0..group.len()).collect();
        order.sort_by_key(|&idx| group[idx].mesh_asset_id);
        let mut last_mesh_asset_id: Option<i32> = None;
        let mut j = 0;
        while j < order.len() {
            let run_start = j;
            let first_idx = order[run_start];
            let mesh_id = group[first_idx].mesh_asset_id;
            let mut run_end = j + 1;
            while run_end < order.len() && group[order[run_end]].mesh_asset_id == mesh_id {
                run_end += 1;
            }
            let run_len = run_end - run_start;
            let run_has_stencil =
                (run_start..run_end).any(|k| group[order[k]].stencil_state.is_some());
            let first_range = group[first_idx].submesh_index_range;
            let same_index_range =
                (run_start..run_end).all(|k| group[order[k]].submesh_index_range == first_range);
            let use_instancing = run_len > 1
                && pipeline.supports_instancing()
                && !is_stencil_pipeline
                && !run_has_stencil
                && !is_native_ui
                && !matches!(pipeline_variant, PipelineVariant::PbrHostAlbedo)
                && run_len as u32 <= crate::gpu::MAX_INSTANCE_RUN
                && same_index_range;

            let Some(buffers) = params.mesh_buffer_cache.get(&mesh_id) else {
                j = run_end;
                continue;
            };
            if last_mesh_asset_id != Some(mesh_id) {
                pipeline.set_mesh_buffers(pass, buffers);
                last_mesh_asset_id = Some(mesh_id);
            }

            if use_instancing {
                pipeline.bind_draw(pass, Some(first_idx as u32), params.frame_index, None);
                pipeline.draw_mesh_indexed_instanced(pass, buffers, run_len as u32, first_range);
            } else {
                for idx in order[run_start..run_end].iter().copied() {
                    let d = &group[idx];
                    let draw_bind = pbr_host_albedo_draw_bind(
                        pipeline_variant,
                        pipeline.as_ref(),
                        d,
                        params.material_property_store,
                        params.render_config,
                        params.asset_registry,
                        params.device,
                        params.queue,
                        params.texture2d_gpu,
                        params.texture2d_last_uploaded_version,
                        params.native_ui_material_bind_cache,
                        params.pbr_host_albedo_bind_cache,
                    );
                    pipeline.bind_draw(pass, Some(idx as u32), params.frame_index, draw_bind);
                    match pipeline_variant {
                        PipelineVariant::NativeUiUnlit { material_id }
                        | PipelineVariant::NativeUiUnlitStencil { material_id } => {
                            if let Some(ui) = native_ui_unlit_ref {
                                let ui_lookup = MaterialDrawContext::for_non_skinned_draw(
                                    material_id,
                                    d.mesh_renderer_property_block_slot0_id,
                                )
                                .property_lookup;
                                let (_, main_packed, mask_packed) =
                                    crate::assets::ui_unlit_material_uniform(
                                        params.material_property_store,
                                        ui_lookup,
                                        &params.render_config.ui_unlit_property_ids,
                                    );
                                let main_id = (main_packed != 0)
                                    .then(|| texture2d_asset_id_from_packed(main_packed))
                                    .flatten();
                                let mask_id = (mask_packed != 0)
                                    .then(|| texture2d_asset_id_from_packed(mask_packed))
                                    .flatten();
                                if let Some((id, tex)) = main_id.and_then(|id| {
                                    params.asset_registry.get_texture(id).map(|tex| (id, tex))
                                }) {
                                    let _ = ensure_texture2d_gpu_view(
                                        params.device,
                                        params.queue,
                                        params.texture2d_gpu,
                                        params.texture2d_last_uploaded_version,
                                        params.native_ui_material_bind_cache,
                                        params.pbr_host_albedo_bind_cache,
                                        id,
                                        tex,
                                    );
                                }
                                if let Some((id, tex)) = mask_id.and_then(|id| {
                                    params.asset_registry.get_texture(id).map(|tex| (id, tex))
                                }) {
                                    let _ = ensure_texture2d_gpu_view(
                                        params.device,
                                        params.queue,
                                        params.texture2d_gpu,
                                        params.texture2d_last_uploaded_version,
                                        params.native_ui_material_bind_cache,
                                        params.pbr_host_albedo_bind_cache,
                                        id,
                                        tex,
                                    );
                                }
                                let main_view = main_id
                                    .and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
                                let mask_view = mask_id
                                    .and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
                                let main_key = main_id.unwrap_or(0);
                                let mask_key = mask_id.unwrap_or(0);
                                if params.render_config.log_native_ui_routing
                                    && !params.render_config.native_ui_routing_metrics
                                {
                                    logger::trace!(
                                        "native_ui: ui_unlit bind material_block={} main_tex_id={:?} mask_tex_id={:?} main_view_ok={} mask_view_ok={}",
                                        material_id,
                                        main_id,
                                        mask_id,
                                        main_view.is_some(),
                                        mask_view.is_some(),
                                    );
                                }
                                params
                                    .native_ui_material_bind_cache
                                    .write_ui_unlit_material_bind(
                                        params.device,
                                        params.queue,
                                        pass,
                                        ui.material_bind_group_layout(),
                                        ui.material_uniform_buffer(),
                                        ui.linear_sampler(),
                                        params.material_property_store,
                                        ui_lookup,
                                        &params.render_config.ui_unlit_property_ids,
                                        main_view,
                                        mask_view,
                                        main_key,
                                        mask_key,
                                    );
                            }
                        }
                        PipelineVariant::NativeUiTextUnlit { material_id }
                        | PipelineVariant::NativeUiTextUnlitStencil { material_id } => {
                            if let Some(ui) = native_ui_text_ref {
                                let ui_lookup = MaterialDrawContext::for_non_skinned_draw(
                                    material_id,
                                    d.mesh_renderer_property_block_slot0_id,
                                )
                                .property_lookup;
                                let (_, font_packed) =
                                    crate::assets::ui_text_unlit_material_uniform(
                                        params.material_property_store,
                                        ui_lookup,
                                        &params.render_config.ui_text_unlit_property_ids,
                                    );
                                let font_id = (font_packed != 0)
                                    .then(|| texture2d_asset_id_from_packed(font_packed))
                                    .flatten();
                                if let Some((id, tex)) = font_id.and_then(|id| {
                                    params.asset_registry.get_texture(id).map(|tex| (id, tex))
                                }) {
                                    let _ = ensure_texture2d_gpu_view(
                                        params.device,
                                        params.queue,
                                        params.texture2d_gpu,
                                        params.texture2d_last_uploaded_version,
                                        params.native_ui_material_bind_cache,
                                        params.pbr_host_albedo_bind_cache,
                                        id,
                                        tex,
                                    );
                                }
                                let font_view = font_id
                                    .and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
                                let font_key = font_id.unwrap_or(0);
                                if params.render_config.log_native_ui_routing
                                    && !params.render_config.native_ui_routing_metrics
                                {
                                    logger::trace!(
                                        "native_ui: ui_text_unlit bind material_block={} font_atlas_id={:?} font_view_ok={}",
                                        material_id,
                                        font_id,
                                        font_view.is_some(),
                                    );
                                }
                                params
                                    .native_ui_material_bind_cache
                                    .write_ui_text_unlit_material_bind(
                                        params.device,
                                        params.queue,
                                        pass,
                                        ui.material_uniform_buffer(),
                                        ui.linear_sampler(),
                                        ui.material_bind_group_layout(),
                                        params.material_property_store,
                                        ui_lookup,
                                        &params.render_config.ui_text_unlit_property_ids,
                                        font_view,
                                        font_key,
                                    );
                            }
                        }
                        PipelineVariant::Material { material_id } => {
                            if let Some(w) = world_unlit_ref {
                                let ui_lookup = MaterialDrawContext::for_non_skinned_draw(
                                    material_id,
                                    d.mesh_renderer_property_block_slot0_id,
                                )
                                .property_lookup;
                                let (_, main_packed, mask_packed) =
                                    crate::assets::world_unlit_material_uniform(
                                        params.material_property_store,
                                        ui_lookup,
                                        &params.render_config.world_unlit_property_ids,
                                    );
                                let main_id = (main_packed != 0)
                                    .then(|| texture2d_asset_id_from_packed(main_packed))
                                    .flatten();
                                let mask_id = (mask_packed != 0)
                                    .then(|| texture2d_asset_id_from_packed(mask_packed))
                                    .flatten();
                                if let Some((id, tex)) = main_id.and_then(|id| {
                                    params.asset_registry.get_texture(id).map(|tex| (id, tex))
                                }) {
                                    let _ = ensure_texture2d_gpu_view(
                                        params.device,
                                        params.queue,
                                        params.texture2d_gpu,
                                        params.texture2d_last_uploaded_version,
                                        params.native_ui_material_bind_cache,
                                        params.pbr_host_albedo_bind_cache,
                                        id,
                                        tex,
                                    );
                                }
                                if let Some((id, tex)) = mask_id.and_then(|id| {
                                    params.asset_registry.get_texture(id).map(|tex| (id, tex))
                                }) {
                                    let _ = ensure_texture2d_gpu_view(
                                        params.device,
                                        params.queue,
                                        params.texture2d_gpu,
                                        params.texture2d_last_uploaded_version,
                                        params.native_ui_material_bind_cache,
                                        params.pbr_host_albedo_bind_cache,
                                        id,
                                        tex,
                                    );
                                }
                                let main_view = main_id
                                    .and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
                                let mask_view = mask_id
                                    .and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
                                let main_key = main_id.unwrap_or(0);
                                let mask_key = mask_id.unwrap_or(0);
                                params
                                    .native_ui_material_bind_cache
                                    .write_world_unlit_material_bind(
                                        params.device,
                                        params.queue,
                                        pass,
                                        w.material_bind_group_layout(),
                                        w.material_uniform_buffer(),
                                        w.linear_sampler(),
                                        params.material_property_store,
                                        ui_lookup,
                                        &params.render_config.world_unlit_property_ids,
                                        main_view,
                                        mask_view,
                                        main_key,
                                        mask_key,
                                    );
                            }
                        }
                        _ => {}
                    }
                    if let Some(ref stencil) = d.stencil_state {
                        pass.set_stencil_reference(stencil.reference as u32);
                    } else if is_stencil_pipeline {
                        debug_assert!(
                            d.stencil_state.is_some(),
                            "Overlay stencil draws must have stencil_state"
                        );
                    }
                    pipeline.draw_mesh_indexed(pass, buffers, d.submesh_index_range);
                }
            }
            j = run_end;
        }

        i += group_end;
    }
}
