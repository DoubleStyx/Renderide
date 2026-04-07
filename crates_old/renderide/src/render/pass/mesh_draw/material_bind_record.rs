//! Per-material bind recording for non-skinned mesh draws (Strangler Fig seam).
//!
//! Host routes that sample textures or push material uniforms (`NativeUi*`, [`PipelineVariant::Material`])
//! perform additional `set_bind_group` / buffer writes after [`crate::gpu::pipeline::RenderPipeline::bind_draw`].
//! This module centralizes that logic so [`super::record_non_skinned`] stays focused on batching and instancing.
//!
//! **Deferred (not here):** renumbering WGSL bind groups to put shared scene data at `@group(0)` and
//! material-owned data at `@group(1)` — that requires new pipeline layouts and shaders per family.

use super::types::{BatchedDraw, MeshDrawParams};
use crate::assets::{MaterialPropertyStore, MaterialPropertyValue, texture2d_asset_id_from_packed};
use crate::config::RenderConfig;
use crate::gpu::PbrHostAlbedoMaterialBindKey;
use crate::gpu::pipeline::{
    PbrHostAlbedoPipeline, UiTextUnlitNativePipeline, UiUnlitNativePipeline, WorldUnlitPipeline,
};
use crate::gpu::state::ensure_texture2d_gpu_view;
use crate::gpu::{PipelineVariant, RenderPipeline};
use crate::render::pass::material_draw_context::MaterialDrawContext;
use std::collections::HashMap;

/// Returns true when [`bind_material_resources_for_non_skinned_draw`] may perform material GPU writes
/// (native UI / world-unlit texture binds). Used by tests to lock routing invariants.
///
/// Not invoked from production code; keep in sync with the `match` in
/// [`bind_material_resources_for_non_skinned_draw`].
#[allow(dead_code)]
pub(super) fn pipeline_variant_uses_material_resource_binds(variant: PipelineVariant) -> bool {
    matches!(
        variant,
        PipelineVariant::NativeUiUnlit { .. }
            | PipelineVariant::NativeUiUnlitStencil { .. }
            | PipelineVariant::NativeUiTextUnlit { .. }
            | PipelineVariant::NativeUiTextUnlitStencil { .. }
            | PipelineVariant::Material { .. }
    )
}

/// Resolves per-draw bind group 0 for [`PipelineVariant::PbrHostAlbedo`] (host `_MainTex`).
///
/// Takes disjoint store/registry/cache handles instead of [`MeshDrawParams`] so the borrow
/// checker can overlap with other `params` uses in [`super::record_non_skinned::record_non_skinned_draws`].
#[allow(clippy::too_many_arguments)]
pub(super) fn pbr_host_albedo_draw_bind<'a>(
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
    material_gpu_resources: &'a mut crate::gpu::MaterialGpuResources,
    pbr_host_albedo_bind_cache: &'a mut HashMap<PbrHostAlbedoMaterialBindKey, wgpu::BindGroup>,
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
        material_gpu_resources,
        pbr_host_albedo_bind_cache,
        tid,
        tex,
    );
    let view = texture2d_gpu.get(&tid).map(|(_, v)| v)?;
    let cache_key: PbrHostAlbedoMaterialBindKey = (d.material_asset_id, tid);
    let bg = pbr_host_albedo_bind_cache
        .entry(cache_key)
        .or_insert_with(|| pbr.create_albedo_bind_group(device, view));
    Some(&*bg)
}

/// Binds per-material resources after `bind_draw` for native UI and world-unlit paths.
///
/// Must preserve the exact `set_bind_group` / [`crate::gpu::MaterialGpuResources`] call order from
/// the pre-refactor inline match in `record_non_skinned`.
pub(super) fn bind_material_resources_for_non_skinned_draw(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline_variant: PipelineVariant,
    d: &BatchedDraw,
    params: &mut MeshDrawParams<'_>,
    native_ui_unlit: Option<&UiUnlitNativePipeline>,
    native_ui_text: Option<&UiTextUnlitNativePipeline>,
    world_unlit: Option<&WorldUnlitPipeline>,
) {
    match pipeline_variant {
        PipelineVariant::NativeUiUnlit { material_id }
        | PipelineVariant::NativeUiUnlitStencil { material_id } => {
            if let Some(ui) = native_ui_unlit {
                let ui_lookup = MaterialDrawContext::for_non_skinned_draw(
                    material_id,
                    d.mesh_renderer_property_block_slot0_id,
                )
                .property_lookup;
                let (_, main_packed, mask_packed) = crate::assets::ui_unlit_material_uniform(
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
                if let Some((id, tex)) = main_id
                    .and_then(|id| params.asset_registry.get_texture(id).map(|tex| (id, tex)))
                {
                    let _ = ensure_texture2d_gpu_view(
                        params.device,
                        params.queue,
                        params.texture2d_gpu,
                        params.texture2d_last_uploaded_version,
                        params.material_gpu_resources,
                        params.pbr_host_albedo_bind_cache,
                        id,
                        tex,
                    );
                }
                if let Some((id, tex)) = mask_id
                    .and_then(|id| params.asset_registry.get_texture(id).map(|tex| (id, tex)))
                {
                    let _ = ensure_texture2d_gpu_view(
                        params.device,
                        params.queue,
                        params.texture2d_gpu,
                        params.texture2d_last_uploaded_version,
                        params.material_gpu_resources,
                        params.pbr_host_albedo_bind_cache,
                        id,
                        tex,
                    );
                }
                let main_view =
                    main_id.and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
                let mask_view =
                    mask_id.and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
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
                params.material_gpu_resources.write_ui_unlit_material_bind(
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
            if let Some(ui) = native_ui_text {
                let ui_lookup = MaterialDrawContext::for_non_skinned_draw(
                    material_id,
                    d.mesh_renderer_property_block_slot0_id,
                )
                .property_lookup;
                let (_, font_packed) = crate::assets::ui_text_unlit_material_uniform(
                    params.material_property_store,
                    ui_lookup,
                    &params.render_config.ui_text_unlit_property_ids,
                );
                let font_id = (font_packed != 0)
                    .then(|| texture2d_asset_id_from_packed(font_packed))
                    .flatten();
                if let Some((id, tex)) = font_id
                    .and_then(|id| params.asset_registry.get_texture(id).map(|tex| (id, tex)))
                {
                    let _ = ensure_texture2d_gpu_view(
                        params.device,
                        params.queue,
                        params.texture2d_gpu,
                        params.texture2d_last_uploaded_version,
                        params.material_gpu_resources,
                        params.pbr_host_albedo_bind_cache,
                        id,
                        tex,
                    );
                }
                let font_view =
                    font_id.and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
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
                    .material_gpu_resources
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
            if let Some(w) = world_unlit {
                let ui_lookup = MaterialDrawContext::for_non_skinned_draw(
                    material_id,
                    d.mesh_renderer_property_block_slot0_id,
                )
                .property_lookup;
                let (_, main_packed, mask_packed) = crate::assets::world_unlit_material_uniform(
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
                if let Some((id, tex)) = main_id
                    .and_then(|id| params.asset_registry.get_texture(id).map(|tex| (id, tex)))
                {
                    let _ = ensure_texture2d_gpu_view(
                        params.device,
                        params.queue,
                        params.texture2d_gpu,
                        params.texture2d_last_uploaded_version,
                        params.material_gpu_resources,
                        params.pbr_host_albedo_bind_cache,
                        id,
                        tex,
                    );
                }
                if let Some((id, tex)) = mask_id
                    .and_then(|id| params.asset_registry.get_texture(id).map(|tex| (id, tex)))
                {
                    let _ = ensure_texture2d_gpu_view(
                        params.device,
                        params.queue,
                        params.texture2d_gpu,
                        params.texture2d_last_uploaded_version,
                        params.material_gpu_resources,
                        params.pbr_host_albedo_bind_cache,
                        id,
                        tex,
                    );
                }
                let main_view =
                    main_id.and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
                let mask_view =
                    mask_id.and_then(|id| params.texture2d_gpu.get(&id).map(|(_, v)| v));
                let main_key = main_id.unwrap_or(0);
                let mask_key = mask_id.unwrap_or(0);
                params
                    .material_gpu_resources
                    .write_world_unlit_material_bind(
                        params.device,
                        params.queue,
                        pass,
                        w.material_bind_group_layout(),
                        w.linear_sampler(),
                        params.material_property_store,
                        ui_lookup,
                        &params.render_config.world_unlit_property_ids,
                        main_view,
                        mask_view,
                        main_key,
                        mask_key,
                        material_id,
                    );
            }
        }
        _ => {}
    }
}
