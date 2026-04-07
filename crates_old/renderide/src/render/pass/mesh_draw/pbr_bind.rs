//! PBR scene bind groups, MRT bind-group routing, and host uniform extras for mesh draws.

use super::types::{BatchedDraw, MeshDrawParams};
use crate::assets::{MaterialPropertyStore, MaterialPropertyValue, texture2d_asset_id_from_packed};
use crate::config::RenderConfig;
use crate::gpu::pipeline::{RtShadowSceneBind, RtShadowUniforms, SceneUniforms};
use crate::gpu::{NonSkinnedUniformUpload, PipelineVariant, RenderPipeline};
use crate::render::pass::material_draw_context::MaterialDrawContext;

/// Writes host [`MaterialPropertyStore`] channels into a non-skinned uniform ring slot for PBR draws.
pub fn fill_pbr_host_uniform_extras(
    upload: &mut NonSkinnedUniformUpload,
    store: &MaterialPropertyStore,
    rc: &RenderConfig,
    draw: &BatchedDraw,
) {
    if !rc.pbr_bind_host_material_properties {
        return;
    }
    let lookup = MaterialDrawContext::for_non_skinned_draw(
        draw.material_asset_id,
        draw.mesh_renderer_property_block_slot0_id,
    )
    .property_lookup;
    if rc.pbr_host_color_property_id >= 0
        && let Some(MaterialPropertyValue::Float4(c)) =
            store.get_merged(lookup, rc.pbr_host_color_property_id)
    {
        upload.host_base_color = [c[0], c[1], c[2], c[3]];
    }
    let mut mr_active = false;
    let mut metallic = 0.5_f32;
    let mut roughness = 0.5_f32;
    if rc.pbr_host_metallic_property_id >= 0
        && let Some(MaterialPropertyValue::Float(m)) =
            store.get_merged(lookup, rc.pbr_host_metallic_property_id)
    {
        metallic = m.clamp(0.0, 1.0);
        mr_active = true;
    }
    if rc.pbr_host_smoothness_property_id >= 0
        && let Some(MaterialPropertyValue::Float(g)) =
            store.get_merged(lookup, rc.pbr_host_smoothness_property_id)
    {
        let g = g.clamp(0.0, 1.0);
        roughness = 1.0 - g;
        mr_active = true;
    }
    if mr_active {
        upload.host_metallic_roughness = [metallic, roughness, 1.0, 0.0];
    }
    if rc.pbr_bind_host_main_texture
        && rc.pbr_host_main_tex_property_id >= 0
        && let Some(MaterialPropertyValue::Texture(packed)) =
            store.get_merged(lookup, rc.pbr_host_main_tex_property_id)
        && texture2d_asset_id_from_packed(*packed).is_some()
    {
        upload.host_metallic_roughness[3] = 1.0;
    }
}

fn pbr_variant_is_ray_query(variant: PipelineVariant) -> bool {
    matches!(
        variant,
        PipelineVariant::PbrRayQuery
            | PipelineVariant::PbrMRTRayQuery
            | PipelineVariant::SkinnedPbrRayQuery
            | PipelineVariant::SkinnedPbrMRTRayQuery
    )
}

/// Gets or creates a PBR scene bind group from the cache.
/// Invalidates cache when light or cluster buffer versions change.
#[allow(clippy::too_many_arguments)]
pub fn get_or_create_pbr_scene_bind_group<'a>(
    params: &'a mut MeshDrawParams,
    pipeline: &dyn RenderPipeline,
    variant: PipelineVariant,
    view_position: [f32; 3],
    view_space_z_coeffs: [f32; 4],
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
    light_buffer: &wgpu::Buffer,
    cluster_light_counts: &wgpu::Buffer,
    cluster_light_indices: &wgpu::Buffer,
    scene_tlas: Option<std::ptr::NonNull<wgpu::Tlas>>,
) -> Option<&'a wgpu::BindGroup> {
    if *params.last_pbr_scene_cache_light_version != params.light_buffer_version
        || *params.last_pbr_scene_cache_cluster_version != params.cluster_buffer_version
        || *params.last_pbr_scene_cache_tlas_generation != params.pbr_tlas_generation
        || *params.last_pbr_scene_cache_rt_shadow_atlas_generation
            != params.rt_shadow_atlas_generation
    {
        params.pbr_scene_bind_group_cache.clear();
        *params.last_pbr_scene_cache_light_version = params.light_buffer_version;
        *params.last_pbr_scene_cache_cluster_version = params.cluster_buffer_version;
        *params.last_pbr_scene_cache_tlas_generation = params.pbr_tlas_generation;
        *params.last_pbr_scene_cache_rt_shadow_atlas_generation = params.rt_shadow_atlas_generation;
    }
    let scene = SceneUniforms {
        view_position,
        _pad0: 0.0,
        view_space_z_coeffs,
        cluster_count_x,
        cluster_count_y,
        cluster_count_z,
        near_clip,
        far_clip,
        light_count,
        viewport_width,
        viewport_height,
    };
    pipeline.write_scene_uniform(params.queue, bytemuck::bytes_of(&scene));
    let tlas_ref = scene_tlas.map(|p| {
        // SAFETY: `p` comes from `RayTracingState::tlas` for this frame; mesh pass does not drop
        // or replace TLAS until after this encode (see `RenderGraph::execute`).
        unsafe { p.as_ref() }
    });
    if pbr_variant_is_ray_query(variant) {
        let rb = params.rt_shadow_bind.as_ref()?;
        let u = RtShadowUniforms {
            soft_shadow_sample_count: rb.soft_samples.clamp(1, 16),
            soft_cone_scale: rb.cone_scale,
            frame_counter: params.frame_index as u32,
            shadow_mode: rb.shadow_mode,
            full_viewport_width: rb.full_viewport_width,
            full_viewport_height: rb.full_viewport_height,
            shadow_atlas_width: rb.atlas_width.max(1),
            shadow_atlas_height: rb.atlas_height.max(1),
            gbuffer_origin: rb.gbuffer_origin,
            _pad0: 0.0,
        };
        params
            .queue
            .write_buffer(rb.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }
    let bg = params
        .pbr_scene_bind_group_cache
        .entry(variant)
        .or_insert_with(|| {
            let rt_shadow = if pbr_variant_is_ray_query(variant) {
                let rb = params.rt_shadow_bind.as_ref().expect(
                    "ray-query PBR scene bind group requires MeshDrawParams::rt_shadow_bind",
                );
                Some(RtShadowSceneBind {
                    uniform_buffer: rb.uniform_buffer,
                    atlas_view: rb.atlas_view,
                    sampler: rb.sampler,
                })
            } else {
                None
            };
            pipeline
                .create_scene_bind_group(
                    params.device,
                    params.queue,
                    view_position,
                    view_space_z_coeffs,
                    cluster_count_x,
                    cluster_count_y,
                    cluster_count_z,
                    near_clip,
                    far_clip,
                    light_count,
                    viewport_width,
                    viewport_height,
                    light_buffer,
                    cluster_light_counts,
                    cluster_light_indices,
                    tlas_ref,
                    rt_shadow,
                )
                .expect("PBR pipeline must create scene bind group")
        });
    Some(bg)
}

/// Debug MRT pipelines use group 1 for [`crate::gpu::pipeline::mrt::MrtGbufferOriginUniform`]; PBR MRT uses group 1 for scene data instead.
pub fn pipeline_uses_standalone_mrt_gbuffer_origin_bind_group(
    variant: &crate::gpu::PipelineVariant,
) -> bool {
    matches!(
        variant,
        crate::gpu::PipelineVariant::NormalDebugMRT
            | crate::gpu::PipelineVariant::UvDebugMRT
            | crate::gpu::PipelineVariant::SkinnedMRT
    )
}
