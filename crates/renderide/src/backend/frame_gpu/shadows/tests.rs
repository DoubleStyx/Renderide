use std::mem::size_of;

use glam::{Mat4, Vec3};
use hashbrown::HashMap;

use super::{
    PaddedShadowCasterDraw, PaddedShadowLayerUniforms, clamp_shadow_resolution,
    clamp_shadow_texture_resolution, shadow_atlas_array_view_descriptor,
    shadow_atlas_layer_view_descriptor, shadow_pipeline_state,
};
use crate::backend::frame_resource_manager::ShadowRenderView;
use crate::gpu::{SHADOW_VIEW_KIND_DIRECTIONAL, SHADOW_VIEW_KIND_POINT, SHADOW_VIEW_KIND_SPOT};
use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

fn limits(max_texture_dimension_2d: u32, max_texture_array_layers: u32) -> crate::gpu::GpuLimits {
    crate::gpu::GpuLimits::synthetic_for_tests(
        wgpu::Limits {
            max_texture_dimension_2d,
            max_texture_array_layers,
            ..Default::default()
        },
        wgpu::Features::empty(),
        HashMap::new(),
    )
}

fn dummy_draw_item() -> crate::world_mesh::WorldMeshDrawItem {
    dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 0,
        collect_order: 0,
        alpha_blended: false,
    })
}

fn shadow_view(kind: u32) -> ShadowRenderView {
    ShadowRenderView::for_tests(
        kind,
        Mat4::from_scale(Vec3::splat(2.0)),
        Vec3::new(1.0, 2.0, 3.0),
        12.0,
        0.25,
    )
}

#[test]
fn shadow_resolution_clamps_to_device_limit() {
    let limits = limits(1024, 8);
    assert_eq!(clamp_shadow_resolution(&limits, 0), 1);
    assert_eq!(clamp_shadow_resolution(&limits, 512), 512);
    assert_eq!(clamp_shadow_resolution(&limits, 2048), 1024);
}

#[test]
fn shadow_texture_resolution_clamps_to_atlas_budget() {
    let limits = limits(8192, 64);

    assert_eq!(
        clamp_shadow_texture_resolution(&limits, 4096, 16, wgpu::TextureFormat::Depth32Float),
        2896
    );
}

#[test]
fn shadow_atlas_array_view_is_sampled_only() {
    let format = wgpu::TextureFormat::Depth24Plus;
    let desc = shadow_atlas_array_view_descriptor(4, format);

    assert_eq!(desc.format, Some(format));
    assert_eq!(desc.dimension, Some(wgpu::TextureViewDimension::D2Array));
    assert_eq!(desc.usage, Some(wgpu::TextureUsages::TEXTURE_BINDING));
    assert_eq!(desc.aspect, wgpu::TextureAspect::DepthOnly);
    assert_eq!(desc.base_mip_level, 0);
    assert_eq!(desc.mip_level_count, Some(1));
    assert_eq!(desc.base_array_layer, 0);
    assert_eq!(desc.array_layer_count, Some(4));
}

#[test]
fn shadow_atlas_layer_view_is_render_attachment_only() {
    let format = wgpu::TextureFormat::Depth16Unorm;
    let desc = shadow_atlas_layer_view_descriptor(3, format);

    assert_eq!(desc.format, Some(format));
    assert_eq!(desc.dimension, Some(wgpu::TextureViewDimension::D2));
    assert_eq!(desc.usage, Some(wgpu::TextureUsages::RENDER_ATTACHMENT));
    assert_eq!(desc.aspect, wgpu::TextureAspect::DepthOnly);
    assert_eq!(desc.base_mip_level, 0);
    assert_eq!(desc.mip_level_count, Some(1));
    assert_eq!(desc.base_array_layer, 3);
    assert_eq!(desc.array_layer_count, Some(1));
}

#[test]
fn shadow_pipeline_state_uses_selected_depth_format() {
    let format = wgpu::TextureFormat::Depth24Plus;
    let pipeline = shadow_pipeline_state(format);

    assert_eq!(pipeline.pass_desc.depth_stencil_format, Some(format));
}

#[test]
fn shadow_caster_uniform_stride_matches_dynamic_offset_stride() {
    assert_eq!(size_of::<PaddedShadowCasterDraw>(), PER_DRAW_UNIFORM_STRIDE);
    assert_eq!(
        size_of::<PaddedShadowLayerUniforms>(),
        PER_DRAW_UNIFORM_STRIDE
    );
}

#[test]
fn shadow_caster_draw_uniforms_pack_model_data() {
    let mut item = dummy_draw_item();
    let model = Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0));
    item.rigid_world_matrix = Some(model);

    let slot = PaddedShadowCasterDraw::new(&item);

    assert_eq!(slot.model, model.to_cols_array());
}

#[test]
fn radial_shadow_layer_uniforms_pack_light_data() {
    for kind in [SHADOW_VIEW_KIND_POINT, SHADOW_VIEW_KIND_SPOT] {
        let slot = PaddedShadowLayerUniforms::new(&shadow_view(kind));

        assert_eq!(
            slot.view_proj,
            Mat4::from_scale(Vec3::splat(2.0)).to_cols_array()
        );
        assert_eq!(slot.light_position_range, [1.0, 2.0, 3.0, 12.0]);
        assert_eq!(slot.shadow_params[0], 0.25);
    }
}

#[test]
fn projected_shadow_layer_uniforms_do_not_pack_radial_bias() {
    let slot = PaddedShadowLayerUniforms::new(&shadow_view(SHADOW_VIEW_KIND_DIRECTIONAL));

    assert_eq!(slot.light_position_range, [0.0; 4]);
    assert_eq!(slot.shadow_params[0], 0.0);
}

#[test]
fn shadow_caster_uniforms_use_identity_model_for_world_space_positions() {
    let mut item = dummy_draw_item();
    item.world_space_deformed = true;
    item.rigid_world_matrix = Some(Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0)));

    let slot = PaddedShadowCasterDraw::new(&item);

    assert_eq!(slot.model, Mat4::IDENTITY.to_cols_array());
}
