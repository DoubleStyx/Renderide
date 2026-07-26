use std::mem::size_of;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use hashbrown::HashMap;

use super::{
    PaddedShadowCasterDraw, PaddedShadowLayerUniforms, clamp_shadow_resolution,
    clamp_shadow_texture_resolution, select_shadow_atlas_split_workload,
    shadow_atlas_array_view_descriptor, shadow_atlas_layer_view_descriptor, shadow_pipeline_state,
};
use crate::backend::frame_resource_manager::ShadowRenderView;
use crate::gpu::{SHADOW_VIEW_KIND_DIRECTIONAL, SHADOW_VIEW_KIND_POINT, SHADOW_VIEW_KIND_SPOT};
use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::render_phase::RenderPhaseSet;
use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
use crate::world_mesh::{DrawGroup, WorldMeshDrawItem, WorldMeshPhase};

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

fn dummy_draw_item() -> WorldMeshDrawItem {
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
fn shadow_split_workload_preserves_layer_order_and_balances_chunks() {
    let groups = super::SHADOW_ATLAS_PARALLEL_MIN_VISIBLE_GROUPS;

    let workload =
        select_shadow_atlas_split_workload(5, groups, 300, true, 2).expect("split workload");

    assert_eq!(workload.unit_count, 5);
    assert_eq!(workload.estimated_work, groups + 300);
    assert_eq!(workload.chunk_size, 3);
}

#[test]
fn shadow_split_workload_rejects_unsafe_or_tiny_fanout() {
    let groups = super::SHADOW_ATLAS_PARALLEL_MIN_VISIBLE_GROUPS;

    assert_eq!(
        select_shadow_atlas_split_workload(1, groups, 300, true, 4),
        None
    );
    assert_eq!(
        select_shadow_atlas_split_workload(2, groups - 1, 300, true, 4),
        None
    );
    assert_eq!(
        select_shadow_atlas_split_workload(2, groups, 300, true, 1),
        None
    );
    assert_eq!(
        select_shadow_atlas_split_workload(2, groups, 300, false, 4),
        None
    );
}

#[test]
fn shadow_caster_uniforms_use_identity_model_for_world_space_positions() {
    let mut item = dummy_draw_item();
    item.world_space_deformed = true;
    item.rigid_world_matrix = Some(Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0)));

    let slot = PaddedShadowCasterDraw::new(&item);

    assert_eq!(slot.model, Mat4::IDENTITY.to_cols_array());
}

#[test]
fn shrink_window_requires_sustained_below_capacity_demand() {
    let mut window = super::ShadowAtlasShrinkWindow::default();

    for _ in 0..(super::SHADOW_ATLAS_SHRINK_SYNC_COUNT - 1) {
        assert_eq!(window.note(1024, 4, 4096, 16), None);
    }
    assert_eq!(window.note(1024, 4, 4096, 16), Some((1024, 4)));
}

#[test]
fn shrink_window_resets_on_growth_and_tracks_peak_demand() {
    let mut window = super::ShadowAtlasShrinkWindow::default();

    for _ in 0..10 {
        assert_eq!(window.note(1024, 4, 4096, 16), None);
    }
    assert_eq!(window.note(4096, 16, 4096, 16), None);
    for _ in 0..(super::SHADOW_ATLAS_SHRINK_SYNC_COUNT - 1) {
        assert_eq!(window.note(2048, 8, 4096, 16), None);
    }
    assert_eq!(window.note(1024, 4, 4096, 16), Some((2048, 8)));
}

#[test]
fn shrink_window_skips_marginal_savings() {
    let mut window = super::ShadowAtlasShrinkWindow::default();

    for _ in 0..(super::SHADOW_ATLAS_SHRINK_SYNC_COUNT - 1) {
        assert_eq!(window.note(4096, 15, 4096, 16), None);
    }
    assert_eq!(window.note(4096, 15, 4096, 16), None);
}

fn indirect_key(
    generation: u64,
    slab_slot_offset: usize,
    draws: Arc<[WorldMeshDrawItem]>,
    visible_groups: Arc<RenderPhaseSet<WorldMeshPhase, DrawGroup>>,
    view: &ShadowRenderView,
) -> super::ShadowIndirectPlanKey {
    super::ShadowIndirectPlanKey {
        allocation_generation: generation,
        layers: vec![super::ShadowIndirectLayerKey {
            layer: view.layer,
            slab_slot_offset,
            view_signature: view.view_signature,
            draws,
            visible_groups,
        }],
    }
}

#[test]
fn shadow_indirect_key_requires_strong_packet_identity_and_arena_generation() {
    let draws: Arc<[WorldMeshDrawItem]> = Arc::from([dummy_draw_item()]);
    let groups = Arc::new(RenderPhaseSet::new());
    let view = shadow_view(SHADOW_VIEW_KIND_SPOT);
    let first = indirect_key(7, 4, Arc::clone(&draws), Arc::clone(&groups), &view);
    let exact = indirect_key(7, 4, Arc::clone(&draws), Arc::clone(&groups), &view);
    assert!(first.matches(&exact));

    let replaced_draws = indirect_key(
        7,
        4,
        Arc::from([dummy_draw_item()]),
        Arc::clone(&groups),
        &view,
    );
    assert!(!first.matches(&replaced_draws));

    let replaced_groups = indirect_key(
        7,
        4,
        Arc::clone(&draws),
        Arc::new(RenderPhaseSet::new()),
        &view,
    );
    assert!(!first.matches(&replaced_groups));

    let moved_arena = indirect_key(8, 4, Arc::clone(&draws), Arc::clone(&groups), &view);
    assert!(!first.matches(&moved_arena));
}

#[test]
fn shadow_indirect_key_tracks_slab_and_exact_view_signature() {
    let draws: Arc<[WorldMeshDrawItem]> = Arc::from([dummy_draw_item()]);
    let groups = Arc::new(RenderPhaseSet::new());
    let spot = shadow_view(SHADOW_VIEW_KIND_SPOT);
    let point = shadow_view(SHADOW_VIEW_KIND_POINT);
    let first = indirect_key(3, 0, Arc::clone(&draws), Arc::clone(&groups), &spot);
    let moved_slab = indirect_key(3, 1, Arc::clone(&draws), Arc::clone(&groups), &spot);
    let changed_view = indirect_key(3, 0, draws, groups, &point);

    assert!(!first.matches(&moved_slab));
    assert!(!first.matches(&changed_view));
}
