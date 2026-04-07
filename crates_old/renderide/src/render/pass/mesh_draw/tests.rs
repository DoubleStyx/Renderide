//! Unit tests for mesh draw pipeline routing and PBR host uniform fill.

use super::material_bind_record::pipeline_variant_uses_material_resource_binds;
use super::pbr_bind::fill_pbr_host_uniform_extras;
use super::pipeline::{mesh_pipeline_variant_for_mrt, overlay_pipeline_variant_for_orthographic};
use super::types::BatchedDraw;
use crate::assets::{MaterialPropertyStore, MaterialPropertyValue};
use crate::config::RenderConfig;
use crate::gpu::{NonSkinnedUniformUpload, PipelineVariant};
use nalgebra::Matrix4;

#[test]
fn fill_pbr_host_uniform_reads_merged_color() {
    let mut store = MaterialPropertyStore::new();
    store.set_material(10, 5, MaterialPropertyValue::Float4([0.2, 0.4, 0.6, 1.0]));
    let rc = RenderConfig {
        pbr_host_color_property_id: 5,
        ..RenderConfig::default()
    };
    let draw = BatchedDraw {
        mesh_asset_id: 1,
        mvp: Matrix4::identity(),
        model: Matrix4::identity(),
        material_asset_id: 10,
        pipeline_variant: PipelineVariant::Pbr,
        is_overlay: false,
        stencil_state: None,
        mesh_renderer_property_block_slot0_id: None,
        submesh_index_range: None,
    };
    let mut u = NonSkinnedUniformUpload::new(draw.mvp, draw.model);
    fill_pbr_host_uniform_extras(&mut u, &store, &rc, &draw);
    assert!((u.host_base_color[0] - 0.2).abs() < 1e-5);
    assert!((u.host_base_color[3] - 1.0).abs() < 1e-5);
}

#[test]
fn fill_pbr_host_uniform_sets_albedo_flag_when_main_tex_bound() {
    let mut store = MaterialPropertyStore::new();
    store.set_material(10, 9, MaterialPropertyValue::Texture(1));
    let rc = RenderConfig {
        pbr_bind_host_material_properties: true,
        pbr_bind_host_main_texture: true,
        pbr_host_main_tex_property_id: 9,
        ..RenderConfig::default()
    };
    let draw = BatchedDraw {
        mesh_asset_id: 1,
        mvp: Matrix4::identity(),
        model: Matrix4::identity(),
        material_asset_id: 10,
        pipeline_variant: PipelineVariant::PbrHostAlbedo,
        is_overlay: false,
        stencil_state: None,
        mesh_renderer_property_block_slot0_id: None,
        submesh_index_range: None,
    };
    let mut u = NonSkinnedUniformUpload::new(draw.mvp, draw.model);
    fill_pbr_host_uniform_extras(&mut u, &store, &rc, &draw);
    assert!((u.host_metallic_roughness[3] - 1.0).abs() < 1e-5);
}

#[test]
fn overlay_pipeline_variant_orthographic_maps_normal_debug() {
    let v = overlay_pipeline_variant_for_orthographic(&PipelineVariant::NormalDebug, true);
    assert_eq!(v, PipelineVariant::OverlayNoDepthNormalDebug);
}

#[test]
fn overlay_pipeline_variant_orthographic_preserves_when_false() {
    let v = overlay_pipeline_variant_for_orthographic(&PipelineVariant::NormalDebug, false);
    assert_eq!(v, PipelineVariant::NormalDebug);
}

#[test]
fn overlay_pipeline_variant_orthographic_preserves_stencil_variants() {
    let v =
        overlay_pipeline_variant_for_orthographic(&PipelineVariant::OverlayStencilMaskWrite, true);
    assert_eq!(v, PipelineVariant::OverlayStencilMaskWrite);
}

#[test]
fn mesh_pipeline_variant_mrt_upgrades_when_use_mrt() {
    let v = mesh_pipeline_variant_for_mrt(&PipelineVariant::NormalDebug, true, false, true, false);
    assert_eq!(v, PipelineVariant::NormalDebugMRT);
}

#[test]
fn mesh_pipeline_variant_pbr_upgrades_when_use_pbr() {
    let v = mesh_pipeline_variant_for_mrt(&PipelineVariant::NormalDebug, false, true, true, false);
    assert_eq!(v, PipelineVariant::Pbr);
}

#[test]
fn mesh_pipeline_variant_pbr_ray_query_when_flag_set() {
    let v = mesh_pipeline_variant_for_mrt(&PipelineVariant::NormalDebug, false, true, true, true);
    assert_eq!(v, PipelineVariant::PbrRayQuery);
    let mrt = mesh_pipeline_variant_for_mrt(&PipelineVariant::NormalDebug, true, true, true, true);
    assert_eq!(mrt, PipelineVariant::PbrMRTRayQuery);
}

#[test]
fn mesh_pipeline_variant_fallback_when_no_pbr_scene() {
    let v = mesh_pipeline_variant_for_mrt(&PipelineVariant::Pbr, false, true, false, false);
    assert_eq!(v, PipelineVariant::NormalDebug);
}

#[test]
fn mesh_pipeline_variant_pbr_host_albedo_tracks_pbr_paths() {
    let mrt =
        mesh_pipeline_variant_for_mrt(&PipelineVariant::PbrHostAlbedo, true, true, true, false);
    assert_eq!(mrt, PipelineVariant::PbrMRT);
    let non_mrt =
        mesh_pipeline_variant_for_mrt(&PipelineVariant::PbrHostAlbedo, false, true, true, false);
    assert_eq!(non_mrt, PipelineVariant::Pbr);
    let no_scene =
        mesh_pipeline_variant_for_mrt(&PipelineVariant::PbrHostAlbedo, false, true, false, false);
    assert_eq!(no_scene, PipelineVariant::NormalDebug);
}

#[test]
fn mesh_pipeline_variant_material_downgrades_to_mrt_debug() {
    let v = mesh_pipeline_variant_for_mrt(
        &PipelineVariant::Material { material_id: 1 },
        true,
        false,
        true,
        false,
    );
    assert_eq!(v, PipelineVariant::NormalDebugMRT);
}

#[test]
fn mesh_pipeline_variant_material_preserved_without_mrt() {
    let v = mesh_pipeline_variant_for_mrt(
        &PipelineVariant::Material { material_id: 1 },
        false,
        true,
        true,
        false,
    );
    assert_eq!(v, PipelineVariant::Material { material_id: 1 });
}

#[test]
fn mesh_pipeline_variant_native_ui_unlit_mirrors_material_mrt_rule() {
    let v = mesh_pipeline_variant_for_mrt(
        &PipelineVariant::NativeUiUnlit { material_id: 2 },
        true,
        false,
        true,
        false,
    );
    assert_eq!(v, PipelineVariant::NormalDebugMRT);
    let v2 = mesh_pipeline_variant_for_mrt(
        &PipelineVariant::NativeUiUnlit { material_id: 2 },
        false,
        true,
        true,
        false,
    );
    assert_eq!(v2, PipelineVariant::NativeUiUnlit { material_id: 2 });
}

#[test]
fn mesh_pipeline_variant_native_ui_text_unlit_mirrors_material_mrt_rule() {
    let v = mesh_pipeline_variant_for_mrt(
        &PipelineVariant::NativeUiTextUnlit { material_id: 3 },
        true,
        true,
        true,
        false,
    );
    assert_eq!(v, PipelineVariant::NormalDebugMRT);
}

#[test]
fn material_resource_binds_only_native_ui_and_material_variants() {
    assert!(pipeline_variant_uses_material_resource_binds(
        PipelineVariant::NativeUiUnlit { material_id: 1 }
    ));
    assert!(pipeline_variant_uses_material_resource_binds(
        PipelineVariant::NativeUiUnlitStencil { material_id: 1 }
    ));
    assert!(pipeline_variant_uses_material_resource_binds(
        PipelineVariant::NativeUiTextUnlit { material_id: 1 }
    ));
    assert!(pipeline_variant_uses_material_resource_binds(
        PipelineVariant::NativeUiTextUnlitStencil { material_id: 1 }
    ));
    assert!(pipeline_variant_uses_material_resource_binds(
        PipelineVariant::Material { material_id: 1 }
    ));
}

#[test]
fn material_resource_binds_false_for_global_and_pbr_variants() {
    assert!(!pipeline_variant_uses_material_resource_binds(
        PipelineVariant::Pbr
    ));
    assert!(!pipeline_variant_uses_material_resource_binds(
        PipelineVariant::PbrHostAlbedo
    ));
    assert!(!pipeline_variant_uses_material_resource_binds(
        PipelineVariant::NormalDebug
    ));
    assert!(!pipeline_variant_uses_material_resource_binds(
        PipelineVariant::SkinnedPbr
    ));
    assert!(!pipeline_variant_uses_material_resource_binds(
        PipelineVariant::OverlayStencilContent
    ));
}
