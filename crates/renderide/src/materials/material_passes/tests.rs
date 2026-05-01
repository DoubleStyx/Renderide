//! Cross-submodule integration tests for the `material_passes` module.
//!
//! Tests of behavior that spans `MaterialPipelinePropertyIds`, `MaterialBlendMode`,
//! `MaterialRenderState`, and `MaterialPassDesc` live here; per-submodule unit tests live next to
//! the code they cover.

use super::super::render_state::{
    MaterialCullOverride, MaterialRenderState, material_render_state_for_lookup,
};
use super::*;
use crate::materials::host_data::{
    MaterialDictionary, MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
    PropertyIdRegistry,
};

#[test]
fn resolves_unity_src_dst_blend_properties() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let src = reg.intern("_SrcBlend");
    let dst = reg.intern("_DstBlend");
    store.set_material(43, src, MaterialPropertyValue::Float(1.0));
    store.set_material(43, dst, MaterialPropertyValue::Float(1.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 43,
        mesh_property_block_slot0: None,
    };
    assert_eq!(
        material_blend_mode_for_lookup(&dict, lookup, &ids),
        MaterialBlendMode::UnityBlend { src: 1, dst: 1 }
    );
}

#[test]
fn resolves_xiexe_src_dst_base_blend_properties() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let src = reg.intern("_SrcBlendBase");
    let dst = reg.intern("_DstBlendBase");
    store.set_material(430, src, MaterialPropertyValue::Float(5.0));
    store.set_material(430, dst, MaterialPropertyValue::Float(10.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 430,
        mesh_property_block_slot0: None,
    };
    assert_eq!(
        material_blend_mode_for_lookup(&dict, lookup, &ids),
        MaterialBlendMode::UnityBlend { src: 5, dst: 10 }
    );
}

#[test]
fn resolves_unity_stencil_and_color_mask_properties() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let stencil = reg.intern("_Stencil");
    let comp = reg.intern("_StencilComp");
    let op = reg.intern("_StencilOp");
    let fail = reg.intern("_StencilFail");
    let zfail = reg.intern("_StencilZFail");
    let read = reg.intern("_StencilReadMask");
    let write = reg.intern("_StencilWriteMask");
    let color_mask = reg.intern("_ColorMask");
    store.set_material(44, stencil, MaterialPropertyValue::Float(3.0));
    store.set_material(44, comp, MaterialPropertyValue::Float(8.0));
    store.set_material(44, op, MaterialPropertyValue::Float(2.0));
    store.set_material(44, fail, MaterialPropertyValue::Float(5.0));
    store.set_material(44, zfail, MaterialPropertyValue::Float(3.0));
    store.set_material(44, read, MaterialPropertyValue::Float(127.0));
    store.set_material(44, write, MaterialPropertyValue::Float(63.0));
    store.set_material(44, color_mask, MaterialPropertyValue::Float(0.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 44,
        mesh_property_block_slot0: None,
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert!(state.stencil.enabled);
    assert_eq!(state.stencil_reference(), 3);
    assert_eq!(state.stencil.compare, 8);
    assert_eq!(state.stencil.pass_op, 2);
    assert_eq!(state.stencil.fail_op, 5);
    assert_eq!(state.stencil.depth_fail_op, 3);
    assert_eq!(state.stencil.read_mask, 127);
    assert_eq!(state.stencil.write_mask, 63);
    assert_eq!(
        state.color_writes(wgpu::ColorWrites::ALL),
        wgpu::ColorWrites::empty()
    );
    assert_eq!(
        state.stencil_state().front.pass_op,
        wgpu::StencilOperation::Replace
    );
    assert_eq!(
        state.stencil_state().front.fail_op,
        wgpu::StencilOperation::Invert
    );
    assert_eq!(
        state.stencil_state().front.depth_fail_op,
        wgpu::StencilOperation::IncrementClamp
    );
}

#[test]
fn property_block_overrides_stencil_reference() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let stencil = reg.intern("_Stencil");
    store.set_material(45, stencil, MaterialPropertyValue::Float(1.0));
    store.set_property_block(450, stencil, MaterialPropertyValue::Float(5.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 45,
        mesh_property_block_slot0: Some(450),
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(state.stencil_reference(), 5);
}

#[test]
fn stencil_comp_zero_disables_stencil_state() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let stencil = reg.intern("_Stencil");
    let comp = reg.intern("_StencilComp");
    store.set_material(46, stencil, MaterialPropertyValue::Float(7.0));
    store.set_material(46, comp, MaterialPropertyValue::Float(0.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 46,
        mesh_property_block_slot0: None,
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert!(!state.stencil.enabled);
    assert_eq!(state.stencil_state(), wgpu::StencilState::default());
}

#[test]
fn zwrite_property_overrides_pass_depth_write() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let zwrite = reg.intern("_ZWrite");
    store.set_material(47, zwrite, MaterialPropertyValue::Float(0.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 47,
        mesh_property_block_slot0: None,
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(state.depth_write, Some(false));
    assert!(!state.depth_write(true));
    assert!(!state.depth_write(false));

    store.set_property_block(470, zwrite, MaterialPropertyValue::Float(1.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 47,
        mesh_property_block_slot0: Some(470),
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(state.depth_write, Some(true));
    assert!(state.depth_write(false));
}

#[test]
fn ztest_property_overrides_pass_depth_compare_for_reverse_z() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let ztest = reg.intern("_ZTest");
    // FrooxEngine `ZTest.Always = 6` inverts to wgpu `Always` under reverse-Z.
    store.set_material(48, ztest, MaterialPropertyValue::Float(6.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 48,
        mesh_property_block_slot0: None,
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(state.depth_compare, Some(6));
    assert_eq!(
        state.depth_compare(wgpu::CompareFunction::GreaterEqual),
        wgpu::CompareFunction::Always
    );

    // FrooxEngine `ZTest.LessOrEqual = 2` inverts to wgpu `GreaterEqual` under reverse-Z.
    store.set_property_block(480, ztest, MaterialPropertyValue::Float(2.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 48,
        mesh_property_block_slot0: Some(480),
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(
        state.depth_compare(wgpu::CompareFunction::Always),
        wgpu::CompareFunction::GreaterEqual
    );
}

#[test]
fn offset_properties_override_pass_depth_bias_for_reverse_z() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let factor = reg.intern("_OffsetFactor");
    let units = reg.intern("_OffsetUnits");
    store.set_material(49, factor, MaterialPropertyValue::Float(-1.0));
    store.set_material(49, units, MaterialPropertyValue::Float(-2.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 49,
        mesh_property_block_slot0: None,
    };

    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(
        state
            .depth_offset
            .map(super::super::render_state::MaterialDepthOffsetState::factor),
        Some(-1.0)
    );
    assert_eq!(
        state
            .depth_offset
            .map(super::super::render_state::MaterialDepthOffsetState::units),
        Some(-2)
    );
    let bias = state.depth_bias(7, 0.25);
    assert_eq!(bias.constant, 2);
    assert_eq!(bias.slope_scale, 1.0);
    assert_eq!(bias.clamp, 0.0);

    store.set_property_block(490, units, MaterialPropertyValue::Float(3.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 49,
        mesh_property_block_slot0: Some(490),
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    let bias = state.depth_bias(7, 0.25);
    assert_eq!(bias.constant, -3);
    assert_eq!(bias.slope_scale, 1.0);
}

#[test]
fn forward_pass_uses_unity_separate_alpha_blend() {
    let pass = MaterialPassDesc {
        material_state: MaterialPassState::Forward,
        ..default_pass(DefaultPassParams {
            use_alpha_blending: false,
            depth_write: true,
        })
    };

    let materialized =
        materialized_pass_for_blend_mode(&pass, MaterialBlendMode::UnityBlend { src: 5, dst: 10 });
    let blend = materialized.blend.expect("alpha blend");

    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    assert_eq!(blend.color.operation, wgpu::BlendOperation::Add);
    assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Max);
}

#[test]
fn overlay_pass_uses_unity_rgb_blend_and_keeps_alpha_max() {
    let pass = pass_from_kind(PassKind::OverlayFront, "fs_overlay");
    let materialized =
        materialized_pass_for_blend_mode(&pass, MaterialBlendMode::UnityBlend { src: 5, dst: 10 });
    let blend = materialized.blend.expect("overlay blend");

    assert_eq!(materialized.material_state, MaterialPassState::Overlay);
    assert!(materialized.depth_write);
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    assert_eq!(blend.color.operation, wgpu::BlendOperation::Add);
    assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Max);
}

#[test]
fn forward_transparent_defaults_to_unity_alpha_blend() {
    let pass = pass_from_kind(PassKind::ForwardTransparent, "fs_forward_base");
    let blend = pass.blend.expect("transparent default blend");

    assert!(!pass.depth_write);
    assert_eq!(pass.cull_mode, None);
    assert_eq!(pass.write_mask, wgpu::ColorWrites::ALL);
    assert_eq!(pass.material_state, MaterialPassState::Forward);
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    assert_eq!(blend.color.operation, wgpu::BlendOperation::Add);
    assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Max);
}

#[test]
fn overlay_pass_preserves_explicit_blend_one_zero_for_alpha_max() {
    let pass = pass_from_kind(PassKind::OverlayBehind, "fs_overlay");
    let materialized =
        materialized_pass_for_blend_mode(&pass, MaterialBlendMode::UnityBlend { src: 1, dst: 0 });
    let blend = materialized.blend.expect("overlay blend");

    assert_eq!(materialized.depth_compare, wgpu::CompareFunction::Less);
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::Zero);
    assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Max);
}

#[test]
fn overlay_always_pass_matches_fixed_overlay_shader_state() {
    let pass = pass_from_kind(PassKind::OverlayAlways, "fs_main");
    let blend = pass.blend.expect("overlay blend");

    assert_eq!(pass.name, "overlay_always");
    assert_eq!(pass.material_state, MaterialPassState::Static);
    assert_eq!(pass.depth_compare, wgpu::CompareFunction::Always);
    assert!(!pass.depth_write);
    assert_eq!(pass.write_mask, wgpu::ColorWrites::ALL);
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
}

#[test]
fn transparent_forward_host_blend_override_materializes() {
    let pass = pass_from_kind(PassKind::ForwardTransparent, "fs_forward_base");

    let stem_default = materialized_pass_for_blend_mode(&pass, MaterialBlendMode::StemDefault);
    assert!(stem_default.blend.is_some());
    assert!(!stem_default.depth_write);
    assert_eq!(stem_default.write_mask, wgpu::ColorWrites::ALL);

    let opaque = materialized_pass_for_blend_mode(&pass, MaterialBlendMode::Opaque);
    assert_eq!(opaque.blend, None);
    assert!(opaque.depth_write);
    assert_eq!(opaque.write_mask, wgpu::ColorWrites::COLOR);
}

#[test]
fn transparent_fixed_cull_ignores_host_cull() {
    let state = MaterialRenderState {
        cull_override: MaterialCullOverride::Off,
        ..MaterialRenderState::default()
    };

    assert_eq!(
        pass_from_kind(PassKind::ForwardTransparent, "fs_forward_base").resolved_cull_mode(state),
        None
    );
    assert_eq!(
        pass_from_kind(PassKind::ForwardTransparentCullFront, "fs_back_faces")
            .resolved_cull_mode(state),
        Some(wgpu::Face::Front)
    );
    assert_eq!(
        pass_from_kind(PassKind::ForwardTransparentCullBack, "fs_front_faces")
            .resolved_cull_mode(state),
        Some(wgpu::Face::Back)
    );
}

#[test]
fn cull_property_resolves_off_front_back() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let cull = reg.intern("_Cull");

    store.set_material(50, cull, MaterialPropertyValue::Float(0.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 50,
        mesh_property_block_slot0: None,
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(state.cull_override, MaterialCullOverride::Off);
    assert_eq!(state.resolved_cull_mode(Some(wgpu::Face::Back)), None);

    store.set_material(50, cull, MaterialPropertyValue::Float(1.0));
    let dict = MaterialDictionary::new(&store);
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(state.cull_override, MaterialCullOverride::Front);
    assert_eq!(
        state.resolved_cull_mode(Some(wgpu::Face::Back)),
        Some(wgpu::Face::Front)
    );

    store.set_material(50, cull, MaterialPropertyValue::Float(2.0));
    let dict = MaterialDictionary::new(&store);
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(state.cull_override, MaterialCullOverride::Back);
    assert_eq!(
        state.resolved_cull_mode(Some(wgpu::Face::Back)),
        Some(wgpu::Face::Back)
    );
}

#[test]
fn property_block_overrides_cull() {
    let reg = PropertyIdRegistry::new();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut store = MaterialPropertyStore::new();
    let cull = reg.intern("_Cull");
    store.set_material(52, cull, MaterialPropertyValue::Float(2.0));
    store.set_property_block(520, cull, MaterialPropertyValue::Float(0.0));
    let dict = MaterialDictionary::new(&store);
    let lookup = MaterialPropertyLookupIds {
        material_asset_id: 52,
        mesh_property_block_slot0: Some(520),
    };
    let state = material_render_state_for_lookup(&dict, lookup, &ids);
    assert_eq!(state.cull_override, MaterialCullOverride::Off);
}

#[test]
fn default_pass_opaque_culls_back_faces() {
    let pass = default_pass(DefaultPassParams {
        use_alpha_blending: false,
        depth_write: true,
    });
    assert_eq!(pass.cull_mode, Some(wgpu::Face::Back));
}

#[test]
fn default_pass_alpha_blended_disables_culling() {
    let pass = default_pass(DefaultPassParams {
        use_alpha_blending: true,
        depth_write: false,
    });
    assert_eq!(pass.cull_mode, None);
}

#[test]
fn unspecified_cull_preserves_opaque_back_face_default() {
    let state = MaterialRenderState::default();
    assert_eq!(state.cull_override, MaterialCullOverride::Unspecified);
    assert_eq!(
        state.resolved_cull_mode(
            default_pass(DefaultPassParams {
                use_alpha_blending: false,
                depth_write: true,
            })
            .cull_mode,
        ),
        Some(wgpu::Face::Back)
    );
}
