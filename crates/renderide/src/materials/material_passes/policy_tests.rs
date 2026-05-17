//! No-GPU coverage for pass-scoped material render-state policy.

use super::super::render_state::{
    MaterialCullOverride, MaterialDepthCompareDomain, MaterialDepthOffsetState,
    MaterialRenderState, MaterialStencilState,
};
use super::*;

/// Builds a render-state override set that exercises every pass policy field.
fn override_state(depth_write: bool) -> MaterialRenderState {
    MaterialRenderState {
        stencil: MaterialStencilState {
            enabled: true,
            reference: 9,
            compare: 3,
            pass_op: 2,
            fail_op: 1,
            depth_fail_op: 4,
            read_mask: 0xf0,
            write_mask: 0x0f,
        },
        color_mask: Some(15),
        depth_write: Some(depth_write),
        depth_compare: Some(6),
        depth_offset: MaterialDepthOffsetState::new(2.0, 3),
        cull_override: MaterialCullOverride::Off,
    }
}

/// Asserts the resolved render-state fields most sensitive to pass-policy regressions.
fn assert_resolved_pass(
    pass: MaterialPassDesc,
    state: MaterialRenderState,
    color_writes: wgpu::ColorWrites,
    depth_write: bool,
    depth_compare: wgpu::CompareFunction,
    cull_mode: Option<wgpu::Face>,
) {
    assert_eq!(pass.resolved_color_writes(state), color_writes);
    assert_eq!(pass.resolved_depth_write(state), depth_write);
    assert_eq!(pass.resolved_depth_compare(state), depth_compare);
    assert_eq!(pass.resolved_cull_mode(state), cull_mode);
    assert_eq!(
        pass.resolved_stencil_state(state).front.pass_op,
        wgpu::StencilOperation::Replace
    );
    let bias = pass.resolved_depth_bias(state);
    assert_eq!(bias.constant, -3);
    assert_eq!(bias.slope_scale, -2.0);
}

/// Verifies each pass descriptor admits only the material overrides listed in the policy table.
#[test]
fn pass_policy_resolves_expected_material_overrides_by_descriptor() {
    let disabled_depth = override_state(false);
    let enabled_depth = override_state(true);

    assert_resolved_pass(
        depth_prepass("fs_depth_only"),
        disabled_depth,
        COLOR_WRITES_NONE,
        true,
        wgpu::CompareFunction::Always,
        None,
    );
    assert_resolved_pass(
        stencil_pass("fs_stencil"),
        enabled_depth,
        wgpu::ColorWrites::ALL,
        true,
        wgpu::CompareFunction::Always,
        None,
    );
    assert_resolved_pass(
        forward_pass("fs_main"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        None,
    );
    assert_resolved_pass(
        forward_filter_pass("fs_main"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        None,
    );
    assert_resolved_pass(
        forward_two_sided_pass("fs_main"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        None,
    );
    assert_resolved_pass(
        forward_transparent_cull_front_pass("fs_back_faces"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        Some(wgpu::Face::Front),
    );
    assert_resolved_pass(
        outline_pass("fs_outline"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        Some(wgpu::Face::Front),
    );
    assert_resolved_pass(
        overlay_behind_pass("fs_overlay"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Less,
        None,
    );
    let overlay_always = overlay_always_pass("fs_overlay");
    assert_eq!(
        overlay_always.resolved_color_writes(enabled_depth),
        wgpu::ColorWrites::ALL
    );
    assert!(!overlay_always.resolved_depth_write(enabled_depth));
    assert_eq!(
        overlay_always.resolved_depth_compare(enabled_depth),
        wgpu::CompareFunction::Always
    );
    assert_eq!(
        overlay_always.resolved_cull_mode(enabled_depth),
        Some(wgpu::Face::Back)
    );
    assert_eq!(
        overlay_always.resolved_stencil_state(enabled_depth),
        wgpu::StencilState::default()
    );
    assert_eq!(
        overlay_always.resolved_depth_bias(enabled_depth),
        wgpu::DepthBiasState::default()
    );

    let fur_blend = forward_alpha_blend_zwrite_pass("fs_fur");
    assert_eq!(
        fur_blend.resolved_color_writes(disabled_depth),
        wgpu::ColorWrites::ALL
    );
    assert!(fur_blend.resolved_depth_write(disabled_depth));
    assert_eq!(
        fur_blend.resolved_depth_compare(disabled_depth),
        crate::gpu::MAIN_FORWARD_DEPTH_COMPARE
    );
}

/// Verifies fixed transparent RGB passes preserve Unity-authored state even when host overrides exist.
#[test]
fn transparent_rgb_pass_ignores_material_render_state_overrides() {
    let pass = transparent_rgb_pass("fs_circle");
    let override_state = override_state(true);

    assert_eq!(
        pass.resolved_color_writes(override_state),
        wgpu::ColorWrites::COLOR
    );
    assert!(!pass.resolved_depth_write(override_state));
    assert_eq!(
        pass.resolved_depth_compare(override_state),
        crate::gpu::MAIN_FORWARD_DEPTH_COMPARE
    );
    assert_eq!(pass.resolved_cull_mode(override_state), None);
    assert_eq!(
        pass.resolved_stencil_state(override_state),
        wgpu::StencilState::default()
    );
    assert_eq!(
        pass.resolved_depth_bias(override_state),
        wgpu::DepthBiasState::default()
    );

    let blend = pass
        .blend
        .expect("transparent RGB pass should have static alpha blending");
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    assert_eq!(pass.material_state, MaterialPassState::Static);
}

/// Verifies volume passes keep cull/depth fixed while accepting stencil state.
#[test]
fn volume_front_pass_policy_preserves_authored_volume_state() {
    let state = override_state(true);
    let pass = volume_front_pass("fs_volume");

    assert_eq!(pass.resolved_color_writes(state), wgpu::ColorWrites::ALL);
    assert!(!pass.resolved_depth_write(state));
    assert_eq!(
        pass.resolved_depth_compare(state),
        wgpu::CompareFunction::Always
    );
    assert_eq!(pass.resolved_cull_mode(state), Some(wgpu::Face::Front));
    assert_eq!(
        pass.resolved_stencil_state(state).front.pass_op,
        wgpu::StencilOperation::Replace
    );
    assert_eq!(
        pass.resolved_depth_bias(state),
        wgpu::DepthBiasState::default()
    );
}

/// Verifies volume shaders declare the fixed front-culled volume pass.
#[test]
fn volume_stems_use_volume_front_pass() {
    for stem in ["fogboxvolume_default", "volumeunlit_default"] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(passes.len(), 1, "{stem}");
        assert_eq!(passes[0].name, "volume_front", "{stem}");
        assert_eq!(
            passes[0].depth_compare,
            wgpu::CompareFunction::Always,
            "{stem}"
        );
        assert!(!passes[0].depth_write, "{stem}");
        assert_eq!(passes[0].cull_mode, Some(wgpu::Face::Front), "{stem}");
        assert_eq!(passes[0].write_mask, wgpu::ColorWrites::ALL, "{stem}");
    }
}

/// Asserts that a UI unlit stem uses material-controlled filter pass state.
fn assert_ui_unlit_filter_pass(stem: &str) {
    let passes = crate::embedded_shaders::embedded_target_passes(stem);
    assert_eq!(passes.len(), 1, "{stem}");
    assert_eq!(passes[0].name, "forward_filter", "{stem}");
    assert_eq!(
        passes[0].material_state,
        MaterialPassState::Filter,
        "{stem}"
    );
    assert_eq!(
        passes[0].depth_compare_domain,
        MaterialDepthCompareDomain::FrooxZTest,
        "{stem}"
    );

    let materialized = materialized_embedded_pass_for_blend_mode(
        stem,
        &passes[0],
        MaterialBlendMode::UnityBlend { src: 1, dst: 0 },
    );
    let blend = materialized.blend.expect(stem);
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One, "{stem}");
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::Zero, "{stem}");
    assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One, "{stem}");
    assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::One, "{stem}");
    assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Max, "{stem}");

    let zwrite_off = MaterialRenderState {
        depth_write: Some(false),
        ..MaterialRenderState::default()
    };
    assert!(!materialized.resolved_depth_write(zwrite_off), "{stem}");

    let less_or_equal_ztest = MaterialRenderState {
        depth_compare: Some(2),
        ..MaterialRenderState::default()
    };
    assert_eq!(
        materialized.resolved_depth_compare(less_or_equal_ztest),
        wgpu::CompareFunction::GreaterEqual,
        "{stem}"
    );

    let always_ztest = MaterialRenderState {
        depth_compare: Some(ZTEST_ALWAYS),
        ..MaterialRenderState::default()
    };
    assert_eq!(
        materialized.resolved_depth_compare(always_ztest),
        wgpu::CompareFunction::Always,
        "{stem}"
    );

    let depth_offset = MaterialRenderState {
        depth_offset: MaterialDepthOffsetState::new(1.0, 100),
        ..MaterialRenderState::default()
    };
    let bias = materialized.resolved_depth_bias(depth_offset);
    assert_eq!(bias.constant, -100, "{stem}");
    assert_eq!(bias.slope_scale, -1.0, "{stem}");
}

/// Verifies UI unlit stems keep Unity-style filter render state.
#[test]
fn ui_unlit_stems_use_filter_pass_material_state() {
    for stem in [
        "ui_unlit_default",
        "ui_textunlit_default",
        "ui_circlesegment_default",
    ] {
        assert_ui_unlit_filter_pass(stem);
    }
}

/// Verifies opaque PBS DualSided stems use their source-authored material cull fallback.
#[test]
fn pbs_dualsided_opaque_stems_apply_material_cull_overrides() {
    for stem in ["pbsdualsided_default", "pbsdualsidedspecular_default"] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(passes.len(), 1, "{stem} should declare one forward pass");
        assert_eq!(passes[0].name, "forward_two_sided", "{stem}");
        assert_eq!(passes[0].cull_mode, None, "{stem}");

        for (cull_override, expected_cull) in [
            (MaterialCullOverride::Front, Some(wgpu::Face::Front)),
            (MaterialCullOverride::Back, Some(wgpu::Face::Back)),
            (MaterialCullOverride::Off, None),
        ] {
            let state = MaterialRenderState {
                cull_override,
                ..MaterialRenderState::default()
            };
            assert_eq!(
                passes[0].resolved_cull_mode(state),
                expected_cull,
                "{stem} must apply host {cull_override:?} over authored Cull Off"
            );
        }
    }
}

/// Verifies PBSRim transparent zwrite variants preserve their depth-only stem before transparent color.
#[test]
fn pbsrim_zwrite_stems_keep_depth_prepass_before_transparent_forward() {
    for stem in [
        "pbsrimtransparentzwrite_default",
        "pbsrimtransparentzwritespecular_default",
    ] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(passes.len(), 2, "{stem} should declare two passes");
        assert_eq!(passes[0].name, "depth_prepass");
        assert_eq!(passes[1].name, "forward_transparent_cull_back");
        assert_eq!(passes[1].cull_mode, Some(wgpu::Face::Back), "{stem}");

        let state = MaterialRenderState {
            color_mask: Some(15),
            depth_write: Some(false),
            ..MaterialRenderState::default()
        };
        let blend = MaterialBlendMode::UnityBlend { src: 1, dst: 10 };
        let depth_prepass = materialized_pass_for_blend_mode(&passes[0], blend);
        let forward = materialized_pass_for_blend_mode(&passes[1], blend);

        assert!(depth_prepass.resolved_depth_write(state), "{stem}");
        assert_eq!(
            depth_prepass.resolved_color_writes(state),
            COLOR_WRITES_NONE,
            "{stem}"
        );
        assert!(!forward.resolved_depth_write(state), "{stem}");
        assert!(forward.blend.is_some(), "{stem}");
    }
}

/// Asserts that a shader stem declares one premultiplied transparent pass.
fn assert_one_transparent_forward_pass(stem: &str) {
    let passes = crate::embedded_shaders::embedded_target_passes(stem);
    assert_eq!(
        passes.len(),
        1,
        "{stem} should declare one transparent forward pass"
    );
    assert_eq!(passes[0].name, "forward_transparent", "{stem}");
    assert!(!passes[0].depth_write, "{stem}");
    assert!(passes[0].blend.is_some(), "{stem}");
    assert_eq!(
        passes[0].material_state,
        MaterialPassState::TransparentForward,
        "{stem}"
    );
    let opaque = materialized_pass_for_blend_mode(&passes[0], MaterialBlendMode::Opaque);
    let blend = opaque.blend.expect(stem);
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One, "{stem}");
    assert_eq!(
        blend.color.dst_factor,
        wgpu::BlendFactor::OneMinusSrcAlpha,
        "{stem}"
    );
    assert!(!opaque.depth_write, "{stem}");
    assert_eq!(opaque.write_mask, wgpu::ColorWrites::ALL, "{stem}");
}

/// Asserts that a shader stem keeps its depth prepass before back-culled transparent color output.
fn assert_depth_prepass_before_back_culled_transparent_forward(stem: &str) {
    let passes = crate::embedded_shaders::embedded_target_passes(stem);
    assert_eq!(
        passes.len(),
        2,
        "{stem} should declare depth prepass then transparent forward pass"
    );
    assert_eq!(passes[0].name, "depth_prepass", "{stem}");
    assert!(passes[0].depth_write, "{stem}");
    assert_eq!(passes[0].write_mask, COLOR_WRITES_NONE, "{stem}");
    assert_eq!(passes[1].name, "forward_transparent_cull_back", "{stem}");
    assert_eq!(passes[1].cull_mode, Some(wgpu::Face::Back), "{stem}");
    assert!(!passes[1].depth_write, "{stem}");
    assert!(passes[1].blend.is_some(), "{stem}");
    assert_eq!(
        passes[1].material_state,
        MaterialPassState::TransparentForward,
        "{stem}"
    );
    let opaque = materialized_pass_for_blend_mode(&passes[1], MaterialBlendMode::Opaque);
    let blend = opaque.blend.expect(stem);
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One, "{stem}");
    assert_eq!(
        blend.color.dst_factor,
        wgpu::BlendFactor::OneMinusSrcAlpha,
        "{stem}"
    );
    assert!(!opaque.depth_write, "{stem}");
    assert_eq!(opaque.write_mask, wgpu::ColorWrites::ALL, "{stem}");
}

/// Asserts that a shader stem declares one back-face-culled transparent pass.
fn assert_one_back_face_culled_transparent_pass(stem: &str) {
    let passes = crate::embedded_shaders::embedded_target_passes(stem);
    assert_eq!(
        passes.len(),
        1,
        "{stem} should declare one back-face-culled transparent forward pass"
    );
    assert_eq!(passes[0].name, "forward_transparent_cull_back", "{stem}");
    assert_eq!(passes[0].cull_mode, Some(wgpu::Face::Back), "{stem}");
    assert!(!passes[0].depth_write, "{stem}");
    assert!(passes[0].blend.is_some(), "{stem}");
    assert_eq!(
        passes[0].material_state,
        MaterialPassState::TransparentForward,
        "{stem}"
    );
    let opaque = materialized_pass_for_blend_mode(&passes[0], MaterialBlendMode::Opaque);
    let blend = opaque.blend.expect(stem);
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One, "{stem}");
    assert_eq!(
        blend.color.dst_factor,
        wgpu::BlendFactor::OneMinusSrcAlpha,
        "{stem}"
    );
    assert!(!opaque.depth_write, "{stem}");
    assert_eq!(opaque.write_mask, wgpu::ColorWrites::ALL, "{stem}");
}

/// Asserts that a shader stem declares the back-face then front-face transparent pass pair.
fn assert_dualsided_transparent_pass_pair(stem: &str) {
    let passes = crate::embedded_shaders::embedded_target_passes(stem);
    assert_eq!(
        passes.len(),
        2,
        "{stem} should declare back-face then front-face transparent passes"
    );
    assert_eq!(passes[0].name, "forward_transparent_cull_front", "{stem}");
    assert_eq!(passes[0].cull_mode, Some(wgpu::Face::Front), "{stem}");
    assert!(passes[0].blend.is_some(), "{stem}");
    assert_eq!(
        passes[0].material_state,
        MaterialPassState::TransparentForward,
        "{stem}"
    );
    assert_eq!(passes[1].name, "forward_transparent_cull_back", "{stem}");
    assert_eq!(passes[1].cull_mode, Some(wgpu::Face::Back), "{stem}");
    assert!(passes[1].blend.is_some(), "{stem}");
    assert_eq!(
        passes[1].material_state,
        MaterialPassState::TransparentForward,
        "{stem}"
    );
    for pass in passes {
        let opaque = materialized_pass_for_blend_mode(pass, MaterialBlendMode::Opaque);
        let blend = opaque.blend.expect(stem);
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One, "{stem}");
        assert_eq!(
            blend.color.dst_factor,
            wgpu::BlendFactor::OneMinusSrcAlpha,
            "{stem}"
        );
        assert!(!opaque.depth_write, "{stem}");
        assert_eq!(opaque.write_mask, wgpu::ColorWrites::ALL, "{stem}");
    }
}

/// Verifies all PBS transparent stems declare transparent defaults instead of opaque forward aliases.
#[test]
fn pbs_transparent_stems_keep_transparent_pass_defaults() {
    for stem in [
        "pbsdisplacetransparent_default",
        "pbsdisplacespeculartransparent_default",
        "pbsdistancelerptransparent_default",
        "pbsdistancelerpspeculartransparent_default",
        "pbsslicetransparent_default",
        "pbsslicetransparentspecular_default",
        "pbstriplanartransparent_default",
        "pbstriplanartransparentspecular_default",
    ] {
        assert_one_transparent_forward_pass(stem);
    }

    for stem in [
        "pbsrimtransparentzwrite_default",
        "pbsrimtransparentzwritespecular_default",
    ] {
        assert_depth_prepass_before_back_culled_transparent_forward(stem);
    }

    for stem in [
        "pbsrimtransparent_default",
        "pbsrimtransparentspecular_default",
        "pbsvertexcolortransparent_default",
        "pbsvertexcolortransparentspecular_default",
    ] {
        assert_one_back_face_culled_transparent_pass(stem);
    }

    for stem in [
        "pbsdualsidedtransparent_default",
        "pbsdualsidedtransparentspecular_default",
    ] {
        assert_dualsided_transparent_pass_pair(stem);
    }
}

/// Verifies outlined XSToon stems keep outline before forward rendering.
#[test]
fn xstoon_outlined_stems_keep_outline_before_forward() {
    for stem in [
        "xstoon2.0-outlined_default",
        "xstoon2.0_outlined_default",
        "xstoon2.0-dithered-outlined_default",
        "xstoon2.0-cutouta2c-outlined_default",
    ] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(passes.len(), 2, "{stem} should declare outline + forward");
        assert_eq!(passes[0].name, "outline", "{stem}");
        assert_eq!(passes[1].name, "forward", "{stem}");
    }
}

/// Verifies opaque XSToon stems keep a single forward pass.
#[test]
fn xstoon_opaque_stems_keep_single_forward_pass() {
    for stem in [
        "xstoon2.0_default",
        "xstoon2.0-cutout_default",
        "xstoon2.0-cutouta2c_default",
        "xstoon2.0-cutouta2cmasked_default",
        "xstoon2.0-dithered_default",
    ] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(
            passes.len(),
            1,
            "{stem} should declare a single forward pass"
        );
        assert_eq!(passes[0].name, "forward", "{stem}");
    }
}

/// Verifies XSToon fade uses its source-authored alpha blend state.
#[test]
fn xstoon_fade_uses_source_alpha_blend_pass_state() {
    let fade_passes = crate::embedded_shaders::embedded_target_passes("xstoon2.0-fade_default");
    assert_eq!(fade_passes.len(), 1, "xstoon2.0-fade_default");
    assert_eq!(fade_passes[0].name, "forward_alpha_blend");
    assert_eq!(fade_passes[0].cull_mode, Some(wgpu::Face::Back));
    assert!(!fade_passes[0].depth_write);
    let fade_blend = fade_passes[0].blend.expect("xstoon2.0-fade blend");
    assert_eq!(fade_blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(
        fade_blend.color.dst_factor,
        wgpu::BlendFactor::OneMinusSrcAlpha
    );
    assert_eq!(fade_blend.alpha.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(
        fade_blend.alpha.dst_factor,
        wgpu::BlendFactor::OneMinusSrcAlpha
    );
}

/// Verifies XSToon transparent uses its source-authored premultiplied blend state.
#[test]
fn xstoon_transparent_uses_source_premultiplied_pass_state() {
    let transparent_passes =
        crate::embedded_shaders::embedded_target_passes("xstoon2.0-transparent_default");
    assert_eq!(transparent_passes.len(), 1, "xstoon2.0-transparent_default");
    assert_eq!(
        transparent_passes[0].name,
        "forward_premultiplied_transparent"
    );
    assert_eq!(transparent_passes[0].cull_mode, Some(wgpu::Face::Back));
    assert!(!transparent_passes[0].depth_write);
    let transparent_blend = transparent_passes[0]
        .blend
        .expect("xstoon2.0-transparent blend");
    assert_eq!(transparent_blend.color.src_factor, wgpu::BlendFactor::One);
    assert_eq!(
        transparent_blend.color.dst_factor,
        wgpu::BlendFactor::OneMinusSrcAlpha
    );
}

/// Verifies XSToon stenciler keeps its source-authored color and depth state.
#[test]
fn xstoon_stenciler_uses_source_stencil_pass_state() {
    let stencil_passes = crate::embedded_shaders::embedded_target_passes("xstoonstenciler_default");
    assert_eq!(stencil_passes.len(), 1, "xstoonstenciler_default");
    assert_eq!(stencil_passes[0].name, "stencil", "xstoonstenciler_default");
    assert_eq!(stencil_passes[0].cull_mode, Some(wgpu::Face::Front));
    assert_eq!(stencil_passes[0].write_mask, wgpu::ColorWrites::ALL);
    assert_eq!(
        stencil_passes[0].depth_compare_domain,
        MaterialDepthCompareDomain::UnityCompareFunction
    );
    assert!(!stencil_passes[0].depth_write);
}

/// Verifies XSToon alpha-to-coverage variants request the matching pipeline state.
#[test]
fn xstoon_a2c_stems_enable_alpha_to_coverage() {
    for stem in [
        "xstoon2.0-cutouta2c_default",
        "xstoon2.0-cutouta2cmasked_default",
    ] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(
            passes.len(),
            1,
            "{stem} should declare a single forward pass"
        );
        assert!(passes[0].alpha_to_coverage, "{stem}");
    }

    let outlined =
        crate::embedded_shaders::embedded_target_passes("xstoon2.0-cutouta2c-outlined_default");
    assert_eq!(outlined.len(), 2, "xstoon2.0-cutouta2c-outlined_default");
    assert!(
        outlined.iter().all(|pass| pass.alpha_to_coverage),
        "xstoon2.0-cutouta2c-outlined_default"
    );

    for stem in [
        "xstoon2.0-cutout_default",
        "xstoon2.0-dithered_default",
        "xstoon2.0-dithered-outlined_default",
    ] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert!(passes.iter().all(|pass| !pass.alpha_to_coverage), "{stem}");
    }
}
