//! No-GPU coverage for pass-scoped material render-state policy.

use super::super::render_state::{
    MaterialCullOverride, MaterialDepthOffsetState, MaterialRenderState, MaterialStencilState,
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

/// Verifies each pass kind admits only the material overrides listed in the policy table.
#[test]
fn pass_policy_resolves_expected_material_overrides_by_kind() {
    let disabled_depth = override_state(false);
    let enabled_depth = override_state(true);

    assert_resolved_pass(
        pass_from_kind(PassKind::DepthPrepass, "fs_depth_only"),
        disabled_depth,
        COLOR_WRITES_NONE,
        true,
        wgpu::CompareFunction::Always,
        None,
    );
    assert_resolved_pass(
        pass_from_kind(PassKind::Stencil, "fs_stencil"),
        enabled_depth,
        COLOR_WRITES_NONE,
        true,
        wgpu::CompareFunction::Always,
        None,
    );
    assert_resolved_pass(
        pass_from_kind(PassKind::Forward, "fs_main"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        None,
    );
    assert_resolved_pass(
        pass_from_kind(PassKind::ForwardTwoSided, "fs_main"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        None,
    );
    assert_resolved_pass(
        pass_from_kind(PassKind::ForwardTransparentCullFront, "fs_back_faces"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        Some(wgpu::Face::Front),
    );
    assert_resolved_pass(
        pass_from_kind(PassKind::ForwardTransparentVolume, "fs_volume_fog"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        Some(wgpu::Face::Front),
    );
    assert_resolved_pass(
        pass_from_kind(PassKind::Outline, "fs_outline"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Always,
        Some(wgpu::Face::Front),
    );
    assert_resolved_pass(
        pass_from_kind(PassKind::OverlayBehind, "fs_overlay"),
        disabled_depth,
        wgpu::ColorWrites::ALL,
        false,
        wgpu::CompareFunction::Less,
        None,
    );
    let overlay_always = pass_from_kind(PassKind::OverlayAlways, "fs_overlay");
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
}

/// Verifies fixed transparent RGB passes preserve Unity-authored state even when host overrides exist.
#[test]
fn transparent_rgb_pass_ignores_material_render_state_overrides() {
    let pass = pass_from_kind(PassKind::TransparentRgb, "fs_circle");
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

/// Verifies PBSRim transparent zwrite variants preserve their depth-only stem before color.
#[test]
fn pbsrim_zwrite_stems_keep_depth_prepass_before_forward() {
    for stem in [
        "pbsrimtransparentzwrite_default",
        "pbsrimtransparentzwritespecular_default",
    ] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(passes.len(), 2, "{stem} should declare two passes");
        assert_eq!(passes[0].name, "depth_prepass");
        assert_eq!(passes[1].name, "forward");

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

/// Verifies opaque PBS DualSided stems preserve authored Cull Off regardless of host `_Cull`.
#[test]
fn pbs_dualsided_opaque_stems_preserve_authored_cull_off() {
    for stem in ["pbsdualsided_default", "pbsdualsidedspecular_default"] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(passes.len(), 1, "{stem} should declare one forward pass");
        assert_eq!(passes[0].name, "forward_two_sided", "{stem}");
        assert_eq!(passes[0].cull_mode, None, "{stem}");

        for cull_override in [
            MaterialCullOverride::Front,
            MaterialCullOverride::Back,
            MaterialCullOverride::Off,
        ] {
            let state = MaterialRenderState {
                cull_override,
                ..MaterialRenderState::default()
            };
            assert_eq!(
                passes[0].resolved_cull_mode(state),
                None,
                "{stem} must keep authored Cull Off when host sends {cull_override:?}"
            );
        }
    }
}

/// Verifies selected PBS transparent stems declare transparent defaults instead of opaque forward aliases.
#[test]
fn selected_pbs_transparent_stems_keep_transparent_pass_defaults() {
    for stem in [
        "pbsdisplacetransparent_default",
        "pbsdisplacespeculartransparent_default",
        "pbsdistancelerptransparent_default",
        "pbsdistancelerpspeculartransparent_default",
    ] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(
            passes.len(),
            1,
            "{stem} should declare one transparent forward pass"
        );
        assert_eq!(passes[0].name, "forward_transparent", "{stem}");
        assert!(!passes[0].depth_write, "{stem}");
        assert!(passes[0].blend.is_some(), "{stem}");
    }

    for stem in [
        "pbsdualsidedtransparent_default",
        "pbsdualsidedtransparentspecular_default",
    ] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(
            passes.len(),
            2,
            "{stem} should declare back-face then front-face transparent passes"
        );
        assert_eq!(passes[0].name, "forward_transparent_cull_front", "{stem}");
        assert_eq!(passes[0].cull_mode, Some(wgpu::Face::Front), "{stem}");
        assert!(passes[0].blend.is_some(), "{stem}");
        assert_eq!(passes[1].name, "forward_transparent_cull_back", "{stem}");
        assert_eq!(passes[1].cull_mode, Some(wgpu::Face::Back), "{stem}");
        assert!(passes[1].blend.is_some(), "{stem}");
    }
}

/// Verifies the XSToon family keeps its expected forward / outline / stencil topology.
#[test]
fn xstoon_stems_keep_expected_outline_and_stencil_pass_order() {
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

    for stem in [
        "xstoon2.0_default",
        "xstoon2.0-cutout_default",
        "xstoon2.0-cutouta2c_default",
        "xstoon2.0-cutouta2cmasked_default",
        "xstoon2.0-dithered_default",
        "xstoon2.0-fade_default",
        "xstoon2.0-transparent_default",
    ] {
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        assert_eq!(
            passes.len(),
            1,
            "{stem} should declare a single forward pass"
        );
        assert_eq!(passes[0].name, "forward", "{stem}");
    }

    let stencil_passes = crate::embedded_shaders::embedded_target_passes("xstoonstenciler_default");
    assert_eq!(stencil_passes.len(), 1, "xstoonstenciler_default");
    assert_eq!(stencil_passes[0].name, "stencil", "xstoonstenciler_default");
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
