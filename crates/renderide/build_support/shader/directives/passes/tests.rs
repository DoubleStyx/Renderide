use super::*;

/// Pass directives bind explicit metadata to the following fragment entry point.
#[test]
fn pass_directive_extracts_fragment_entry_and_state() -> Result<(), BuildError> {
    let passes = parse_pass_directives(
        r#"
//#pass type=forward name=outline vs=vs_outline blend=off cull=front zwrite=material(on) ztest=material_froox(main) offset=material(0,0)
@fragment
fn fs_outline() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
        "test.wgsl",
    )?;

    let pass = &passes[0];
    assert_eq!(pass.pass_type, BuildPassType::Forward);
    assert_eq!(pass.name, "outline");
    assert_eq!(pass.fragment_entry, "fs_outline");
    assert_eq!(pass.vertex_entry, "vs_outline");
    assert_eq!(pass.blend, BuildBlend::Off);
    assert_eq!(pass.cull_mode, BuildCullMode::Front);
    assert!(pass.render_state_policy.depth_write);
    assert!(!pass.render_state_policy.cull);
    Ok(())
}

/// Pass directives can opt into hardware alpha-to-coverage.
#[test]
fn pass_directive_extracts_alpha_to_coverage() -> Result<(), BuildError> {
    let passes = parse_pass_directives(
        r#"
//#pass type=forward a2c=true
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
        "test.wgsl",
    )?;

    assert!(passes[0].alpha_to_coverage);
    assert_eq!(passes[0].name, "forward");
    Ok(())
}

/// Pass directives can select Unity `CompareFunction` decoding for material `_ZTest`.
#[test]
fn pass_directive_extracts_ztest_domain() -> Result<(), BuildError> {
    let passes = parse_pass_directives(
        r#"
//#pass type=forward name=stencil ztest=material_unity(main)
@fragment
fn fs_stencil() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
        "stencil.wgsl",
    )?;

    assert_eq!(
        passes[0].depth_compare_domain,
        BuildDepthCompareDomain::UnityCompareFunction
    );
    assert!(passes[0].render_state_policy.depth_compare);
    assert!(pass_literal(&passes[0]).contains(
        "depth_compare_domain: crate::materials::MaterialDepthCompareDomain::UnityCompareFunction"
    ));
    Ok(())
}

/// Explicit metadata replaces former fixed-state cartesian pass aliases.
#[test]
fn pass_directive_parses_explicit_fixed_state_metadata() -> Result<(), BuildError> {
    let passes = parse_pass_directives(
        r#"
//#pass type=forward name=transparent_rgb blend=alpha zwrite=off ztest=main cull=off color_mask=rgb stencil=off offset=0,0
@fragment
fn fs_circle() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
//#pass type=forward name=volume_front blend=material_overlay zwrite=off ztest=always cull=front color_mask=rgba offset=0,0
@fragment
fn fs_volume() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
        "test.wgsl",
    )?;

    assert_eq!(passes[0].blend, BuildBlend::Alpha);
    assert_eq!(passes[0].write_mask, BuildColorWrites::Rgb);
    assert_eq!(passes[0].material_state, BuildMaterialPassState::Static);
    assert_eq!(
        passes[0].render_state_policy,
        BuildRenderStatePolicy {
            color_mask: false,
            depth_write: false,
            depth_compare: false,
            cull: false,
            stencil: false,
            depth_offset: false,
        }
    );
    assert_eq!(passes[1].blend, BuildBlend::Overlay);
    assert_eq!(passes[1].material_state, BuildMaterialPassState::Overlay);
    assert_eq!(passes[1].depth_compare, BuildDepthCompare::Always);
    assert!(!passes[1].render_state_policy.depth_compare);
    Ok(())
}

#[test]
fn pass_directive_parses_static_additive_blend() -> Result<(), BuildError> {
    let passes = parse_pass_directives(
        r#"
//#pass type=forward blend=additive
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
        "additive.wgsl",
    )?;

    assert_eq!(passes[0].blend, BuildBlend::Additive);
    assert_eq!(passes[0].write_mask, BuildColorWrites::Rgba);
    assert!(pass_literal(&passes[0]).contains("PASS_BLEND_ONE_ONE"));
    Ok(())
}

/// Transparent material state carries premultiplied defaults and still allows material overrides.
#[test]
fn pass_directive_parses_transparent_material_state() -> Result<(), BuildError> {
    let passes = parse_pass_directives(
        r#"
//#pass type=forward name=forward_transparent blend=transparent_material zwrite=material(off) cull=material(off) color_mask=material(rgba)
@fragment
fn fs_transparent() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
        "transparent.wgsl",
    )?;

    let pass = &passes[0];
    assert_eq!(pass.blend, BuildBlend::Premultiplied);
    assert_eq!(pass.write_mask, BuildColorWrites::Rgba);
    assert!(!pass.depth_write);
    assert_eq!(
        pass.material_state,
        BuildMaterialPassState::TransparentForward
    );
    assert!(pass.render_state_policy.depth_write);
    assert!(pass.render_state_policy.cull);
    Ok(())
}

/// Old cartesian pass tokens are rejected instead of silently selecting presets.
#[test]
fn pass_directive_rejects_old_preset_token() {
    let err = parse_pass_directives(
        r#"
//#pass forward_alpha_blend
@fragment
fn fs_fur() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
        "furfx.wgsl",
    )
    .expect_err("old pass aliases must be rejected");

    assert!(err.to_string().contains("key=value metadata"));
}

/// Static Unity pass offsets are converted to reverse-Z wgpu depth-bias defaults.
#[test]
fn pass_directive_extracts_static_unity_offset() -> Result<(), BuildError> {
    let passes = parse_pass_directives(
        r#"
//#pass type=forward offset=2,2
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
        "null.wgsl",
    )?;

    assert_eq!(passes[0].depth_bias_slope_scale_bits, (-2.0f32).to_bits());
    assert_eq!(passes[0].depth_bias_constant, -2);
    assert!(!passes[0].render_state_policy.depth_offset);
    Ok(())
}

/// Zero Unity slope offset stays a canonical zero in generated pass literals.
#[test]
fn pass_directive_canonicalizes_zero_unity_offset_factor() -> Result<(), BuildError> {
    let passes = parse_pass_directives(
        r#"
//#pass type=forward offset=0,1
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
        "newunlitshader.wgsl",
    )?;

    assert_eq!(passes[0].depth_bias_slope_scale_bits, 0.0f32.to_bits());
    assert_eq!(passes[0].depth_bias_constant, -1);
    assert!(pass_literal(&passes[0]).contains("depth_bias_constant: -1"));
    Ok(())
}
