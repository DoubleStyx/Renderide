//! Shader source audits for realtime shadow behavior.

use super::*;

#[test]
fn point_and_spot_shadow_receivers_use_radial_distance_compare() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/modules/lighting/shadows.wgsl"))?;

    for required in [
        "fn radial_shadow_kind",
        "kind == ft::SHADOW_VIEW_KIND_POINT || kind == ft::SHADOW_VIEW_KIND_SPOT",
        "fn radial_shadow_compare_depth",
        "length(world_pos - light.position.xyz) / range",
        "!radial_shadow && (ndc.z < 0.0 || ndc.z > 1.0)",
        "compare_depth = radial_shadow_compare_depth(light, biased_world_pos);",
        "compare_depth = projected_shadow_compare_depth(light, shadow_view, ndc);",
    ] {
        assert!(
            src.contains(required),
            "radial shadow receiver path must contain `{required}`"
        );
    }

    Ok(())
}

#[test]
fn shadow_receiver_applies_atlas_rect_normal_bias_and_soft_filtering() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/modules/lighting/shadows.wgsl"))?;

    for required in [
        "const SHADOW_TYPE_SOFT: u32 = 2u;",
        "fn receiver_position",
        "world_pos + normalize(world_normal) * bias",
        "shadow_view.atlas_rect.xy + local_uv * shadow_view.atlas_rect.zw",
        "fn sample_soft_shadow",
        "for (var y: i32 = -1; y <= 1; y = y + 1)",
        "if (light.shadow_type == SHADOW_TYPE_SOFT)",
        "fn visibility(light: ft::GpuLight, world_pos: vec3<f32>, world_normal: vec3<f32>)",
    ] {
        assert!(
            src.contains(required),
            "shadow receiver must contain `{required}`"
        );
    }

    Ok(())
}

#[test]
fn radial_shadow_caster_writes_radial_fragment_depth() -> io::Result<()> {
    let src = source_file(
        manifest_dir()
            .join("shaders/passes/backend")
            .join("world_mesh_radial_shadow_caster.wgsl"),
    )?;

    for required in [
        "Radial-distance shadow caster for world meshes.",
        "point and spot shadows compare consistently",
        "-> @builtin(frag_depth) f32",
        "light_position_range: vec4<f32>",
        "let radial_depth = (length(in.world_pos - draw.light_position_range.xyz) + bias) / range;",
        "return clamp(radial_depth, 0.0, 1.0);",
    ] {
        assert!(
            src.contains(required),
            "radial shadow caster must contain `{required}`"
        );
    }

    Ok(())
}
