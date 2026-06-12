//! Shader source audits for the Null fallback material.

use super::*;

#[test]
fn null_material_checker_remaps_world_space_streams_to_model_space() -> io::Result<()> {
    let src = material_source("null.wgsl")?;

    for required in [
        "#import renderide::draw::types as dt",
        "fn checker_model_position(draw: dt::PerDrawUniforms, pos: vec4<f32>) -> vec3<f32>",
        "if (!dt::position_stream_is_world_space(draw))",
        "let world_relative = pos.xyz - draw.model[3].xyz;",
        "dot(world_relative, inv_x)",
        "out.checker = checker_model_position(d, pos) * 5.0;",
    ] {
        assert!(
            src.contains(required),
            "null.wgsl must contain `{required}`"
        );
    }

    assert!(
        !src.contains("out.checker = pos.xyz * 5.0;"),
        "null.wgsl must not anchor world-space deformed streams directly in world coordinates"
    );

    Ok(())
}
