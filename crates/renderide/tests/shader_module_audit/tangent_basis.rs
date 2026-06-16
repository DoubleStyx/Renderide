//! Shader source audits for mesh tangent-basis invariants.

use super::*;

#[test]
fn mesh_world_tangent_applies_model_transform_parity() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/modules/mesh/vertex.wgsl"))?;

    for required in [
        "fn model_handedness(draw: dt::PerDrawUniforms) -> f32",
        "dt::position_stream_is_world_space(draw)",
        "dot(draw.model[0].xyz, cross(draw.model[1].xyz, draw.model[2].xyz))",
        "let tangent_sign = select(1.0, -1.0, t.w < 0.0);",
        "tangent_sign * model_handedness(draw)",
    ] {
        assert!(
            src.contains(required),
            "mesh/vertex.wgsl must contain `{required}`"
        );
    }

    assert!(
        !src.contains("raw tangent handedness is preserved"),
        "world_tangent docs must not describe raw tangent handedness preservation"
    );

    Ok(())
}

#[test]
fn mesh_normals_do_not_use_model_vector_helper_path() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/modules/mesh/vertex.wgsl"))?;

    for forbidden in [
        "fn model_world_normal(",
        "fn world_model_normal_vertex_main(",
        "normalize(model_vector(draw, n.xyz))",
        "out.world_n = model_world_normal(draw, n);",
    ] {
        assert!(
            !src.contains(forbidden),
            "mesh/vertex.wgsl must not contain model-matrix normal path `{forbidden}`"
        );
    }

    for required in [
        "fn world_normal_for_view(draw: dt::PerDrawUniforms, n: vec4<f32>, view_idx: u32) -> vec3<f32>",
        "return world_normal(draw, n);",
        "out.world_n = world_normal_for_view(draw, n, view_idx);",
    ] {
        assert!(
            src.contains(required),
            "mesh/vertex.wgsl must route world vertex normals through the inverse-transpose helper `{required}`"
        );
    }

    Ok(())
}

#[test]
fn pbs_tbn_reorthonormalizes_interpolated_frame() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/modules/pbs/normal.wgsl"))?;

    for required in [
        "let n = world_n * inverseSqrt(n_len_sq);",
        "let t_ortho = world_t.xyz - n * dot(world_t.xyz, n);",
        "let t = t_ortho * inverseSqrt(t_len_sq);",
        "rmath::safe_normalize(cross(n, t)",
    ] {
        assert!(
            src.contains(required),
            "pbs/normal.wgsl must contain `{required}`"
        );
    }

    for forbidden in [
        "let n = world_n;",
        "let t = world_t.xyz;",
        "let b = cross(n, t) * sign;",
    ] {
        assert!(
            !src.contains(forbidden),
            "pbs/normal.wgsl must not contain `{forbidden}`"
        );
    }

    Ok(())
}

#[test]
fn fresnel_normal_mapped_shaders_use_normal_matrix_vertex_path() -> io::Result<()> {
    for file_name in ["fresnellerp.wgsl", "overlayfresnel.wgsl"] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("mv::world_vertex_main(instance_index, view_idx, pos, n, t, uv)")
                && src.contains("mv::world_vertex_main(instance_index, 0u, pos, n, t, uv)"),
            "{file_name} must use the standard world vertex helper so normals use the normal matrix"
        );
        assert!(
            !src.contains("world_model_normal_vertex_main"),
            "{file_name} must not use the removed model-matrix normal helper"
        );
    }

    Ok(())
}

#[test]
fn no_shader_references_model_matrix_normal_vertex_helper() -> io::Result<()> {
    let mut offenders = Vec::new();

    for relative_dir in ["shaders/materials", "shaders/modules", "shaders/passes"] {
        for path in wgsl_files_recursive(relative_dir)? {
            let src = source_file(&path)?;
            if src.contains("world_model_normal_vertex_main") {
                offenders.push(file_label(&path));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "WGSL must not reference the removed model-matrix normal helper:\n  {}",
        offenders.join("\n  ")
    );

    Ok(())
}

#[test]
fn custom_mesh_tbn_shaders_route_through_shared_parity() -> io::Result<()> {
    let xiexe = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/surface.wgsl"))?;
    assert!(
        xiexe.contains("let world_tangent = mv::world_tangent(d, tangent);"),
        "Xiexe Toon 2 must use the shared parity-aware world tangent helper"
    );
    assert!(
        !xiexe.contains("tangent.w,"),
        "Xiexe Toon 2 must not preserve raw tangent.w directly"
    );

    let matcap = material_source("matcap.wgsl")?;
    assert!(
        matcap.contains("pnorm::orthonormal_tbn(world_n, mv::world_tangent(d, tangent))"),
        "Matcap must build its mesh TBN through shared parity-aware helpers"
    );
    for forbidden in [
        "let tangent_sign = select",
        "tangent.w < 0.0",
        "cross(world_n, world_t)",
    ] {
        assert!(
            !matcap.contains(forbidden),
            "Matcap must not contain local mesh TBN fragment `{forbidden}`"
        );
    }

    Ok(())
}

#[test]
fn matcap_shader_uses_active_view_projection_basis() -> io::Result<()> {
    let matcap = material_source("matcap.wgsl")?;

    for required in [
        "let view_layer = view_idx;",
        "let vp = mv::select_view_proj(d, view_layer);",
        "let basis = vb::from_view_projection(vp);",
        "out.view_x = basis.x;",
        "out.view_y = basis.y;",
    ] {
        assert!(
            matcap.contains(required),
            "Matcap must derive its lookup basis from the active view projection via `{required}`"
        );
    }

    for forbidden in [
        "stereo_center",
        "frag_screen_uv",
        "screen_uv",
        "frag_pos.xy",
    ] {
        assert!(
            !matcap.contains(forbidden),
            "Matcap must not derive matcap UVs from stereo-center or screen-space fragment `{forbidden}`"
        );
    }

    Ok(())
}

#[test]
fn mesh_tangent_handedness_is_not_recomputed_in_material_roots() -> io::Result<()> {
    let allowed = [
        "shaders/modules/mesh/vertex.wgsl",
        "shaders/modules/pbs/normal.wgsl",
        "shaders/passes/compute/mesh_skinning.wgsl",
    ];
    let mut offenders = Vec::new();

    for relative_dir in ["shaders/materials", "shaders/modules", "shaders/passes"] {
        for path in wgsl_files_recursive(relative_dir)? {
            let label = file_label(&path);
            if allowed.contains(&label.as_str()) {
                continue;
            }
            let src = source_file(&path)?;
            for forbidden in [
                "let tangent_sign = select",
                "tangent.w < 0.0",
                "t.w < 0.0",
                "cross(world_n, world_t) *",
                "cross(n, t) * sign",
            ] {
                if src.contains(forbidden) {
                    offenders.push(format!("{label} still contains `{forbidden}`"));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "mesh tangent-basis handedness must stay centralized:\n  {}",
        offenders.join("\n  ")
    );

    Ok(())
}

#[test]
fn skinning_derives_tangent_handedness_from_deformed_bitangent() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/passes/compute/mesh_skinning.wgsl"))?;

    for required in [
        "b_bind = cross(n_bind, t_bind) * bind_sign;",
        "next.bitangent += weight * (linear * b_bind);",
        "let bb = safe_normalize(accum.bitangent / accum.weight_sum, b_bind);",
        "dot(cross(nn, tt), bb) < 0.0",
    ] {
        assert!(
            src.contains(required),
            "mesh_skinning.wgsl must contain `{required}`"
        );
    }

    Ok(())
}

#[test]
fn procedural_tangent_frames_are_explicitly_exempt_from_mesh_parity() -> io::Result<()> {
    let ggx = module_source("ibl/ggx_prefilter.wgsl")?;
    assert!(
        ggx.contains("fn tangent_to_world(local_dir: vec3<f32>, n: vec3<f32>) -> vec3<f32>")
            && ggx.contains("let tangent = normalize(cross(up, n));")
            && !ggx.contains("renderide::mesh::vertex"),
        "GGX prefilter builds a procedural sampling basis and must not import mesh tangent parity"
    );

    let gtao = source_file(manifest_dir().join("shaders/passes/post/gtao_main.wgsl"))?;
    assert!(
        gtao.contains("accepted.x * cross(l, t)")
            && gtao.contains("accepted.y * cross(t, r)")
            && !gtao.contains("renderide::mesh::vertex"),
        "GTAO builds a screen-space procedural basis and must not import mesh tangent parity"
    );

    Ok(())
}
