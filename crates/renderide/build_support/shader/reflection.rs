//! Build-time reflection for device-independent embedded shader metadata.

use std::collections::BTreeMap;

use naga::{AddressSpace, Binding, Module, ShaderStage, Type, TypeInner, VectorSize};

use super::directives::BuildPassDirective;
use super::error::BuildError;
use super::model::{
    BuildShaderReflection, BuildSnapshotRequirements, BuildVertexInput, BuildVertexInputFormat,
    BuildVertexStreamMask, ShaderSourceClass,
};

const UV_SHADER_LOCATIONS: [u32; 8] = [2, 5, 6, 7, 8, 9, 10, 11];

/// Reflects stable shader metadata from one final flattened WGSL target.
pub(super) fn reflect_embedded_target(
    target_stem: &str,
    wgsl: &str,
    pass_directives: &[BuildPassDirective],
    source_class: ShaderSourceClass,
) -> Result<BuildShaderReflection, BuildError> {
    if source_class != ShaderSourceClass::Material {
        return Ok(BuildShaderReflection::default());
    }
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| {
        BuildError::Message(format!(
            "reflect flattened WGSL {target_stem}: {}",
            e.emit_to_string(wgsl)
        ))
    })?;
    let vertex_entries = pass_directives
        .iter()
        .map(|pass| pass.vertex_entry.as_str())
        .collect::<Vec<_>>();
    let vertex_inputs = reflect_vertex_entry_inputs(&module, &vertex_entries, target_stem)?;
    let snapshot_requirements = reflect_snapshot_requirements(&module);
    Ok(BuildShaderReflection {
        vertex_stream_mask: vertex_stream_mask_from_inputs(&vertex_inputs),
        snapshot_requirements,
        uses_renderide_variant_bits: wgsl.contains("renderide_static_variant_bits"),
        supports_generic_depth_prepass: supports_generic_depth_prepass(
            wgsl,
            pass_directives,
            snapshot_requirements,
        ),
    })
}

fn reflect_vertex_entry_inputs(
    module: &Module,
    entry_names: &[&str],
    target_stem: &str,
) -> Result<Vec<BuildVertexInput>, BuildError> {
    let mut inputs_by_location = BTreeMap::new();
    for entry_name in entry_names {
        let entry = module
            .entry_points
            .iter()
            .find(|entry| entry.stage == ShaderStage::Vertex && entry.name == *entry_name)
            .ok_or_else(|| {
                BuildError::Message(format!(
                    "{target_stem}: vertex entry point `{entry_name}` was not found during build reflection"
                ))
            })?;
        for arg in &entry.function.arguments {
            let Some(Binding::Location { location, .. }) = arg.binding else {
                continue;
            };
            let format = vertex_input_format(module, arg.ty);
            match inputs_by_location.insert(location, format) {
                Some(existing) if existing != format => {
                    return Err(BuildError::Message(format!(
                        "{target_stem}: vertex input @location({location}) has incompatible formats {existing:?} and {format:?} across reflected pass vertex entries"
                    )));
                }
                _ => {}
            }
        }
    }
    Ok(inputs_by_location
        .into_iter()
        .map(|(location, format)| BuildVertexInput { location, format })
        .collect())
}

fn vertex_input_format(module: &Module, ty: naga::Handle<Type>) -> BuildVertexInputFormat {
    match &module.types[ty].inner {
        TypeInner::Vector { size, scalar }
            if scalar.kind == naga::ScalarKind::Float && *size == VectorSize::Bi =>
        {
            BuildVertexInputFormat::Float32x2
        }
        TypeInner::Vector { size, scalar }
            if scalar.kind == naga::ScalarKind::Float && *size == VectorSize::Tri =>
        {
            BuildVertexInputFormat::Float32x3
        }
        TypeInner::Vector { size, scalar }
            if scalar.kind == naga::ScalarKind::Float && *size == VectorSize::Quad =>
        {
            BuildVertexInputFormat::Float32x4
        }
        _ => BuildVertexInputFormat::Unsupported,
    }
}

fn vertex_stream_mask_from_inputs(inputs: &[BuildVertexInput]) -> BuildVertexStreamMask {
    let mut mask = BuildVertexStreamMask::default();
    for input in inputs {
        match (input.location, input.format) {
            (2, BuildVertexInputFormat::Float32x2) => mask.uv0 = true,
            (3, BuildVertexInputFormat::Float32x4) => mask.color = true,
            (4, BuildVertexInputFormat::Float32x4) => mask.tangent = true,
            (5, BuildVertexInputFormat::Float32x2) => mask.uv1 = true,
            (6, BuildVertexInputFormat::Float32x2) => mask.uv2 = true,
            (7, BuildVertexInputFormat::Float32x2) => mask.uv3 = true,
            (location, format) if uv_channel_from_location(location).is_some() => {
                apply_uv_requirement(&mut mask, location, format);
            }
            _ => {}
        }
    }
    mask
}

fn apply_uv_requirement(
    mask: &mut BuildVertexStreamMask,
    location: u32,
    format: BuildVertexInputFormat,
) {
    let Some(channel) = uv_channel_from_location(location) else {
        return;
    };
    let supported = matches!(
        format,
        BuildVertexInputFormat::Float32x2
            | BuildVertexInputFormat::Float32x3
            | BuildVertexInputFormat::Float32x4
    );
    if !supported {
        return;
    }
    match channel {
        0 => mask.uv0 = true,
        1 => mask.uv1 = true,
        2 => mask.uv2 = true,
        3 => mask.uv3 = true,
        _ => {}
    }
    if channel >= 4 {
        mask.wide_high_uvs = true;
    } else if format != BuildVertexInputFormat::Float32x2 {
        mask.wide_low_uvs = true;
    }
}

fn uv_channel_from_location(location: u32) -> Option<usize> {
    UV_SHADER_LOCATIONS
        .iter()
        .position(|candidate| *candidate == location)
}

fn reflect_snapshot_requirements(module: &Module) -> BuildSnapshotRequirements {
    let mut requirements = BuildSnapshotRequirements::default();
    for (_, gv) in module.global_variables.iter() {
        let Some(binding) = gv.binding else {
            continue;
        };
        match (binding.group, binding.binding) {
            (0, 4 | 5) => requirements.uses_scene_depth = true,
            (0, 6 | 7) => requirements.uses_scene_color = true,
            (1, _) => {
                requirements.requires_intersection_pass |=
                    group1_uniform_declares_field(module, gv, "_IntersectColor");
            }
            _ => {}
        }
    }
    requirements
}

fn group1_uniform_declares_field(module: &Module, gv: &naga::GlobalVariable, field: &str) -> bool {
    let (space, data_ty) = resource_data_ty(module, gv);
    if space != AddressSpace::Uniform {
        return false;
    }
    let TypeInner::Struct { members, .. } = &module.types[data_ty].inner else {
        return false;
    };
    members
        .iter()
        .any(|member| member.name.as_deref() == Some(field))
}

fn resource_data_ty(
    module: &Module,
    gv: &naga::GlobalVariable,
) -> (AddressSpace, naga::Handle<Type>) {
    match &module.types[gv.ty].inner {
        TypeInner::Pointer { base, space } => (*space, *base),
        _ => (gv.space, gv.ty),
    }
}

fn supports_generic_depth_prepass(
    wgsl: &str,
    pass_directives: &[BuildPassDirective],
    snapshot_requirements: BuildSnapshotRequirements,
) -> bool {
    let [pass] = pass_directives else {
        return false;
    };
    pass.is_generic_depth_prepass_candidate()
        && !wgsl.contains("discard")
        && snapshot_requirements == BuildSnapshotRequirements::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::directives::{
        BuildBlend, BuildColorWrites, BuildCullMode, BuildDepthCompare, BuildDepthCompareDomain,
        BuildMaterialPassState, BuildPassDirective, BuildPassType, BuildRenderStatePolicy,
    };

    fn forward_pass() -> BuildPassDirective {
        BuildPassDirective {
            pass_type: BuildPassType::Forward,
            name: "forward".to_string(),
            fragment_entry: "fs_main".to_string(),
            vertex_entry: "vs_main".to_string(),
            alpha_to_coverage: false,
            depth_compare_domain: BuildDepthCompareDomain::FrooxZTest,
            depth_compare: BuildDepthCompare::Main,
            depth_write: true,
            cull_mode: BuildCullMode::Back,
            blend: BuildBlend::Off,
            write_mask: BuildColorWrites::Rgb,
            depth_bias_slope_scale_bits: 0.0f32.to_bits(),
            depth_bias_constant: 0,
            material_state: BuildMaterialPassState::Forward,
            render_state_policy: BuildRenderStatePolicy::ALL_MATERIAL,
        }
    }

    fn material_wgsl(extra_globals: &str, fragment_body: &str) -> String {
        format!(
            r#"
struct Material {{
    _IntersectColor: vec4<f32>,
}}

@group(1) @binding(0)
var<uniform> material: Material;

{extra_globals}

struct VsOut {{
    @builtin(position) position: vec4<f32>,
}};

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(8) uv4: vec3<f32>,
) -> VsOut {{
    var out: VsOut;
    out.position = position + vec4<f32>(uv0, uv4.x, 0.0) * 0.0;
    return out;
}}

@fragment
fn fs_main() -> @location(0) vec4<f32> {{
    {fragment_body}
}}
"#
        )
    }

    #[test]
    fn reflects_stable_material_metadata() -> Result<(), BuildError> {
        let wgsl = material_wgsl(
            "@group(0) @binding(6) var scene_color: texture_2d<f32>;",
            "return vec4<f32>(1.0);",
        );
        let reflection = reflect_embedded_target(
            "test",
            &wgsl,
            &[forward_pass()],
            ShaderSourceClass::Material,
        )?;

        assert!(reflection.vertex_stream_mask.uv0);
        assert!(!reflection.vertex_stream_mask.uv1);
        assert!(reflection.vertex_stream_mask.wide_high_uvs);
        assert!(reflection.snapshot_requirements.uses_scene_color);
        assert!(reflection.snapshot_requirements.requires_intersection_pass);
        assert!(!reflection.supports_generic_depth_prepass);
        Ok(())
    }

    #[test]
    fn generic_depth_prepass_requires_opaque_single_pass_without_snapshots()
    -> Result<(), BuildError> {
        let wgsl = material_wgsl("", "return vec4<f32>(1.0);");
        let reflection = reflect_embedded_target(
            "test",
            &wgsl,
            &[forward_pass()],
            ShaderSourceClass::Material,
        )?;

        assert!(!reflection.supports_generic_depth_prepass);

        let no_intersection_wgsl =
            wgsl.replace("_IntersectColor: vec4<f32>,", "_Color: vec4<f32>,");
        let reflection = reflect_embedded_target(
            "test",
            &no_intersection_wgsl,
            &[forward_pass()],
            ShaderSourceClass::Material,
        )?;
        assert!(reflection.supports_generic_depth_prepass);
        Ok(())
    }
}
