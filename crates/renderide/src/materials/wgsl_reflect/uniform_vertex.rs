//! `@group(1)` uniform struct reflection and material vertex input analysis.

use hashbrown::HashMap;
use std::collections::BTreeMap;

use naga::proc::Layouter;
use naga::{AddressSpace, Binding, Module, ShaderStage, TypeInner, VectorSize};

use super::resource::resource_data_ty;
use super::types::{
    ReflectError, ReflectedMaterialUniformBlock, ReflectedUniformField, ReflectedUniformScalarKind,
    ReflectedVertexInput, ReflectedVertexInputFormat,
};

/// `true` when `@group(1)` uniform struct includes `_IntersectColor` (PBS intersect materials).
pub(super) fn material_uniform_requires_intersection_subpass(
    material_uniform: Option<&ReflectedMaterialUniformBlock>,
) -> bool {
    material_uniform.is_some_and(|u| u.fields.contains_key("_IntersectColor"))
}

pub(super) fn reflect_group1_global_binding_names(module: &Module) -> HashMap<u32, String> {
    let mut out = HashMap::new();
    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group != 1 {
            continue;
        }
        let Some(name) = gv.name.as_deref() else {
            continue;
        };
        out.insert(rb.binding, name.to_string());
    }
    out
}

fn vertex_input_format(
    module: &Module,
    ty: naga::Handle<naga::Type>,
) -> ReflectedVertexInputFormat {
    match &module.types[ty].inner {
        TypeInner::Vector { size, scalar }
            if scalar.kind == naga::ScalarKind::Float && *size == VectorSize::Bi =>
        {
            ReflectedVertexInputFormat::Float32x2
        }
        TypeInner::Vector { size, scalar }
            if scalar.kind == naga::ScalarKind::Float && *size == VectorSize::Tri =>
        {
            ReflectedVertexInputFormat::Float32x3
        }
        TypeInner::Vector { size, scalar }
            if scalar.kind == naga::ScalarKind::Float && *size == VectorSize::Quad =>
        {
            ReflectedVertexInputFormat::Float32x4
        }
        _ => ReflectedVertexInputFormat::Unsupported,
    }
}

pub(super) fn reflect_vertex_entry_inputs(
    module: &Module,
    entry_names: &[&str],
) -> Result<Vec<ReflectedVertexInput>, ReflectError> {
    let mut inputs_by_location = BTreeMap::new();
    for entry_name in entry_names {
        let ep = module
            .entry_points
            .iter()
            .find(|e| e.stage == ShaderStage::Vertex && e.name == *entry_name)
            .ok_or_else(|| ReflectError::VertexEntryPointMissing {
                entry: (*entry_name).to_string(),
            })?;
        for arg in &ep.function.arguments {
            let Some(Binding::Location { location, .. }) = arg.binding else {
                continue;
            };
            let format = vertex_input_format(module, arg.ty);
            match inputs_by_location.insert(location, format) {
                Some(existing) if existing != format => {
                    return Err(ReflectError::VertexInputFormatConflict {
                        location,
                        first: existing,
                        second: format,
                    });
                }
                _ => {}
            }
        }
    }
    Ok(inputs_by_location
        .into_iter()
        .map(|(location, format)| ReflectedVertexInput { location, format })
        .collect())
}

pub(super) fn reflect_vs_main_vertex_inputs(module: &Module) -> Vec<ReflectedVertexInput> {
    let ep = module
        .entry_points
        .iter()
        .find(|e| e.stage == ShaderStage::Vertex && e.name == "vs_main");
    let Some(ep) = ep else {
        return Vec::new();
    };
    let func = &ep.function;
    let mut inputs = Vec::new();
    for arg in &func.arguments {
        if let Some(Binding::Location { location, .. }) = arg.binding {
            inputs.push(ReflectedVertexInput {
                location,
                format: vertex_input_format(module, arg.ty),
            });
        }
    }
    inputs.sort_by_key(|input| input.location);
    inputs
}

fn uniform_member_kind(
    module: &Module,
    ty: naga::Handle<naga::Type>,
) -> ReflectedUniformScalarKind {
    match &module.types[ty].inner {
        TypeInner::Scalar(sc) => match sc.kind {
            naga::ScalarKind::Float => ReflectedUniformScalarKind::F32,
            naga::ScalarKind::Uint => ReflectedUniformScalarKind::U32,
            naga::ScalarKind::Sint => ReflectedUniformScalarKind::Unsupported,
            naga::ScalarKind::Bool => ReflectedUniformScalarKind::Unsupported,
            naga::ScalarKind::AbstractInt | naga::ScalarKind::AbstractFloat => {
                ReflectedUniformScalarKind::Unsupported
            }
        },
        TypeInner::Vector { size, scalar } => {
            if *size == VectorSize::Quad && scalar.kind == naga::ScalarKind::Float {
                ReflectedUniformScalarKind::Vec4
            } else {
                ReflectedUniformScalarKind::Unsupported
            }
        }
        _ => ReflectedUniformScalarKind::Unsupported,
    }
}

/// Finds the first `@group(1)` `var<uniform>` with a struct type and records member offsets/sizes.
pub(super) fn reflect_first_group1_uniform_struct(
    module: &Module,
    layouter: &Layouter,
) -> Option<ReflectedMaterialUniformBlock> {
    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group != 1 {
            continue;
        }
        let (space, data_ty) = resource_data_ty(module, gv);
        if space != AddressSpace::Uniform {
            continue;
        }
        let inner = &module.types[data_ty].inner;
        let TypeInner::Struct { members, .. } = inner else {
            continue;
        };
        let mut fields = HashMap::new();
        for m in members {
            let Some(name) = m.name.as_deref() else {
                continue;
            };
            let size = layouter[m.ty].size;
            let kind = uniform_member_kind(module, m.ty);
            fields.insert(
                name.to_string(),
                ReflectedUniformField {
                    offset: m.offset,
                    size,
                    kind,
                },
            );
        }
        let total_size = layouter[data_ty].size;
        return Some(ReflectedMaterialUniformBlock {
            binding: rb.binding,
            total_size,
            fields,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use naga::front::wgsl::parse_str;

    use super::*;

    #[test]
    fn reflects_exact_vertex_input_formats() {
        let module = parse_str(
            r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(5) uv1: vec3<f32>,
    @location(8) uv4: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.position = position + normal * 0.0 + color * 0.0 + vec4<f32>(uv0, uv1.x, uv4.w) * 0.0;
    return out;
}
"#,
        )
        .expect("parse synthetic vertex shader");

        let inputs = reflect_vs_main_vertex_inputs(&module);
        assert!(inputs.contains(&ReflectedVertexInput {
            location: 2,
            format: ReflectedVertexInputFormat::Float32x2,
        }));
        assert!(inputs.contains(&ReflectedVertexInput {
            location: 3,
            format: ReflectedVertexInputFormat::Float32x4,
        }));
        assert!(inputs.contains(&ReflectedVertexInput {
            location: 5,
            format: ReflectedVertexInputFormat::Float32x3,
        }));
        assert!(inputs.contains(&ReflectedVertexInput {
            location: 8,
            format: ReflectedVertexInputFormat::Float32x4,
        }));
    }

    #[test]
    fn distinguishes_location_three_uv_from_vertex_color() {
        let module = parse_str(
            r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(3) uv: vec2<f32>,
) -> VsOut {
    var out: VsOut;
    out.position = position + vec4<f32>(uv, 0.0, 0.0) * 0.0;
    return out;
}
"#,
        )
        .expect("parse synthetic vertex shader");

        let inputs = reflect_vs_main_vertex_inputs(&module);
        assert!(inputs.contains(&ReflectedVertexInput {
            location: 3,
            format: ReflectedVertexInputFormat::Float32x2,
        }));
        assert!(!inputs.contains(&ReflectedVertexInput {
            location: 3,
            format: ReflectedVertexInputFormat::Float32x4,
        }));
    }

    #[test]
    fn reflects_requested_vertex_entry_without_vs_main() {
        let module = parse_str(
            r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_fur_layer(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VsOut {
    var out: VsOut;
    out.position = position + normal * 0.0 + vec4<f32>(uv0, 0.0, 0.0) * 0.0;
    return out;
}
"#,
        )
        .expect("parse synthetic vertex shader");

        let inputs =
            reflect_vertex_entry_inputs(&module, &["vs_fur_layer"]).expect("reflect vertex entry");
        assert_eq!(
            inputs,
            vec![
                ReflectedVertexInput {
                    location: 0,
                    format: ReflectedVertexInputFormat::Float32x4,
                },
                ReflectedVertexInput {
                    location: 1,
                    format: ReflectedVertexInputFormat::Float32x4,
                },
                ReflectedVertexInput {
                    location: 2,
                    format: ReflectedVertexInputFormat::Float32x2,
                },
            ]
        );
    }

    #[test]
    fn reflects_union_of_requested_vertex_entries() {
        let module = parse_str(
            r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_base(
    @location(0) position: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VsOut {
    var out: VsOut;
    out.position = position + vec4<f32>(uv0, 0.0, 0.0) * 0.0;
    return out;
}

@vertex
fn vs_tangent(
    @location(0) position: vec4<f32>,
    @location(4) tangent: vec4<f32>,
    @location(5) uv1: vec2<f32>,
) -> VsOut {
    var out: VsOut;
    out.position = position + tangent * 0.0 + vec4<f32>(uv1, 0.0, 0.0) * 0.0;
    return out;
}
"#,
        )
        .expect("parse synthetic vertex shader");

        let inputs =
            reflect_vertex_entry_inputs(&module, &["vs_base", "vs_tangent"]).expect("reflect");
        assert_eq!(
            inputs,
            vec![
                ReflectedVertexInput {
                    location: 0,
                    format: ReflectedVertexInputFormat::Float32x4,
                },
                ReflectedVertexInput {
                    location: 2,
                    format: ReflectedVertexInputFormat::Float32x2,
                },
                ReflectedVertexInput {
                    location: 4,
                    format: ReflectedVertexInputFormat::Float32x4,
                },
                ReflectedVertexInput {
                    location: 5,
                    format: ReflectedVertexInputFormat::Float32x2,
                },
            ]
        );
    }

    #[test]
    fn rejects_conflicting_requested_vertex_entry_formats() {
        let module = parse_str(
            r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_uv(@location(2) uv0: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.position = vec4<f32>(uv0, 0.0, 1.0);
    return out;
}

@vertex
fn vs_color(@location(2) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    out.position = color;
    return out;
}
"#,
        )
        .expect("parse synthetic vertex shader");

        let err = reflect_vertex_entry_inputs(&module, &["vs_uv", "vs_color"])
            .expect_err("location format conflict should fail");
        assert!(matches!(
            err,
            ReflectError::VertexInputFormatConflict {
                location: 2,
                first: ReflectedVertexInputFormat::Float32x2,
                second: ReflectedVertexInputFormat::Float32x4,
            }
        ));
    }
}
