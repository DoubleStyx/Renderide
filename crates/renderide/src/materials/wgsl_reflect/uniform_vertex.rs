//! `@group(1)` uniform struct reflection and `vs_main` vertex input analysis.

use hashbrown::HashMap;

use naga::proc::Layouter;
use naga::{AddressSpace, Binding, Module, ShaderStage, TypeInner, VectorSize};

use super::resource::resource_data_ty;
use super::types::{
    ReflectedMaterialUniformBlock, ReflectedUniformField, ReflectedUniformScalarKind,
};

/// `true` when `@group(1)` uniform struct includes `_IntersectColor` (PBS intersect materials).
pub(super) fn material_uniform_requires_intersection_subpass(
    material_uniform: &Option<ReflectedMaterialUniformBlock>,
) -> bool {
    material_uniform
        .as_ref()
        .is_some_and(|u| u.fields.contains_key("_IntersectColor"))
}

/// `true` when `@group(1)` uniform struct includes `_GrabPass` (scene-color snapshot materials like blur).
pub(super) fn material_uniform_requires_grab_pass_subpass(
    material_uniform: &Option<ReflectedMaterialUniformBlock>,
) -> bool {
    material_uniform
        .as_ref()
        .is_some_and(|u| u.fields.contains_key("_GrabPass"))
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

pub(super) fn reflect_vs_main_max_vertex_location(module: &Module) -> Option<u32> {
    let ep = module
        .entry_points
        .iter()
        .find(|e| e.stage == ShaderStage::Vertex && e.name == "vs_main")?;
    let func = &ep.function;
    let mut max: Option<u32> = None;
    for arg in &func.arguments {
        if let Some(Binding::Location { location, .. }) = arg.binding {
            max = Some(max.map_or(location, |m| m.max(location)));
        }
    }
    max
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
        for m in members.iter() {
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
