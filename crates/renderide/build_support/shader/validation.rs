//! Naga validation and Renderide shader contract checks.

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use super::directives::BuildPassDirective;
use super::error::BuildError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LocationSignature {
    interpolation: Option<naga::Interpolation>,
    sampling: Option<naga::Sampling>,
    scalar_kind: Option<naga::ScalarKind>,
}

fn scalar_kind_for_type(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Option<naga::ScalarKind> {
    match &module.types[ty].inner {
        naga::TypeInner::Scalar(s) => Some(s.kind),
        naga::TypeInner::Vector { scalar, .. } => Some(scalar.kind),
        _ => None,
    }
}

fn extract_location_signature(
    module: &naga::Module,
    binding: &naga::Binding,
    ty: naga::Handle<naga::Type>,
) -> Option<(u32, LocationSignature)> {
    let naga::Binding::Location {
        location,
        interpolation,
        sampling,
        ..
    } = *binding
    else {
        return None;
    };
    Some((
        location,
        LocationSignature {
            interpolation,
            sampling,
            scalar_kind: scalar_kind_for_type(module, ty),
        },
    ))
}

fn vertex_output_signatures(
    module: &naga::Module,
    vertex_entry: &naga::EntryPoint,
) -> std::collections::BTreeMap<u32, LocationSignature> {
    let mut out = std::collections::BTreeMap::new();
    let Some(result) = &vertex_entry.function.result else {
        return out;
    };
    if let Some(binding) = &result.binding
        && let Some((loc, sig)) = extract_location_signature(module, binding, result.ty)
    {
        out.insert(loc, sig);
        return out;
    }
    if let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner {
        for member in members {
            if let Some(binding) = &member.binding
                && let Some((loc, sig)) = extract_location_signature(module, binding, member.ty)
            {
                out.insert(loc, sig);
            }
        }
    }
    out
}

fn fragment_input_signatures(
    module: &naga::Module,
    fragment_entry: &naga::EntryPoint,
) -> std::collections::BTreeMap<u32, LocationSignature> {
    let mut out = std::collections::BTreeMap::new();
    for arg in &fragment_entry.function.arguments {
        if let Some(binding) = &arg.binding
            && let Some((loc, sig)) = extract_location_signature(module, binding, arg.ty)
        {
            out.insert(loc, sig);
        }
    }
    out
}

fn is_vertex_compatible_with_fragment(
    module: &naga::Module,
    vertex_entry: &naga::EntryPoint,
    fragment_entry: &naga::EntryPoint,
) -> bool {
    let vs = vertex_output_signatures(module, vertex_entry);
    let fs = fragment_input_signatures(module, fragment_entry);
    fs.into_iter().all(|(loc, fs_sig)| {
        let Some(vs_sig) = vs.get(&loc).copied() else {
            return false;
        };
        // Fragment inputs must receive the same scalar category and interpolation shape.
        vs_sig.scalar_kind == fs_sig.scalar_kind
            && vs_sig.interpolation == fs_sig.interpolation
            && vs_sig.sampling == fs_sig.sampling
    })
}

fn resolve_implicit_vertex_entry(
    module: &naga::Module,
    pass: &BuildPassDirective,
) -> Option<String> {
    let fragment_entry = module.entry_points.iter().find(|e| {
        e.stage == naga::ShaderStage::Fragment && e.name == pass.fragment_entry
    })?;
    let mut matches = module
        .entry_points
        .iter()
        .filter(|e| e.stage == naga::ShaderStage::Vertex)
        .filter(|candidate| is_vertex_compatible_with_fragment(module, candidate, fragment_entry))
        .map(|entry| entry.name.clone())
        .collect::<Vec<_>>();
    matches.sort();
    matches.dedup();
    match matches.len() {
        0 => None,
        1 => matches.into_iter().next(),
        _ => matches
            .iter()
            .find(|name| name.as_str() == "vs_main")
            .cloned()
            .or_else(|| matches.into_iter().next()),
    }
}

/// Checks that `module` declares the entry points required by `passes`.
pub(super) fn validate_entry_points(
    module: &naga::Module,
    label: &str,
    passes: &mut [BuildPassDirective],
) -> Result<(), BuildError> {
    if passes.is_empty() {
        let has_compute = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Compute);
        if has_compute {
            return Ok(());
        }
        let has_vs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == "vs_main");
        let has_any_fs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Fragment);
        if !has_vs || !has_any_fs {
            return Err(BuildError::Message(format!(
                "{label}: expected a vs_main vertex entry point and at least one @fragment \
                 entry point (vertex={has_vs} fragment={has_any_fs})",
            )));
        }
        return Ok(());
    }
    for pass in passes {
        if !pass.vertex_entry_explicit {
            if let Some(inferred) = resolve_implicit_vertex_entry(module, pass) {
                pass.vertex_entry = inferred;
            } else {
                return Err(BuildError::Message(format!(
                    "{label}: pass `{:?}` fragment `{}` has no compatible vertex entry; add `vs=<entry>` to //#pass",
                    pass.kind, pass.fragment_entry
                )));
            }
        }
        let has_vs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == pass.vertex_entry.as_str());
        let has_fs = module.entry_points.iter().any(|e| {
            e.stage == naga::ShaderStage::Fragment && e.name == pass.fragment_entry.as_str()
        });
        if !has_vs || !has_fs {
            return Err(BuildError::Message(format!(
                "{label}: pass `{:?}` expected entry points {} and {} (vertex={has_vs} fragment={has_fs})",
                pass.kind, pass.vertex_entry, pass.fragment_entry
            )));
        }
    }
    Ok(())
}

/// Canonical Unity pipeline-state property names that must never appear in material uniforms.
const PIPELINE_STATE_PROPERTY_NAMES: &[&str] = &[
    "_SrcBlend",
    "_SrcBlendBase",
    "_SrcBlendAdd",
    "_DstBlend",
    "_DstBlendBase",
    "_DstBlendAdd",
    "_ZWrite",
    "_ZTest",
    "_Cull",
    "_Stencil",
    "_StencilComp",
    "_StencilOp",
    "_StencilFail",
    "_StencilZFail",
    "_StencilReadMask",
    "_StencilWriteMask",
    "_ColorMask",
    "_OffsetFactor",
    "_OffsetUnits",
];

/// Rejects any material whose `@group(1) @binding(0)` uniform contains pipeline-state fields.
pub(super) fn validate_no_pipeline_state_uniform_fields(
    module: &naga::Module,
    label: &str,
) -> Result<(), BuildError> {
    for (_, var) in module.global_variables.iter() {
        let Some(binding) = &var.binding else {
            continue;
        };
        if binding.group != 1 || binding.binding != 0 {
            continue;
        }
        if !matches!(var.space, naga::AddressSpace::Uniform) {
            continue;
        }
        let ty = &module.types[var.ty];
        let naga::TypeInner::Struct { ref members, .. } = ty.inner else {
            continue;
        };
        for member in members {
            let Some(name) = member.name.as_deref() else {
                continue;
            };
            if PIPELINE_STATE_PROPERTY_NAMES.contains(&name) {
                let struct_name = ty.name.as_deref().unwrap_or("<unnamed>");
                return Err(BuildError::Message(format!(
                    "{label}: material uniform struct `{struct_name}` declares pipeline-state \
                     field `{name}` at @group(1) @binding(0). Pipeline-state properties \
                     flow through MaterialBlendMode + MaterialRenderState and are baked into \
                     MaterialPipelineCacheKey; remove the field from the WGSL struct."
                )));
            }
        }
    }
    Ok(())
}

/// Validates a naga module and flattens it back to WGSL.
pub(super) fn module_to_wgsl(module: &naga::Module, label: &str) -> Result<String, BuildError> {
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(module)
        .map_err(|e| BuildError::Message(format!("validate {label}: {e}")))?;
    naga::back::wgsl::write_string(module, &info, WriterFlags::EXPLICIT_TYPES)
        .map_err(|e| BuildError::Message(format!("wgsl out {label}: {e}")))
}
