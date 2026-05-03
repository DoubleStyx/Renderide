//! Per-source shader composition.

use naga::valid::Capabilities;
use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderType};

use super::directives::parse_pass_directives;
use super::error::BuildError;
use super::model::{CompiledShader, CompiledShaderTarget, ShaderJob, ShaderVariant};
use super::modules::{ShaderModuleSources, register_composable_modules};
use super::source::shader_source_for_compile;
use super::validation::{
    module_to_wgsl, validate_entry_points, validate_no_pipeline_state_uniform_fields,
};

/// Composes one source variant through naga-oil.
fn compose_source_variant(
    modules: &ShaderModuleSources,
    source: &str,
    file_path: &str,
    variant: ShaderVariant,
) -> Result<naga::Module, BuildError> {
    let mut composer = Composer::default().with_capabilities(Capabilities::all());
    register_composable_modules(&mut composer, modules)?;
    composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path,
            shader_type: ShaderType::Wgsl,
            shader_defs: std::collections::HashMap::from_iter(variant.shader_defs()),
            ..Default::default()
        })
        .map_err(|e| BuildError::Message(format!("compose {file_path}: {e}")))
}

/// Checks the `@builtin(view_index)` contract for variant-sensitive outputs.
fn validate_view_index_contract(
    target_stem: &str,
    wgsl: &str,
    variant: ShaderVariant,
) -> Result<(), BuildError> {
    let has = wgsl.contains("@builtin(view_index)");
    if variant.expects_view_index() != has {
        return Err(BuildError::Message(format!(
            "{target_stem}: expected @builtin(view_index) {} in output (multiview shader_defs contract)",
            if variant.expects_view_index() {
                "present"
            } else {
                "absent"
            }
        )));
    }
    Ok(())
}

/// Compiles one source shader into one or two flattened WGSL targets without writing files.
pub(super) fn compile_shader_job(
    modules: &ShaderModuleSources,
    job: &ShaderJob,
) -> Result<CompiledShader, BuildError> {
    let source_path = &job.source_path;
    let stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| BuildError::Message(format!("invalid stem: {}", source_path.display())))?;
    let (source, file_path) = shader_source_for_compile(source_path)?;
    let mut pass_directives = parse_pass_directives(&source, &file_path)?;
    if job.validation.require_pass_directive && pass_directives.is_empty() {
        return Err(BuildError::Message(format!(
            "{file_path}: material WGSL must declare at least one //#pass directive (e.g. //#pass forward)"
        )));
    }

    let default_module =
        compose_source_variant(modules, &source, &file_path, ShaderVariant::Default)?;
    let multiview_module =
        compose_source_variant(modules, &source, &file_path, ShaderVariant::Multiview)?;
    validate_entry_points(
        &default_module,
        &format!("{stem} ({})", ShaderVariant::Default.label()),
        &mut pass_directives,
    )?;
    validate_entry_points(
        &multiview_module,
        &format!("{stem} ({})", ShaderVariant::Multiview.label()),
        &mut pass_directives,
    )?;
    validate_no_pipeline_state_uniform_fields(
        &default_module,
        &format!("{stem} ({})", ShaderVariant::Default.label()),
    )?;
    validate_no_pipeline_state_uniform_fields(
        &multiview_module,
        &format!("{stem} ({})", ShaderVariant::Multiview.label()),
    )?;

    let default_wgsl = module_to_wgsl(
        &default_module,
        &format!("{stem} ({})", ShaderVariant::Default.label()),
    )?;
    let multiview_wgsl = module_to_wgsl(
        &multiview_module,
        &format!("{stem} ({})", ShaderVariant::Multiview.label()),
    )?;

    let targets = if default_wgsl == multiview_wgsl {
        vec![CompiledShaderTarget {
            target_stem: stem.to_string(),
            wgsl: default_wgsl,
        }]
    } else {
        let variants = [
            (ShaderVariant::Default, default_wgsl),
            (ShaderVariant::Multiview, multiview_wgsl),
        ];
        let mut targets = Vec::with_capacity(variants.len());
        for (variant, wgsl) in variants {
            let target_stem = variant.target_stem(stem);
            if job.validation.validate_view_index {
                validate_view_index_contract(&target_stem, &wgsl, variant)?;
            }
            targets.push(CompiledShaderTarget { target_stem, wgsl });
        }
        targets
    };

    Ok(CompiledShader {
        compile_order: job.compile_order,
        source_class: job.source_class,
        pass_directives,
        targets,
    })
}
