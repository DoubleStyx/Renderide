//! Per-source shader composition.

use naga::ShaderStage;
use naga::valid::Capabilities;
use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderType};

use super::directives::{BuildPassDirective, parse_pass_directives};
use super::error::BuildError;
use super::mirror_once::rewrite_material_mirror_once_wgsl;
use super::model::{
    CompiledShader, CompiledShaderTarget, ShaderJob, ShaderSourceClass, ShaderVariant,
};
use super::modules::{ShaderModuleSources, register_composable_modules};
use super::source::shader_source_for_compile;
use super::validation::{
    module_to_wgsl, validate_entry_points, validate_no_pipeline_state_uniform_fields,
    validate_pass_interfaces,
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
        .map_err(|e| {
            BuildError::Message(format!(
                "compose {file_path}: {}",
                e.emit_to_string(&composer)
            ))
        })
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

fn remapped_pass_directives_for_output(
    source_module: &naga::Module,
    output_wgsl: &str,
    pass_directives: &[BuildPassDirective],
    label: &str,
) -> Result<Vec<BuildPassDirective>, BuildError> {
    if pass_directives.is_empty() {
        return Ok(Vec::new());
    }
    let output_module = naga::front::wgsl::parse_str(output_wgsl).map_err(|e| {
        BuildError::Message(format!(
            "parse flattened WGSL {label}: {}",
            e.emit_to_string(output_wgsl)
        ))
    })?;
    let vertex_names =
        entry_point_name_pairs(source_module, &output_module, ShaderStage::Vertex, label)?;
    let fragment_names =
        entry_point_name_pairs(source_module, &output_module, ShaderStage::Fragment, label)?;
    pass_directives
        .iter()
        .map(|pass| {
            let mut remapped = pass.clone();
            remapped.vertex_entry =
                remapped_entry_point_name(&vertex_names, &pass.vertex_entry, label, "vertex")?;
            remapped.fragment_entry = remapped_entry_point_name(
                &fragment_names,
                &pass.fragment_entry,
                label,
                "fragment",
            )?;
            Ok(remapped)
        })
        .collect()
}

/// Converts a composed module to WGSL and applies material-only post-processing.
fn flattened_wgsl_for_job(
    module: &naga::Module,
    stem: &str,
    variant: ShaderVariant,
    source_class: ShaderSourceClass,
) -> Result<String, BuildError> {
    let label = format!("{stem} ({})", variant.label());
    let wgsl = module_to_wgsl(module, &label)?;
    if source_class == ShaderSourceClass::Material {
        rewrite_material_mirror_once_wgsl(&wgsl, &label)
    } else {
        Ok(wgsl)
    }
}

/// Runs source-level validation on both composed shader variants.
fn validate_composed_variants(
    stem: &str,
    pass_directives: &[BuildPassDirective],
    default_module: &naga::Module,
    multiview_module: &naga::Module,
) -> Result<(), BuildError> {
    validate_entry_points(
        default_module,
        &format!("{stem} ({})", ShaderVariant::Default.label()),
        pass_directives,
    )?;
    validate_pass_interfaces(
        default_module,
        &format!("{stem} ({})", ShaderVariant::Default.label()),
        pass_directives,
    )?;
    validate_entry_points(
        multiview_module,
        &format!("{stem} ({})", ShaderVariant::Multiview.label()),
        pass_directives,
    )?;
    validate_pass_interfaces(
        multiview_module,
        &format!("{stem} ({})", ShaderVariant::Multiview.label()),
        pass_directives,
    )?;
    validate_no_pipeline_state_uniform_fields(
        default_module,
        &format!("{stem} ({})", ShaderVariant::Default.label()),
    )?;
    validate_no_pipeline_state_uniform_fields(
        multiview_module,
        &format!("{stem} ({})", ShaderVariant::Multiview.label()),
    )
}

fn entry_point_name_pairs(
    source_module: &naga::Module,
    output_module: &naga::Module,
    stage: ShaderStage,
    label: &str,
) -> Result<Vec<(String, String)>, BuildError> {
    let source_names = entry_point_names(source_module, stage);
    let output_names = entry_point_names(output_module, stage);
    if source_names.len() != output_names.len() {
        return Err(BuildError::Message(format!(
            "{label}: flattened WGSL changed {stage:?} entry point count from {} to {}",
            source_names.len(),
            output_names.len(),
        )));
    }
    Ok(source_names.into_iter().zip(output_names).collect())
}

fn entry_point_names(module: &naga::Module, stage: ShaderStage) -> Vec<String> {
    module
        .entry_points
        .iter()
        .filter(|entry| entry.stage == stage)
        .map(|entry| entry.name.clone())
        .collect()
}

fn remapped_entry_point_name(
    name_pairs: &[(String, String)],
    requested: &str,
    label: &str,
    stage_name: &str,
) -> Result<String, BuildError> {
    name_pairs
        .iter()
        .find_map(|(source, output)| (source == requested).then(|| output.clone()))
        .ok_or_else(|| {
            BuildError::Message(format!(
                "{label}: requested {stage_name} entry point `{requested}` was not found before WGSL flattening"
            ))
        })
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
    let compile_source = shader_source_for_compile(source_path)?;
    let source = compile_source.source;
    let file_path = compile_source.file_path;
    let pass_directives = parse_pass_directives(&source, &file_path)?;
    if job.validation.require_pass_directive && pass_directives.is_empty() {
        return Err(BuildError::Message(format!(
            "{file_path}: material WGSL must declare at least one //#pass directive (e.g. //#pass forward)"
        )));
    }

    let default_module =
        compose_source_variant(modules, &source, &file_path, ShaderVariant::Default)?;
    let multiview_module =
        compose_source_variant(modules, &source, &file_path, ShaderVariant::Multiview)?;
    validate_composed_variants(stem, &pass_directives, &default_module, &multiview_module)?;

    let default_wgsl = flattened_wgsl_for_job(
        &default_module,
        stem,
        ShaderVariant::Default,
        job.source_class,
    )?;
    let multiview_wgsl = flattened_wgsl_for_job(
        &multiview_module,
        stem,
        ShaderVariant::Multiview,
        job.source_class,
    )?;

    let targets = if default_wgsl == multiview_wgsl {
        let pass_directives = remapped_pass_directives_for_output(
            &default_module,
            &default_wgsl,
            &pass_directives,
            &format!("{stem} ({})", ShaderVariant::Default.label()),
        )?;
        vec![CompiledShaderTarget {
            target_stem: stem.to_string(),
            wgsl: default_wgsl,
            pass_directives,
        }]
    } else {
        let variants = [
            (ShaderVariant::Default, &default_module, default_wgsl),
            (ShaderVariant::Multiview, &multiview_module, multiview_wgsl),
        ];
        let mut targets = Vec::with_capacity(variants.len());
        for (variant, module, wgsl) in variants {
            let target_stem = variant.target_stem(stem);
            if job.validation.validate_view_index {
                validate_view_index_contract(&target_stem, &wgsl, variant)?;
            }
            let pass_directives = remapped_pass_directives_for_output(
                module,
                &wgsl,
                &pass_directives,
                &format!("{stem} ({})", variant.label()),
            )?;
            targets.push(CompiledShaderTarget {
                target_stem,
                wgsl,
                pass_directives,
            });
        }
        targets
    };

    Ok(CompiledShader {
        compile_order: job.compile_order,
        source_class: job.source_class,
        pass_directives,
        texture_defaults: compile_source.texture_defaults,
        material_defaults: compile_source.material_defaults,
        wgpu_features: compile_source.wgpu_features,
        targets,
    })
}
