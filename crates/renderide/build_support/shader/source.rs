//! Shader source discovery and source-alias loading.

use std::fs;
use std::path::{Path, PathBuf};

use super::directives::{
    MaterialDefaultDirective, RenderQueueDirective, TextureDefaultDirective, WgpuFeatureDirective,
    parse_material_default_directives, parse_render_queue_directive, parse_source_alias,
    parse_texture_default_directives, parse_wgpu_feature_directives,
};
use super::error::BuildError;
use super::model::{ShaderJob, ShaderSourceClass};

/// WGSL source selected for composition plus source-level metadata from wrapper/alias directives.
pub(super) struct ShaderCompileSource {
    /// Source text passed to naga-oil.
    pub source: String,
    /// File path label passed to naga-oil and source diagnostics.
    pub file_path: String,
    /// Texture defaults parsed from the source or merged from alias + wrapper directives.
    pub texture_defaults: Vec<TextureDefaultDirective>,
    /// Material uniform defaults parsed from the source or merged from alias + wrapper directives.
    pub material_defaults: Vec<MaterialDefaultDirective>,
    /// Required wgpu features parsed from the source or merged from alias + wrapper directives.
    pub wgpu_features: Vec<WgpuFeatureDirective>,
    /// Shader default render queue parsed from the source or inherited from an alias.
    pub default_render_queue: i32,
}

const DEFAULT_SHADER_RENDER_QUEUE: i32 = 2000;

/// Lists every `.wgsl` file directly under `dir`, sorted lexicographically.
pub(super) fn list_wgsl_files(dir: &Path) -> Result<Vec<PathBuf>, BuildError> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", dir.display())))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    paths.sort();
    Ok(paths)
}

/// Discovers all source shaders that must be compiled, in deterministic order.
pub(super) fn discover_shader_jobs(shader_root: &Path) -> Result<Vec<ShaderJob>, BuildError> {
    let mut jobs = Vec::new();
    for source_class in ShaderSourceClass::ALL {
        let dir = shader_root.join(source_class.source_subdir());
        if !dir.is_dir() {
            continue;
        }
        for source_path in list_wgsl_files(&dir)? {
            jobs.push(ShaderJob {
                compile_order: jobs.len(),
                source_class,
                source_path,
                validation: source_class.validation(),
            });
        }
    }
    Ok(jobs)
}

fn merge_texture_defaults(
    mut base: Vec<TextureDefaultDirective>,
    overrides: Vec<TextureDefaultDirective>,
) -> Vec<TextureDefaultDirective> {
    for override_default in overrides {
        if let Some(existing) = base
            .iter_mut()
            .find(|existing| existing.property == override_default.property)
        {
            *existing = override_default;
        } else {
            base.push(override_default);
        }
    }
    base
}

/// Merges material uniform defaults, with wrapper directives overriding alias defaults.
fn merge_material_defaults(
    mut base: Vec<MaterialDefaultDirective>,
    overrides: Vec<MaterialDefaultDirective>,
) -> Vec<MaterialDefaultDirective> {
    for override_default in overrides {
        if let Some(existing) = base
            .iter_mut()
            .find(|existing| existing.property == override_default.property)
        {
            *existing = override_default;
        } else {
            base.push(override_default);
        }
    }
    base
}

/// Merges required feature directives, preserving first-seen order.
fn merge_wgpu_features(
    mut base: Vec<WgpuFeatureDirective>,
    overrides: Vec<WgpuFeatureDirective>,
) -> Vec<WgpuFeatureDirective> {
    for required in overrides {
        if !base
            .iter()
            .any(|existing| existing.feature == required.feature)
        {
            base.push(required);
        }
    }
    base
}

/// Selects the wrapper render queue override, then alias default, then Unity's Geometry default.
fn merge_render_queue(
    base: Option<RenderQueueDirective>,
    override_queue: Option<RenderQueueDirective>,
) -> i32 {
    override_queue
        .or(base)
        .map_or(DEFAULT_SHADER_RENDER_QUEUE, |directive| directive.queue)
}

/// Loads the WGSL source used for composition, following `//#source_alias` when present.
pub(super) fn shader_source_for_compile(
    source_path: &Path,
) -> Result<ShaderCompileSource, BuildError> {
    let wrapper_source = fs::read_to_string(source_path)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", source_path.display())))?;
    let wrapper_file_path = source_path.to_str().ok_or_else(|| {
        BuildError::Message(format!(
            "shader path must be UTF-8: {}",
            source_path.display()
        ))
    })?;
    let wrapper_defaults = parse_texture_default_directives(&wrapper_source, wrapper_file_path)?;
    let wrapper_material_defaults =
        parse_material_default_directives(&wrapper_source, wrapper_file_path)?;
    let wrapper_wgpu_features = parse_wgpu_feature_directives(&wrapper_source, wrapper_file_path)?;
    let wrapper_render_queue = parse_render_queue_directive(&wrapper_source, wrapper_file_path)?;
    let Some(alias) = parse_source_alias(&wrapper_source, wrapper_file_path)? else {
        return Ok(ShaderCompileSource {
            source: wrapper_source,
            file_path: wrapper_file_path.to_string(),
            texture_defaults: wrapper_defaults,
            material_defaults: wrapper_material_defaults,
            wgpu_features: wrapper_wgpu_features,
            default_render_queue: merge_render_queue(None, wrapper_render_queue),
        });
    };
    let alias_path = source_path.with_file_name(format!("{alias}.wgsl"));
    if alias_path == source_path {
        return Err(BuildError::Message(format!(
            "{wrapper_file_path}: `//#source_alias` cannot point at itself"
        )));
    }
    let alias_source = fs::read_to_string(&alias_path)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", alias_path.display())))?;
    let alias_file_path = alias_path.to_str().ok_or_else(|| {
        BuildError::Message(format!(
            "shader alias path must be UTF-8: {}",
            alias_path.display()
        ))
    })?;
    let alias_defaults = parse_texture_default_directives(&alias_source, alias_file_path)?;
    let alias_material_defaults =
        parse_material_default_directives(&alias_source, alias_file_path)?;
    let alias_wgpu_features = parse_wgpu_feature_directives(&alias_source, alias_file_path)?;
    let alias_render_queue = parse_render_queue_directive(&alias_source, alias_file_path)?;
    Ok(ShaderCompileSource {
        source: alias_source,
        file_path: alias_file_path.to_string(),
        texture_defaults: merge_texture_defaults(alias_defaults, wrapper_defaults),
        material_defaults: merge_material_defaults(
            alias_material_defaults,
            wrapper_material_defaults,
        ),
        wgpu_features: merge_wgpu_features(alias_wgpu_features, wrapper_wgpu_features),
        default_render_queue: merge_render_queue(alias_render_queue, wrapper_render_queue),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// Source discovery follows the algebraic class order and new shader layout.
    #[test]
    fn discovers_jobs_in_class_order() -> Result<(), BuildError> {
        let root = tempfile::tempdir()?;
        for subdir in ["materials", "passes/post", "passes/compute"] {
            fs::create_dir_all(root.path().join(subdir))?;
        }
        fs::write(root.path().join("passes/compute/compute_b.wgsl"), "")?;
        fs::write(root.path().join("materials/mat_b.wgsl"), "")?;
        fs::write(root.path().join("materials/mat_a.wgsl"), "")?;
        fs::write(root.path().join("passes/post/post_a.wgsl"), "")?;

        let jobs = discover_shader_jobs(root.path())?;
        let names = jobs
            .iter()
            .map(|job| {
                job.source_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("")
            })
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            ["mat_a.wgsl", "mat_b.wgsl", "post_a.wgsl", "compute_b.wgsl"]
        );
        assert_eq!(jobs[0].source_class, ShaderSourceClass::Material);
        assert_eq!(jobs[2].source_class, ShaderSourceClass::Post);
        assert_eq!(jobs[3].source_class, ShaderSourceClass::Compute);
        Ok(())
    }

    #[test]
    fn source_alias_inherits_and_overrides_texture_defaults() -> Result<(), BuildError> {
        let root = tempfile::tempdir()?;
        let alias_path = root.path().join("base.wgsl");
        let wrapper_path = root.path().join("wrapper.wgsl");
        fs::write(
            &alias_path,
            r#"
//#texture_default _MainTex white
//#texture_default _MaskTex black
"#,
        )?;
        fs::write(
            &wrapper_path,
            r#"
//#source_alias base
//#texture_default _MaskTex white
//#texture_default _OtherTex grey
"#,
        )?;

        let source = shader_source_for_compile(&wrapper_path)?;

        assert_eq!(source.file_path, alias_path.to_string_lossy().as_ref());
        assert_eq!(
            source
                .texture_defaults
                .iter()
                .map(|d| (d.property.as_str(), d.kind))
                .collect::<Vec<_>>(),
            [
                (
                    "_MainTex",
                    super::super::directives::TextureDefaultKind::White
                ),
                (
                    "_MaskTex",
                    super::super::directives::TextureDefaultKind::White
                ),
                (
                    "_OtherTex",
                    super::super::directives::TextureDefaultKind::Gray
                ),
            ]
        );
        Ok(())
    }

    #[test]
    fn source_alias_inherits_and_overrides_material_defaults() -> Result<(), BuildError> {
        let root = tempfile::tempdir()?;
        let alias_path = root.path().join("base.wgsl");
        let wrapper_path = root.path().join("wrapper.wgsl");
        fs::write(
            &alias_path,
            r#"
//#mat_default _GlossMapScale float 1.0
//#mat_default _Tint vec4 1.0 1.0 1.0 1.0
"#,
        )?;
        fs::write(
            &wrapper_path,
            r#"
//#source_alias base
//#mat_default _GlossMapScale float 0.5
//#mat_default _OcclusionStrength float 1.0
"#,
        )?;

        let source = shader_source_for_compile(&wrapper_path)?;

        assert_eq!(source.file_path, alias_path.to_string_lossy().as_ref());
        assert_eq!(
            source
                .material_defaults
                .iter()
                .map(|d| (d.property.as_str(), d.value))
                .collect::<Vec<_>>(),
            [
                (
                    "_GlossMapScale",
                    super::super::directives::MaterialDefaultValue::float_bits(0.5f32.to_bits()),
                ),
                (
                    "_Tint",
                    super::super::directives::MaterialDefaultValue::vec4_bits([
                        1.0f32.to_bits(),
                        1.0f32.to_bits(),
                        1.0f32.to_bits(),
                        1.0f32.to_bits(),
                    ]),
                ),
                (
                    "_OcclusionStrength",
                    super::super::directives::MaterialDefaultValue::float_bits(1.0f32.to_bits()),
                ),
            ]
        );
        Ok(())
    }

    #[test]
    fn source_alias_inherits_required_wgpu_features() -> Result<(), BuildError> {
        let root = tempfile::tempdir()?;
        let alias_path = root.path().join("base.wgsl");
        let wrapper_path = root.path().join("wrapper.wgsl");
        fs::write(&alias_path, "//#wgpu_feature shader_barycentrics\n")?;
        fs::write(
            &wrapper_path,
            r#"
//#source_alias base
//#wgpu_feature shader_barycentrics
"#,
        )?;

        let source = shader_source_for_compile(&wrapper_path)?;

        assert_eq!(source.wgpu_features.len(), 1);
        assert_eq!(
            source.wgpu_features[0].feature,
            super::super::directives::BuildWgpuFeature::ShaderBarycentrics
        );
        Ok(())
    }

    #[test]
    fn source_without_render_queue_uses_geometry_default() -> Result<(), BuildError> {
        let root = tempfile::tempdir()?;
        let source_path = root.path().join("material.wgsl");
        fs::write(&source_path, "")?;

        let source = shader_source_for_compile(&source_path)?;

        assert_eq!(source.default_render_queue, 2000);
        Ok(())
    }

    #[test]
    fn source_alias_inherits_and_overrides_render_queue() -> Result<(), BuildError> {
        let root = tempfile::tempdir()?;
        let alias_path = root.path().join("base.wgsl");
        let inherited_path = root.path().join("inherited.wgsl");
        let override_path = root.path().join("override.wgsl");
        fs::write(&alias_path, "//#render_queue Transparent+500\n")?;
        fs::write(&inherited_path, "//#source_alias base\n")?;
        fs::write(
            &override_path,
            r#"
//#source_alias base
//#render_queue AlphaTest+200
"#,
        )?;

        let inherited = shader_source_for_compile(&inherited_path)?;
        let overridden = shader_source_for_compile(&override_path)?;

        assert_eq!(inherited.default_render_queue, 3500);
        assert_eq!(overridden.default_render_queue, 2650);
        Ok(())
    }
}
