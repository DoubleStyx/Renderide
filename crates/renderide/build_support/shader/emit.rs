//! Flattened WGSL and generated Rust emission.

use std::fs;
use std::path::Path;

use super::error::BuildError;
use super::model::{CompiledShader, ShaderSourceClass};
use super::shader_package_schema::SHADER_PACKAGE_MANIFEST_FILE;

#[cfg(test)]
const DEFAULT_SHADER_RENDER_QUEUE: i32 = 2000;

/// Per-source-class composed shader output for generated Rust compatibility lists.
#[derive(Debug)]
pub(super) struct ComposedShaders {
    material_stems: Vec<String>,
    post_stems: Vec<String>,
    backend_stems: Vec<String>,
    compute_stems: Vec<String>,
    present_stems: Vec<String>,
}

impl ComposedShaders {
    /// Creates empty shader-output accumulators.
    pub(super) const fn new() -> Self {
        Self {
            material_stems: Vec::new(),
            post_stems: Vec::new(),
            backend_stems: Vec::new(),
            compute_stems: Vec::new(),
            present_stems: Vec::new(),
        }
    }

    /// Records one compiled shader source into embedded shader registries.
    pub(super) fn record_compiled_shader(&mut self, compiled: &CompiledShader) {
        for target in &compiled.targets {
            self.push_stem(compiled.source_class, target.target_stem.clone());
        }
    }

    /// Appends one compiled target stem to its source-class list.
    fn push_stem(&mut self, source_class: ShaderSourceClass, stem: String) {
        match source_class {
            ShaderSourceClass::Material => self.material_stems.push(stem),
            ShaderSourceClass::Post => self.post_stems.push(stem),
            ShaderSourceClass::Backend => self.backend_stems.push(stem),
            ShaderSourceClass::Compute => self.compute_stems.push(stem),
            ShaderSourceClass::Present => self.present_stems.push(stem),
        }
    }
}

/// Removes generated `.wgsl` inspection outputs so deleted/renamed shader sources do not linger.
pub(super) fn clean_target_dir(target_dir: &Path) -> Result<(), BuildError> {
    fs::create_dir_all(target_dir)?;
    for entry in fs::read_dir(target_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "wgsl")
            || path
                .file_name()
                .is_some_and(|name| name == SHADER_PACKAGE_MANIFEST_FILE)
        {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

/// Writes flattened WGSL inspection files for one compiled shader source.
pub(super) fn write_compiled_shader_targets(
    compiled: &CompiledShader,
    target_dir: &Path,
) -> Result<(), BuildError> {
    for target in &compiled.targets {
        let out_path = target_dir.join(format!("{}.wgsl", target.target_stem));
        fs::write(&out_path, &target.wgsl)?;
    }
    Ok(())
}

/// Serially emits files and embedded registry data for one compiled shader source.
pub(super) fn emit_compiled_shader(
    compiled: &CompiledShader,
    target_dir: &Path,
    out: &mut ComposedShaders,
) -> Result<(), BuildError> {
    write_compiled_shader_targets(compiled, target_dir)?;
    out.record_compiled_shader(compiled);
    Ok(())
}

/// Generated Rust definitions for material uniform defaults.
fn embedded_material_default_type_defs() -> &'static str {
    r#"/// Material uniform fallback value kind parsed from `//#mat_default` directives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddedMaterialDefaultKind {
    /// Unity float property default.
    Float,
    /// Unity vector/color property default.
    Vec4,
}

/// Material uniform fallback value parsed from `//#mat_default` directives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbeddedMaterialDefaultValue {
    /// Unity material property default kind.
    pub kind: EmbeddedMaterialDefaultKind,
    /// Unity material property default values. Float defaults use only the first element.
    pub values: [f32; 4],
}

impl EmbeddedMaterialDefaultValue {
    /// Creates a float material default.
    pub const fn float(value: f32) -> Self {
        Self {
            kind: EmbeddedMaterialDefaultKind::Float,
            values: [value, 0.0, 0.0, 0.0],
        }
    }

    /// Creates a vec4 material default.
    pub const fn vec4(values: [f32; 4]) -> Self {
        Self {
            kind: EmbeddedMaterialDefaultKind::Vec4,
            values,
        }
    }
}

/// One reflected material uniform property and its Unity fallback value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbeddedMaterialDefault {
    /// Reflected host material uniform property name.
    pub property: &'static str,
    /// Unity fallback value for the uniform property.
    pub value: EmbeddedMaterialDefaultValue,
}
"#
}

/// Generated Rust definitions for embedded shader descriptors.
fn embedded_shader_descriptor_type_defs() -> &'static str {
    r#"/// Mesh streams required by the reflected material vertex entries.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EmbeddedVertexStreamMask {
    /// UV0 stream at `@location(2)`.
    pub uv0: bool,
    /// Vertex color stream at `@location(3)`.
    pub color: bool,
    /// Tangent stream at `@location(4)`.
    pub tangent: bool,
    /// UV1 stream at `@location(5)`.
    pub uv1: bool,
    /// UV2 stream at `@location(6)`.
    pub uv2: bool,
    /// UV3 stream at `@location(7)`.
    pub uv3: bool,
    /// Packed UV0-UV3 stream for 3D/4D low UV inputs.
    pub wide_low_uvs: bool,
    /// Packed UV4-UV7 stream for high UV inputs.
    pub wide_high_uvs: bool,
}

/// Scene-snapshot and intersection requirements reflected at build time.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EmbeddedSnapshotRequirements {
    /// True when the shader declares a scene-color snapshot binding.
    pub uses_scene_color: bool,
    /// True when the shader declares a scene-depth snapshot binding.
    pub uses_scene_depth: bool,
    /// True when the material uniform block declares intersection tint.
    pub requires_intersection_pass: bool,
}

/// Stable reflected metadata for a compiled shader target.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EmbeddedShaderReflection {
    /// Mesh streams required by the material vertex entries.
    pub vertex_stream_mask: EmbeddedVertexStreamMask,
    /// Scene-snapshot and intersection requirements.
    pub snapshot_requirements: EmbeddedSnapshotRequirements,
    /// Whether the target decodes `_RenderideVariantBits`.
    pub uses_renderide_variant_bits: bool,
    /// Whether the single forward pass can be mirrored by the generic depth prepass.
    pub supports_generic_depth_prepass: bool,
}
"#
}

/// Generated Rust lookup wrappers over the embedded shader descriptor table.
fn embedded_shader_lookup_fn_defs() -> &'static str {
    r#"/// Flattened WGSL for `stem` from the runtime shader package.
pub fn embedded_target_wgsl(stem: &str) -> Option<&'static str> {
    crate::shader_package::target_wgsl(stem)
}

/// Declared render passes for `stem`, parsed from package TOML metadata.
pub fn embedded_target_passes(stem: &str) -> &'static [crate::materials::MaterialPassDesc] {
    crate::shader_package::material_passes(stem)
}

/// Shader default render queue for `stem`, parsed from package TOML metadata.
pub fn embedded_target_default_render_queue(stem: &str) -> i32 {
    crate::shader_package::material_default_render_queue(stem)
}

/// Required device features for `stem`, parsed from package TOML metadata.
pub fn embedded_target_required_features(stem: &str) -> wgpu::Features {
    crate::shader_package::target_required_features(stem)
}

/// Declared texture fallbacks for `stem`, parsed from package TOML metadata.
pub fn embedded_target_texture_defaults(stem: &str) -> &'static [EmbeddedTextureDefault] {
    crate::shader_package::material_texture_defaults(stem)
}

/// Declared material uniform fallbacks for `stem`, parsed from package TOML metadata.
pub fn embedded_target_material_defaults(stem: &str) -> &'static [EmbeddedMaterialDefault] {
    crate::shader_package::material_uniform_defaults(stem)
}

/// Stable reflection metadata for `stem`, reflected from WGSL with Naga at runtime.
pub fn embedded_target_reflection(stem: &str) -> EmbeddedShaderReflection {
    crate::shader_package::material_reflection(stem)
}
"#
}

/// Renders generated `embedded_shaders.rs`.
pub(super) fn render_embedded_shaders_rs(c: &ComposedShaders) -> String {
    let stems_list = |stems: &[String]| {
        stems
            .iter()
            .map(|s| format!("    \"{s}\","))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        r#"// Generated by `build.rs` - do not edit.

/// Returns embedded WGSL for a compile-time known shader target stem.
macro_rules! embedded_wgsl {{
    ($stem:literal) => {{
        match $crate::embedded_shaders::embedded_target_wgsl($stem) {{
            Some(source) => source,
            None => {{
                logger::error!("shader package target missing: {{}}", $stem);
                ""
            }}
        }}
    }};
}}

pub(crate) use embedded_wgsl;

/// Texture fallback token parsed from `//#texture_default` material directives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddedTextureDefaultKind {{
    /// Unity `"white"` default texture.
    White,
    /// Unity `"black"` default texture.
    Black,
    /// Unity `"gray"` / `"grey"` default texture.
    Gray,
    /// Unity `"bump"` default texture.
    Bump,
    /// Unity `"red"` default texture.
    Red,
    /// Empty Unity texture default (`""`), resolved as Unity's gray placeholder.
    Empty,
}}

const _: &[EmbeddedTextureDefaultKind] = &[
    EmbeddedTextureDefaultKind::White,
    EmbeddedTextureDefaultKind::Black,
    EmbeddedTextureDefaultKind::Gray,
    EmbeddedTextureDefaultKind::Bump,
    EmbeddedTextureDefaultKind::Red,
    EmbeddedTextureDefaultKind::Empty,
];

/// One reflected material texture property and its Unity fallback token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddedTextureDefault {{
    /// Reflected host material texture property name.
    pub property: &'static str,
    /// Unity fallback token for the texture property.
    pub kind: EmbeddedTextureDefaultKind,
}}

{material_default_type_defs}

{descriptor_type_defs}

{lookup_fn_defs}

/// Material target stems (composed from `shaders/materials/*.wgsl`).
#[cfg(test)]
pub const COMPILED_MATERIAL_STEMS: &[&str] = &[
{material_stems}
];
"#,
        material_default_type_defs = embedded_material_default_type_defs(),
        descriptor_type_defs = embedded_shader_descriptor_type_defs(),
        lookup_fn_defs = embedded_shader_lookup_fn_defs(),
        material_stems = stems_list(&c.material_stems),
    )
}

#[cfg(test)]
mod tests {
    use crate::shader::directives::{
        BuildAlphaToCoverageMode, BuildBlend, BuildColorWrites, BuildCullMode, BuildDepthCompare,
        BuildDepthCompareDomain, BuildMaterialPassState, BuildPassDirective, BuildPassType,
        BuildRenderStatePolicy, MaterialDefaultDirective, MaterialDefaultValue,
        TextureDefaultDirective, TextureDefaultKind, WgpuFeatureDirective,
    };
    use crate::shader::model::{CompiledShader, CompiledShaderTarget, ShaderSourceClass};

    use super::*;

    /// Single- and dual-target shader outputs keep the emitted target shape.
    #[test]
    fn compiled_shader_emits_single_and_dual_targets() -> Result<(), BuildError> {
        let target_dir = tempfile::tempdir()?;
        let mut composed = ComposedShaders::new();
        let single = fake_compiled_shader(
            0,
            ShaderSourceClass::Material,
            &[("single", "single wgsl")],
            FakeCompiledShaderMetadata::default(),
        );
        let dual = fake_compiled_shader(
            1,
            ShaderSourceClass::Post,
            &[
                ("dual_default", "default wgsl"),
                ("dual_multiview", "multiview wgsl"),
            ],
            FakeCompiledShaderMetadata::default(),
        );

        emit_compiled_shader(&single, target_dir.path(), &mut composed)?;
        emit_compiled_shader(&dual, target_dir.path(), &mut composed)?;

        assert!(target_dir.path().join("single.wgsl").is_file());
        assert!(target_dir.path().join("dual_default.wgsl").is_file());
        assert!(target_dir.path().join("dual_multiview.wgsl").is_file());
        assert_eq!(composed.material_stems, ["single"]);
        assert_eq!(composed.post_stems, ["dual_default", "dual_multiview"]);
        Ok(())
    }

    /// Generated Rust stays small and delegates shader lookups to the runtime package.
    #[test]
    fn compiled_shader_emits_package_facade() -> Result<(), BuildError> {
        let target_dir = tempfile::tempdir()?;
        let mut composed = ComposedShaders::new();
        let compiled = fake_compiled_shader(
            0,
            ShaderSourceClass::Material,
            &[("outline_default", "wgsl body")],
            FakeCompiledShaderMetadata {
                pass_directives: pass_metadata_directives(),
                texture_defaults: vec![
                    TextureDefaultDirective {
                        property: "_MainTex".to_string(),
                        kind: TextureDefaultKind::White,
                    },
                    TextureDefaultDirective {
                        property: "_EmissionMap".to_string(),
                        kind: TextureDefaultKind::Black,
                    },
                ],
                material_defaults: vec![MaterialDefaultDirective {
                    property: "_GlossMapScale".to_string(),
                    value: MaterialDefaultValue::float_bits(1.0f32.to_bits()),
                }],
                wgpu_features: vec![WgpuFeatureDirective {
                    feature: crate::shader::directives::BuildWgpuFeature::ShaderBarycentrics,
                }],
                default_render_queue: 3500,
            },
        );

        emit_compiled_shader(&compiled, target_dir.path(), &mut composed)?;
        let embedded = render_embedded_shaders_rs(&composed);

        assert!(embedded.contains("COMPILED_MATERIAL_STEMS"));
        assert!(embedded.contains("EmbeddedTextureDefaultKind"));
        assert!(embedded.contains("embedded_target_texture_defaults"));
        assert!(embedded.contains("embedded_target_material_defaults"));
        assert!(embedded.contains("embedded_target_default_render_queue"));
        assert!(embedded.contains("embedded_target_required_features"));
        assert!(!embedded.contains("pub struct EmbeddedShaderTargetDesc"));
        assert!(!embedded.contains("pub fn embedded_target_desc"));
        assert!(embedded.contains("pub fn embedded_target_reflection"));
        assert!(embedded.contains("macro_rules! embedded_wgsl"));
        assert!(embedded.contains("crate::shader_package::target_wgsl"));
        assert!(!embedded.contains("wgsl body"));
        assert!(!embedded.contains("pub const OUTLINE_DEFAULT_WGSL"));
        Ok(())
    }

    fn pass_metadata_directives() -> Vec<BuildPassDirective> {
        vec![
            BuildPassDirective {
                pass_type: BuildPassType::Forward,
                name: "forward".to_string(),
                fragment_entry: "fs_main".to_string(),
                vertex_entry: "vs_main".to_string(),
                alpha_to_coverage: BuildAlphaToCoverageMode::Always,
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
            },
            BuildPassDirective {
                pass_type: BuildPassType::Forward,
                name: "outline".to_string(),
                fragment_entry: "fs_outline".to_string(),
                vertex_entry: "vs_outline".to_string(),
                alpha_to_coverage: BuildAlphaToCoverageMode::Off,
                depth_compare_domain: BuildDepthCompareDomain::FrooxZTest,
                depth_compare: BuildDepthCompare::Main,
                depth_write: true,
                cull_mode: BuildCullMode::Front,
                blend: BuildBlend::Off,
                write_mask: BuildColorWrites::Rgb,
                depth_bias_slope_scale_bits: 0.0f32.to_bits(),
                depth_bias_constant: 0,
                material_state: BuildMaterialPassState::Static,
                render_state_policy: BuildRenderStatePolicy {
                    color_mask: true,
                    depth_write: true,
                    depth_compare: true,
                    cull: false,
                    stencil: true,
                    depth_offset: true,
                },
            },
        ]
    }

    /// Stale WGSL inspection outputs are removed before current targets are emitted.
    #[test]
    fn clean_target_dir_removes_stale_wgsl_only() -> Result<(), BuildError> {
        let target_dir = tempfile::tempdir()?;
        fs::write(target_dir.path().join("old.wgsl"), "old")?;
        fs::write(
            target_dir.path().join(SHADER_PACKAGE_MANIFEST_FILE),
            "stale",
        )?;
        fs::write(target_dir.path().join("keep.txt"), "keep")?;

        clean_target_dir(target_dir.path())?;

        assert!(!target_dir.path().join("old.wgsl").exists());
        assert!(
            !target_dir
                .path()
                .join(SHADER_PACKAGE_MANIFEST_FILE)
                .exists()
        );
        assert!(target_dir.path().join("keep.txt").is_file());
        Ok(())
    }

    struct FakeCompiledShaderMetadata {
        pass_directives: Vec<BuildPassDirective>,
        texture_defaults: Vec<TextureDefaultDirective>,
        material_defaults: Vec<MaterialDefaultDirective>,
        wgpu_features: Vec<WgpuFeatureDirective>,
        default_render_queue: i32,
    }

    impl Default for FakeCompiledShaderMetadata {
        fn default() -> Self {
            Self {
                pass_directives: Vec::new(),
                texture_defaults: Vec::new(),
                material_defaults: Vec::new(),
                wgpu_features: Vec::new(),
                default_render_queue: DEFAULT_SHADER_RENDER_QUEUE,
            }
        }
    }

    fn fake_compiled_shader(
        compile_order: usize,
        source_class: ShaderSourceClass,
        targets: &[(&str, &str)],
        metadata: FakeCompiledShaderMetadata,
    ) -> CompiledShader {
        let FakeCompiledShaderMetadata {
            pass_directives,
            texture_defaults,
            material_defaults,
            wgpu_features,
            default_render_queue,
        } = metadata;
        let target_pass_directives = pass_directives.clone();
        CompiledShader {
            compile_order,
            source_stem: targets
                .first()
                .map_or("shader", |(target_stem, _)| *target_stem)
                .trim_end_matches("_default")
                .trim_end_matches("_multiview")
                .to_string(),
            source_class,
            pass_directives,
            texture_defaults,
            material_defaults,
            wgpu_features,
            default_render_queue,
            targets: targets
                .iter()
                .map(|(target_stem, wgsl)| CompiledShaderTarget {
                    target_stem: (*target_stem).to_string(),
                    wgsl: (*wgsl).to_string(),
                    pass_directives: target_pass_directives.clone(),
                })
                .collect(),
        }
    }
}
