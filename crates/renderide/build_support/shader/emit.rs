//! Flattened WGSL and generated Rust emission.

use std::fs;
use std::path::Path;

use super::directives::{
    BuildPassDirective, MaterialDefaultDirective, TextureDefaultDirective, WgpuFeatureDirective,
    material_default_literal, pass_literal, texture_default_literal, wgpu_features_literal,
};
use super::error::BuildError;
use super::model::{CompiledShader, ShaderSourceClass};

const DEFAULT_SHADER_RENDER_QUEUE: i32 = 2000;

/// Escapes `s` as a Rust `str` literal token.
fn rust_string_literal_token(s: &str) -> String {
    format!("{s:?}")
}

/// Per-source-class composed shader output and generated Rust accumulators.
#[derive(Debug)]
pub(super) struct ComposedShaders {
    material_stems: Vec<String>,
    post_stems: Vec<String>,
    backend_stems: Vec<String>,
    compute_stems: Vec<String>,
    present_stems: Vec<String>,
    embedded_arms: String,
    embedded_macro_arms: String,
    embedded_pass_arms: String,
    embedded_render_queue_arms: String,
    embedded_required_features_arms: String,
    embedded_texture_default_arms: String,
    embedded_material_default_arms: String,
}

struct EmbeddedTargetEmit<'a> {
    target_stem: &'a str,
    wgsl: &'a str,
    pass_directives: &'a [BuildPassDirective],
    default_render_queue: i32,
    wgpu_features: &'a [WgpuFeatureDirective],
    texture_defaults: &'a [TextureDefaultDirective],
    material_defaults: &'a [MaterialDefaultDirective],
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
            embedded_arms: String::new(),
            embedded_macro_arms: String::new(),
            embedded_pass_arms: String::new(),
            embedded_render_queue_arms: String::new(),
            embedded_required_features_arms: String::new(),
            embedded_texture_default_arms: String::new(),
            embedded_material_default_arms: String::new(),
        }
    }

    /// Records one compiled shader source into embedded shader registries.
    pub(super) fn record_compiled_shader(&mut self, compiled: &CompiledShader) {
        for target in &compiled.targets {
            self.emit_embedded_target(EmbeddedTargetEmit {
                target_stem: &target.target_stem,
                wgsl: &target.wgsl,
                pass_directives: &target.pass_directives,
                default_render_queue: compiled.default_render_queue,
                wgpu_features: &compiled.wgpu_features,
                texture_defaults: &compiled.texture_defaults,
                material_defaults: &compiled.material_defaults,
            });
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

    /// Emits generated Rust registry fragments for one compiled target.
    fn emit_embedded_target(&mut self, target: EmbeddedTargetEmit<'_>) {
        use std::fmt::Write as _;

        let EmbeddedTargetEmit {
            target_stem,
            wgsl,
            pass_directives,
            default_render_queue,
            wgpu_features,
            texture_defaults,
            material_defaults,
        } = target;
        let lit = rust_string_literal_token(wgsl);
        let _ = writeln!(
            self.embedded_arms,
            "        \"{target_stem}\" => Some({lit}),"
        );
        let _ = writeln!(
            self.embedded_macro_arms,
            "            \"{target_stem}\" => {lit},"
        );
        if !pass_directives.is_empty() {
            let pass_literals = pass_directives
                .iter()
                .map(pass_literal)
                .collect::<Vec<_>>()
                .join(",\n            ");
            let _ = writeln!(
                self.embedded_pass_arms,
                "        \"{target_stem}\" => const {{ &[\n            {pass_literals},\n        ] }},"
            );
        }
        if default_render_queue != DEFAULT_SHADER_RENDER_QUEUE {
            let _ = writeln!(
                self.embedded_render_queue_arms,
                "        \"{target_stem}\" => {default_render_queue},"
            );
        }
        if !wgpu_features.is_empty() {
            let features = wgpu_features_literal(wgpu_features);
            let _ = writeln!(
                self.embedded_required_features_arms,
                "        \"{target_stem}\" => {features},"
            );
        }
        if !texture_defaults.is_empty() {
            let default_literals = texture_defaults
                .iter()
                .map(texture_default_literal)
                .collect::<Vec<_>>()
                .join(",\n            ");
            let _ = writeln!(
                self.embedded_texture_default_arms,
                "        \"{target_stem}\" => const {{ &[\n            {default_literals},\n        ] }},"
            );
        }
        if !material_defaults.is_empty() {
            let default_literals = material_defaults
                .iter()
                .map(material_default_literal)
                .collect::<Vec<_>>()
                .join(",\n            ");
            let _ = writeln!(
                self.embedded_material_default_arms,
                "        \"{target_stem}\" => const {{ &[\n            {default_literals},\n        ] }},"
            );
        }
    }
}

/// Removes generated `.wgsl` inspection outputs so deleted/renamed shader sources do not linger.
pub(super) fn clean_target_dir(target_dir: &Path) -> Result<(), BuildError> {
    fs::create_dir_all(target_dir)?;
    for entry in fs::read_dir(target_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "wgsl") {
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
        match $stem {{
{embedded_macro_arms}            _ => "",
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

/// Flattened WGSL for `stem` (also written under `shaders/target/{{stem}}.wgsl` at build time).
#[expect(clippy::too_many_lines, reason = "match arm per embedded shader target; scales with shader count")]
pub fn embedded_target_wgsl(stem: &str) -> Option<&'static str> {{
    match stem {{
{embedded_arms}        _ => None,
    }}
}}

/// Declared render passes for `stem`, parsed from `//#pass` directives in the source WGSL.
#[expect(clippy::too_many_lines, reason = "match arm per embedded shader target; scales with shader count")]
pub fn embedded_target_passes(stem: &str) -> &'static [crate::materials::MaterialPassDesc] {{
    match stem {{
{embedded_pass_arms}        _ => &[],
    }}
}}

/// Shader default render queue for `stem`, parsed from `//#render_queue` directives.
#[expect(clippy::too_many_lines, reason = "match arm per embedded shader target; scales with shader count")]
pub fn embedded_target_default_render_queue(stem: &str) -> i32 {{
    match stem {{
{embedded_render_queue_arms}        _ => 2000,
    }}
}}

/// Required device features for `stem`, parsed from `//#wgpu_feature` directives in the source WGSL.
pub fn embedded_target_required_features(stem: &str) -> wgpu::Features {{
    match stem {{
{embedded_required_features_arms}        _ => wgpu::Features::empty(),
    }}
}}

/// Declared texture fallbacks for `stem`, parsed from `//#texture_default` directives in the source WGSL.
#[expect(clippy::too_many_lines, reason = "match arm per embedded shader target; scales with shader count")]
pub fn embedded_target_texture_defaults(stem: &str) -> &'static [EmbeddedTextureDefault] {{
    match stem {{
{embedded_texture_default_arms}        _ => &[],
    }}
}}

/// Declared material uniform fallbacks for `stem`, parsed from `//#mat_default` directives in the source WGSL.
#[expect(clippy::too_many_lines, reason = "match arm per embedded shader target; scales with shader count")]
pub fn embedded_target_material_defaults(stem: &str) -> &'static [EmbeddedMaterialDefault] {{
    match stem {{
{embedded_material_default_arms}        _ => &[],
    }}
}}

/// Material target stems (composed from `shaders/materials/*.wgsl`).
#[cfg(test)]
pub const COMPILED_MATERIAL_STEMS: &[&str] = &[
{material_stems}
];
"#,
        embedded_arms = c.embedded_arms,
        embedded_macro_arms = c.embedded_macro_arms,
        embedded_pass_arms = c.embedded_pass_arms,
        embedded_render_queue_arms = c.embedded_render_queue_arms,
        embedded_required_features_arms = c.embedded_required_features_arms,
        embedded_texture_default_arms = c.embedded_texture_default_arms,
        embedded_material_default_arms = c.embedded_material_default_arms,
        material_default_type_defs = embedded_material_default_type_defs(),
        material_stems = stems_list(&c.material_stems),
    )
}

#[cfg(test)]
mod tests {
    use crate::shader::directives::{
        BuildBlend, BuildColorWrites, BuildCullMode, BuildDepthCompare, BuildDepthCompareDomain,
        BuildMaterialPassState, BuildPassDirective, BuildPassType, BuildRenderStatePolicy,
        MaterialDefaultDirective, MaterialDefaultValue, TextureDefaultDirective,
        TextureDefaultKind, WgpuFeatureDirective,
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

    /// Embedded pass metadata stays attached to emitted shader targets.
    #[test]
    fn compiled_shader_preserves_pass_metadata() -> Result<(), BuildError> {
        let target_dir = tempfile::tempdir()?;
        let mut composed = ComposedShaders::new();
        let compiled = fake_compiled_shader(
            0,
            ShaderSourceClass::Material,
            &[("outline_default", "wgsl body")],
            FakeCompiledShaderMetadata {
                pass_directives: vec![
                    BuildPassDirective {
                        pass_type: BuildPassType::Forward,
                        name: "forward".to_string(),
                        fragment_entry: "fs_main".to_string(),
                        vertex_entry: "vs_main".to_string(),
                        alpha_to_coverage: true,
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
                        alpha_to_coverage: false,
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
                ],
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

        assert!(embedded.contains("pass_type: crate::materials::PassType::Forward"));
        assert!(embedded.contains("alpha_to_coverage: true"));
        assert!(embedded.contains(
            "name: \"outline\", pass_type: crate::materials::PassType::Forward, vertex_entry: \"vs_outline\", fragment_entry: \"fs_outline\""
        ));
        assert!(embedded.contains("COMPILED_MATERIAL_STEMS"));
        assert!(embedded.contains("EmbeddedTextureDefaultKind"));
        assert!(embedded.contains(
            "EmbeddedTextureDefault { property: \"_MainTex\", kind: EmbeddedTextureDefaultKind::White }"
        ));
        assert!(embedded.contains(
            "EmbeddedMaterialDefault { property: \"_GlossMapScale\", value: EmbeddedMaterialDefaultValue::float(f32::from_bits(0x3f80_0000)) }"
        ));
        assert!(embedded.contains("embedded_target_texture_defaults"));
        assert!(embedded.contains("embedded_target_material_defaults"));
        assert!(embedded.contains("embedded_target_default_render_queue"));
        assert!(embedded.contains("embedded_target_required_features"));
        assert!(embedded.contains("\"outline_default\" => 3500,"));
        assert!(embedded.contains("wgpu::Features::SHADER_BARYCENTRICS"));
        assert!(embedded.contains("macro_rules! embedded_wgsl"));
        assert!(embedded.contains("\"outline_default\" => \"wgsl body\","));
        assert!(!embedded.contains("pub const OUTLINE_DEFAULT_WGSL"));
        Ok(())
    }

    /// Stale WGSL inspection outputs are removed before current targets are emitted.
    #[test]
    fn clean_target_dir_removes_stale_wgsl_only() -> Result<(), BuildError> {
        let target_dir = tempfile::tempdir()?;
        fs::write(target_dir.path().join("old.wgsl"), "old")?;
        fs::write(target_dir.path().join("keep.txt"), "keep")?;

        clean_target_dir(target_dir.path())?;

        assert!(!target_dir.path().join("old.wgsl").exists());
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
