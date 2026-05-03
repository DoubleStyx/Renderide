//! Flattened WGSL and generated Rust emission.

use std::fs;
use std::path::Path;

use super::directives::{BuildPassDirective, pass_literal};
use super::error::BuildError;
use super::model::{CompiledShader, ShaderSourceClass};

/// Escapes `s` as a Rust `str` literal token.
fn rust_string_literal_token(s: &str) -> String {
    format!("{s:?}")
}

/// Maps a shader stem to a SCREAMING_SNAKE_CASE Rust identifier for WGSL constants.
fn stem_to_const_ident(stem: &str) -> String {
    let mut out = String::with_capacity(stem.len());
    for c in stem.chars() {
        match c {
            c if c.is_ascii_alphanumeric() => out.push(c.to_ascii_uppercase()),
            '_' => out.push('_'),
            '.' => out.push_str("_DOT_"),
            '-' => out.push_str("_DASH_"),
            _ => out.push('_'),
        }
    }
    out
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
    embedded_pass_arms: String,
    embedded_consts: String,
    embedded_targets: String,
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
            embedded_pass_arms: String::new(),
            embedded_consts: String::new(),
            embedded_targets: String::new(),
        }
    }

    /// Records one compiled shader source into embedded shader registries.
    pub(super) fn record_compiled_shader(&mut self, compiled: &CompiledShader) {
        for target in &compiled.targets {
            self.emit_embedded_target(
                &target.target_stem,
                &target.wgsl,
                compiled.source_class,
                &compiled.pass_directives,
            );
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
    fn emit_embedded_target(
        &mut self,
        target_stem: &str,
        wgsl: &str,
        source_class: ShaderSourceClass,
        pass_directives: &[BuildPassDirective],
    ) {
        use std::fmt::Write as _;

        let lit = rust_string_literal_token(wgsl);
        let _ = writeln!(
            self.embedded_arms,
            "        \"{target_stem}\" => Some({lit}),"
        );
        let const_ident = stem_to_const_ident(target_stem);
        let _ = writeln!(
            self.embedded_consts,
            "/// Composed WGSL for the `{target_stem}` embedded shader target.\npub const {const_ident}_WGSL: &str = {lit};"
        );
        let class_variant = source_class.embedded_class_variant();
        let _ = writeln!(
            self.embedded_targets,
            "    EmbeddedShaderTarget {{ stem: \"{target_stem}\", class: EmbeddedShaderClass::{class_variant} }},"
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

{embedded_consts}
/// Logical source class for an embedded shader target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddedShaderClass {{
    /// Material shader target.
    Material,
    /// Post-processing shader target.
    Post,
    /// Backend utility shader target.
    Backend,
    /// Compute shader target.
    Compute,
    /// Presentation shader target.
    Present,
}}

/// Metadata for one embedded shader target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddedShaderTarget {{
    /// Composed shader target stem.
    pub stem: &'static str,
    /// Source class for the target.
    pub class: EmbeddedShaderClass,
}}

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

/// All embedded shader targets in deterministic build order.
pub const COMPILED_SHADER_TARGETS: &[EmbeddedShaderTarget] = &[
{embedded_targets}
];

/// Material target stems (composed from `shaders/materials/*.wgsl`).
pub const COMPILED_MATERIAL_STEMS: &[&str] = &[
{material_stems}
];

/// Post-processing target stems (composed from `shaders/passes/post/*.wgsl`).
pub const COMPILED_POST_STEMS: &[&str] = &[
{post_stems}
];

/// Backend target stems (composed from `shaders/passes/backend/*.wgsl`).
pub const COMPILED_BACKEND_STEMS: &[&str] = &[
{backend_stems}
];

/// Compute target stems (composed from `shaders/passes/compute/*.wgsl`).
pub const COMPILED_COMPUTE_STEMS: &[&str] = &[
{compute_stems}
];

/// Present target stems (composed from `shaders/passes/present/*.wgsl`).
pub const COMPILED_PRESENT_STEMS: &[&str] = &[
{present_stems}
];
"#,
        embedded_arms = c.embedded_arms,
        embedded_pass_arms = c.embedded_pass_arms,
        embedded_consts = c.embedded_consts,
        embedded_targets = c.embedded_targets,
        material_stems = stems_list(&c.material_stems),
        post_stems = stems_list(&c.post_stems),
        backend_stems = stems_list(&c.backend_stems),
        compute_stems = stems_list(&c.compute_stems),
        present_stems = stems_list(&c.present_stems),
    )
}

#[cfg(test)]
mod tests {
    use crate::shader::directives::{BuildPassDirective, BuildPassKind};
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
            Vec::new(),
        );
        let dual = fake_compiled_shader(
            1,
            ShaderSourceClass::Post,
            &[
                ("dual_default", "default wgsl"),
                ("dual_multiview", "multiview wgsl"),
            ],
            Vec::new(),
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
            vec![
                BuildPassDirective {
                    kind: BuildPassKind::Forward,
                    fragment_entry: "fs_main".to_string(),
                    vertex_entry: "vs_main".to_string(),
                    vertex_entry_explicit: false,
                    alpha_to_coverage: true,
                },
                BuildPassDirective {
                    kind: BuildPassKind::Outline,
                    fragment_entry: "fs_outline".to_string(),
                    vertex_entry: "vs_outline".to_string(),
                    vertex_entry_explicit: true,
                    alpha_to_coverage: false,
                },
            ],
        );

        emit_compiled_shader(&compiled, target_dir.path(), &mut composed)?;
        let embedded = render_embedded_shaders_rs(&composed);

        assert!(
            embedded.contains("pass_from_kind(crate::materials::PassKind::Forward, \"fs_main\")")
        );
        assert!(embedded.contains("alpha_to_coverage: true"));
        assert!(embedded.contains(
            "MaterialPassDesc { vertex_entry: \"vs_outline\", ..crate::materials::pass_from_kind(crate::materials::PassKind::Outline, \"fs_outline\") }"
        ));
        assert!(embedded.contains("EmbeddedShaderClass::Material"));
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

    fn fake_compiled_shader(
        compile_order: usize,
        source_class: ShaderSourceClass,
        targets: &[(&str, &str)],
        pass_directives: Vec<BuildPassDirective>,
    ) -> CompiledShader {
        CompiledShader {
            compile_order,
            source_class,
            pass_directives,
            targets: targets
                .iter()
                .map(|(target_stem, wgsl)| CompiledShaderTarget {
                    target_stem: (*target_stem).to_string(),
                    wgsl: (*wgsl).to_string(),
                })
                .collect(),
        }
    }
}
