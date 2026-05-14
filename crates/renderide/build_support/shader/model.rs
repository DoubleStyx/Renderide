//! Algebraic shader build model.

use std::path::PathBuf;

use hashbrown::HashMap;
use naga_oil::compose::ShaderDefValue;

use super::directives::BuildPassDirective;

/// Validation toggles applied per shader source class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ComposeValidation {
    /// Enforce the default/multiview `@builtin(view_index)` contract when a source fans out.
    pub validate_view_index: bool,
    /// Require at least one `//#pass` directive.
    pub require_pass_directive: bool,
}

/// Logical source class for a WGSL file.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum ShaderSourceClass {
    /// Host-routed material shader.
    Material,
    /// Post-processing fullscreen shader.
    Post,
    /// Backend utility raster shader.
    Backend,
    /// Compute shader.
    Compute,
    /// Presentation shader.
    Present,
}

impl ShaderSourceClass {
    /// Deterministic source discovery order.
    pub const ALL: [Self; 5] = [
        Self::Material,
        Self::Post,
        Self::Backend,
        Self::Compute,
        Self::Present,
    ];

    /// Source directory below `crates/renderide/shaders`.
    pub const fn source_subdir(self) -> &'static str {
        match self {
            Self::Material => "materials",
            Self::Post => "passes/post",
            Self::Backend => "passes/backend",
            Self::Compute => "passes/compute",
            Self::Present => "passes/present",
        }
    }

    /// Validation policy for this source class.
    pub const fn validation(self) -> ComposeValidation {
        match self {
            Self::Material => ComposeValidation {
                validate_view_index: true,
                require_pass_directive: true,
            },
            Self::Post | Self::Backend | Self::Present => ComposeValidation {
                validate_view_index: true,
                require_pass_directive: false,
            },
            Self::Compute => ComposeValidation {
                validate_view_index: false,
                require_pass_directive: false,
            },
        }
    }
}

/// One shader source discovered for build-time composition.
#[derive(Clone, Debug)]
pub(super) struct ShaderJob {
    /// Deterministic global ordering matching source traversal.
    pub compile_order: usize,
    /// Logical source class.
    pub source_class: ShaderSourceClass,
    /// Absolute path to the source WGSL file.
    pub source_path: PathBuf,
    /// Validation policy for this source.
    pub validation: ComposeValidation,
}

/// Variant of one source under shader-def composition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ShaderVariant {
    /// Composition without the `MULTIVIEW` shader def.
    Default,
    /// Composition with the `MULTIVIEW` shader def.
    Multiview,
}

impl ShaderVariant {
    /// Returns shader defs for naga-oil composition.
    ///
    /// `#ifdef MULTIVIEW` is true when the key exists, regardless of its boolean value, so the
    /// default variant must omit the key entirely.
    pub fn shader_defs(self) -> HashMap<String, ShaderDefValue> {
        let mut defs = HashMap::new();
        if self == Self::Multiview {
            defs.insert("MULTIVIEW".to_string(), ShaderDefValue::Bool(true));
        }
        defs
    }

    /// Human-readable label used in build errors.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Default => "MULTIVIEW=false",
            Self::Multiview => "MULTIVIEW=true",
        }
    }

    /// Output target stem for this variant.
    pub fn target_stem(self, source_stem: &str) -> String {
        match self {
            Self::Default => format!("{source_stem}_default"),
            Self::Multiview => format!("{source_stem}_multiview"),
        }
    }

    /// Whether this variant should contain `@builtin(view_index)` in variant-sensitive outputs.
    pub const fn expects_view_index(self) -> bool {
        matches!(self, Self::Multiview)
    }
}

/// One flattened WGSL target emitted for a compiled source shader.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CompiledShaderTarget {
    /// Target stem used for both `shaders/target/{stem}.wgsl` and the embedded registry.
    pub target_stem: String,
    /// Fully flattened WGSL source text.
    pub wgsl: String,
    /// Pass metadata remapped to the entry point names emitted in [`Self::wgsl`].
    pub pass_directives: Vec<BuildPassDirective>,
}

/// Full build-time output for one source shader prior to serial file emission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CompiledShader {
    /// Deterministic global ordering matching source traversal.
    pub compile_order: usize,
    /// Logical source class.
    pub source_class: ShaderSourceClass,
    /// Parsed pass metadata embedded alongside material WGSL.
    pub pass_directives: Vec<BuildPassDirective>,
    /// One or two output targets depending on whether multiview changes the WGSL.
    pub targets: Vec<CompiledShaderTarget>,
}
