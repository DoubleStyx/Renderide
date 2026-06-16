//! TOML schema for runtime shader packages.

use serde::{Deserialize, Serialize};

/// Current shader package manifest schema version.
pub const SHADER_PACKAGE_MANIFEST_VERSION: u32 = 1;

/// Filename used for the shader package manifest.
pub const SHADER_PACKAGE_MANIFEST_FILE: &str = "shader_manifest.toml";

/// Stable non-cryptographic hash for WGSL package contents.
#[must_use]
pub fn stable_source_hash(source: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for byte in source.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

/// Root shader package manifest.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShaderPackageManifest {
    /// Manifest schema version.
    pub version: u32,
    /// Generated shader targets.
    pub targets: Vec<ShaderTargetManifest>,
    /// Material shader asset-name routes.
    pub routes: Vec<ShaderRouteManifest>,
}

/// One generated target entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShaderTargetManifest {
    /// Target stem used by renderer lookups.
    pub stem: String,
    /// Logical shader class.
    pub class: ShaderTargetClass,
    /// WGSL file path relative to the manifest directory.
    pub file: String,
    /// Stable hash of the WGSL source contents.
    pub wgsl_hash: String,
    /// Required device features.
    #[serde(default)]
    pub required_features: ShaderRequiredFeatures,
    /// Material-only metadata that cannot be reflected from WGSL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialShaderManifest>,
}

/// Logical shader target class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShaderTargetClass {
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

/// Required device features for a target.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShaderRequiredFeatures {
    /// Fragment shader barycentric coordinates are required.
    pub shader_barycentrics: bool,
}

/// One material route from Unity shader asset name to default material target stem.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShaderRouteManifest {
    /// Normalized Unity shader asset lookup key.
    pub asset: String,
    /// Default material target stem.
    pub stem: String,
}

/// Material-only metadata for one generated target.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MaterialShaderManifest {
    /// Declared material passes in submission order.
    pub passes: Vec<MaterialPassManifest>,
    /// Default Unity render queue.
    pub default_render_queue: i32,
    /// Texture fallback directives.
    pub texture_defaults: Vec<TextureDefaultManifest>,
    /// Material uniform fallback directives.
    pub material_defaults: Vec<MaterialDefaultManifest>,
}

/// One material pass entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialPassManifest {
    /// Semantic pass role.
    pub pass_type: MaterialPassTypeManifest,
    /// Debug label.
    pub name: String,
    /// Vertex entry point.
    pub vertex_entry: String,
    /// Fragment entry point.
    pub fragment_entry: String,
    /// Alpha-to-coverage policy.
    pub alpha_to_coverage: MaterialAlphaToCoverageManifest,
    /// Depth compare enum domain.
    pub depth_compare_domain: MaterialDepthCompareDomainManifest,
    /// Depth compare fallback.
    pub depth_compare: MaterialDepthCompareManifest,
    /// Default depth write state.
    pub depth_write: bool,
    /// Default cull mode.
    pub cull_mode: MaterialCullModeManifest,
    /// Default blend state.
    pub blend: MaterialBlendManifest,
    /// Default color write mask.
    pub write_mask: MaterialColorWritesManifest,
    /// Slope-scaled depth bias bits.
    pub depth_bias_slope_scale_bits: u32,
    /// Constant depth bias.
    pub depth_bias_constant: i32,
    /// Material pass-state override mode.
    pub material_state: MaterialPassStateManifest,
    /// Material render-state override policy.
    pub render_state_policy: MaterialRenderStatePolicyManifest,
}

/// Material pass role.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialPassTypeManifest {
    /// Normal forward pass.
    Forward,
    /// Authored depth-only prepass.
    DepthPrepass,
}

/// Alpha-to-coverage policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialAlphaToCoverageManifest {
    /// Disabled.
    Off,
    /// Always enabled for MSAA targets.
    Always,
    /// Enabled only for alpha-test queue materials.
    Cutout,
}

/// Depth compare enum domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialDepthCompareDomainManifest {
    /// FrooxEngine ZTest layout.
    FrooxZTest,
    /// Unity CompareFunction layout.
    UnityCompareFunction,
}

/// Depth compare fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialDepthCompareManifest {
    /// Renderer main reverse-Z compare.
    Main,
    /// Always passes.
    Always,
    /// Less compare.
    Less,
    /// Greater-equal compare.
    GreaterEqual,
}

/// Cull mode fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialCullModeManifest {
    /// Cull back faces.
    Back,
    /// Cull front faces.
    Front,
    /// Disable culling.
    Off,
}

/// Blend state fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialBlendManifest {
    /// Disable blending.
    Off,
    /// Straight alpha.
    Alpha,
    /// Additive.
    Additive,
    /// Premultiplied alpha.
    Premultiplied,
    /// Overlay color/no-op plus max alpha.
    Overlay,
}

/// Color write mask fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialColorWritesManifest {
    /// Write RGBA.
    Rgba,
    /// Write RGB.
    Rgb,
    /// Write no color channels.
    None,
}

/// Material pass-state override mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialPassStateManifest {
    /// Static pass state.
    Static,
    /// Material-driven forward blend.
    Forward,
    /// Transparent forward state.
    TransparentForward,
    /// Overlay state.
    Overlay,
    /// Filter state.
    Filter,
}

/// Per-field material render-state override policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialRenderStatePolicyManifest {
    /// Whether `_ColorMask` overrides the pass color write mask.
    pub color_mask: bool,
    /// Whether `_ZWrite` overrides the pass depth-write flag.
    pub depth_write: bool,
    /// Whether `_ZTest` overrides the pass depth compare function.
    pub depth_compare: bool,
    /// Whether `_Cull` overrides the pass cull mode.
    pub cull: bool,
    /// Whether `_Stencil*` properties override the pass stencil state.
    pub stencil: bool,
    /// Whether `_OffsetFactor` / `_OffsetUnits` override the pass depth bias.
    pub depth_offset: bool,
}

/// Texture fallback metadata.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureDefaultManifest {
    /// Texture property name.
    pub property: String,
    /// Fallback texture token.
    pub kind: TextureDefaultKindManifest,
}

/// Texture fallback kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureDefaultKindManifest {
    /// White texture.
    White,
    /// Black texture.
    Black,
    /// Gray texture.
    Gray,
    /// Normal-map bump texture.
    Bump,
    /// Red texture.
    Red,
    /// Empty Unity texture default.
    Empty,
}

/// Material uniform fallback metadata.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialDefaultManifest {
    /// Uniform property name.
    pub property: String,
    /// Uniform fallback value.
    pub value: MaterialDefaultValueManifest,
}

/// Material uniform fallback value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialDefaultValueManifest {
    /// Fallback value kind.
    pub kind: MaterialDefaultKindManifest,
    /// Raw `f32` bits. Float defaults use only the first element.
    pub bits: [u32; 4],
}

/// Material uniform fallback value kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialDefaultKindManifest {
    /// Single float.
    Float,
    /// Four-lane vector.
    Vec4,
}
