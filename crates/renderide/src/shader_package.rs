//! Runtime shader package loading and compatibility lookups.

pub(crate) mod schema;

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use hashbrown::{HashMap, HashSet};
use thiserror::Error;

use crate::embedded_shaders::{
    EmbeddedMaterialDefault, EmbeddedMaterialDefaultValue, EmbeddedShaderReflection,
    EmbeddedSnapshotRequirements, EmbeddedTextureDefault, EmbeddedTextureDefaultKind,
    EmbeddedVertexStreamMask,
};
use crate::materials::{
    COLOR_WRITES_NONE, MaterialAlphaToCoverageMode, MaterialDepthCompareDomain, MaterialPassDesc,
    MaterialPassState, MaterialRenderStatePolicy, PASS_BLEND_ONE_ONE,
    PASS_BLEND_ONE_ONE_MINUS_SRC_ALPHA, PASS_BLEND_OVERLAY_NOOP_COLOR_MAX_ALPHA,
    PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA, PassType, ReflectedRasterLayout,
    ReflectedVertexInputFormat, SnapshotRequirements,
    reflect_raster_material_wgsl_with_vertex_entries,
};

use self::schema::{
    MaterialAlphaToCoverageManifest, MaterialBlendManifest, MaterialColorWritesManifest,
    MaterialCullModeManifest, MaterialDefaultKindManifest, MaterialDefaultManifest,
    MaterialDepthCompareDomainManifest, MaterialDepthCompareManifest, MaterialPassManifest,
    MaterialPassStateManifest, MaterialPassTypeManifest, SHADER_PACKAGE_MANIFEST_FILE,
    SHADER_PACKAGE_MANIFEST_VERSION, ShaderPackageManifest, ShaderRequiredFeatures,
    TextureDefaultKindManifest, TextureDefaultManifest, stable_source_hash,
};

const DEFAULT_RENDER_QUEUE: i32 = 2000;
const SHADER_PACKAGE_DIR_ENV: &str = "RENDERIDE_SHADER_PACKAGE_DIR";
const UNIX_SYSTEM_SHADER_PACKAGE_DIR: &str = "/usr/share/renderide/shaders";

/// Runtime shader package load failure.
#[derive(Debug, Error)]
pub(crate) enum ShaderPackageError {
    /// No shader package manifest was found in any candidate directory.
    #[error("no shader package manifest found; searched {searched:?}")]
    MissingPackage {
        /// Candidate directories searched.
        searched: Vec<PathBuf>,
    },
    /// A shader package file could not be read.
    #[error("read {path}: {source}")]
    Read {
        /// File path.
        path: PathBuf,
        /// Source IO error.
        source: std::io::Error,
    },
    /// Manifest TOML could not be decoded.
    #[error("parse {path}: {source}")]
    ParseManifest {
        /// Manifest path.
        path: PathBuf,
        /// TOML decode error.
        source: toml::de::Error,
    },
    /// Manifest schema version is unsupported.
    #[error("unsupported shader package manifest version {actual}; expected {expected}")]
    UnsupportedVersion {
        /// Manifest version read from disk.
        actual: u32,
        /// Supported manifest version.
        expected: u32,
    },
    /// Two manifest target entries use the same stem.
    #[error("duplicate shader package target stem `{0}`")]
    DuplicateTarget(String),
    /// Two manifest route entries use the same asset key.
    #[error("duplicate shader package route `{0}`")]
    DuplicateRoute(String),
    /// A route points at a target that is not present.
    #[error("shader package route `{asset}` points at missing target `{stem}`")]
    MissingRouteTarget {
        /// Route asset key.
        asset: String,
        /// Missing target stem.
        stem: String,
    },
    /// A target's WGSL file hash did not match the manifest.
    #[error("shader package target `{stem}` hash mismatch: expected {expected}, got {actual}")]
    HashMismatch {
        /// Target stem.
        stem: String,
        /// Hash recorded in the manifest.
        expected: String,
        /// Hash computed from disk.
        actual: String,
    },
}

/// Loaded runtime shader package.
#[derive(Debug)]
pub(crate) struct ShaderPackage {
    targets: HashMap<String, ShaderTarget>,
    routes: HashMap<String, Arc<str>>,
}

#[derive(Debug)]
struct ShaderTarget {
    wgsl: Arc<str>,
    required_features: wgpu::Features,
    material: Option<MaterialTargetMetadata>,
    reflection: OnceLock<EmbeddedShaderReflection>,
}

#[derive(Debug)]
struct MaterialTargetMetadata {
    passes: Vec<MaterialPassDesc>,
    default_render_queue: i32,
    texture_defaults: Vec<EmbeddedTextureDefault>,
    material_defaults: Vec<EmbeddedMaterialDefault>,
}

impl ShaderPackage {
    fn load_from_dir(dir: &Path) -> Result<Self, ShaderPackageError> {
        let manifest_path = dir.join(SHADER_PACKAGE_MANIFEST_FILE);
        let manifest_text = read_to_string(&manifest_path)?;
        let manifest: ShaderPackageManifest =
            toml::from_str(&manifest_text).map_err(|source| ShaderPackageError::ParseManifest {
                path: manifest_path.clone(),
                source,
            })?;
        if manifest.version != SHADER_PACKAGE_MANIFEST_VERSION {
            return Err(ShaderPackageError::UnsupportedVersion {
                actual: manifest.version,
                expected: SHADER_PACKAGE_MANIFEST_VERSION,
            });
        }

        let mut targets = HashMap::new();
        for target in manifest.targets {
            if targets.contains_key(&target.stem) {
                return Err(ShaderPackageError::DuplicateTarget(target.stem));
            }
            let path = dir.join(&target.file);
            let source = read_to_string(&path)?;
            let actual = stable_source_hash(&source);
            if actual != target.wgsl_hash {
                return Err(ShaderPackageError::HashMismatch {
                    stem: target.stem,
                    expected: target.wgsl_hash,
                    actual,
                });
            }
            let material = target.material.map(material_target_metadata);
            targets.insert(
                target.stem,
                ShaderTarget {
                    wgsl: Arc::from(source.into_boxed_str()),
                    required_features: target.required_features.to_wgpu_features(),
                    material,
                    reflection: OnceLock::new(),
                },
            );
        }

        let mut routes = HashMap::new();
        for route in manifest.routes {
            if routes.contains_key(&route.asset) {
                return Err(ShaderPackageError::DuplicateRoute(route.asset));
            }
            if !targets.contains_key(&route.stem) {
                return Err(ShaderPackageError::MissingRouteTarget {
                    asset: route.asset,
                    stem: route.stem,
                });
            }
            routes.insert(route.asset, Arc::from(route.stem.into_boxed_str()));
        }

        Ok(Self { targets, routes })
    }
}

impl ShaderRequiredFeatures {
    fn to_wgpu_features(self) -> wgpu::Features {
        let mut features = wgpu::Features::empty();
        if self.shader_barycentrics {
            features |= wgpu::Features::SHADER_BARYCENTRICS;
        }
        features
    }
}

fn read_to_string(path: &Path) -> Result<String, ShaderPackageError> {
    std::fs::read_to_string(path).map_err(|source| ShaderPackageError::Read {
        path: path.to_path_buf(),
        source,
    })
}

fn material_target_metadata(material: schema::MaterialShaderManifest) -> MaterialTargetMetadata {
    MaterialTargetMetadata {
        passes: material
            .passes
            .into_iter()
            .map(material_pass_desc)
            .collect(),
        default_render_queue: material.default_render_queue,
        texture_defaults: material
            .texture_defaults
            .into_iter()
            .map(texture_default)
            .collect(),
        material_defaults: material
            .material_defaults
            .into_iter()
            .map(material_default)
            .collect(),
    }
}

fn material_pass_desc(pass: MaterialPassManifest) -> MaterialPassDesc {
    MaterialPassDesc {
        name: leak_string(pass.name),
        pass_type: pass_type(pass.pass_type),
        vertex_entry: leak_string(pass.vertex_entry),
        fragment_entry: leak_string(pass.fragment_entry),
        depth_compare: depth_compare(pass.depth_compare),
        depth_compare_domain: depth_compare_domain(pass.depth_compare_domain),
        depth_write: pass.depth_write,
        cull_mode: cull_mode(pass.cull_mode),
        blend: blend(pass.blend),
        write_mask: color_writes(pass.write_mask),
        depth_bias_slope_scale: f32::from_bits(pass.depth_bias_slope_scale_bits),
        depth_bias_constant: pass.depth_bias_constant,
        alpha_to_coverage: alpha_to_coverage(pass.alpha_to_coverage),
        material_state: material_pass_state(pass.material_state),
        render_state_policy: material_render_state_policy(pass.render_state_policy),
    }
}

fn leak_string(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn pass_type(value: MaterialPassTypeManifest) -> PassType {
    match value {
        MaterialPassTypeManifest::Forward => PassType::Forward,
        MaterialPassTypeManifest::DepthPrepass => PassType::DepthPrepass,
    }
}

fn alpha_to_coverage(value: MaterialAlphaToCoverageManifest) -> MaterialAlphaToCoverageMode {
    match value {
        MaterialAlphaToCoverageManifest::Off => MaterialAlphaToCoverageMode::Off,
        MaterialAlphaToCoverageManifest::Always => MaterialAlphaToCoverageMode::Always,
        MaterialAlphaToCoverageManifest::Cutout => MaterialAlphaToCoverageMode::Cutout,
    }
}

fn depth_compare_domain(value: MaterialDepthCompareDomainManifest) -> MaterialDepthCompareDomain {
    match value {
        MaterialDepthCompareDomainManifest::FrooxZTest => MaterialDepthCompareDomain::FrooxZTest,
        MaterialDepthCompareDomainManifest::UnityCompareFunction => {
            MaterialDepthCompareDomain::UnityCompareFunction
        }
    }
}

fn depth_compare(value: MaterialDepthCompareManifest) -> wgpu::CompareFunction {
    match value {
        MaterialDepthCompareManifest::Main => crate::gpu::MAIN_FORWARD_DEPTH_COMPARE,
        MaterialDepthCompareManifest::Always => wgpu::CompareFunction::Always,
        MaterialDepthCompareManifest::Less => wgpu::CompareFunction::Less,
        MaterialDepthCompareManifest::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
    }
}

fn cull_mode(value: MaterialCullModeManifest) -> Option<wgpu::Face> {
    match value {
        MaterialCullModeManifest::Back => Some(wgpu::Face::Back),
        MaterialCullModeManifest::Front => Some(wgpu::Face::Front),
        MaterialCullModeManifest::Off => None,
    }
}

fn blend(value: MaterialBlendManifest) -> Option<wgpu::BlendState> {
    match value {
        MaterialBlendManifest::Off => None,
        MaterialBlendManifest::Alpha => Some(PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA),
        MaterialBlendManifest::Additive => Some(PASS_BLEND_ONE_ONE),
        MaterialBlendManifest::Premultiplied => Some(PASS_BLEND_ONE_ONE_MINUS_SRC_ALPHA),
        MaterialBlendManifest::Overlay => Some(PASS_BLEND_OVERLAY_NOOP_COLOR_MAX_ALPHA),
    }
}

fn color_writes(value: MaterialColorWritesManifest) -> wgpu::ColorWrites {
    match value {
        MaterialColorWritesManifest::Rgba => wgpu::ColorWrites::ALL,
        MaterialColorWritesManifest::Rgb => wgpu::ColorWrites::COLOR,
        MaterialColorWritesManifest::None => COLOR_WRITES_NONE,
    }
}

fn material_pass_state(value: MaterialPassStateManifest) -> MaterialPassState {
    match value {
        MaterialPassStateManifest::Static => MaterialPassState::Static,
        MaterialPassStateManifest::Forward => MaterialPassState::Forward,
        MaterialPassStateManifest::TransparentForward => MaterialPassState::TransparentForward,
        MaterialPassStateManifest::Overlay => MaterialPassState::Overlay,
        MaterialPassStateManifest::Filter => MaterialPassState::Filter,
    }
}

fn material_render_state_policy(
    value: schema::MaterialRenderStatePolicyManifest,
) -> MaterialRenderStatePolicy {
    MaterialRenderStatePolicy {
        color_mask: value.color_mask,
        depth_write: value.depth_write,
        depth_compare: value.depth_compare,
        cull: value.cull,
        stencil: value.stencil,
        depth_offset: value.depth_offset,
    }
}

fn texture_default(default: TextureDefaultManifest) -> EmbeddedTextureDefault {
    EmbeddedTextureDefault {
        property: leak_string(default.property),
        kind: texture_default_kind(default.kind),
    }
}

fn texture_default_kind(value: TextureDefaultKindManifest) -> EmbeddedTextureDefaultKind {
    match value {
        TextureDefaultKindManifest::White => EmbeddedTextureDefaultKind::White,
        TextureDefaultKindManifest::Black => EmbeddedTextureDefaultKind::Black,
        TextureDefaultKindManifest::Gray => EmbeddedTextureDefaultKind::Gray,
        TextureDefaultKindManifest::Bump => EmbeddedTextureDefaultKind::Bump,
        TextureDefaultKindManifest::Red => EmbeddedTextureDefaultKind::Red,
        TextureDefaultKindManifest::Empty => EmbeddedTextureDefaultKind::Empty,
    }
}

fn material_default(default: MaterialDefaultManifest) -> EmbeddedMaterialDefault {
    EmbeddedMaterialDefault {
        property: leak_string(default.property),
        value: material_default_value(default.value),
    }
}

fn material_default_value(
    value: schema::MaterialDefaultValueManifest,
) -> EmbeddedMaterialDefaultValue {
    match value.kind {
        MaterialDefaultKindManifest::Float => {
            EmbeddedMaterialDefaultValue::float(f32::from_bits(value.bits[0]))
        }
        MaterialDefaultKindManifest::Vec4 => EmbeddedMaterialDefaultValue::vec4([
            f32::from_bits(value.bits[0]),
            f32::from_bits(value.bits[1]),
            f32::from_bits(value.bits[2]),
            f32::from_bits(value.bits[3]),
        ]),
    }
}

fn global_package() -> Option<&'static ShaderPackage> {
    static PACKAGE: OnceLock<Result<ShaderPackage, ShaderPackageError>> = OnceLock::new();
    match PACKAGE.get_or_init(load_default_package) {
        Ok(package) => Some(package),
        Err(error) => {
            logger::error!("shader package load failed: {error}");
            None
        }
    }
}

fn load_default_package() -> Result<ShaderPackage, ShaderPackageError> {
    let candidates = package_candidate_dirs();
    for dir in &candidates {
        if dir.join(SHADER_PACKAGE_MANIFEST_FILE).is_file() {
            logger::info!("shader package: loading {}", dir.display());
            return ShaderPackage::load_from_dir(dir);
        }
    }
    Err(ShaderPackageError::MissingPackage {
        searched: candidates,
    })
}

fn package_candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(path) = std::env::var(SHADER_PACKAGE_DIR_ENV)
        && !path.trim().is_empty()
    {
        dirs.push(PathBuf::from(path));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dirs.push(parent.join("shaders"));
    }
    #[cfg(target_family = "unix")]
    {
        dirs.push(PathBuf::from(UNIX_SYSTEM_SHADER_PACKAGE_DIR));
    }
    if let Some(path) = option_env!("RENDERIDE_SHADER_PACKAGE_DIR_DEFAULT")
        && !path.trim().is_empty()
    {
        dirs.push(PathBuf::from(path));
    }
    dedup_dirs(dirs)
}

fn dedup_dirs(dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for dir in dirs {
        let key = dir.to_string_lossy().to_string();
        if seen.insert(key) {
            deduped.push(dir);
        }
    }
    deduped
}

/// Returns WGSL source for a target stem.
pub(crate) fn target_wgsl(stem: &str) -> Option<&'static str> {
    let package = global_package()?;
    package.targets.get(stem).map(|target| target.wgsl.as_ref())
}

/// Returns declared material passes for a target stem.
pub(crate) fn material_passes(stem: &str) -> &'static [MaterialPassDesc] {
    let Some(package) = global_package() else {
        return &[];
    };
    package
        .targets
        .get(stem)
        .and_then(|target| target.material.as_ref())
        .map_or(&[], |material| material.passes.as_slice())
}

/// Returns the default render queue for a material target stem.
pub(crate) fn material_default_render_queue(stem: &str) -> i32 {
    let Some(package) = global_package() else {
        return DEFAULT_RENDER_QUEUE;
    };
    package
        .targets
        .get(stem)
        .and_then(|target| target.material.as_ref())
        .map_or(DEFAULT_RENDER_QUEUE, |material| {
            material.default_render_queue
        })
}

/// Returns required device features for a target stem.
pub(crate) fn target_required_features(stem: &str) -> wgpu::Features {
    let Some(package) = global_package() else {
        return wgpu::Features::empty();
    };
    package
        .targets
        .get(stem)
        .map_or(wgpu::Features::empty(), |target| target.required_features)
}

/// Returns texture defaults for a material target stem.
pub(crate) fn material_texture_defaults(stem: &str) -> &'static [EmbeddedTextureDefault] {
    let Some(package) = global_package() else {
        return &[];
    };
    package
        .targets
        .get(stem)
        .and_then(|target| target.material.as_ref())
        .map_or(&[], |material| material.texture_defaults.as_slice())
}

/// Returns material uniform defaults for a material target stem.
pub(crate) fn material_uniform_defaults(stem: &str) -> &'static [EmbeddedMaterialDefault] {
    let Some(package) = global_package() else {
        return &[];
    };
    package
        .targets
        .get(stem)
        .and_then(|target| target.material.as_ref())
        .map_or(&[], |material| material.material_defaults.as_slice())
}

/// Returns Naga-reflected material metadata for a target stem.
pub(crate) fn material_reflection(stem: &str) -> EmbeddedShaderReflection {
    let Some(package) = global_package() else {
        return EmbeddedShaderReflection::default();
    };
    let Some(target) = package.targets.get(stem) else {
        return EmbeddedShaderReflection::default();
    };
    *target
        .reflection
        .get_or_init(|| reflect_material_target(stem, target))
}

/// Returns the default material target stem for a normalized shader asset key.
pub(crate) fn default_material_stem_for_asset_key(asset: &str) -> Option<Arc<str>> {
    let package = global_package()?;
    package.routes.get(asset).cloned()
}

/// Returns the first configured shader package directory for runtime development reload.
pub(crate) fn default_package_dir() -> Option<PathBuf> {
    let candidates = package_candidate_dirs();
    candidates
        .iter()
        .find(|dir| dir.join(SHADER_PACKAGE_MANIFEST_FILE).is_file())
        .cloned()
        .or_else(|| candidates.into_iter().next())
}

fn reflect_material_target(stem: &str, target: &ShaderTarget) -> EmbeddedShaderReflection {
    let Some(material) = target.material.as_ref() else {
        return EmbeddedShaderReflection::default();
    };
    let vertex_entries = material
        .passes
        .iter()
        .map(|pass| pass.vertex_entry)
        .collect::<Vec<_>>();
    match reflect_raster_material_wgsl_with_vertex_entries(&target.wgsl, &vertex_entries) {
        Ok(reflected) => reflection_from_layout(&target.wgsl, &material.passes, &reflected),
        Err(error) => {
            logger::warn!("shader package: material reflection failed for {stem}: {error}");
            EmbeddedShaderReflection::default()
        }
    }
}

fn reflection_from_layout(
    wgsl: &str,
    passes: &[MaterialPassDesc],
    reflected: &ReflectedRasterLayout,
) -> EmbeddedShaderReflection {
    let snapshot_requirements = reflected.snapshot_requirements();
    EmbeddedShaderReflection {
        vertex_stream_mask: vertex_stream_mask_from_layout(reflected),
        snapshot_requirements: EmbeddedSnapshotRequirements {
            uses_scene_color: snapshot_requirements.uses_scene_color,
            uses_scene_depth: snapshot_requirements.uses_scene_depth,
            requires_intersection_pass: snapshot_requirements.requires_intersection_pass,
        },
        uses_renderide_variant_bits: wgsl.contains("renderide_static_variant_bits"),
        supports_generic_depth_prepass: supports_generic_depth_prepass(
            wgsl,
            passes,
            snapshot_requirements,
        ),
    }
}

fn vertex_stream_mask_from_layout(reflected: &ReflectedRasterLayout) -> EmbeddedVertexStreamMask {
    let mut mask = EmbeddedVertexStreamMask::default();
    for input in &reflected.vs_vertex_inputs {
        match (input.location, input.format) {
            (2, ReflectedVertexInputFormat::Float32x2) => mask.uv0 = true,
            (3, ReflectedVertexInputFormat::Float32x4) => mask.color = true,
            (4, ReflectedVertexInputFormat::Float32x4) => mask.tangent = true,
            (5, ReflectedVertexInputFormat::Float32x2) => mask.uv1 = true,
            (6, ReflectedVertexInputFormat::Float32x2) => mask.uv2 = true,
            (7, ReflectedVertexInputFormat::Float32x2) => mask.uv3 = true,
            (location, format) if uv_channel_from_location(location).is_some() => {
                apply_uv_requirement(&mut mask, location, format);
            }
            _ => {}
        }
    }
    mask
}

fn apply_uv_requirement(
    mask: &mut EmbeddedVertexStreamMask,
    location: u32,
    format: ReflectedVertexInputFormat,
) {
    let Some(channel) = uv_channel_from_location(location) else {
        return;
    };
    let supported = matches!(
        format,
        ReflectedVertexInputFormat::Float32x2
            | ReflectedVertexInputFormat::Float32x3
            | ReflectedVertexInputFormat::Float32x4
    );
    if !supported {
        return;
    }
    match channel {
        0 => mask.uv0 = true,
        1 => mask.uv1 = true,
        2 => mask.uv2 = true,
        3 => mask.uv3 = true,
        _ => {}
    }
    if channel >= 4 {
        mask.wide_high_uvs = true;
    } else if format != ReflectedVertexInputFormat::Float32x2 {
        mask.wide_low_uvs = true;
    }
}

fn uv_channel_from_location(location: u32) -> Option<usize> {
    [2, 5, 6, 7, 8, 9, 10, 11]
        .iter()
        .position(|candidate| *candidate == location)
}

fn supports_generic_depth_prepass(
    wgsl: &str,
    passes: &[MaterialPassDesc],
    snapshot_requirements: SnapshotRequirements,
) -> bool {
    let [pass] = passes else {
        return false;
    };
    pass.pass_type == PassType::Forward
        && pass.blend.is_none()
        && pass.depth_write
        && pass.alpha_to_coverage == MaterialAlphaToCoverageMode::Off
        && !wgsl.contains("discard")
        && snapshot_requirements == SnapshotRequirements::default()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::schema::{
        ShaderPackageManifest, ShaderRequiredFeatures, ShaderTargetClass, ShaderTargetManifest,
        stable_source_hash,
    };
    use super::*;

    #[test]
    fn stable_source_hash_is_deterministic() {
        assert_eq!(stable_source_hash("abc"), stable_source_hash("abc"));
        assert_ne!(stable_source_hash("abc"), stable_source_hash("abd"));
    }

    #[test]
    fn package_loader_rejects_hash_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("bad.wgsl"), "fn main() {}\n").expect("write shader");
        let manifest = ShaderPackageManifest {
            version: SHADER_PACKAGE_MANIFEST_VERSION,
            targets: vec![ShaderTargetManifest {
                stem: "bad".to_string(),
                class: ShaderTargetClass::Compute,
                file: "bad.wgsl".to_string(),
                wgsl_hash: "not-the-hash".to_string(),
                required_features: ShaderRequiredFeatures::default(),
                material: None,
            }],
            routes: Vec::new(),
        };
        std::fs::write(
            dir.path().join(SHADER_PACKAGE_MANIFEST_FILE),
            toml::to_string(&manifest).expect("manifest"),
        )
        .expect("write manifest");

        let err = ShaderPackage::load_from_dir(dir.path()).expect_err("hash mismatch");

        assert!(matches!(err, ShaderPackageError::HashMismatch { .. }));
    }

    #[test]
    fn package_candidates_do_not_include_source_tree_target_dir() {
        let source_tree_target = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/target");

        assert!(
            !package_candidate_dirs()
                .into_iter()
                .any(|dir| dir == source_tree_target)
        );
    }

    #[cfg(target_family = "unix")]
    #[test]
    fn package_candidates_include_unix_system_share_dir() {
        let system_dir = Path::new(UNIX_SYSTEM_SHADER_PACKAGE_DIR);

        assert!(
            package_candidate_dirs()
                .into_iter()
                .any(|dir| dir == system_dir)
        );
    }
}
