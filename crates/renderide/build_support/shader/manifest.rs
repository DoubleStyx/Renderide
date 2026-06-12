//! Shader package manifest emission.

use std::fs;
use std::path::Path;

use super::directives::{
    BuildAlphaToCoverageMode, BuildBlend, BuildColorWrites, BuildCullMode, BuildDepthCompare,
    BuildDepthCompareDomain, BuildMaterialPassState, BuildPassDirective, BuildPassType,
    BuildRenderStatePolicy, BuildWgpuFeature, MaterialDefaultDirective, MaterialDefaultKind,
    TextureDefaultDirective, TextureDefaultKind, WgpuFeatureDirective,
};
use super::error::BuildError;
use super::model::{CompiledShader, ShaderSourceClass};
use super::shader_package_schema::{
    MaterialAlphaToCoverageManifest, MaterialBlendManifest, MaterialColorWritesManifest,
    MaterialCullModeManifest, MaterialDefaultKindManifest, MaterialDefaultManifest,
    MaterialDefaultValueManifest, MaterialDepthCompareDomainManifest, MaterialDepthCompareManifest,
    MaterialPassManifest, MaterialPassStateManifest, MaterialPassTypeManifest,
    MaterialRenderStatePolicyManifest, MaterialShaderManifest, SHADER_PACKAGE_MANIFEST_FILE,
    SHADER_PACKAGE_MANIFEST_VERSION, ShaderPackageManifest, ShaderRequiredFeatures,
    ShaderRouteManifest, ShaderTargetClass, ShaderTargetManifest, TextureDefaultKindManifest,
    TextureDefaultManifest, stable_source_hash,
};

/// Writes `shader_manifest.toml` beside generated target WGSL files.
pub(super) fn write_shader_package_manifest(
    compiled: &[CompiledShader],
    target_dir: &Path,
) -> Result<(), BuildError> {
    let mut manifest = ShaderPackageManifest {
        version: SHADER_PACKAGE_MANIFEST_VERSION,
        targets: Vec::new(),
        routes: Vec::new(),
    };
    for shader in compiled {
        record_shader_manifest(shader, &mut manifest);
    }
    manifest.targets.sort_by(|a, b| a.stem.cmp(&b.stem));
    manifest.routes.sort_by(|a, b| a.asset.cmp(&b.asset));
    let toml = toml::to_string_pretty(&manifest)
        .map_err(|e| BuildError::Message(format!("serialize shader package manifest: {e}")))?;
    fs::write(target_dir.join(SHADER_PACKAGE_MANIFEST_FILE), toml)?;
    Ok(())
}

fn record_shader_manifest(shader: &CompiledShader, manifest: &mut ShaderPackageManifest) {
    for target in &shader.targets {
        let target_manifest = ShaderTargetManifest {
            stem: target.target_stem.clone(),
            class: target_class(shader.source_class),
            file: format!("{}.wgsl", target.target_stem),
            wgsl_hash: stable_source_hash(&target.wgsl),
            required_features: required_features(&shader.wgpu_features),
            material: (shader.source_class == ShaderSourceClass::Material).then(|| {
                material_manifest(
                    &target.pass_directives,
                    shader.default_render_queue,
                    &shader.texture_defaults,
                    &shader.material_defaults,
                )
            }),
        };
        manifest.targets.push(target_manifest);
    }

    if shader.source_class == ShaderSourceClass::Material {
        let default_stem = format!("{}_default", shader.source_stem);
        if shader
            .targets
            .iter()
            .any(|target| target.target_stem == default_stem)
        {
            manifest.routes.push(ShaderRouteManifest {
                asset: shader.source_stem.clone(),
                stem: default_stem,
            });
        }
    }
}

fn target_class(source_class: ShaderSourceClass) -> ShaderTargetClass {
    match source_class {
        ShaderSourceClass::Material => ShaderTargetClass::Material,
        ShaderSourceClass::Post => ShaderTargetClass::Post,
        ShaderSourceClass::Backend => ShaderTargetClass::Backend,
        ShaderSourceClass::Compute => ShaderTargetClass::Compute,
        ShaderSourceClass::Present => ShaderTargetClass::Present,
    }
}

fn required_features(features: &[WgpuFeatureDirective]) -> ShaderRequiredFeatures {
    ShaderRequiredFeatures {
        shader_barycentrics: features
            .iter()
            .any(|feature| feature.feature == BuildWgpuFeature::ShaderBarycentrics),
    }
}

fn material_manifest(
    passes: &[BuildPassDirective],
    default_render_queue: i32,
    texture_defaults: &[TextureDefaultDirective],
    material_defaults: &[MaterialDefaultDirective],
) -> MaterialShaderManifest {
    MaterialShaderManifest {
        passes: passes.iter().map(pass_manifest).collect(),
        default_render_queue,
        texture_defaults: texture_defaults
            .iter()
            .map(texture_default_manifest)
            .collect(),
        material_defaults: material_defaults
            .iter()
            .map(material_default_manifest)
            .collect(),
    }
}

fn pass_manifest(pass: &BuildPassDirective) -> MaterialPassManifest {
    MaterialPassManifest {
        pass_type: pass_type(pass.pass_type),
        name: pass.name.clone(),
        vertex_entry: pass.vertex_entry.clone(),
        fragment_entry: pass.fragment_entry.clone(),
        alpha_to_coverage: alpha_to_coverage(pass.alpha_to_coverage),
        depth_compare_domain: depth_compare_domain(pass.depth_compare_domain),
        depth_compare: depth_compare(pass.depth_compare),
        depth_write: pass.depth_write,
        cull_mode: cull_mode(pass.cull_mode),
        blend: blend(pass.blend),
        write_mask: color_writes(pass.write_mask),
        depth_bias_slope_scale_bits: pass.depth_bias_slope_scale_bits,
        depth_bias_constant: pass.depth_bias_constant,
        material_state: material_pass_state(pass.material_state),
        render_state_policy: render_state_policy(pass.render_state_policy),
    }
}

fn pass_type(value: BuildPassType) -> MaterialPassTypeManifest {
    match value {
        BuildPassType::Forward => MaterialPassTypeManifest::Forward,
        BuildPassType::DepthPrepass => MaterialPassTypeManifest::DepthPrepass,
    }
}

fn alpha_to_coverage(value: BuildAlphaToCoverageMode) -> MaterialAlphaToCoverageManifest {
    match value {
        BuildAlphaToCoverageMode::Off => MaterialAlphaToCoverageManifest::Off,
        BuildAlphaToCoverageMode::Always => MaterialAlphaToCoverageManifest::Always,
        BuildAlphaToCoverageMode::Cutout => MaterialAlphaToCoverageManifest::Cutout,
    }
}

fn depth_compare_domain(value: BuildDepthCompareDomain) -> MaterialDepthCompareDomainManifest {
    match value {
        BuildDepthCompareDomain::FrooxZTest => MaterialDepthCompareDomainManifest::FrooxZTest,
        BuildDepthCompareDomain::UnityCompareFunction => {
            MaterialDepthCompareDomainManifest::UnityCompareFunction
        }
    }
}

fn depth_compare(value: BuildDepthCompare) -> MaterialDepthCompareManifest {
    match value {
        BuildDepthCompare::Main => MaterialDepthCompareManifest::Main,
        BuildDepthCompare::Always => MaterialDepthCompareManifest::Always,
        BuildDepthCompare::Less => MaterialDepthCompareManifest::Less,
        BuildDepthCompare::GreaterEqual => MaterialDepthCompareManifest::GreaterEqual,
    }
}

fn cull_mode(value: BuildCullMode) -> MaterialCullModeManifest {
    match value {
        BuildCullMode::Back => MaterialCullModeManifest::Back,
        BuildCullMode::Front => MaterialCullModeManifest::Front,
        BuildCullMode::Off => MaterialCullModeManifest::Off,
    }
}

fn blend(value: BuildBlend) -> MaterialBlendManifest {
    match value {
        BuildBlend::Off => MaterialBlendManifest::Off,
        BuildBlend::Alpha => MaterialBlendManifest::Alpha,
        BuildBlend::Additive => MaterialBlendManifest::Additive,
        BuildBlend::Premultiplied => MaterialBlendManifest::Premultiplied,
        BuildBlend::Overlay => MaterialBlendManifest::Overlay,
    }
}

fn color_writes(value: BuildColorWrites) -> MaterialColorWritesManifest {
    match value {
        BuildColorWrites::Rgba => MaterialColorWritesManifest::Rgba,
        BuildColorWrites::Rgb => MaterialColorWritesManifest::Rgb,
        BuildColorWrites::None => MaterialColorWritesManifest::None,
    }
}

fn material_pass_state(value: BuildMaterialPassState) -> MaterialPassStateManifest {
    match value {
        BuildMaterialPassState::Static => MaterialPassStateManifest::Static,
        BuildMaterialPassState::Forward => MaterialPassStateManifest::Forward,
        BuildMaterialPassState::TransparentForward => MaterialPassStateManifest::TransparentForward,
        BuildMaterialPassState::Overlay => MaterialPassStateManifest::Overlay,
        BuildMaterialPassState::Filter => MaterialPassStateManifest::Filter,
    }
}

fn render_state_policy(value: BuildRenderStatePolicy) -> MaterialRenderStatePolicyManifest {
    MaterialRenderStatePolicyManifest {
        color_mask: value.color_mask,
        depth_write: value.depth_write,
        depth_compare: value.depth_compare,
        cull: value.cull,
        stencil: value.stencil,
        depth_offset: value.depth_offset,
    }
}

fn texture_default_manifest(default: &TextureDefaultDirective) -> TextureDefaultManifest {
    TextureDefaultManifest {
        property: default.property.clone(),
        kind: texture_default_kind(default.kind),
    }
}

fn texture_default_kind(value: TextureDefaultKind) -> TextureDefaultKindManifest {
    match value {
        TextureDefaultKind::White => TextureDefaultKindManifest::White,
        TextureDefaultKind::Black => TextureDefaultKindManifest::Black,
        TextureDefaultKind::Gray => TextureDefaultKindManifest::Gray,
        TextureDefaultKind::Bump => TextureDefaultKindManifest::Bump,
        TextureDefaultKind::Red => TextureDefaultKindManifest::Red,
        TextureDefaultKind::Empty => TextureDefaultKindManifest::Empty,
    }
}

fn material_default_manifest(default: &MaterialDefaultDirective) -> MaterialDefaultManifest {
    MaterialDefaultManifest {
        property: default.property.clone(),
        value: MaterialDefaultValueManifest {
            kind: material_default_kind(default.value.kind),
            bits: default.value.bits,
        },
    }
}

fn material_default_kind(value: MaterialDefaultKind) -> MaterialDefaultKindManifest {
    match value {
        MaterialDefaultKind::Float => MaterialDefaultKindManifest::Float,
        MaterialDefaultKind::Vec4 => MaterialDefaultKindManifest::Vec4,
    }
}
