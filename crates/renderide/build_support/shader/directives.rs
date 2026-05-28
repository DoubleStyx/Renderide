//! WGSL source directive parsing.

#[path = "directives/defaults.rs"]
mod defaults;
#[path = "directives/features.rs"]
mod features;
#[path = "directives/passes.rs"]
mod passes;
#[path = "directives/source_alias.rs"]
mod source_alias;

pub(super) use defaults::{
    MaterialDefaultDirective, TextureDefaultDirective, material_default_literal,
    parse_material_default_directives, parse_texture_default_directives, texture_default_literal,
};
#[cfg(test)]
pub(super) use defaults::{MaterialDefaultValue, TextureDefaultKind};
#[cfg(test)]
pub(super) use features::BuildWgpuFeature;
pub(super) use features::{
    WgpuFeatureDirective, parse_wgpu_feature_directives, wgpu_features_literal,
};
#[cfg(test)]
pub(super) use passes::{
    BuildBlend, BuildColorWrites, BuildCullMode, BuildDepthCompare, BuildDepthCompareDomain,
    BuildMaterialPassState, BuildPassType, BuildRenderStatePolicy,
};
pub(super) use passes::{BuildPassDirective, parse_pass_directives, pass_literal};
pub(super) use source_alias::parse_source_alias;
