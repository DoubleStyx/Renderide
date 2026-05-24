//! Embedded raster materials: WGSL reflection, texture resolution, uniform packing, and `@group(1)` bind groups.

mod bind_kind;
mod default_stem;
mod embedded_material_bind_error;
mod layout;
mod material_bind;
mod snapshot_requirements;
pub(in crate::materials) mod stem_metadata;
mod texture_pools;
pub(crate) mod texture_resolve;
mod uniform_pack;
mod wrap_mode_bits;

pub use default_stem::embedded_default_stem_for_shader_asset_name;
pub use embedded_material_bind_error::EmbeddedMaterialBindError;
pub use material_bind::EmbeddedMaterialBindResources;
pub(crate) use material_bind::EmbeddedMaterialBindShader;
pub use snapshot_requirements::{SceneColorSnapshotMode, SnapshotRequirements};
pub(crate) use stem_metadata::{
    EmbeddedStemQuery, EmbeddedTangentFallbackMode, embedded_stem_depth_prepass_pass,
    embedded_stem_pipeline_pass_count,
};
pub use texture_pools::EmbeddedTexturePools;
