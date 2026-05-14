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

pub use default_stem::embedded_default_stem_for_shader_asset_name;
pub use embedded_material_bind_error::EmbeddedMaterialBindError;
pub use material_bind::EmbeddedMaterialBindResources;
pub(crate) use material_bind::EmbeddedMaterialBindShader;
pub use snapshot_requirements::SnapshotRequirements;
pub use stem_metadata::{
    EmbeddedTangentFallbackMode, embedded_stem_depth_prepass_pass,
    embedded_stem_needs_color_stream, embedded_stem_needs_extended_vertex_streams,
    embedded_stem_needs_tangent_stream, embedded_stem_needs_uv0_stream,
    embedded_stem_needs_uv1_stream, embedded_stem_needs_uv2_stream, embedded_stem_needs_uv3_stream,
    embedded_stem_needs_wide_uv_stream, embedded_stem_pipeline_pass_count,
    embedded_stem_requires_intersection_pass, embedded_stem_tangent_fallback_mode,
    embedded_stem_uses_alpha_blending, embedded_stem_uses_raw_normal_payload,
    embedded_stem_uses_raw_tangent_payload, embedded_stem_uses_scene_color_snapshot,
    embedded_stem_uses_scene_depth_snapshot, embedded_stem_uses_ui_transparent_fallback,
};
pub use texture_pools::EmbeddedTexturePools;
