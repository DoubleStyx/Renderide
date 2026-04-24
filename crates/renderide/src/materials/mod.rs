//! AAA-style materials: WGSL templates + overrides, pipeline cache, and routing.
//!
//! Host material **properties** live in [`crate::assets::material::MaterialPropertyStore`] (IPC
//! batches). **Shader program choice** (which embedded WGSL target to use) is routed via [`MaterialRouter`]
//! from host shader asset ids updated by [`crate::assets::shader::resolve_shader_upload`].

mod cache;
mod embedded_raster_pipeline;
mod embedded_shader_stem;
mod family;
mod material_pass_tables;
mod material_passes;
mod material_property_binding;
mod pipeline_build_error;
mod pipeline_kind;
pub(crate) mod raster_pipeline;
mod registry;
mod render_state;
mod resolve_raster;
mod router;
mod wgsl;
mod wgsl_reflect;

/// Pipeline cache keyed by shader route / layout fingerprint.
pub use cache::{MaterialPipelineCache, MaterialPipelineCacheKey, MaterialPipelineSet};

/// Unity shader names → embedded WGSL stems and permutation flags.
pub use embedded_raster_pipeline::{
    embedded_composed_stem_for_permutation, embedded_stem_needs_color_stream,
    embedded_stem_needs_extended_vertex_streams, embedded_stem_needs_uv0_stream,
    embedded_stem_pipeline_pass_count, embedded_stem_requires_grab_pass,
    embedded_stem_requires_intersection_pass, embedded_stem_uses_alpha_blending,
    embedded_wgsl_needs_color_stream, embedded_wgsl_needs_extended_vertex_streams,
    embedded_wgsl_needs_uv0_stream, embedded_wgsl_requires_grab_pass,
    embedded_wgsl_requires_intersection_pass,
};
pub use embedded_shader_stem::{
    embedded_default_stem_for_unity_name, embedded_stem_for_unity_name,
};

/// Pipeline family descriptors, per-property GPU layout, and raster kind flags.
pub use family::MaterialPipelineDesc;
pub use material_passes::{
    default_pass, default_pass_for_blend_mode, material_blend_mode_for_lookup,
    material_blend_mode_from_maps, materialized_pass_for_blend_mode, pass_from_kind,
    MaterialBlendMode, MaterialPassDesc, MaterialPassState, MaterialPipelinePropertyIds, PassKind,
    COLOR_WRITES_NONE,
};
pub use material_property_binding::MaterialPropertyGpuLayout;
pub use pipeline_build_error::PipelineBuildError;
pub use pipeline_kind::RasterPipelineKind;
pub use render_state::{
    material_render_state_for_lookup, material_render_state_from_maps, MaterialCullOverride,
    MaterialDepthOffsetState, MaterialRenderState, MaterialStencilState,
};

/// Naga reflection: composed WGSL → `wgpu` bind layouts, uniform block layout, stem fingerprints.
pub use wgsl_reflect::{
    reflect_raster_material_requires_grab_pass, reflect_raster_material_requires_intersection_pass,
    reflect_raster_material_wgsl, reflect_vertex_shader_needs_color_stream,
    reflect_vertex_shader_needs_uv0_stream, validate_per_draw_group2, ReflectError,
    ReflectedMaterialUniformBlock, ReflectedRasterLayout, ReflectedUniformField,
    ReflectedUniformScalarKind,
};

/// Shader route table, optional material asset registry, and WGSL composition patches.
pub use registry::MaterialRegistry;
pub use resolve_raster::resolve_raster_pipeline;
pub use router::{MaterialRouter, ShaderRouteEntry};
pub use wgsl::{compose_wgsl, WgslPatch};

pub use crate::pipelines::raster::DebugWorldNormalsFamily;
