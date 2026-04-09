//! AAA-style materials: WGSL templates + overrides, pipeline cache, and routing.
//!
//! Host material **properties** live in [`crate::assets::material::MaterialPropertyStore`] (IPC
//! batches). **Shader program choice** (which embedded WGSL target to use) is routed via [`MaterialRouter`]
//! from host shader asset ids updated by [`crate::assets::shader::resolve_shader_upload`].

mod cache;
mod embedded_raster_pipeline;
mod embedded_shader_stem;
mod family;
mod material_property_binding;
mod pipeline_kind;
pub(crate) mod raster_pipeline;
mod registry;
mod resolve_raster;
mod router;
mod wgsl;
mod wgsl_reflect;
pub use cache::{MaterialPipelineCache, MaterialPipelineCacheKey};
pub use embedded_raster_pipeline::{
    embedded_composed_stem_for_permutation, embedded_stem_needs_uv0_stream,
    embedded_wgsl_needs_uv0_stream,
};
pub use embedded_shader_stem::{
    embedded_default_stem_for_unity_name, embedded_stem_for_unity_name,
};
pub use family::MaterialPipelineDesc;
pub use material_property_binding::MaterialPropertyGpuLayout;
pub use pipeline_kind::RasterPipelineKind;
pub use registry::MaterialRegistry;
pub use resolve_raster::resolve_raster_pipeline;
pub use router::{MaterialRouter, ShaderRouteEntry};
pub use wgsl::{compose_wgsl, WgslPatch};
pub use wgsl_reflect::{
    reflect_raster_material_wgsl, reflect_vertex_shader_needs_uv0_stream, validate_per_draw_group2,
    ReflectError, ReflectedMaterialUniformBlock, ReflectedRasterLayout, ReflectedUniformField,
    ReflectedUniformScalarKind,
};

pub use crate::pipelines::raster::DebugWorldNormalsFamily;
