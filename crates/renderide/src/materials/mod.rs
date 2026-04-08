//! AAA-style materials: WGSL templates + overrides, per-family pipeline builders, and cache.
//!
//! Host material **properties** live in [`crate::assets::material::MaterialPropertyStore`] (IPC
//! batches). **Shader program choice** (which WGSL family to use) is routed via [`MaterialRouter`]
//! from host shader asset ids updated by [`crate::assets::shader::resolve_shader_upload`].

mod builtin_solid;
mod cache;
mod family;
mod manifest_stem;
mod material_property_binding;
mod registry;
mod resolve_raster;
mod router;
mod stem_manifest;
mod wgsl;
mod wgsl_reflect;

pub use builtin_solid::{SolidColorFamily, SOLID_COLOR_FAMILY_ID};
pub use cache::{MaterialPipelineCache, MaterialPipelineCacheKey};
pub use family::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
pub use manifest_stem::{ManifestStemMaterialFamily, MANIFEST_RASTER_FAMILY_ID};
pub use material_property_binding::{MaterialPropertyGpuLayout, WorldUnlitPropertyIds};
pub use registry::MaterialRegistry;
pub use resolve_raster::resolve_raster_family;
pub use router::{MaterialRouter, ShaderRouteEntry};
pub use stem_manifest::{
    manifest_stem_for_unity_name, ShaderManifest, ShaderManifestMaterialEntry, StemResolver,
};
pub use wgsl::{compose_wgsl, WgslPatch};
pub use wgsl_reflect::{
    reflect_raster_material_wgsl, validate_per_draw_group2, ReflectError, ReflectedRasterLayout,
};

pub use crate::pipelines::raster::{DebugWorldNormalsFamily, DEBUG_WORLD_NORMALS_FAMILY_ID};
