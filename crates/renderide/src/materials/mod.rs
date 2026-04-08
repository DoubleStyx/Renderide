//! AAA-style materials: WGSL templates + overrides, per-family pipeline builders, and cache.
//!
//! Host material **properties** live in [`crate::assets::material::MaterialPropertyStore`] (IPC
//! batches). **Shader program choice** (which WGSL family to use) is routed via [`MaterialRouter`]
//! from host shader asset ids updated by [`crate::assets::shader::resolve_shader_upload`].

mod builtin_solid;
mod cache;
mod family;
mod material_property_binding;
mod registry;
mod resolve_raster;
mod router;
mod wgsl;

pub use builtin_solid::{SolidColorFamily, SOLID_COLOR_FAMILY_ID};
pub use cache::{MaterialPipelineCache, MaterialPipelineCacheKey};
pub use family::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
pub use material_property_binding::MaterialPropertyGpuLayout;
pub use registry::MaterialRegistry;
pub use resolve_raster::resolve_raster_family;
pub use router::{MaterialRouter, ShaderRouteEntry};
pub use wgsl::{compose_wgsl, WgslPatch};

pub use crate::pipelines::raster::{DebugWorldNormalsFamily, DEBUG_WORLD_NORMALS_FAMILY_ID};
