//! AAA-style materials: WGSL templates + overrides, pipeline cache, and routing.
//!
//! Host material **properties** live in [`crate::assets::material::MaterialPropertyStore`] (IPC
//! batches). **Shader program choice** (which embedded WGSL target to use) is routed via [`MaterialRouter`]
//! from host shader asset ids updated by [`crate::assets::shader::resolve_shader_upload`].
//!
//! # Pipeline-state vs. shader-uniform boundary
//!
//! The host writes material properties as a flat `(property_id → value)` store. Each property
//! lands in exactly one of three places:
//!
//! | Property kind | Examples | Resolved by | Lives in |
//! |---|---|---|---|
//! | Pipeline state | `_SrcBlend`, `_DstBlend`, `_ZWrite`, `_ZTest`, `_Cull`, `_Stencil*`, `_ColorMask`, `_OffsetFactor`, `_OffsetUnits` | [`MaterialBlendMode`] + [`MaterialRenderState`] | [`MaterialPipelineCacheKey`] (`wgpu::RenderPipeline` build) |
//! | Shader uniform — value | `_Color`, `_Tint`, `_Cutoff`, `_Glossiness`, `*_ST` | Host property store, packed by reflection | `@group(1) @binding(0)` material struct |
//! | Shader uniform — keyword | `_NORMALMAP`, `_ALPHATEST_ON`, `_ALPHABLEND_ON` | Host property OR [`crate::backend::embedded::uniform_pack`] inference (Unity routes these through `ShaderKeywords`, never on the wire) | `@group(1) @binding(0)` material struct |
//! | Texture | `_MainTex`, `_NormalMap`, … | Host texture pools, bound by reflection | `@group(1) @binding(N)` |
//!
//! **Pipeline-state property names must NEVER appear in a shader's `@group(1) @binding(0)`
//! uniform struct.** They are dead weight there: shaders never read them, but the host writes
//! them and reflection allocates uniform space for them. The canonical list lives in
//! [`MaterialPipelinePropertyIds::new`]; the build script in `crates/renderide/build.rs` rejects
//! any material WGSL that violates this contract via `validate_no_pipeline_state_uniform_fields`.
//! Two materials sharing a shader but differing in any pipeline-state property correctly resolve
//! to distinct cached pipelines because [`MaterialPipelineCacheKey`] includes the resolved
//! [`MaterialBlendMode`] and [`MaterialRenderState`].
//!
//! `_BlendMode` itself is not on the wire — FrooxEngine translates `MaterialProvider.SetBlendMode`
//! to `_SrcBlend` / `_DstBlend` factors, and [`MaterialBlendMode::from_unity_blend_factors`]
//! reconstructs the mode here.
//!
//! # Relationship with [`crate::pipelines`]
//!
//! Pipeline primitives — [`crate::pipelines::ShaderPermutation`] (static feature flags such as
//! multiview) and the [`crate::pipelines::raster::null`] debug fallback — live in
//! [`crate::pipelines`]. This module *composes* those primitives into material-driven render
//! pipelines via [`MaterialPipelineCache`], keyed by [`MaterialPipelineCacheKey`] (shader route
//! + permutation + attachment formats + resolved render state). Build new material-side wiring
//! here; add new permutation flags or fallback pipelines under [`crate::pipelines`].

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
    DefaultPassParams, MaterialBlendMode, MaterialPassDesc, MaterialPassState,
    MaterialPipelinePropertyIds, PassKind, COLOR_WRITES_NONE,
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
    reflect_vertex_shader_needs_uv0_stream, validate_layout_against_limits,
    validate_per_draw_group2, validate_vertex_layout_against_limits, ReflectError,
    ReflectedMaterialUniformBlock, ReflectedRasterLayout, ReflectedUniformField,
    ReflectedUniformScalarKind,
};

/// Shader route table, optional material asset registry, and WGSL composition patches.
pub use registry::MaterialRegistry;
pub use resolve_raster::resolve_raster_pipeline;
pub use router::{MaterialRouter, ShaderRouteEntry};
pub use wgsl::{compose_wgsl, WgslPatch};

pub use crate::pipelines::raster::NullFamily;
