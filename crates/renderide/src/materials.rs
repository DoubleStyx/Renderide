//! AAA-style materials: WGSL templates + overrides, pipeline cache, and routing.
//!
//! Host material **properties** live in [`host_data::MaterialPropertyStore`] (IPC batches).
//! **Shader program choice** (which embedded WGSL target to use) is routed via [`MaterialRouter`]
//! from host shader asset ids updated by [`crate::assets::shader::resolve_shader_upload`].
//!
//! # Pipeline-state vs. shader-uniform boundary
//!
//! The host writes material properties as a flat `(property_id -> value)` store. Each property
//! lands in exactly one of three places:
//!
//! | Property kind | Examples | Resolved by | Lives in |
//! |---|---|---|---|
//! | Pipeline state | `_SrcBlend`, `_DstBlend`, `_ZWrite`, `_ZTest`, `_Cull`, `_Stencil*`, `_ColorMask`, `_OffsetFactor`, `_OffsetUnits` | [`MaterialBlendMode`] + [`MaterialRenderState`] | [`MaterialPipelineCacheKey`] (`wgpu::RenderPipeline` build) |
//! | Shader uniform -- value | `_Color`, `_Tint`, `_Cutoff`, `_Glossiness`, `*_ST` | Host property store, packed by reflection | `@group(1) @binding(0)` material struct |
//! | Shader uniform -- keyword | `_NORMALMAP`, `_ALPHATEST_ON`, `_ALPHABLEND_ON`, `_RenderideVariantBits` | Reflected keyword uniforms use compatibility inference; `_RenderideVariantBits` is the raw parsed variant bitmask and is decoded only by WGSL | `@group(1) @binding(0)` material struct |
//! | Texture | `_MainTex`, `_NormalMap`, ... | Host texture pools, bound by reflection | `@group(1) @binding(N)` |
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
//! `_BlendMode` itself is not on the wire -- FrooxEngine translates `MaterialProvider.SetBlendMode`
//! to `_SrcBlend` / `_DstBlend` factors, and [`MaterialBlendMode::from_unity_blend_factors`]
//! reconstructs the mode here.
//!
//! # Pass system
//!
//! Every material WGSL under `shaders/materials/*.wgsl` declares one or more Unity-style `//#pass`
//! comment directives, each sitting directly above an `@fragment` entry point. The build script
//! parses each directive into a static [`MaterialPassDesc`] table per stem. Each desc becomes one
//! `wgpu::RenderPipeline`; the forward encode loop dispatches all pipelines for every draw that
//! binds the material, in declared order.
//!
//! The directive does three things at once:
//! 1. Selects which `@fragment` entry points become pipelines and what to label them.
//! 2. Declares explicit pipeline state: blend, depth write/test, cull, color mask, stencil, offset.
//! 3. Counts the draws per material -- N directives => N pipelines => N `draw_indexed` calls.
//!
//! Recognized pass types are intentionally small: `type=forward` for raster material draws and
//! `type=depth_prepass` for authored depth-only prepasses. Specialized Unity behavior is expressed
//! through metadata such as `blend=transparent_material`, `zwrite=material(off)`, `cull=front`,
//! `color_mask=0`, and `offset=material(0,0)` rather than by multiplying pass kind variants.
//!
//! **Why explicit pass metadata exists when the host already sends pipeline state.** The host's
//! IPC sends one `_SrcBlend`/`_ZWrite`/`_Cull`/etc. set per material -- not per pass. The directive
//! fills the gap host properties cannot fill: multi-draw structure, auxiliary-pass state, and
//! source-authored state Unity exposes in ShaderLab pass blocks. Each field records whether the
//! matching host property may override the authored fallback; `depth_prepass`, for example, accepts
//! stencil / depth-test / offset state but preserves its authored `ZWrite On` and `ColorMask 0`.
//!
//! **Every material WGSL must declare at least one `//#pass`** -- the build script rejects empty
//! declarations. The runtime has no implicit "default forward" fallback; what you see in the
//! WGSL is the entire pipeline topology of the material.
//!
//! # Pipeline primitives
//!
//! The static-feature vocabulary lives next to the material code that consumes it:
//! [`shader_permutation::ShaderPermutation`] selects WGSL variants (e.g. multiview), and
//! [`null_pipeline::NullFamily`] is the debug fallback used when host pipeline build fails. This
//! module composes those primitives into material-driven render pipelines via
//! [`MaterialPipelineCache`], keyed by [`MaterialPipelineCacheKey`] (shader route + permutation
//! + attachment formats + resolved render state).

mod asset_graph;
mod cache;
pub(crate) mod embedded;
pub(crate) mod host_data;
mod material_passes;
mod null_pipeline;
mod pipeline_build_error;
mod pipeline_kind;
mod pipeline_property_resolver;
pub(crate) mod raster_pipeline;
mod registry;
mod render_queue;
mod render_state;
mod router;
pub(crate) mod shader_permutation;
mod system;
#[cfg(test)]
mod wgsl;
mod wgsl_reflect;

pub(crate) use asset_graph::{
    GlobalUniformValueType, MaterialShaderGraphDiagnosticSnapshot, MaterialShaderHotReloadReport,
};
#[cfg(test)]
pub(crate) use cache::MaterialPipelineCache;
/// Pipeline cache keyed by shader route / layout fingerprint.
pub(crate) use cache::{
    MaterialPipelineCacheDiagnosticSnapshot, MaterialPipelineSet, MaterialPipelineVariantSpec,
};

/// Embedded raster materials: bind groups, texture pools, uniform packing, and stem-metadata queries.
pub(crate) use embedded::EmbeddedMaterialBindShader;
pub(crate) use embedded::{
    EmbeddedMaterialBindResources, EmbeddedTangentFallbackMode, EmbeddedTexturePools,
    SnapshotRequirements, embedded_default_stem_for_shader_asset_name,
    embedded_stem_depth_prepass_pass, embedded_stem_needs_color_stream,
    embedded_stem_needs_extended_vertex_streams, embedded_stem_needs_tangent_stream,
    embedded_stem_needs_uv0_stream, embedded_stem_needs_uv1_stream, embedded_stem_needs_uv2_stream,
    embedded_stem_needs_uv3_stream, embedded_stem_needs_wide_uv_stream,
    embedded_stem_pipeline_pass_count, embedded_stem_requires_intersection_pass,
    embedded_stem_tangent_fallback_mode, embedded_stem_uses_alpha_blending,
    embedded_stem_uses_blended_depth_write, embedded_stem_uses_raw_normal_payload,
    embedded_stem_uses_raw_tangent_payload, embedded_stem_uses_scene_color_snapshot,
    embedded_stem_uses_scene_depth_snapshot, embedded_stem_uses_two_sided_transparency,
    embedded_stem_uses_ui_transparent_fallback,
};

pub(crate) use material_passes::{
    COLOR_WRITES_NONE, MaterialBlendMode, MaterialPassDesc, MaterialPassState,
    MaterialPipelinePropertyIds, MaterialRenderStatePolicy, PASS_BLEND_ONE_ONE,
    PASS_BLEND_ONE_ONE_MINUS_SRC_ALPHA, PASS_BLEND_OVERLAY_NOOP_COLOR_MAX_ALPHA,
    PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA, PassType, material_blend_mode_from_maps,
    materialized_embedded_pass_for_blend_mode, materialized_pass_for_blend_mode,
};
pub(crate) use material_passes::{PropertyMapRef, first_float_from_maps, first_vec4_from_maps};
pub(crate) use pipeline_build_error::PipelineBuildError;
pub(crate) use pipeline_kind::RasterPipelineKind;
/// Pipeline family descriptors, per-property GPU layout, and raster kind flags.
pub(crate) use raster_pipeline::MaterialPipelineDesc;
pub(crate) use render_queue::{UNITY_RENDER_QUEUE_ALPHA_TEST, render_queue_is_transparent};
#[cfg(test)]
pub(crate) use render_queue::{
    UNITY_RENDER_QUEUE_GEOMETRY, UNITY_RENDER_QUEUE_OVERLAY, UNITY_RENDER_QUEUE_TRANSPARENT,
};
pub(crate) use render_queue::{
    fallback_render_queue_for_material, material_render_queue_from_maps,
};
#[cfg(test)]
pub(crate) use render_state::MaterialDepthOffsetState;
pub(crate) use render_state::{
    MaterialDepthCompareDomain, MaterialDepthCompareOverride, MaterialRenderState, RasterFrontFace,
    RasterPrimitiveTopology, material_render_state_for_lookup, material_render_state_from_maps,
};

#[cfg(test)]
pub(crate) use wgsl_reflect::{ReflectedMaterialUniformBlock, ReflectedVertexInput};
/// Naga reflection: composed WGSL -> `wgpu` bind layouts, uniform block layout, stem fingerprints.
pub(crate) use wgsl_reflect::{
    ReflectedRasterLayout, ReflectedUniformField, ReflectedUniformScalarKind,
    ReflectedVertexInputFormat, reflect_raster_material_wgsl, validate_layout_against_limits,
    validate_per_draw_group2, validate_vertex_layout_against_limits,
};

/// Null/fallback raster family used when host pipeline build fails.
pub(crate) use null_pipeline::NullFamily;

/// Cached resolver that interns [`MaterialPipelinePropertyIds`] once per
/// [`crate::materials::host_data::PropertyIdRegistry`].
pub(crate) use pipeline_property_resolver::PipelinePropertyResolver;

/// Shader route table, optional material asset registry, and WGSL composition patches.
pub(crate) use registry::{MaterialPipelineResolution, MaterialRegistry};
pub(crate) use router::{MaterialRouter, resolve_raster_pipeline};

/// Static shader feature flags (multiview, etc.) keyed into the pipeline cache.
pub(crate) use shader_permutation::{SHADER_PERM_MULTIVIEW_STEREO, ShaderPermutation};

pub(crate) use system::{MaterialSystem, MaterialSystemDiagnosticSnapshot};
