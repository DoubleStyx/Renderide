//! Per-pass pipeline descriptor for multi-pass material shaders.
//!
//! A material stem may declare multiple passes via Unity-style `//#pass` tags parsed in `build.rs`
//! and embedded alongside the composed WGSL (see [`crate::embedded_shaders::embedded_target_passes`]).
//! Each tag sits directly above an `@fragment` entry point and names a small semantic [`PassType`].
//! Blend, depth, cull, color-mask, stencil, and offset state are declared explicitly on the tag.
//! Every descriptor becomes one `wgpu::RenderPipeline`; the forward encode loop dispatches all
//! pipelines for every draw that binds the material, in declared order.
//!
//! Each state field records both an authored fallback and whether the matching host runtime
//! material property (`_ZWrite`, `_ZTest`, `_Cull`, `_ColorMask`, `_OffsetFactor`, `_OffsetUnits`,
//! `_SrcBlend`, `_DstBlend`, stencil) may override that fallback.
//!
//! Embedded material WGSL must declare at least one `//#pass` tag. [`default_pass`] remains only for
//! fallback / null pipelines that do not come from embedded material source.

mod blend_mode;
mod pass_kind;
mod property_ids;
pub(in crate::materials) mod wire_tables;

#[cfg(test)]
mod policy_tests;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use blend_mode::material_blend_mode_for_lookup;
pub use blend_mode::{MaterialBlendMode, material_blend_mode_from_maps};
pub(crate) use pass_kind::materialized_embedded_pass_for_blend_mode;
pub(crate) use pass_kind::{
    COLOR_WRITES_NONE, MaterialRenderStatePolicy, PASS_BLEND_ONE_ONE,
    PASS_BLEND_ONE_ONE_MINUS_SRC_ALPHA, PASS_BLEND_OVERLAY_NOOP_COLOR_MAX_ALPHA,
    PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA,
};
pub use pass_kind::{
    DefaultPassParams, MaterialPassDesc, MaterialPassState, PassType, default_pass,
    materialized_pass_for_blend_mode,
};
pub use property_ids::MaterialPipelinePropertyIds;
#[cfg(test)]
pub(crate) use test_support::{
    depth_prepass, forward_alpha_blend_pass, forward_alpha_blend_zwrite_pass, forward_filter_pass,
    forward_pass, forward_premultiplied_transparent_pass, forward_transparent_cull_back_pass,
    forward_transparent_cull_front_pass, forward_transparent_pass, forward_two_sided_pass,
    outline_pass, overlay_always_pass, overlay_behind_pass, overlay_front_pass, stencil_pass,
    transparent_rgb_pass, volume_front_pass,
};

pub(crate) use blend_mode::{PropertyMapRef, first_float_from_maps, first_vec4_from_maps};
