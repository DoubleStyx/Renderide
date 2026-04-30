//! Per-pass pipeline descriptor for multi-pass material shaders.
//!
//! A material stem may declare multiple passes via `//#material <kind>` tags parsed in `build.rs`
//! and embedded alongside the composed WGSL (see [`crate::embedded_shaders::embedded_target_passes`]).
//! Each tag sits directly above an `@fragment` entry point and names one [`PassKind`]; the build
//! script turns each tag into a [`MaterialPassDesc`] via [`pass_from_kind`]. Every descriptor becomes
//! one `wgpu::RenderPipeline`; the forward encode loop dispatches all pipelines for every draw that
//! binds the material, in declared order.
//!
//! Render-state fields (depth compare, depth write, cull, blend, write mask) live in
//! [`pass_from_kind`]'s per-kind defaults plus a per-pass `MaterialRenderStatePolicy` that decides
//! which host runtime material properties (`_ZWrite`, `_ZTest`, `_Cull`, `_ColorMask`,
//! `_OffsetFactor`, `_OffsetUnits`, `_SrcBlend`, `_DstBlend`, stencil) may override those defaults.
//! Shaders carry no depth / blend / cull metadata of their own.
//!
//! Single-pass materials that declare no `//#material` tag fall through to [`default_pass`],
//! preserving the pre-multi-pass opaque default exactly.

mod blend_mode;
mod pass_kind;
mod property_ids;

#[cfg(test)]
mod policy_tests;
#[cfg(test)]
mod tests;

pub use blend_mode::{
    MaterialBlendMode, material_blend_mode_for_lookup, material_blend_mode_from_maps,
};
pub use pass_kind::{
    COLOR_WRITES_NONE, DefaultPassParams, MaterialPassDesc, MaterialPassState, PassKind,
    default_pass, materialized_pass_for_blend_mode, pass_from_kind,
};
pub use property_ids::MaterialPipelinePropertyIds;

pub(crate) use blend_mode::{PropertyMapRef, first_float_from_maps};
