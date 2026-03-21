//! Pipeline abstraction: RenderPipeline trait, PipelineManager, and concrete implementations.
//!
//! # Structure
//! - [`core`]: `RenderPipeline` trait, `UniformData`, and shared ring-buffer constants.
//! - [`builder`]: shared construction helpers (primitive state, depth-stencil, color targets,
//!   bind group layouts). Reduces per-pipeline boilerplate until runtime pipeline assembly lands.
//! - [`shaders`]: WGSL source strings, split by pipeline family into a `shaders/` subdirectory.
//! - [`uniforms`]: GPU-side uniform struct layouts (`Uniforms`, `SkinnedUniforms`, `SceneUniforms`).
//! - [`ring_buffer`]: `UniformRingBuffer` and `SkinnedUniformRingBuffer` for batched uniform upload.
//! - One file per concrete pipeline type; MRT variants share `mrt.rs`.

pub(crate) mod builder;
mod core;
pub(crate) mod mrt;
mod normal_debug;
mod overlay_stencil;
mod overlay_stencil_skinned;
mod pbr;
mod pbr_mrt;
mod pbr_ray_query;
mod placeholders;
mod ring_buffer;
pub(crate) mod rt_shadow_uniforms;
mod shaders;
mod skinned;
mod skinned_pbr;
mod uniforms;
mod uv_debug;

pub use core::{
    MAX_BLENDSHAPE_WEIGHTS, MAX_INSTANCE_RUN, NUM_FRAMES_IN_FLIGHT, RenderPipeline, UniformData,
    matrix4_to_wgsl_column_major,
};
pub use mrt::{NormalDebugMRTPipeline, SkinnedMRTPipeline, UvDebugMRTPipeline};
pub use normal_debug::NormalDebugPipeline;
pub use overlay_stencil::{
    OverlayStencilMaskClearPipeline, OverlayStencilMaskWritePipeline, OverlayStencilPipeline,
};
pub use overlay_stencil_skinned::{
    OverlayStencilMaskClearSkinnedPipeline, OverlayStencilMaskWriteSkinnedPipeline,
    OverlayStencilSkinnedPipeline,
};
pub use pbr::PbrPipeline;
pub use pbr_mrt::PbrMRTPipeline;
pub use pbr_ray_query::{
    PbrMrtRayQueryPipeline, PbrRayQueryPipeline, SkinnedPbrMrtRayQueryPipeline,
    SkinnedPbrRayQueryPipeline,
};
pub use placeholders::MaterialPipeline;
pub use rt_shadow_uniforms::{
    RT_SHADOW_MODE_ATLAS, RT_SHADOW_MODE_TRACE, RtShadowSceneBind, RtShadowUniforms,
};
pub use skinned::SkinnedPipeline;
pub use skinned_pbr::{SkinnedPbrMRTPipeline, SkinnedPbrPipeline};
pub use uniforms::SceneUniforms;
pub use uv_debug::UvDebugPipeline;
