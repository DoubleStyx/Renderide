//! Pipeline abstraction: RenderPipeline trait, PipelineManager, and concrete implementations.
//!
//! Extension point for pipelines, materials, PBR.

mod core;
mod mrt;
mod normal_debug;
mod overlay_stencil;
mod overlay_stencil_skinned;
mod pbr;
mod pbr_mrt;
mod placeholders;
mod ring_buffer;
mod shaders;
mod skinned;
mod skinned_pbr;
mod uniforms;
mod uv_debug;

pub use core::{
    MAX_BLENDSHAPE_WEIGHTS, MAX_INSTANCE_RUN, RenderPipeline, UniformData,
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
pub use placeholders::MaterialPipeline;
pub use skinned::SkinnedPipeline;
pub use skinned_pbr::{SkinnedPbrMRTPipeline, SkinnedPbrPipeline};
pub use uniforms::SceneUniforms;
pub use uv_debug::UvDebugPipeline;
