//! GPU device, adapter, and swapchain configuration.

mod blendshape_bind_chunks;
mod context;
#[cfg(feature = "debug-hud")]
mod frame_cpu_gpu_timing;
pub mod hi_z_build;
pub mod mesh_preprocess;
mod per_draw_uniforms;
pub mod vr_mirror_blit;

pub mod frame_globals;

pub use blendshape_bind_chunks::plan_blendshape_bind_chunks;
pub use context::{instance_flags_for_gpu_init, GpuContext};
pub use frame_globals::FrameGpuUniforms;
pub use mesh_preprocess::MeshPreprocessPipelines;
pub use per_draw_uniforms::{
    write_per_draw_uniform_slab, PaddedPerDrawUniforms, INITIAL_PER_DRAW_UNIFORM_SLOTS,
    PER_DRAW_UNIFORM_STRIDE,
};
pub use vr_mirror_blit::{VrMirrorBlitResources, VR_MIRROR_EYE_LAYER};
