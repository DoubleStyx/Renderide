//! GPU device, adapter, and swapchain configuration.

mod blendshape_bind_chunks;
mod context;
pub mod mesh_preprocess;
mod per_draw_uniforms;

pub use blendshape_bind_chunks::plan_blendshape_bind_chunks;
pub use context::GpuContext;
pub use mesh_preprocess::MeshPreprocessPipelines;
pub use per_draw_uniforms::{
    write_per_draw_uniform_slab, PaddedPerDrawUniforms, INITIAL_PER_DRAW_UNIFORM_SLOTS,
    PER_DRAW_UNIFORM_STRIDE,
};
