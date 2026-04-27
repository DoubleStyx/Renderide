//! GPU hierarchical depth pyramid construction and readback for Hi-Z occlusion culling.
//!
//! Used by [`crate::backend::OcclusionSystem`] and [`crate::render_graph::passes::HiZBuildPass`].

mod hi_z_encode;
mod hi_z_gpu;
mod hi_z_pipelines;
mod readback_ring;

pub use hi_z_encode::{encode_hi_z_build, HiZBuildRecord};
pub use hi_z_gpu::HiZGpuState;
pub(crate) use hi_z_gpu::HIZ_MAX_MIPS;
