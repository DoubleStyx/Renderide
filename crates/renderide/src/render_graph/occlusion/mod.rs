//! GPU hierarchical depth pyramid construction and readback for Hi-Z occlusion culling.
//!
//! Used by [`crate::backend::OcclusionSystem`] and [`crate::render_graph::passes::HiZBuildPass`].

mod hi_z_gpu;

pub use hi_z_gpu::{encode_hi_z_build, HiZGpuState};
