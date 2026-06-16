//! CPU-side Hi-Z helpers: data types, mip-pyramid math, GPU-readback decoding, and the
//! AABB-vs-pyramid occlusion test driving CPU world-mesh culling.

pub mod pyramid;
pub mod query;
pub mod readback;
pub mod snapshot;

pub use pyramid::{hi_z_pyramid_dimensions, mip_levels_for_extent};
pub use query::{hi_z_view_proj_matrices, mesh_fully_occluded_in_hiz, stereo_hiz_keeps_draw};
#[cfg(test)]
pub(crate) use snapshot::HiZCpuSnapshot;
pub use snapshot::HiZCullData;
