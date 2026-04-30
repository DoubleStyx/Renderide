//! Hi-Z occlusion subsystem: CPU helpers (mip layout, readback unpacking, screen-space tests),
//! GPU pyramid build (`gpu`), and the [`OcclusionSystem`] facade that owns per-view temporal state.

pub(crate) mod cpu;
pub mod gpu;
mod system;

pub use cpu::pyramid::{
    HI_Z_PYRAMID_MAX_LONG_EDGE, hi_z_pyramid_dimensions, mip_dimensions, mip_levels_for_extent,
};
pub use cpu::readback::{hi_z_snapshot_from_linear_linear, unpack_linear_rows_to_mips};
pub use cpu::snapshot::{HiZCpuSnapshot, HiZCullData, HiZStereoCpuSnapshot};
pub use cpu::test::{hi_z_view_proj_matrices, mesh_fully_occluded_in_hiz, stereo_hiz_keeps_draw};
pub use gpu::{HiZBuildRecord, HiZGpuState, HiZHistoryTarget, encode_hi_z_build};
pub(crate) use system::HiZBuildInput;
pub use system::OcclusionSystem;
