//! Errors from [`super::FrameGpuResources::new`](crate::backend::FrameGpuResources::new).

use thiserror::Error;

/// Frame-global GPU buffers could not be allocated or wired for the initial `(1x1)` cluster grid.
#[derive(Debug, Error)]
pub enum FrameGpuInitError {
    /// Lights storage [`wgpu::Buffer`] would exceed device limits.
    #[error("lights storage size {size} exceeds GPU storage/buffer limits")]
    LightsStorageExceedsLimits {
        /// Requested storage size in bytes.
        size: u64,
    },
    /// [`crate::backend::cluster_gpu::ClusterBufferCache::ensure_buffers`] failed for the bootstrap viewport.
    #[error("cluster buffers: ensure_buffers failed for 1x1 viewport")]
    ClusterEnsureFailed,
    /// [`crate::backend::cluster_gpu::ClusterBufferCache::get_buffers`] failed after ensure.
    #[error("cluster buffers: get_buffers failed for 1x1 viewport")]
    ClusterGetBuffersFailed,
}
