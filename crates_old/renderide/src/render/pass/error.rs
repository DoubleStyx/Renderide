//! Errors produced when executing render passes on the GPU.

/// Errors that can occur during render pass execution.
#[derive(Debug)]
pub enum RenderPassError {
    /// Wrapper for wgpu surface errors when acquiring the current texture.
    Surface(wgpu::SurfaceError),
    /// Cached mesh draws were not provided to the pass.
    MissingCachedMeshDraws,
    /// MRT views were required but not provided.
    MissingMrtViews,
}

impl From<wgpu::SurfaceError> for RenderPassError {
    fn from(e: wgpu::SurfaceError) -> Self {
        RenderPassError::Surface(e)
    }
}
