//! [`MaterialPipelineDesc`]: swapchain-relevant state used to build a [`wgpu::RenderPipeline`] for materials.

use std::num::NonZeroU32;

/// Swapchain-relevant state needed to build a [`wgpu::RenderPipeline`].
#[derive(Clone, Copy, Debug)]
pub struct MaterialPipelineDesc {
    /// Primary color attachment format (for example swapchain format).
    pub surface_format: wgpu::TextureFormat,
    /// Optional depth attachment (meshes / MRT later).
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    /// MSAA sample count (1 = off).
    pub sample_count: u32,
    /// When set, must match the render pass and pipeline (e.g. `0b11` for two multiview layers).
    pub multiview_mask: Option<NonZeroU32>,
}
