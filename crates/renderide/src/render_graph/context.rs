//! Per-frame context passed to each [`super::RenderPass`].

use std::sync::{Arc, Mutex};

use super::frame_params::FrameRenderParams;

/// Immutable GPU handles and mutable encoder for one frame’s recording.
pub struct RenderPassContext<'a> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Submission queue (same mutex as [`crate::gpu::GpuContext::queue`]).
    pub queue: &'a Arc<Mutex<wgpu::Queue>>,
    /// Command encoder for this frame (all passes share one encoder in v1).
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Swapchain view when this frame acquired the surface; [`None`] for offscreen-only graphs.
    pub backbuffer: Option<&'a wgpu::TextureView>,
    /// Depth attachment for the main forward pass when configured.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend when the graph participates in mesh drawing.
    pub frame: Option<&'a mut FrameRenderParams<'a>>,
}
