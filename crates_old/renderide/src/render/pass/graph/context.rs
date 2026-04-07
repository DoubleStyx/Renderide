//! Frame- and pass-level execution context for the render graph.

use nalgebra::Matrix4;

use crate::render::batch::SpaceDrawBatch;
use crate::render::target::RenderTarget;
use crate::render::view::ViewParams;
use crate::session::Session;

use crate::render::pass::mesh_prep::{CachedMeshDraws, CachedMeshDrawsRef};

use super::views::RenderTargetViews;

/// Per-pass context passed to `RenderPass::execute`.
pub struct RenderPassContext<'a> {
    /// GPU state including device, queue, mesh cache, and depth texture.
    pub gpu: &'a mut crate::gpu::GpuState,
    /// Session for scene, assets, and view state.
    pub session: &'a Session,
    /// Draw batches for this frame.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager for mesh pipelines.
    pub pipeline_manager: &'a mut crate::gpu::PipelineManager,
    /// Frame index for ring buffer offset; advanced once per frame by the graph.
    pub frame_index: u64,
    /// Viewport dimensions (width, height).
    pub viewport: (u32, u32),
    /// Primary projection matrix; passes build view-proj per batch as needed.
    pub proj: Matrix4<f32>,
    /// Optional overlay projection override (borrowed for the frame; owned by [`RenderGraphContext`]).
    /// When `Some`, overlay batches use this instead of `proj` (e.g. orthographic for screen-space UI).
    pub overlay_projection_override: Option<&'a ViewParams>,
    /// Current color and depth attachments.
    pub render_target: RenderTargetViews<'a>,
    /// Command encoder for this frame; pass records into this.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Optional timestamp query set for GPU pass timing.
    pub timestamp_query_set: Option<&'a wgpu::QuerySet>,
    /// Cached mesh draws from a single collect per frame. Mesh and overlay passes use this.
    pub(crate) cached_mesh_draws: Option<CachedMeshDrawsRef<'a>>,
}

/// Frame-level context created when executing the main-view render graph.
pub struct RenderGraphContext<'a> {
    /// GPU state.
    pub gpu: &'a mut crate::gpu::GpuState,
    /// Session.
    pub session: &'a Session,
    /// Draw batches.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager.
    pub pipeline_manager: &'a mut crate::gpu::PipelineManager,
    /// Render target (surface or offscreen).
    pub target: &'a RenderTarget,
    /// Depth view for Surface targets; Offscreen provides its own. Dimensions must match target.
    pub depth_view_override: Option<&'a wgpu::TextureView>,
    /// Viewport (width, height); must match target dimensions.
    pub viewport: (u32, u32),
    /// Primary projection matrix.
    pub proj: Matrix4<f32>,
    /// Optional overlay projection override. When `Some`, overlay pass uses this instead of `proj`.
    pub overlay_projection_override: Option<ViewParams>,
    /// Optional timestamp query set for GPU pass timing.
    pub timestamp_query_set: Option<&'a wgpu::QuerySet>,
    /// Optional resolve buffer for timestamp readback.
    pub timestamp_resolve_buffer: Option<&'a wgpu::Buffer>,
    /// Optional staging buffer for timestamp readback.
    pub timestamp_staging_buffer: Option<&'a wgpu::Buffer>,
    /// When true, the graph allocates or reuses RTAO MRT textures from
    /// [`crate::render::pass::graph::RenderGraph::rtao_mrt_cache`] using [`GpuState::config`](crate::gpu::GpuState::config) color format and [`viewport`](Self::viewport).
    /// Set false for offscreen paths that render without RTAO (e.g. [`crate::render::RenderLoop::render_to_target`]).
    pub enable_rtao_mrt: bool,
    /// Pre-collected mesh draws from the collect phase. When `Some`, skips collect in execute.
    pub(crate) pre_collected: Option<&'a CachedMeshDraws>,
    /// When set, invoked on the graph encoder after passes (and optional timestamp resolve) and before
    /// [`wgpu::Queue::submit`], so extra commands share the same submission (e.g. camera-task readback copy).
    pub before_submit: Option<&'a mut dyn FnMut(&mut wgpu::CommandEncoder)>,
}
