//! Render loop, draw batching, and render graph.
//!
//! ## Overview
//!
//! **Render loop** ([`RenderLoop`]): drives one frame via the graph. Main window uses
//! [`RenderTarget::Surface`] (caller acquires the swapchain texture once per frame; mesh
//! preparation uses that target's dimensions so aspect and viewport stay consistent).
//! Offscreen camera tasks use [`RenderTarget::Offscreen`].
//!
//! **Draw batching** ([`SpaceDrawBatch`]): per-space batches with `view_transform`, sorted by
//! `sort_key` (higher renders on top; matches Unity `sortingOrder`). Draws are ordered by
//! `(is_overlay, -sort_key, pipeline_variant, material_id, mesh_asset_id)`.
//!
//! **Layer filtering**: [`collect_draw_batches`](crate::session::Session::collect_draw_batches)
//! skips `Hidden` layers. Main view includes private overlays (dashboard, loading indicators);
//! `render_private_ui` in [`CameraRenderTask`](crate::shared::CameraRenderTask) controls private
//! overlay inclusion for offscreen renders.
//!
//! **Projection**: Main view uses [`ViewParams::perspective_from_session`]; offscreen
//! [`CameraRenderTask`](crate::shared::CameraRenderTask)s use
//! [`projection_for_params`](pass::projection_for_params). Both use reverse-Z depth.
//!
//! **Overlay pass**: [`OverlayRenderPass`] renders overlays after meshes with `LoadOp::Load`
//! (preserve framebuffer) and alpha blend so UI composites over the scene.
//!
//! **RenderTaskExecutor**: Runs [`CameraRenderTask`](crate::shared::CameraRenderTask)s offscreen.
//! Each task records render plus texture→readback copy in **one** queue submit via
//! [`RenderLoop::render_to_target`](r#loop::RenderLoop::render_to_target); [`GpuFrameScheduler`](crate::gpu::GpuFrameScheduler)
//! (inside [`PipelineManager`](crate::gpu::PipelineManager)) caps concurrent ring-buffer users.
//! [`RenderLoop::drain_pending_camera_task_readbacks`](r#loop::RenderLoop::drain_pending_camera_task_readbacks)
//! completes `map_async` and writes pixels to shared memory (called each tick from the app and session).
//!
//! **Subgraphs**: [`GraphBuilder::add_subgraph`](pass::GraphBuilder::add_subgraph) nests a full
//! [`RenderGraph`](pass::RenderGraph) as one node; [`GraphBuilder::add_edge`](pass::GraphBuilder::add_edge)
//! accepts [`GraphNodeId`](pass::GraphNodeId) (pass or subgraph). The main window still uses a flat
//! graph from [`build_main_render_graph`](pass::build_main_render_graph); this is scaffolding for
//! multi-viewport / mirror / probe passes later.
//!
//! **RTAO MRT**: [`RenderGraph`](pass::RenderGraph) owns lazily created MRT textures when
//! [`RenderGraphContext::enable_rtao_mrt`](pass::RenderGraphContext::enable_rtao_mrt) is true
//! (main window with RTAO enabled and ray tracing available). The main loop rebuilds the graph
//! when that effective flag changes (see [`pass::build_main_render_graph`]) so RTAO passes are
//! omitted when disabled. [`RenderLoop::render_to_target`](r#loop::RenderLoop::render_to_target) sets
//! `enable_rtao_mrt` false so offscreen paths skip RTAO allocation. The main window and each camera
//! task each invoke [`RenderGraph::execute`](pass::RenderGraph::execute) in the same tick when tasks run.
//!
//! ## UI extension point
//!
//! [`OverlayRenderPass`] is the single extension point for UI rendering. Future UI work should
//! extend or configure this pass:
//!
//! - **Orthographic projection**: Set [`RenderGraphContext::overlay_projection_override`] to
//!   orthographic for screen-space UI (Canvas, HUD). Keep `None` for world-space overlays.
//! - **Stencil**: Overlay pass uses Load/Store for stencil; draws with `stencil_state`
//!   use overlay stencil pipelines. See [`crate::stencil`] for GraphicsChunk RenderType
//!   (MaskWrite, Content, MaskClear).
//!   (e.g. scroll rects, clipping).
//! - **Texture binding**: Extend pipeline bind groups or add a pass that binds atlas textures
//!   for UI sprites and materials.

pub mod batch;
pub mod context;
pub mod lights;
pub mod r#loop;
pub mod pass;
pub mod target;
pub mod task;
pub mod view;
pub mod visibility;

pub use crate::shared::RenderingContext;
pub use crate::stencil::{ClipRect, StencilComparison, StencilOperation, StencilState};
pub use batch::{DrawEntry, SpaceDrawBatch};
pub use context::{FramePhase, current_context, set_context, with_context};
pub use r#loop::{PendingCameraTaskReadback, RenderLoop};
pub use pass::{
    ExecutionUnit, GraphBuilder, GraphNodeId, LabeledSubgraph, MeshRenderPass, OverlayRenderPass,
    PassId, PreCollectedFrameData, RenderGraph, RenderGraphContext, RenderPass, RenderPassContext,
    RenderPassError, RenderTargetViews, SubgraphId, SubgraphLabel, build_main_render_graph,
    prepare_mesh_draws_for_view,
};
pub use target::RenderTarget;
pub use task::RenderTaskExecutor;
pub use view::{ViewParams, ViewProjection};
