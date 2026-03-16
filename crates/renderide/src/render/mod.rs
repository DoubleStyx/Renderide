//! Render loop, draw batching, and render graph.
//!
//! ## Overview
//!
//! **Render loop** ([`RenderLoop`]): drives one frame via the graph. Main window uses
//! [`RenderTarget::Surface`]; offscreen camera tasks use [`RenderTarget::Offscreen`].
//!
//! **Draw batching** ([`SpaceDrawBatch`]): per-space batches with `view_transform`, sorted by
//! `sort_key` (higher renders on top; matches Unity `sortingOrder`). Draws are ordered by
//! `(is_overlay, -sort_key, pipeline_variant, material_id, mesh_asset_id)`.
//!
//! **Layer filtering**: [`collect_draw_batches`](crate::session::Session::collect_draw_batches)
//! skips `Hidden` layers. Main view excludes private overlays; `render_private_ui` in
//! [`CameraRenderTask`](crate::shared::CameraRenderTask) controls private overlay inclusion.
//!
//! **Projection**: Main view uses [`ViewParams::perspective_from_session`]; offscreen
//! [`CameraRenderTask`](crate::shared::CameraRenderTask)s use
//! [`projection_for_params`](pass::projection_for_params). Both use reverse-Z depth.
//!
//! **Overlay pass**: [`OverlayRenderPass`] renders overlays after meshes with `LoadOp::Load`
//! (preserve framebuffer) and alpha blend so UI composites over the scene.
//!
//! **RenderTaskExecutor**: Runs [`CameraRenderTask`](crate::shared::CameraRenderTask)s offscreen
//! and copies pixels to shared memory for the host.
//!
//! ## UI extension point
//!
//! [`OverlayRenderPass`] is the single extension point for UI rendering. Future UI work should
//! extend or configure this pass:
//!
//! - **Orthographic projection**: Set [`RenderGraphContext::overlay_projection_override`] to
//!   orthographic for screen-space UI (Canvas, HUD). Keep `None` for world-space overlays.
//! - **Stencil**: Add stencil buffer usage to the overlay pass for GraphicsChunk masking
//!   (e.g. scroll rects, clipping).
//! - **Texture binding**: Extend pipeline bind groups or add a pass that binds atlas textures
//!   for UI sprites and materials.

pub mod batch;
pub mod context;
pub mod pass;
pub mod r#loop;
pub mod task;
pub mod target;
pub mod view;

pub use batch::{DrawEntry, SpaceDrawBatch};
pub use crate::stencil::{ClipRect, StencilComparison, StencilOperation, StencilState};
pub use context::{current_context, set_context, with_context, FramePhase};
pub use crate::shared::RenderingContext;
pub use pass::{
    MeshRenderPass, OverlayRenderPass, RenderGraph, RenderGraphContext, RenderPass,
    RenderPassContext, RenderPassError, RenderTargetViews,
};
pub use r#loop::RenderLoop;
pub use task::RenderTaskExecutor;
pub use target::RenderTarget;
pub use view::{ViewParams, ViewProjection};
