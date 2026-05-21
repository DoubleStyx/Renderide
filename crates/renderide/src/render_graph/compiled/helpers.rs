//! Shared helpers for [`super::CompiledRenderGraph`] execution.
//!
//! Grouped by concern:
//! - [`frame_params`] -- builders that assemble per-pass frame parameters.
//! - [`msaa_views`] -- multisampled / multiview attachment view resolution.
//! - [`transient_extent`] -- transient texture/buffer extent and mip resolution.
//! - [`attachments`] -- raster attachment resolution and the wgpu render-pass lifecycle.

mod attachments;
mod frame_params;
mod msaa_views;
mod transient_extent;

pub(super) use attachments::{
    coalesce_render_pass_template, execute_graph_raster_pass_node,
    frame_sample_count_from_raster_ctx, pass_info_raster_template, resolve_color_attachments,
    resolve_depth_attachment_with_stencil,
};
pub(super) use frame_params::{
    GraphPassFrameViewInputs, ResolvedFrameRenderParamsInputs, frame_render_params_from_resolved,
    frame_render_params_from_shared,
};
pub(super) use msaa_views::resolve_forward_msaa_views_from_graph_resources;
pub(super) use transient_extent::{
    clamp_mip_levels_for_transient_extent, clamp_viewport_for_transient_alloc,
    create_transient_layer_views, resolve_buffer_size, resolve_transient_extent,
};
