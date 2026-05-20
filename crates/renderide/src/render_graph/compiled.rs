//! Compiled DAG: immutable pass order and per-frame execution.

use crate::gpu::{GpuContext, GpuLimits};
use crate::render_graph::GraphExecutionBackend;
use crate::scene::SceneCoordinator;

use super::pass::PassNode;
use super::resources::{
    ImportedBufferDecl, ImportedTextureDecl, TextureHandle, TransientSubresourceDesc,
};
use super::schedule::{FrameSchedule, ScheduleHudSnapshot};
use super::validation::{GraphValidationReport, RenderGraphValidationMode};
use crate::camera::ViewId;

pub(super) mod cache;
mod exec;
mod frame_view;
mod helpers;
mod resource;

#[cfg(test)]
mod dot;

pub(crate) use frame_view::{
    ExternalFrameTargets, ExternalOffscreenTargets, FrameView, FrameViewResourceHints,
    FrameViewTarget, OffscreenColorCopyTarget, OffscreenSampleCountPolicy, ViewPostProcessing,
};
pub(super) use resource::{
    CompileStats, CompiledBufferResource, CompiledPassInfo, CompiledTextureResource,
    ResourceLifetime, ResourceLifetimeLane, ResourceLifetimeSegment,
};

/// Borrows shared across frame-global and per-view [`CompiledRenderGraph::execute_multi_view`] passes.
pub(super) struct MultiViewExecutionContext<'a> {
    /// GPU context (surface, swapchain, submits).
    pub(super) gpu: &'a mut GpuContext,
    /// Scene after cache flush.
    pub(super) scene: &'a SceneCoordinator,
    /// Narrow graph-facing backend access packet.
    pub(super) backend: &'a mut dyn GraphExecutionBackend,
    /// Device for encoders and pipeline state.
    pub(super) device: &'a wgpu::Device,
    /// Limits for pass contexts.
    pub(super) gpu_limits: &'a GpuLimits,
    /// Swapchain color view when a view targets the main window.
    pub(super) backbuffer_view_holder: &'a Option<wgpu::TextureView>,
}

impl CompiledRenderGraph {
    /// Stores main-frame MSAA depth scratch handles used by per-view recording helpers.
    pub(crate) fn set_main_graph_msaa_transient_handles(&mut self, handles: [TextureHandle; 2]) {
        self.main_graph_msaa_transient_handles = Some(handles);
    }

    /// Releases any pass-local view-scoped caches for views that are no longer active.
    pub(crate) fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        if retired_views.is_empty() {
            return;
        }
        for pass in &mut self.passes {
            pass.release_view_resources(retired_views);
        }
    }
}

/// Immutable execution schedule produced by [`super::GraphBuilder::build`].
///
/// ## Pass storage
///
/// Passes are stored as [`PassNode`] enum values, enabling the executor to dispatch to the
/// correct context type (raster/compute) without a runtime `graph_managed_raster()` toggle.
///
/// ## Frame-global contract
///
/// [`super::pass::PassPhase::FrameGlobal`] passes run once per tick in
/// [`CompiledRenderGraph::execute_multi_view_frame_global_passes`]. Host/scene context and
/// resource resolution for that encoder use the **first** [`FrameView`] only.
///
/// ## Submit model
///
/// The executor records frame-global work plus one command buffer per view, drains deferred
/// uploads on the main thread, and submits the assembled batch once per tick.
pub struct CompiledRenderGraph {
    /// Ordered pass nodes in execution order (culled, sorted).
    pub(super) passes: Vec<PassNode>,
    /// `true` when any pass writes an imported frame color target; frame execution
    /// acquires the swapchain once and presents after submit.
    pub needs_surface_acquire: bool,
    /// Build-time stats for tests and profiling hooks.
    pub compile_stats: CompileStats,
    /// Retained pass metadata in execution order.
    pub pass_info: Vec<CompiledPassInfo>,
    /// Compiled transient texture metadata.
    pub transient_textures: Vec<CompiledTextureResource>,
    /// Compiled transient buffer metadata.
    pub transient_buffers: Vec<CompiledBufferResource>,
    /// Lifetime lanes for transient texture alias slots.
    pub texture_lifetime_lanes: Vec<ResourceLifetimeLane>,
    /// Lifetime lanes for transient buffer alias slots.
    pub buffer_lifetime_lanes: Vec<ResourceLifetimeLane>,
    /// Declared subresource views of transient textures. Resolved lazily at execute time via
    /// [`super::context::GraphResolvedResources::subresource_view`]; see
    /// [`super::resources::SubresourceHandle`].
    pub subresources: Vec<TransientSubresourceDesc>,
    /// Imported texture declarations.
    pub imported_textures: Vec<ImportedTextureDecl>,
    /// Imported buffer declarations.
    pub imported_buffers: Vec<ImportedBufferDecl>,
    /// Single source of truth for pass ordering, phase, and wave membership.
    pub schedule: FrameSchedule,
    /// Build-time scheduler summary for diagnostics and HUD overlays.
    pub schedule_hud: ScheduleHudSnapshot,
    /// Build-time validation diagnostics.
    pub validation_report: GraphValidationReport,
    /// Runtime validation policy for this graph.
    pub validation_mode: RenderGraphValidationMode,
    /// When this graph is the main frame graph from [`super::build_main_graph`], transient handles
    /// for the MSAA depth and R32-float depth-resolve scratch resources.
    pub(super) main_graph_msaa_transient_handles: Option<[TextureHandle; 2]>,
}

pub(super) struct ResolvedView<'a> {
    pub(super) depth_texture: &'a wgpu::Texture,
    pub(super) depth_view: &'a wgpu::TextureView,
    pub(super) backbuffer: Option<&'a wgpu::TextureView>,
    pub(super) surface_format: wgpu::TextureFormat,
    pub(super) viewport_px: (u32, u32),
    pub(super) multiview_stereo: bool,
    pub(super) offscreen_write_render_texture_asset_id: Option<i32>,
    pub(super) view_id: ViewId,
    pub(super) sample_count: u32,
    pub(super) post_processing: ViewPostProcessing,
    // MSAA views are now in the per-view blackboard (MsaaViewsSlot), resolved from graph
    // transient textures by the executor via resolve_forward_msaa_views_from_graph_resources.
}
