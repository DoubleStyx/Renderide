//! Compiled DAG: immutable pass order and per-frame execution.

use std::sync::Arc;

use crate::backend::RenderBackend;
use crate::gpu::{GpuContext, GpuLimits};
use crate::scene::SceneCoordinator;

use super::frame_params::{HostCameraFrame, OcclusionViewId};
use super::ids::{GroupId, PassId};
use super::pass::{GroupScope, PassKind, RenderPass};
use super::resources::{
    ImportedBufferDecl, ImportedTextureDecl, ResourceAccess, TextureAttachmentResolve,
    TextureAttachmentTarget, TransientBufferDesc, TransientTextureDesc,
};
use super::world_mesh_draw_prep::{CameraTransformDrawFilter, WorldMeshDrawCollection};

/// Inputs for [`CompiledRenderGraph::execute_offscreen_single_view`] and
/// [`crate::backend::RenderBackend::execute_frame_graph_offscreen_single_view`].
pub struct OffscreenSingleViewExecuteSpec<'a> {
    /// Scene after cache flush.
    pub scene: &'a SceneCoordinator,
    /// Per-view camera and clip data from the host.
    pub host_camera: HostCameraFrame,
    /// Pre-built color/depth views for the render texture.
    pub external: ExternalOffscreenTargets<'a>,
    /// Optional mesh transform filter for secondary cameras.
    pub transform_filter: Option<CameraTransformDrawFilter>,
    /// Optional pre-collected draws when skipping CPU mesh collection.
    pub prefetched_world_mesh_draws: Option<WorldMeshDrawCollection>,
}

/// Single-view color + depth for secondary cameras rendering to a host [`crate::resources::GpuRenderTexture`].
pub struct ExternalOffscreenTargets<'a> {
    /// Host render-texture asset id for `color_view` (used to suppress self-sampling during this pass).
    pub render_texture_asset_id: i32,
    /// Color attachment (`Rgba16Float` for Unity `ARGBHalf` parity).
    pub color_view: &'a wgpu::TextureView,
    /// Depth texture backing `depth_view`.
    pub depth_texture: &'a wgpu::Texture,
    /// Depth-stencil view for the offscreen pass.
    pub depth_view: &'a wgpu::TextureView,
    /// Color/depth attachment extent in physical pixels.
    pub extent_px: (u32, u32),
    /// Color attachment format (must match pipeline targets).
    pub color_format: wgpu::TextureFormat,
}

/// Pre-acquired 2-layer color + depth targets for OpenXR multiview (no window swapchain acquire).
pub struct ExternalFrameTargets<'a> {
    /// `D2Array` color view (`array_layer_count` = 2).
    pub color_view: &'a wgpu::TextureView,
    /// Backing `D2Array` depth texture for copy/snapshot passes.
    pub depth_texture: &'a wgpu::Texture,
    /// `D2Array` depth view (`array_layer_count` = 2).
    pub depth_view: &'a wgpu::TextureView,
    /// Pixel extent per eye (`width`, `height`).
    pub extent_px: (u32, u32),
    /// Color format (must match pipeline targets).
    pub surface_format: wgpu::TextureFormat,
}

/// Where a multi-view frame writes color/depth.
pub enum FrameViewTarget<'a> {
    /// Main window swapchain (acquire + present).
    Swapchain,
    /// OpenXR stereo multiview (pre-acquired array targets).
    ExternalMultiview(ExternalFrameTargets<'a>),
    /// Secondary camera to a host render texture.
    OffscreenRt(ExternalOffscreenTargets<'a>),
}

/// One view to render in a multi-view frame.
pub struct FrameView<'a> {
    /// Clip planes, FOV, and matrix overrides for this view.
    pub host_camera: HostCameraFrame,
    /// Color/depth destination.
    pub target: FrameViewTarget<'a>,
    /// Optional transform filter for secondary cameras.
    pub draw_filter: Option<CameraTransformDrawFilter>,
    /// When set, the world-mesh forward path skips CPU draw collection for this view.
    pub prefetched_world_mesh_draws: Option<WorldMeshDrawCollection>,
}

/// Borrows shared across frame-global and per-view [`CompiledRenderGraph::execute_multi_view`] passes.
pub(super) struct MultiViewExecutionContext<'a> {
    /// GPU context (surface, swapchain, submits).
    gpu: &'a mut GpuContext,
    /// Scene after cache flush.
    scene: &'a SceneCoordinator,
    /// Render backend (materials, occlusion, HUD overlay).
    backend: &'a mut RenderBackend,
    /// Device for encoders and pipeline state.
    device: &'a wgpu::Device,
    /// Limits for [`RenderPassContext`].
    gpu_limits: &'a GpuLimits,
    /// Shared queue handle (wgpu::Queue is internally synchronized).
    queue_arc: &'a Arc<wgpu::Queue>,
    /// Swapchain color view when a view targets the main window.
    backbuffer_view_holder: &'a Option<wgpu::TextureView>,
}

impl<'a> FrameView<'a> {
    /// Hi-Z / occlusion slot for this view.
    pub fn occlusion_view_id(&self) -> OcclusionViewId {
        match &self.target {
            FrameViewTarget::Swapchain | FrameViewTarget::ExternalMultiview(_) => {
                OcclusionViewId::Main
            }
            FrameViewTarget::OffscreenRt(ext) => {
                OcclusionViewId::OffscreenRenderTexture(ext.render_texture_asset_id)
            }
        }
    }
}

/// Statistics emitted when building a [`CompiledRenderGraph`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileStats {
    /// Number of passes in the flattened schedule.
    pub pass_count: usize,
    /// Number of Kahn sweep **waves** (parallel layers) in the build-time DAG sort.
    ///
    /// Runtime execution still walks the compiled pass list in one flat order; this
    /// count is not a separate executor schedule. It is exposed in the debug HUD (with pass count) as
    /// a diagnostic and a hint for future wave-based or parallel record scheduling.
    pub topo_levels: usize,
    /// Number of passes culled because their writes could not reach an import/export.
    pub culled_count: usize,
    /// Number of declared transient texture handles.
    pub transient_texture_count: usize,
    /// Number of physical transient texture slots after lifetime aliasing.
    pub transient_texture_slots: usize,
    /// Number of declared transient buffer handles.
    pub transient_buffer_count: usize,
    /// Number of physical transient buffer slots after lifetime aliasing.
    pub transient_buffer_slots: usize,
    /// Number of imported texture declarations.
    pub imported_texture_count: usize,
    /// Number of imported buffer declarations.
    pub imported_buffer_count: usize,
}

/// Inclusive pass-index lifetime for one transient resource in the retained schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLifetime {
    /// First retained pass index that touches the resource.
    pub first_pass: usize,
    /// Last retained pass index that touches the resource.
    pub last_pass: usize,
}

impl ResourceLifetime {
    /// Returns true when two lifetimes do not overlap.
    pub fn disjoint(self, other: Self) -> bool {
        self.last_pass < other.first_pass || other.last_pass < self.first_pass
    }
}

/// Compiled metadata for a transient texture handle.
#[derive(Clone, Debug)]
pub struct CompiledTextureResource {
    /// Original descriptor.
    pub desc: TransientTextureDesc,
    /// Usage union across retained pass declarations.
    pub usage: wgpu::TextureUsages,
    /// Retained-schedule lifetime.
    pub lifetime: Option<ResourceLifetime>,
    /// Physical alias slot assigned by the compiler.
    pub physical_slot: usize,
}

/// Compiled metadata for a transient buffer handle.
#[derive(Clone, Debug)]
pub struct CompiledBufferResource {
    /// Original descriptor.
    pub desc: TransientBufferDesc,
    /// Usage union across retained pass declarations.
    pub usage: wgpu::BufferUsages,
    /// Retained-schedule lifetime.
    pub lifetime: Option<ResourceLifetime>,
    /// Physical alias slot assigned by the compiler.
    pub physical_slot: usize,
}

/// Compiled setup metadata for one retained pass.
#[derive(Clone, Debug)]
pub struct CompiledPassInfo {
    /// Original pass id in the builder.
    pub id: PassId,
    /// Pass name.
    pub name: String,
    /// Group id.
    pub group: GroupId,
    /// Command kind.
    pub kind: PassKind,
    /// Declared accesses.
    pub(crate) accesses: Vec<ResourceAccess>,
    /// Optional multiview mask for raster passes.
    pub multiview_mask: Option<std::num::NonZeroU32>,
    /// Render-pass attachment template for graph-managed raster passes.
    pub raster_template: Option<RenderPassTemplate>,
}

/// Compiled render-pass attachment template.
#[derive(Clone, Debug)]
pub struct RenderPassTemplate {
    /// Color attachments in declaration order.
    pub color_attachments: Vec<ColorAttachmentTemplate>,
    /// Optional depth/stencil attachment.
    pub depth_stencil_attachment: Option<DepthAttachmentTemplate>,
    /// Optional multiview mask.
    pub multiview_mask: Option<std::num::NonZeroU32>,
}

/// Color attachment template.
#[derive(Clone, Debug)]
pub struct ColorAttachmentTemplate {
    /// Color target handle.
    pub target: TextureAttachmentTarget,
    /// Load operation.
    pub load: wgpu::LoadOp<wgpu::Color>,
    /// Store operation.
    pub store: wgpu::StoreOp,
    /// Optional resolve target.
    pub resolve_to: Option<TextureAttachmentResolve>,
}

/// Depth/stencil attachment template.
#[derive(Clone, Debug)]
pub struct DepthAttachmentTemplate {
    /// Depth/stencil target handle.
    pub target: TextureAttachmentTarget,
    /// Depth operations.
    pub depth: wgpu::Operations<f32>,
    /// Optional stencil operations.
    pub stencil: Option<wgpu::Operations<u32>>,
}

/// Ordered compiled group.
#[derive(Clone, Debug)]
pub struct CompiledGroup {
    /// Group id.
    pub id: GroupId,
    /// Group label.
    pub name: &'static str,
    /// Execution scope.
    pub scope: GroupScope,
    /// Indices into [`CompiledRenderGraph::pass_info`].
    pub pass_indices: Vec<usize>,
}

/// Immutable execution schedule produced by [`super::GraphBuilder::build`].
///
/// After build, pass order and [`Self::needs_surface_acquire`] do not change. Per-frame work is
/// [`Self::execute`] / [`Self::execute_multi_view`]. When any [`PassPhase::FrameGlobal`] passes exist,
/// multi-view records them in one encoder and submits once, then uses **one encoder + submit per
/// [`FrameView`]** for [`PassPhase::PerView`] passes so `wgpu::Queue::write_buffer` work is ordered
/// before each view’s GPU commands.
///
/// ## Frame-global contract
///
/// [`PassPhase::FrameGlobal`] passes run **once** per tick in
/// [`CompiledRenderGraph::execute_multi_view_frame_global_passes`]. Host/scene context and
/// [`super::context::GraphResolvedResources`] resolution for that encoder use the **first**
/// [`FrameView`] only (camera, viewport, and transient pool keys derived from that view).
pub struct CompiledRenderGraph {
    pub(super) passes: Vec<Box<dyn RenderPass>>,
    /// `true` when any pass writes an imported frame color target; frame execution
    /// acquires the swapchain once and presents after submit.
    pub needs_surface_acquire: bool,
    /// Build-time stats for tests and future profiling hooks.
    pub compile_stats: CompileStats,
    /// Ordered groups and retained pass membership.
    pub groups: Vec<CompiledGroup>,
    /// Retained pass metadata in execution order.
    pub pass_info: Vec<CompiledPassInfo>,
    /// Compiled transient texture metadata.
    pub transient_textures: Vec<CompiledTextureResource>,
    /// Compiled transient buffer metadata.
    pub transient_buffers: Vec<CompiledBufferResource>,
    /// Imported texture declarations.
    pub imported_textures: Vec<ImportedTextureDecl>,
    /// Imported buffer declarations.
    pub imported_buffers: Vec<ImportedBufferDecl>,
    /// Indices into [`Self::passes`] for [`PassPhase::FrameGlobal`] passes (execution order).
    pub(super) frame_global_pass_indices: Vec<usize>,
    /// Indices into [`Self::passes`] for [`PassPhase::PerView`] passes (execution order).
    pub(super) per_view_pass_indices: Vec<usize>,
    /// When this graph is the main frame graph from [`super::build_main_graph`], transient handles
    /// for MSAA color/depth/R32 resources so the executor can wire [`super::frame_params::FrameRenderParams`]
    /// once per view.
    pub(super) main_graph_msaa_transient_handles:
        Option<[crate::render_graph::resources::TextureHandle; 3]>,
}

pub(super) struct ResolvedView<'a> {
    depth_texture: &'a wgpu::Texture,
    depth_view: &'a wgpu::TextureView,
    backbuffer: Option<&'a wgpu::TextureView>,
    surface_format: wgpu::TextureFormat,
    viewport_px: (u32, u32),
    multiview_stereo: bool,
    offscreen_write_render_texture_asset_id: Option<i32>,
    occlusion_view: OcclusionViewId,
    sample_count: u32,
    msaa_color_view: Option<wgpu::TextureView>,
    msaa_depth_view: Option<wgpu::TextureView>,
    msaa_depth_resolve_r32_view: Option<wgpu::TextureView>,
    msaa_depth_is_array: bool,
    msaa_stereo_depth_layer_views: Option<[wgpu::TextureView; 2]>,
    msaa_stereo_r32_layer_views: Option<[wgpu::TextureView; 2]>,
}

mod exec;
mod helpers;
