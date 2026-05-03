//! Compiled DAG: immutable pass order and per-frame execution.

use std::sync::Arc;

use crate::backend::BackendGraphAccess;
use crate::gpu::{GpuContext, GpuLimits};
use crate::scene::SceneCoordinator;

use super::error::GraphExecuteError;
use super::frame_params::FrameViewClear;
use super::ids::{GroupId, PassId};
use super::pass::{GroupScope, PassKind, PassMergeHint, PassNode};
use super::resources::{
    ImportedBufferDecl, ImportedTextureDecl, ResourceAccess, TextureAttachmentResolve,
    TextureAttachmentTarget, TransientBufferDesc, TransientSubresourceDesc, TransientTextureDesc,
};
use super::schedule::FrameSchedule;
use crate::camera::{HostCameraFrame, ViewId};
use crate::world_mesh::draw_prep::{CameraTransformDrawFilter, WorldMeshDrawCollection};
use crate::world_mesh::{PrefetchedWorldMeshViewDraws, WorldMeshHelperNeeds};

/// Single-view color + depth for secondary cameras rendering to a host [`crate::gpu_pools::GpuRenderTexture`].
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
    /// Stable logical identity for view-scoped resources and temporal state.
    pub view_id: ViewId,
    /// Clip planes, FOV, and matrix overrides for this view.
    pub host_camera: HostCameraFrame,
    /// Color/depth destination.
    pub target: FrameViewTarget<'a>,
    /// Optional transform filter for secondary cameras.
    pub draw_filter: Option<CameraTransformDrawFilter>,
    /// Background clear/skybox behavior for this view.
    pub clear: FrameViewClear,
    /// Explicit world-mesh draw plan for this view.
    pub world_mesh_draw_plan: WorldMeshDrawPlan,
}

/// Explicit world-mesh draw policy for a [`FrameView`].
pub enum WorldMeshDrawPlan {
    /// Use the supplied collection and skip in-graph CPU scene collection.
    Prefetched(Box<PrefetchedWorldMeshViewDraws>),
    /// Render no world-mesh draws for this view.
    Empty,
}

impl WorldMeshDrawPlan {
    /// Returns the prefetched collection when this plan carries one.
    pub fn as_prefetched(&self) -> Option<&WorldMeshDrawCollection> {
        match self {
            Self::Prefetched(draws) => Some(&draws.collection),
            Self::Empty => None,
        }
    }

    /// Returns the full prefetched per-view packet when this plan carries one.
    pub fn as_prefetched_view_draws(&self) -> Option<&PrefetchedWorldMeshViewDraws> {
        match self {
            Self::Prefetched(draws) => Some(draws),
            Self::Empty => None,
        }
    }

    /// Returns helper-pass requirements derived during draw collection.
    pub fn helper_needs(&self) -> WorldMeshHelperNeeds {
        self.as_prefetched_view_draws()
            .map_or_else(WorldMeshHelperNeeds::default, |draws| draws.helper_needs)
    }
}

/// Borrows shared across frame-global and per-view [`CompiledRenderGraph::execute_multi_view`] passes.
pub(super) struct MultiViewExecutionContext<'a, 'backend> {
    /// GPU context (surface, swapchain, submits).
    pub(super) gpu: &'a mut GpuContext,
    /// Scene after cache flush.
    pub(super) scene: &'a SceneCoordinator,
    /// Narrow graph-facing backend access packet.
    pub(super) backend: &'a mut BackendGraphAccess<'backend>,
    /// Device for encoders and pipeline state.
    pub(super) device: &'a wgpu::Device,
    /// Limits for pass contexts.
    pub(super) gpu_limits: &'a GpuLimits,
    /// Shared queue handle (wgpu::Queue is internally synchronized).
    pub(super) queue_arc: &'a Arc<wgpu::Queue>,
    /// Swapchain color view when a view targets the main window.
    pub(super) backbuffer_view_holder: &'a Option<wgpu::TextureView>,
}

impl FrameViewTarget<'_> {
    /// `true` when this target renders to a 2-layer multiview color attachment.
    pub fn is_multiview_target(&self) -> bool {
        matches!(self, FrameViewTarget::ExternalMultiview(_))
    }

    /// Host render-texture asset id this target writes, or [`None`] when not an offscreen RT.
    pub fn offscreen_rt_asset_id(&self) -> Option<i32> {
        match self {
            FrameViewTarget::OffscreenRt(ext) => Some(ext.render_texture_asset_id),
            FrameViewTarget::Swapchain | FrameViewTarget::ExternalMultiview(_) => None,
        }
    }

    /// Viewport extent in pixels for this target.
    pub fn extent_px(&self, gpu: &GpuContext) -> (u32, u32) {
        match self {
            FrameViewTarget::ExternalMultiview(ext) => ext.extent_px,
            FrameViewTarget::OffscreenRt(ext) => ext.extent_px,
            FrameViewTarget::Swapchain => gpu.surface_extent_px(),
        }
    }

    /// Color attachment format for this target.
    pub fn color_format(&self, gpu: &GpuContext) -> wgpu::TextureFormat {
        match self {
            FrameViewTarget::ExternalMultiview(ext) => ext.surface_format,
            FrameViewTarget::OffscreenRt(ext) => ext.color_format,
            FrameViewTarget::Swapchain => gpu.config_format(),
        }
    }

    /// Depth attachment format for this target. Lazily allocates the swapchain depth target if
    /// needed (the `Swapchain` case requires `&mut`).
    pub fn depth_format(
        &self,
        gpu: &mut GpuContext,
    ) -> Result<wgpu::TextureFormat, GraphExecuteError> {
        match self {
            FrameViewTarget::ExternalMultiview(ext) => Ok(ext.depth_texture.format()),
            FrameViewTarget::OffscreenRt(ext) => Ok(ext.depth_texture.format()),
            FrameViewTarget::Swapchain => {
                let (depth_tex, _) = gpu
                    .ensure_depth_target()
                    .map_err(GraphExecuteError::DepthTarget)?;
                Ok(depth_tex.format())
            }
        }
    }

    /// Effective MSAA sample count for this target. Offscreen RTs are single-sampled.
    pub fn sample_count(&self, gpu: &GpuContext) -> u32 {
        match self {
            FrameViewTarget::ExternalMultiview(_) => gpu.swapchain_msaa_effective_stereo().max(1),
            FrameViewTarget::OffscreenRt(_) => 1,
            FrameViewTarget::Swapchain => gpu.swapchain_msaa_effective().max(1),
        }
    }
}

impl<'a> FrameView<'a> {
    /// Builds a view that renders the main desktop swapchain.
    pub fn for_swapchain(
        host_camera: HostCameraFrame,
        world_mesh_draw_plan: WorldMeshDrawPlan,
    ) -> Self {
        Self {
            view_id: ViewId::Main,
            host_camera,
            target: FrameViewTarget::Swapchain,
            draw_filter: None,
            clear: FrameViewClear::skybox(),
            world_mesh_draw_plan,
        }
    }

    /// Builds a view that renders an OpenXR stereo multiview pair of eye layers.
    pub fn for_hmd(
        host_camera: HostCameraFrame,
        external: ExternalFrameTargets<'a>,
        world_mesh_draw_plan: WorldMeshDrawPlan,
    ) -> Self {
        Self {
            view_id: ViewId::Main,
            host_camera,
            target: FrameViewTarget::ExternalMultiview(external),
            draw_filter: None,
            clear: FrameViewClear::skybox(),
            world_mesh_draw_plan,
        }
    }

    /// Builds a view that renders a secondary camera to a host render texture.
    pub fn for_offscreen_rt(
        view_id: ViewId,
        host_camera: HostCameraFrame,
        external: ExternalOffscreenTargets<'a>,
        draw_filter: Option<CameraTransformDrawFilter>,
        clear: FrameViewClear,
        world_mesh_draw_plan: WorldMeshDrawPlan,
    ) -> Self {
        Self {
            view_id,
            host_camera,
            target: FrameViewTarget::OffscreenRt(external),
            draw_filter,
            clear,
            world_mesh_draw_plan,
        }
    }

    /// Stable logical identity for this view.
    pub fn view_id(&self) -> ViewId {
        self.view_id
    }

    /// `true` when this view both targets a multiview attachment AND the host camera carries stereo
    /// matrices -- i.e. the per-view record path should emit stereo clustering / multiview draws.
    ///
    /// Single source of truth; every caller that gates on "is this the stereo multiview view?"
    /// goes through this method rather than re-deriving the AND-chain.
    pub fn is_multiview_stereo_active(&self) -> bool {
        self.target.is_multiview_target() && self.host_camera.active_stereo().is_some()
    }
}

impl CompiledRenderGraph {
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

/// Statistics emitted when building a [`CompiledRenderGraph`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileStats {
    /// Number of passes in the flattened schedule.
    pub pass_count: usize,
    /// Number of Kahn sweep **waves** (parallel layers) in the build-time DAG sort.
    ///
    /// Runtime execution still walks the compiled pass list in one flat order; this
    /// count is not a separate executor schedule. It is exposed in the debug HUD (with pass count)
    /// as a diagnostic and a hint for future wave-based parallel record scheduling.
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
    /// Backend merge hint declared at setup time. See [`PassMergeHint`].
    ///
    /// The wgpu executor currently ignores this; the field is populated for use by a future
    /// subpass-aware backend without a second migration pass across all call sites.
    pub merge_hint: PassMergeHint,
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
/// ## Pass storage
///
/// Passes are stored as [`PassNode`] enum values, enabling the executor to dispatch to the
/// correct context type (raster/compute/copy/callback) without a runtime `graph_managed_raster()`
/// toggle.
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
    /// Ordered groups and retained pass membership.
    pub groups: Vec<CompiledGroup>,
    /// Retained pass metadata in execution order.
    pub pass_info: Vec<CompiledPassInfo>,
    /// Compiled transient texture metadata.
    pub transient_textures: Vec<CompiledTextureResource>,
    /// Compiled transient buffer metadata.
    pub transient_buffers: Vec<CompiledBufferResource>,
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
    /// When this graph is the main frame graph from [`super::build_main_graph`], transient handles
    /// for the MSAA depth and R32-float depth-resolve scratch resources.
    pub(super) main_graph_msaa_transient_handles:
        Option<[crate::render_graph::resources::TextureHandle; 2]>,
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
    // MSAA views are now in the per-view blackboard (MsaaViewsSlot), resolved from graph
    // transient textures by the executor. ResolvedView no longer carries them.
}

mod exec;
mod helpers;

mod dot;
pub use dot::DotFormat;
