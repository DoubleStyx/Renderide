//! Compiled DAG: immutable pass order and per-frame execution.

use std::sync::{Arc, Mutex};

use winit::window::Window;

use crate::backend::RenderBackend;
use crate::gpu::{GpuContext, GpuLimits};
use crate::present::{acquire_surface_outcome, SurfaceFrameOutcome};
use crate::scene::SceneCoordinator;

use std::collections::HashMap;

use super::context::{
    GraphRasterPassContext, GraphResolvedResources, PostSubmitContext, RenderPassContext,
    ResolvedGraphBuffer, ResolvedGraphTexture, ResolvedImportedBuffer, ResolvedImportedTexture,
};
use super::error::GraphExecuteError;
use super::frame_params::{FrameRenderParams, HostCameraFrame, OcclusionViewId};
use super::ids::{GroupId, PassId};
use super::pass::{GroupScope, PassKind, PassPhase, RenderPass};
use super::resources::{
    BufferImportSource, BufferSizePolicy, FrameTargetRole, ImportSource, ImportedBufferDecl,
    ImportedBufferHandle, ImportedTextureDecl, ImportedTextureHandle, ResourceAccess,
    TextureAttachmentResolve, TextureAttachmentTarget, TextureHandle, TextureResourceHandle,
    TransientBufferDesc, TransientExtent, TransientTextureDesc,
};
use super::transient_pool::{BufferKey, TextureKey};
use super::world_mesh_draw_prep::{CameraTransformDrawFilter, WorldMeshDrawCollection};

/// Inputs for [`CompiledRenderGraph::execute_offscreen_single_view`] and
/// [`crate::backend::RenderBackend::execute_frame_graph_offscreen_single_view`].
pub struct OffscreenSingleViewExecuteSpec<'a> {
    /// Target window (swapchain acquire when the graph needs it; offscreen path may still reference extent).
    pub window: &'a Window,
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
    /// When set, [`crate::render_graph::passes::WorldMeshForwardPass`] skips draw collection.
    pub prefetched_world_mesh_draws: Option<WorldMeshDrawCollection>,
}

/// Borrows shared across frame-global and per-view [`CompiledRenderGraph::execute_multi_view`] passes.
struct MultiViewExecutionContext<'a> {
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
    /// Shared queue (mutex for cross-thread encode if needed).
    queue_arc: &'a Arc<Mutex<wgpu::Queue>>,
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
    /// Number of topological levels (waves); a parallelism hint for future scheduling.
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
/// [`Self::execute`] / [`Self::execute_multi_view`]. Multi-view uses a frame-global submit plus
/// one submit per view so `wgpu::Queue::write_buffer` work in passes is ordered before each view’s GPU work.
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
}

struct ResolvedView<'a> {
    depth_texture: &'a wgpu::Texture,
    depth_view: &'a wgpu::TextureView,
    backbuffer: Option<&'a wgpu::TextureView>,
    surface_format: wgpu::TextureFormat,
    viewport_px: (u32, u32),
    multiview_stereo: bool,
    offscreen_write_render_texture_asset_id: Option<i32>,
    occlusion_view: OcclusionViewId,
    sample_count: u32,
}

/// Builds [`FrameRenderParams`] from a resolved target and per-view host/IPC fields.
fn frame_render_params_from_resolved<'a>(
    scene: &'a SceneCoordinator,
    backend: &'a mut RenderBackend,
    resolved: &ResolvedView<'a>,
    host_camera: HostCameraFrame,
    transform_draw_filter: Option<CameraTransformDrawFilter>,
    prefetched_world_mesh_draws: Option<WorldMeshDrawCollection>,
) -> FrameRenderParams<'a> {
    FrameRenderParams {
        scene,
        backend,
        depth_texture: resolved.depth_texture,
        depth_view: resolved.depth_view,
        surface_format: resolved.surface_format,
        viewport_px: resolved.viewport_px,
        host_camera,
        multiview_stereo: resolved.multiview_stereo,
        transform_draw_filter,
        offscreen_write_render_texture_asset_id: resolved.offscreen_write_render_texture_asset_id,
        prefetched_world_mesh_draws,
        prepared_world_mesh_forward: None,
        occlusion_view: resolved.occlusion_view,
        sample_count: resolved.sample_count,
    }
}

/// Outcome of swapchain acquisition for [`CompiledRenderGraph::execute_multi_view`].
enum MultiViewSwapchainAcquire {
    /// No swapchain view required (no swapchain pass, or graph does not bind the backbuffer).
    NotNeeded,
    /// Skip this frame’s GPU work (timeout, occluded, or swapchain reconfigured).
    SkipPresent,
    /// Surface texture and default view for per-view and present.
    Acquired {
        /// Surface texture presented at the end of multi-view execution when present.
        frame: wgpu::SurfaceTexture,
        /// View used as the swapchain color attachment across views.
        backbuffer_view: wgpu::TextureView,
    },
}

/// Acquires the window swapchain when any [`FrameView`] targets [`FrameViewTarget::Swapchain`].
fn acquire_swapchain_for_multi_view_if_needed(
    needs_swapchain: bool,
    needs_surface_acquire: bool,
    gpu: &mut GpuContext,
    window: &Window,
) -> Result<MultiViewSwapchainAcquire, GraphExecuteError> {
    if !needs_swapchain {
        return Ok(MultiViewSwapchainAcquire::NotNeeded);
    }
    if !needs_surface_acquire {
        return Ok(MultiViewSwapchainAcquire::NotNeeded);
    }
    match acquire_surface_outcome(gpu, window)? {
        SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => {
            Ok(MultiViewSwapchainAcquire::SkipPresent)
        }
        SurfaceFrameOutcome::Acquired(tex) => {
            let backbuffer_view = tex
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            Ok(MultiViewSwapchainAcquire::Acquired {
                frame: tex,
                backbuffer_view,
            })
        }
    }
}

fn resolve_transient_extent(
    extent: TransientExtent,
    viewport_px: (u32, u32),
    array_layers: u32,
) -> TransientExtent {
    match extent {
        TransientExtent::Backbuffer if array_layers > 1 => TransientExtent::MultiLayer {
            width: viewport_px.0.max(1),
            height: viewport_px.1.max(1),
            layers: array_layers,
        },
        TransientExtent::Backbuffer => TransientExtent::Custom {
            width: viewport_px.0.max(1),
            height: viewport_px.1.max(1),
        },
        other => other,
    }
}

fn resolve_buffer_size(size_policy: BufferSizePolicy, viewport_px: (u32, u32)) -> u64 {
    match size_policy {
        BufferSizePolicy::Fixed(size) => size.max(1),
        BufferSizePolicy::PerViewport { bytes_per_px } => u64::from(viewport_px.0.max(1))
            .saturating_mul(u64::from(viewport_px.1.max(1)))
            .saturating_mul(bytes_per_px)
            .max(1),
    }
}

fn create_transient_layer_views(
    texture: &wgpu::Texture,
    key: TextureKey,
) -> Vec<wgpu::TextureView> {
    if key.dimension != wgpu::TextureDimension::D2 || key.array_layers <= 1 {
        return Vec::new();
    }
    (0..key.array_layers)
        .map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("render-graph-transient-layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect()
}

fn pass_info_raster_template(
    pass_info: &[CompiledPassInfo],
    pass_idx: usize,
) -> Result<RenderPassTemplate, GraphExecuteError> {
    let Some(info) = pass_info.get(pass_idx) else {
        return Err(GraphExecuteError::MissingRasterTemplate {
            pass: format!("pass#{pass_idx}"),
        });
    };
    info.raster_template
        .clone()
        .ok_or_else(|| GraphExecuteError::MissingRasterTemplate {
            pass: info.name.clone(),
        })
}

fn frame_sample_count(ctx: &GraphRasterPassContext<'_, '_>) -> u32 {
    ctx.frame
        .as_ref()
        .map(|frame| frame.sample_count.max(1))
        .unwrap_or(1)
}

fn resolve_attachment_target(
    target: TextureAttachmentTarget,
    sample_count: u32,
) -> TextureResourceHandle {
    match target {
        TextureAttachmentTarget::Resource(handle) => handle,
        TextureAttachmentTarget::FrameSampled {
            single_sample,
            multisampled,
        } => {
            if sample_count > 1 {
                multisampled
            } else {
                single_sample
            }
        }
    }
}

fn resolve_attachment_resolve_target(
    target: TextureAttachmentResolve,
    sample_count: u32,
) -> Option<TextureResourceHandle> {
    match target {
        TextureAttachmentResolve::Always(handle) => Some(handle),
        TextureAttachmentResolve::FrameMultisampled(handle) => (sample_count > 1).then_some(handle),
    }
}

fn execute_graph_managed_raster_pass(
    pass: &mut dyn RenderPass,
    template: &RenderPassTemplate,
    graph_resources: &GraphResolvedResources,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &mut GraphRasterPassContext<'_, '_>,
) -> Result<(), GraphExecuteError> {
    let sample_count = frame_sample_count(ctx);
    let mut color_attachments = Vec::with_capacity(template.color_attachments.len());
    for color in &template.color_attachments {
        let target = resolve_attachment_target(color.target, sample_count);
        let view = graph_resources.texture_view(target).ok_or_else(|| {
            GraphExecuteError::MissingGraphAttachment {
                pass: pass.name().to_string(),
                resource: format!("{target:?}"),
            }
        })?;
        let resolve_target = match color
            .resolve_to
            .and_then(|target| resolve_attachment_resolve_target(target, sample_count))
        {
            Some(target) => Some(graph_resources.texture_view(target).ok_or_else(|| {
                GraphExecuteError::MissingGraphAttachment {
                    pass: pass.name().to_string(),
                    resource: format!("{target:?}"),
                }
            })?),
            None => None,
        };
        color_attachments.push(Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target,
            ops: wgpu::Operations {
                load: color.load,
                store: color.store,
            },
            depth_slice: None,
        }));
    }

    let depth_stencil_attachment = if let Some(depth) = &template.depth_stencil_attachment {
        let target = resolve_attachment_target(depth.target, sample_count);
        let view = graph_resources.texture_view(target).ok_or_else(|| {
            GraphExecuteError::MissingGraphAttachment {
                pass: pass.name().to_string(),
                resource: format!("{target:?}"),
            }
        })?;
        Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(depth.depth),
            stencil_ops: pass.graph_raster_stencil_ops(ctx, depth),
        })
    } else {
        None
    };

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("render-graph-raster"),
        color_attachments: &color_attachments,
        depth_stencil_attachment,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: pass.graph_raster_multiview_mask(ctx, template),
    });
    pass.execute_graph_raster(ctx, &mut rpass)?;
    Ok(())
}

impl CompiledRenderGraph {
    /// Ordered pass count.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Whether this graph targets the swapchain this frame.
    pub fn needs_surface_acquire(&self) -> bool {
        self.needs_surface_acquire
    }

    /// Records all passes and submits. Matches [`crate::present::present_clear_frame`] recovery
    /// behavior for surface acquire (timeout/occluded skip, validation reconfigure).
    pub fn execute(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
    ) -> Result<(), GraphExecuteError> {
        self.execute_multi_view(
            gpu,
            window,
            scene,
            backend,
            vec![FrameView {
                host_camera,
                target: FrameViewTarget::Swapchain,
                draw_filter: None,
                prefetched_world_mesh_draws: None,
            }],
        )
    }

    /// Records passes against pre-built multiview array targets (OpenXR swapchain path).
    pub fn execute_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.execute_multi_view(
            gpu,
            window,
            scene,
            backend,
            vec![FrameView {
                host_camera,
                target: FrameViewTarget::ExternalMultiview(external),
                draw_filter: None,
                prefetched_world_mesh_draws: None,
            }],
        )
    }

    /// Renders the graph to a single-view offscreen color/depth target (secondary camera → render texture).
    pub fn execute_offscreen_single_view(
        &mut self,
        gpu: &mut GpuContext,
        backend: &mut RenderBackend,
        spec: OffscreenSingleViewExecuteSpec<'_>,
    ) -> Result<(), GraphExecuteError> {
        let window = spec.window;
        let scene = spec.scene;
        let host_camera = spec.host_camera;
        let external = spec.external;
        let transform_filter = spec.transform_filter;
        let prefetched_world_mesh_draws = spec.prefetched_world_mesh_draws;
        self.execute_multi_view(
            gpu,
            window,
            scene,
            backend,
            vec![FrameView {
                host_camera,
                target: FrameViewTarget::OffscreenRt(external),
                draw_filter: transform_filter,
                prefetched_world_mesh_draws,
            }],
        )
    }

    /// Records all views: one encoder + submit for frame-global work, then one encoder + submit per view.
    ///
    /// Per-view passes use [`wgpu::Queue::write_buffer`] for camera uniforms, per-draw slabs, and
    /// cluster params. Those writes are ordered **before** the next `queue.submit`; a single submit
    /// for all views would leave only the last view’s uploads visible to every view’s GPU commands,
    /// so each view is isolated in its own submit.
    ///
    /// Frame-global passes ([`PassPhase::FrameGlobal`]) run once in the first encoder; per-view
    /// passes run for each [`FrameView`] in order.
    pub fn execute_multi_view(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        mut views: Vec<FrameView<'_>>,
    ) -> Result<(), GraphExecuteError> {
        if views.is_empty() {
            return Ok(());
        }

        let needs_swapchain = views
            .iter()
            .any(|v| matches!(v.target, FrameViewTarget::Swapchain));

        let (frame, backbuffer_view_holder): (
            Option<wgpu::SurfaceTexture>,
            Option<wgpu::TextureView>,
        ) = match acquire_swapchain_for_multi_view_if_needed(
            needs_swapchain,
            self.needs_surface_acquire,
            gpu,
            window,
        )? {
            MultiViewSwapchainAcquire::NotNeeded => (None, None),
            MultiViewSwapchainAcquire::SkipPresent => return Ok(()),
            MultiViewSwapchainAcquire::Acquired {
                frame,
                backbuffer_view,
            } => (Some(frame), Some(backbuffer_view)),
        };

        let device_arc = gpu.device().clone();
        let queue_arc = gpu.queue().clone();
        let device = device_arc.as_ref();
        let gpu_limits_owned = gpu.limits().clone();
        let gpu_limits = gpu_limits_owned.as_ref();

        backend.transient_pool_mut().begin_generation();

        let mut mv_ctx = MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc: &queue_arc,
            backbuffer_view_holder: &backbuffer_view_holder,
        };

        self.execute_multi_view_frame_global_passes(&mut mv_ctx, &views)?;

        // Per-view: separate encoder + submit so queue writes before each submit apply only to this view.
        for view in &mut views {
            self.execute_multi_view_submit_for_one_view(&mut mv_ctx, view)?;
        }

        mv_ctx.backend.transient_pool_mut().gc_tick(120);

        if let Some(f) = frame {
            f.present();
        }
        Ok(())
    }

    /// One per-view encoder, per-view [`PassPhase::PerView`] passes, submit, and Hi-Z bookkeeping.
    fn execute_multi_view_submit_for_one_view(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        view: &mut FrameView<'_>,
    ) -> Result<(), GraphExecuteError> {
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc,
            backbuffer_view_holder,
        } = mv_ctx;

        let prefetched = view.prefetched_world_mesh_draws.take();
        let draw_filter = view.draw_filter.clone();
        let host_camera = view.host_camera;
        let target_is_swapchain = matches!(view.target, FrameViewTarget::Swapchain);
        let resolved = Self::resolve_view_from_target(&view.target, gpu, backbuffer_view_holder)?;
        let graph_resources = self.resolve_graph_resources_for_view(device, backend, &resolved);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-per-view"),
        });
        {
            let mut frame_params = frame_render_params_from_resolved(
                scene,
                backend,
                &resolved,
                host_camera,
                draw_filter,
                prefetched,
            );
            for (pass_idx, pass) in self.passes.iter_mut().enumerate() {
                if pass.phase() == PassPhase::PerView {
                    if pass.graph_managed_raster() {
                        let template = pass_info_raster_template(&self.pass_info, pass_idx)?;
                        let mut ctx = GraphRasterPassContext {
                            device,
                            gpu_limits,
                            queue: queue_arc,
                            backbuffer: resolved.backbuffer,
                            depth_view: Some(resolved.depth_view),
                            frame: Some(&mut frame_params),
                            graph_resources: Some(&graph_resources),
                        };
                        execute_graph_managed_raster_pass(
                            pass.as_mut(),
                            &template,
                            &graph_resources,
                            &mut encoder,
                            &mut ctx,
                        )?;
                    } else {
                        let mut ctx = RenderPassContext {
                            device,
                            gpu_limits,
                            queue: queue_arc,
                            encoder: &mut encoder,
                            backbuffer: resolved.backbuffer,
                            depth_view: Some(resolved.depth_view),
                            frame: Some(&mut frame_params),
                            graph_resources: Some(&graph_resources),
                        };
                        pass.execute(&mut ctx)?;
                    }
                }
            }
        }
        graph_resources.release_to_pool(backend.transient_pool_mut());

        if target_is_swapchain {
            let Some(bb) = backbuffer_view_holder.as_ref() else {
                return Err(GraphExecuteError::MissingSwapchainView);
            };
            let viewport_px = gpu.surface_extent_px();
            let mut queue_lock = queue_arc.lock().expect("queue mutex poisoned");
            if let Err(e) =
                backend.encode_debug_hud_overlay(device, &queue_lock, &mut encoder, bb, viewport_px)
            {
                logger::warn!("debug HUD overlay: {e}");
            }
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands_with_queue(&mut queue_lock, cmd);
        } else {
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands(cmd);
        }

        let occlusion_view = view.occlusion_view_id();
        let mut post_ctx = PostSubmitContext {
            device,
            backend,
            occlusion_view,
            host_camera: view.host_camera,
        };
        for pass in self.passes.iter_mut() {
            if pass.phase() == PassPhase::PerView {
                pass.post_submit(&mut post_ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }
        Ok(())
    }

    /// Runs [`PassPhase::FrameGlobal`] passes once per tick using the first view for host/scene context.
    fn execute_multi_view_frame_global_passes(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc,
            backbuffer_view_holder,
        } = mv_ctx;

        let has_frame_global = self
            .passes
            .iter()
            .any(|p| p.phase() == PassPhase::FrameGlobal);
        if !has_frame_global {
            return Ok(());
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-frame-global"),
        });
        // Frame-global phase (e.g. mesh deform): use first view for host camera / scene context.
        {
            let first = views.first().expect("views non-empty");
            let resolved =
                Self::resolve_view_from_target(&first.target, gpu, backbuffer_view_holder)?;
            let graph_resources = self.resolve_graph_resources_for_view(device, backend, &resolved);
            {
                let mut frame_params = frame_render_params_from_resolved(
                    scene,
                    backend,
                    &resolved,
                    first.host_camera,
                    first.draw_filter.clone(),
                    None,
                );
                for (pass_idx, pass) in self.passes.iter_mut().enumerate() {
                    if pass.phase() == PassPhase::FrameGlobal {
                        if pass.graph_managed_raster() {
                            let template = pass_info_raster_template(&self.pass_info, pass_idx)?;
                            let mut ctx = GraphRasterPassContext {
                                device,
                                gpu_limits,
                                queue: queue_arc,
                                backbuffer: None,
                                depth_view: None,
                                frame: Some(&mut frame_params),
                                graph_resources: Some(&graph_resources),
                            };
                            execute_graph_managed_raster_pass(
                                pass.as_mut(),
                                &template,
                                &graph_resources,
                                &mut encoder,
                                &mut ctx,
                            )?;
                        } else {
                            let mut ctx = RenderPassContext {
                                device,
                                gpu_limits,
                                queue: queue_arc,
                                encoder: &mut encoder,
                                backbuffer: None,
                                depth_view: None,
                                frame: Some(&mut frame_params),
                                graph_resources: Some(&graph_resources),
                            };
                            pass.execute(&mut ctx)?;
                        }
                    }
                }
            }
            graph_resources.release_to_pool(backend.transient_pool_mut());
        }
        let cmd = encoder.finish();
        gpu.submit_tracked_frame_commands(cmd);

        let first = views.first().expect("views non-empty");
        let occlusion_view = first.occlusion_view_id();
        let host_camera = first.host_camera;
        let mut post_ctx = PostSubmitContext {
            device,
            backend,
            occlusion_view,
            host_camera,
        };
        for pass in self.passes.iter_mut() {
            if pass.phase() == PassPhase::FrameGlobal {
                pass.post_submit(&mut post_ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }
        Ok(())
    }

    fn resolve_graph_resources_for_view(
        &self,
        device: &wgpu::Device,
        backend: &mut RenderBackend,
        resolved: &ResolvedView<'_>,
    ) -> GraphResolvedResources {
        let mut resources = GraphResolvedResources::with_capacity(
            self.transient_textures.len(),
            self.transient_buffers.len(),
            self.imported_textures.len(),
            self.imported_buffers.len(),
        );
        self.resolve_transient_textures(
            device,
            backend,
            resolved.viewport_px,
            resolved.surface_format,
            resolved.sample_count,
            resolved.multiview_stereo,
            &mut resources,
        );
        self.resolve_transient_buffers(device, backend, resolved.viewport_px, &mut resources);
        self.resolve_imported_textures(resolved, &mut resources);
        self.resolve_imported_buffers(backend, resolved, &mut resources);
        resources
    }

    fn resolve_transient_textures(
        &self,
        device: &wgpu::Device,
        backend: &mut RenderBackend,
        viewport_px: (u32, u32),
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        multiview_stereo: bool,
        resources: &mut GraphResolvedResources,
    ) {
        let mut physical_slots: HashMap<usize, ResolvedGraphTexture> = HashMap::new();
        for (idx, compiled) in self.transient_textures.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            let resolved = physical_slots
                .entry(compiled.physical_slot)
                .or_insert_with(|| {
                    let array_layers = compiled.desc.array_layers.resolve(multiview_stereo);
                    let key = TextureKey {
                        format: compiled.desc.format.resolve(surface_format),
                        extent: resolve_transient_extent(
                            compiled.desc.extent,
                            viewport_px,
                            array_layers,
                        ),
                        mip_levels: compiled.desc.mip_levels,
                        sample_count: compiled.desc.sample_count.resolve(sample_count),
                        dimension: compiled.desc.dimension,
                        array_layers,
                        usage_bits: compiled.usage.bits() as u64,
                    };
                    let lease = backend.transient_pool_mut().acquire_texture_resource(
                        device,
                        key,
                        compiled.desc.label,
                        compiled.usage,
                    );
                    let layer_views = create_transient_layer_views(&lease.texture, key);
                    ResolvedGraphTexture {
                        pool_id: lease.pool_id,
                        physical_slot: compiled.physical_slot,
                        texture: lease.texture,
                        view: lease.view,
                        layer_views,
                    }
                })
                .clone();
            resources.set_transient_texture(TextureHandle(idx as u32), resolved);
        }
    }

    fn resolve_transient_buffers(
        &self,
        device: &wgpu::Device,
        backend: &mut RenderBackend,
        viewport_px: (u32, u32),
        resources: &mut GraphResolvedResources,
    ) {
        let mut physical_slots: HashMap<usize, ResolvedGraphBuffer> = HashMap::new();
        for (idx, compiled) in self.transient_buffers.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            let resolved = physical_slots
                .entry(compiled.physical_slot)
                .or_insert_with(|| {
                    let key = BufferKey {
                        size_policy: compiled.desc.size_policy,
                        usage_bits: compiled.usage.bits() as u64,
                    };
                    let size = resolve_buffer_size(compiled.desc.size_policy, viewport_px);
                    let lease = backend.transient_pool_mut().acquire_buffer_resource(
                        device,
                        key,
                        compiled.desc.label,
                        compiled.usage,
                        size,
                    );
                    ResolvedGraphBuffer {
                        pool_id: lease.pool_id,
                        physical_slot: compiled.physical_slot,
                        buffer: lease.buffer,
                        size: lease.size,
                    }
                })
                .clone();
            resources.set_transient_buffer(super::resources::BufferHandle(idx as u32), resolved);
        }
    }

    fn resolve_imported_textures(
        &self,
        resolved: &ResolvedView<'_>,
        resources: &mut GraphResolvedResources,
    ) {
        for (idx, import) in self.imported_textures.iter().enumerate() {
            let view = match &import.source {
                ImportSource::FrameTarget(FrameTargetRole::ColorAttachment) => {
                    resolved.backbuffer.cloned()
                }
                ImportSource::FrameTarget(FrameTargetRole::DepthAttachment) => {
                    Some(resolved.depth_view.clone())
                }
                ImportSource::External | ImportSource::PingPong(_) => None,
            };
            if let Some(view) = view {
                resources.set_imported_texture(
                    ImportedTextureHandle(idx as u32),
                    ResolvedImportedTexture { view },
                );
            }
        }
    }

    fn resolve_imported_buffers(
        &self,
        backend: &RenderBackend,
        resolved: &ResolvedView<'_>,
        resources: &mut GraphResolvedResources,
    ) {
        let frame_gpu = backend.frame_resources.frame_gpu();
        let cluster_refs = frame_gpu.and_then(|fgpu| {
            fgpu.cluster_cache.get_buffers(
                resolved.viewport_px,
                crate::backend::CLUSTER_COUNT_Z,
                resolved.multiview_stereo,
            )
        });
        for (idx, import) in self.imported_buffers.iter().enumerate() {
            let buffer = match &import.source {
                BufferImportSource::BackendFrameResource("lights") => {
                    frame_gpu.map(|fgpu| fgpu.lights_buffer.clone())
                }
                BufferImportSource::BackendFrameResource("frame_uniforms") => {
                    frame_gpu.map(|fgpu| fgpu.frame_uniform.clone())
                }
                BufferImportSource::BackendFrameResource("cluster_light_counts") => cluster_refs
                    .as_ref()
                    .map(|refs| refs.cluster_light_counts.clone()),
                BufferImportSource::BackendFrameResource("cluster_light_indices") => cluster_refs
                    .as_ref()
                    .map(|refs| refs.cluster_light_indices.clone()),
                BufferImportSource::BackendFrameResource("per_draw_slab") => backend
                    .frame_resources
                    .per_draw()
                    .map(|per_draw| per_draw.per_draw_storage.clone()),
                BufferImportSource::BackendFrameResource(_) => None,
                BufferImportSource::External | BufferImportSource::PingPong(_) => None,
            };
            if let Some(buffer) = buffer {
                resources.set_imported_buffer(
                    ImportedBufferHandle(idx as u32),
                    ResolvedImportedBuffer { buffer },
                );
            }
        }
    }

    fn resolve_view_from_target<'a>(
        target: &'a FrameViewTarget<'a>,
        gpu: &'a mut GpuContext,
        backbuffer_view_holder: &'a Option<wgpu::TextureView>,
    ) -> Result<ResolvedView<'a>, GraphExecuteError> {
        match target {
            FrameViewTarget::Swapchain => {
                let surface_format = gpu.config_format();
                let viewport_px = gpu.surface_extent_px();
                let bb = backbuffer_view_holder
                    .as_ref()
                    .map(|v| v as &wgpu::TextureView);
                let Some(bb_ref) = bb else {
                    return Err(GraphExecuteError::MissingSwapchainView);
                };
                let sample_count = gpu.swapchain_msaa_effective().max(1);
                let (depth_tex, depth_view) = gpu
                    .ensure_depth_target()
                    .map_err(|_| GraphExecuteError::DepthTarget)?;

                Ok(ResolvedView {
                    depth_texture: depth_tex,
                    depth_view,
                    backbuffer: Some(bb_ref),
                    surface_format,
                    viewport_px,
                    multiview_stereo: false,
                    offscreen_write_render_texture_asset_id: None,
                    occlusion_view: OcclusionViewId::Main,
                    sample_count,
                })
            }
            FrameViewTarget::ExternalMultiview(ext) => {
                let sample_count = gpu.swapchain_msaa_effective_stereo().max(1);
                Ok(ResolvedView {
                    depth_texture: ext.depth_texture,
                    depth_view: ext.depth_view,
                    backbuffer: Some(ext.color_view),
                    surface_format: ext.surface_format,
                    viewport_px: ext.extent_px,
                    multiview_stereo: true,
                    offscreen_write_render_texture_asset_id: None,
                    occlusion_view: OcclusionViewId::Main,
                    sample_count,
                })
            }
            FrameViewTarget::OffscreenRt(ext) => Ok(ResolvedView {
                depth_texture: ext.depth_texture,
                depth_view: ext.depth_view,
                backbuffer: Some(ext.color_view),
                surface_format: ext.color_format,
                viewport_px: ext.extent_px,
                multiview_stereo: false,
                offscreen_write_render_texture_asset_id: Some(ext.render_texture_asset_id),
                occlusion_view: OcclusionViewId::OffscreenRenderTexture(
                    ext.render_texture_asset_id,
                ),
                sample_count: 1,
            }),
        }
    }
}
