//! Compiled DAG: immutable pass order and per-frame execution.

use winit::window::Window;

use crate::backend::RenderBackend;
use crate::gpu::GpuContext;
use crate::present::{acquire_surface_outcome, SurfaceFrameOutcome};
use crate::scene::SceneCoordinator;

use super::context::RenderPassContext;
use super::error::GraphExecuteError;
use super::frame_params::{FrameRenderParams, HostCameraFrame, OcclusionViewId};
use super::pass::{PassPhase, RenderPass};
use super::world_mesh_draw_prep::{CameraTransformDrawFilter, WorldMeshDrawCollection};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileStats {
    /// Number of passes in the flattened schedule.
    pub pass_count: usize,
    /// Number of topological levels (waves); a parallelism hint for future scheduling.
    pub topo_levels: usize,
}

/// Immutable execution schedule produced by [`super::GraphBuilder::build`].
///
/// After build, pass order and [`Self::needs_surface_acquire`] do not change. Per-frame work is
/// [`Self::execute`] / [`Self::execute_multi_view`]. Multi-view uses a frame-global submit plus
/// one submit per view so `wgpu::Queue::write_buffer` work in passes is ordered before each view’s GPU work.
pub struct CompiledRenderGraph {
    pub(super) passes: Vec<Box<dyn RenderPass>>,
    /// `true` when any pass writes [`super::resources::ResourceSlot::Backbuffer`] — frame execution
    /// acquires the swapchain once and presents after submit.
    pub needs_surface_acquire: bool,
    /// Build-time stats for tests and future profiling hooks.
    pub compile_stats: CompileStats,
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
    #[allow(clippy::too_many_arguments)]
    pub fn execute_offscreen_single_view(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
        external: ExternalOffscreenTargets<'_>,
        transform_filter: Option<CameraTransformDrawFilter>,
        prefetched_world_mesh_draws: Option<WorldMeshDrawCollection>,
    ) -> Result<(), GraphExecuteError> {
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
        ) = if !needs_swapchain {
            (None, None)
        } else if self.needs_surface_acquire {
            match acquire_surface_outcome(gpu, window)? {
                SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => {
                    return Ok(());
                }
                SurfaceFrameOutcome::Acquired(tex) => {
                    let view = tex
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    (Some(tex), Some(view))
                }
            }
        } else {
            (None, None)
        };

        let device_arc = gpu.device().clone();
        let queue_arc = gpu.queue().clone();
        let device = device_arc.as_ref();

        let has_frame_global = self
            .passes
            .iter()
            .any(|p| p.phase() == PassPhase::FrameGlobal);
        if has_frame_global {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-graph-frame-global"),
            });
            // Frame-global phase (e.g. mesh deform): use first view for host camera / scene context.
            {
                let first = views.first().expect("views non-empty");
                let resolved =
                    Self::resolve_view_from_target(&first.target, gpu, &backbuffer_view_holder)?;
                let mut frame_params = FrameRenderParams {
                    scene,
                    backend,
                    depth_texture: resolved.depth_texture,
                    depth_view: resolved.depth_view,
                    surface_format: resolved.surface_format,
                    viewport_px: resolved.viewport_px,
                    host_camera: first.host_camera,
                    multiview_stereo: resolved.multiview_stereo,
                    transform_draw_filter: first.draw_filter.clone(),
                    offscreen_write_render_texture_asset_id: resolved
                        .offscreen_write_render_texture_asset_id,
                    prefetched_world_mesh_draws: None,
                    occlusion_view: resolved.occlusion_view,
                };
                let mut ctx = RenderPassContext {
                    device,
                    queue: &queue_arc,
                    encoder: &mut encoder,
                    backbuffer: None,
                    depth_view: None,
                    frame: Some(&mut frame_params),
                };
                for pass in &mut self.passes {
                    if pass.phase() == PassPhase::FrameGlobal {
                        pass.execute(&mut ctx)?;
                    }
                }
            }
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands(cmd);
        }

        // Per-view: separate encoder + submit so queue writes before each submit apply only to this view.
        for view in &mut views {
            let prefetched = view.prefetched_world_mesh_draws.take();
            let draw_filter = view.draw_filter.clone();
            let host_camera = view.host_camera;
            let target_is_swapchain = matches!(view.target, FrameViewTarget::Swapchain);
            let resolved =
                Self::resolve_view_from_target(&view.target, gpu, &backbuffer_view_holder)?;
            let mut frame_params = FrameRenderParams {
                scene,
                backend,
                depth_texture: resolved.depth_texture,
                depth_view: resolved.depth_view,
                surface_format: resolved.surface_format,
                viewport_px: resolved.viewport_px,
                host_camera,
                multiview_stereo: resolved.multiview_stereo,
                transform_draw_filter: draw_filter,
                offscreen_write_render_texture_asset_id: resolved
                    .offscreen_write_render_texture_asset_id,
                prefetched_world_mesh_draws: prefetched,
                occlusion_view: resolved.occlusion_view,
            };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-graph-per-view"),
            });
            let mut ctx = RenderPassContext {
                device,
                queue: &queue_arc,
                encoder: &mut encoder,
                backbuffer: resolved.backbuffer,
                depth_view: Some(resolved.depth_view),
                frame: Some(&mut frame_params),
            };
            for pass in &mut self.passes {
                if pass.phase() == PassPhase::PerView {
                    pass.execute(&mut ctx)?;
                }
            }

            if target_is_swapchain {
                let Some(bb) = backbuffer_view_holder.as_ref() else {
                    return Err(GraphExecuteError::MissingSwapchainView);
                };
                let viewport_px = gpu.surface_extent_px();
                let mut queue_lock = queue_arc.lock().expect("queue mutex poisoned");
                if let Err(e) = backend.encode_debug_hud_overlay(
                    device,
                    &queue_lock,
                    &mut encoder,
                    bb,
                    viewport_px,
                ) {
                    logger::warn!("debug HUD overlay: {e}");
                }
                let cmd = encoder.finish();
                gpu.submit_tracked_frame_commands_with_queue(&mut queue_lock, cmd);
            } else {
                let cmd = encoder.finish();
                gpu.submit_tracked_frame_commands(cmd);
            }

            if Self::should_hi_z_submit_after_pass(&view.host_camera, &view.target) {
                backend
                    .occlusion
                    .hi_z_on_frame_submitted_for_view(device, view.occlusion_view_id());
            }
        }

        if let Some(f) = frame {
            f.present();
        }
        Ok(())
    }

    fn should_hi_z_submit_after_pass(host: &HostCameraFrame, target: &FrameViewTarget<'_>) -> bool {
        match target {
            FrameViewTarget::Swapchain => true,
            FrameViewTarget::ExternalMultiview(_) => !host.suppress_occlusion_temporal,
            FrameViewTarget::OffscreenRt(_) => !host.suppress_occlusion_temporal,
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
                })
            }
            FrameViewTarget::ExternalMultiview(ext) => Ok(ResolvedView {
                depth_texture: ext.depth_texture,
                depth_view: ext.depth_view,
                backbuffer: Some(ext.color_view),
                surface_format: ext.surface_format,
                viewport_px: ext.extent_px,
                multiview_stereo: true,
                offscreen_write_render_texture_asset_id: None,
                occlusion_view: OcclusionViewId::Main,
            }),
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
            }),
        }
    }
}
