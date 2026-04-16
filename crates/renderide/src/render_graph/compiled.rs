//! Compiled DAG: immutable pass order and per-frame execution.

use std::sync::{Arc, Mutex};

use winit::window::Window;

use crate::backend::RenderBackend;
use crate::gpu::{GpuContext, GpuLimits};
use crate::present::{acquire_surface_outcome, SurfaceFrameOutcome};
use crate::scene::SceneCoordinator;

use super::context::RenderPassContext;
use super::error::GraphExecuteError;
use super::frame_params::{FrameRenderParams, HostCameraFrame, OcclusionViewId};
use super::pass::{PassPhase, RenderPass};
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
    sample_count: u32,
    msaa_color_view: Option<wgpu::TextureView>,
    msaa_depth_view: Option<wgpu::TextureView>,
    msaa_depth_resolve_r32_view: Option<wgpu::TextureView>,
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
        occlusion_view: resolved.occlusion_view,
        sample_count: resolved.sample_count,
        msaa_color_view: resolved.msaa_color_view.clone(),
        msaa_depth_view: resolved.msaa_depth_view.clone(),
        msaa_depth_resolve_r32_view: resolved.msaa_depth_resolve_r32_view.clone(),
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

        self.execute_multi_view_frame_global_passes(
            gpu,
            scene,
            backend,
            &views,
            &backbuffer_view_holder,
            device,
            gpu_limits,
            &queue_arc,
        )?;

        // Per-view: separate encoder + submit so queue writes before each submit apply only to this view.
        for view in &mut views {
            self.execute_multi_view_submit_for_one_view(
                view,
                gpu,
                scene,
                backend,
                device,
                gpu_limits,
                &queue_arc,
                &backbuffer_view_holder,
            )?;
        }

        if let Some(f) = frame {
            f.present();
        }
        Ok(())
    }

    /// One per-view encoder, per-view [`PassPhase::PerView`] passes, submit, and Hi-Z bookkeeping.
    #[allow(clippy::too_many_arguments)] // Mirrors per-view submit inputs; bundling hits borrow/lifetime limits in the caller loop.
    fn execute_multi_view_submit_for_one_view(
        &mut self,
        view: &mut FrameView<'_>,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        device: &wgpu::Device,
        gpu_limits: &GpuLimits,
        queue_arc: &Arc<Mutex<wgpu::Queue>>,
        backbuffer_view_holder: &Option<wgpu::TextureView>,
    ) -> Result<(), GraphExecuteError> {
        let prefetched = view.prefetched_world_mesh_draws.take();
        let draw_filter = view.draw_filter.clone();
        let host_camera = view.host_camera;
        let target_is_swapchain = matches!(view.target, FrameViewTarget::Swapchain);
        let resolved = Self::resolve_view_from_target(&view.target, gpu, backbuffer_view_holder)?;
        let mut frame_params = frame_render_params_from_resolved(
            scene,
            backend,
            &resolved,
            host_camera,
            draw_filter,
            prefetched,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-per-view"),
        });
        let mut ctx = RenderPassContext {
            device,
            gpu_limits,
            queue: queue_arc,
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

        if Self::should_hi_z_submit_after_pass(&view.host_camera, &view.target) {
            backend
                .occlusion
                .hi_z_on_frame_submitted_for_view(device, view.occlusion_view_id());
        }
        Ok(())
    }

    /// Runs [`PassPhase::FrameGlobal`] passes once per tick using the first view for host/scene context.
    #[allow(clippy::too_many_arguments)] // Single call site; same bundle as per-view submit.
    fn execute_multi_view_frame_global_passes(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        views: &[FrameView<'_>],
        backbuffer_view_holder: &Option<wgpu::TextureView>,
        device: &wgpu::Device,
        gpu_limits: &GpuLimits,
        queue_arc: &Arc<Mutex<wgpu::Queue>>,
    ) -> Result<(), GraphExecuteError> {
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
            let mut frame_params = frame_render_params_from_resolved(
                scene,
                backend,
                &resolved,
                first.host_camera,
                first.draw_filter.clone(),
                None,
            );
            let mut ctx = RenderPassContext {
                device,
                gpu_limits,
                queue: queue_arc,
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
                let sc_req = gpu.swapchain_msaa_effective();
                // Always run so MSAA-off drops [`GpuContext::msaa_targets`]. Skipping when
                // `sc_req <= 1` would leave the previous frame's multisampled textures alive and
                // `sample_count` would stay stale.
                gpu.ensure_msaa_targets(sc_req, surface_format);
                if sc_req > 1 {
                    let _ = gpu.ensure_msaa_depth_resolve_r32_view();
                }
                let sample_count = gpu.msaa_targets_ref().map(|m| m.sample_count).unwrap_or(1);
                let msaa_color_view = gpu.msaa_targets_ref().map(|m| m.color_view.clone());
                let msaa_depth_view = gpu.msaa_targets_ref().map(|m| m.depth_view.clone());
                let msaa_depth_resolve_r32_view = if sample_count > 1 {
                    gpu.msaa_depth_resolve_r32_view_ref().cloned()
                } else {
                    None
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
                    sample_count,
                    msaa_color_view,
                    msaa_depth_view,
                    msaa_depth_resolve_r32_view,
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
                sample_count: 1,
                msaa_color_view: None,
                msaa_depth_view: None,
                msaa_depth_resolve_r32_view: None,
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
                sample_count: 1,
                msaa_color_view: None,
                msaa_depth_view: None,
                msaa_depth_resolve_r32_view: None,
            }),
        }
    }
}
