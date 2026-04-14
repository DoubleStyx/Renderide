//! Compiled DAG: immutable pass order and per-frame execution.

use winit::window::Window;

use crate::backend::RenderBackend;
use crate::gpu::GpuContext;
use crate::present::{acquire_surface_outcome, SurfaceFrameOutcome};
use crate::scene::SceneCoordinator;

use super::context::RenderPassContext;
use super::error::GraphExecuteError;
use super::frame_params::{FrameRenderParams, HostCameraFrame};
use super::pass::RenderPass;
use super::world_mesh_draw_prep::CameraTransformDrawFilter;

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
/// only [`Self::execute`], which records into one command encoder and submits once (v1).
///
/// Phase 2 may add subgraph expansion, multi-encoder recording, and explicit barrier insertion.
pub struct CompiledRenderGraph {
    pub(super) passes: Vec<Box<dyn RenderPass>>,
    /// `true` when any pass writes [`super::resources::ResourceSlot::Backbuffer`] — frame execution
    /// acquires the swapchain once and presents after submit.
    pub needs_surface_acquire: bool,
    /// Build-time stats for tests and future profiling hooks.
    pub compile_stats: CompileStats,
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
    ///
    /// `scene` and `backend` are passed through [`super::FrameRenderParams`] to mesh passes; the
    /// graph temporarily takes ownership of `backend.frame_graph` via [`RenderBackend::execute_frame_graph`].
    pub fn execute(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
    ) -> Result<(), GraphExecuteError> {
        self.execute_inner(gpu, window, scene, backend, host_camera, None)
    }

    /// Records passes against pre-built multiview array targets (OpenXR swapchain path).
    ///
    /// Does not acquire the window surface or present. Skips the debug HUD overlay on the external
    /// target (mirror/HUD use a separate path).
    pub fn execute_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.execute_inner(gpu, window, scene, backend, host_camera, Some(external))
    }

    /// Renders the graph to a single-view offscreen color/depth target (secondary camera → render texture).
    ///
    /// Does not acquire the swapchain or present. Sets [`FrameRenderParams::multiview_stereo`] to `false`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_offscreen_single_view(
        &mut self,
        gpu: &mut GpuContext,
        _window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
        external: ExternalOffscreenTargets<'_>,
        transform_filter: Option<CameraTransformDrawFilter>,
    ) -> Result<(), GraphExecuteError> {
        let device_arc = gpu.device().clone();
        let queue_arc = gpu.queue().clone();

        let surface_format = external.color_format;
        let viewport_px = external.extent_px;
        let depth_tex_ref = external.depth_texture;
        let depth_ref = external.depth_view;
        let backbuffer_ref = Some(external.color_view);

        let device = device_arc.as_ref();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-offscreen"),
        });

        let mut frame_params = FrameRenderParams {
            scene,
            backend,
            depth_texture: depth_tex_ref,
            depth_view: depth_ref,
            surface_format,
            viewport_px,
            host_camera,
            multiview_stereo: false,
            transform_draw_filter: transform_filter,
            offscreen_write_render_texture_asset_id: Some(external.render_texture_asset_id),
        };

        let mut ctx = RenderPassContext {
            device,
            queue: &queue_arc,
            encoder: &mut encoder,
            backbuffer: backbuffer_ref,
            depth_view: Some(depth_ref),
            frame: Some(&mut frame_params),
        };

        for pass in &mut self.passes {
            pass.execute(&mut ctx)?;
        }

        let cmd = encoder.finish();
        gpu.submit_tracked_frame_commands(cmd);
        if !host_camera.suppress_occlusion_temporal {
            backend.occlusion.hi_z_on_frame_submitted(device);
        }
        Ok(())
    }

    fn execute_inner(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
        external: Option<ExternalFrameTargets<'_>>,
    ) -> Result<(), GraphExecuteError> {
        let (frame, backbuffer_view_holder): (
            Option<wgpu::SurfaceTexture>,
            Option<wgpu::TextureView>,
        ) = if external.is_some() {
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

        let (surface_format, viewport_px, depth_tex_ref, depth_ref, backbuffer_ref): (
            wgpu::TextureFormat,
            (u32, u32),
            &wgpu::Texture,
            &wgpu::TextureView,
            Option<&wgpu::TextureView>,
        ) = if let Some(ext) = external.as_ref() {
            (
                ext.surface_format,
                ext.extent_px,
                ext.depth_texture,
                ext.depth_view,
                Some(ext.color_view),
            )
        } else {
            let surface_format = gpu.config_format();
            let viewport_px = gpu.surface_extent_px();
            let bb = backbuffer_view_holder
                .as_ref()
                .map(|v| v as &wgpu::TextureView);
            let (depth_tex, depth_view) = gpu
                .ensure_depth_target()
                .map_err(|_| GraphExecuteError::DepthTarget)?;
            (surface_format, viewport_px, depth_tex, depth_view, bb)
        };
        let device = device_arc.as_ref();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph"),
        });

        let mut frame_params = FrameRenderParams {
            scene,
            backend,
            depth_texture: depth_tex_ref,
            depth_view: depth_ref,
            surface_format,
            viewport_px,
            host_camera,
            multiview_stereo: external.is_some(),
            transform_draw_filter: None,
            offscreen_write_render_texture_asset_id: None,
        };

        let mut ctx = RenderPassContext {
            device,
            queue: &queue_arc,
            encoder: &mut encoder,
            backbuffer: backbuffer_ref,
            depth_view: Some(depth_ref),
            frame: Some(&mut frame_params),
        };

        for pass in &mut self.passes {
            pass.execute(&mut ctx)?;
        }

        if external.is_some() {
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands(cmd);
            if !host_camera.suppress_occlusion_temporal {
                backend.occlusion.hi_z_on_frame_submitted(device);
            }
        } else if let Some(view) = backbuffer_view_holder.as_ref() {
            let mut queue_lock = queue_arc.lock().expect("queue mutex poisoned");
            if let Err(e) = backend.encode_debug_hud_overlay(
                device,
                &queue_lock,
                &mut encoder,
                view,
                viewport_px,
            ) {
                logger::warn!("debug HUD overlay: {e}");
            }
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands_with_queue(&mut queue_lock, cmd);
            backend.occlusion.hi_z_on_frame_submitted(device);
        } else {
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands(cmd);
            backend.occlusion.hi_z_on_frame_submitted(device);
        }

        if let Some(f) = frame {
            f.present();
        }
        Ok(())
    }
}
