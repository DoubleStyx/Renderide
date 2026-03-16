//! Render loop: executes one frame via the render graph.
//!
//! Extension point for RenderGraph passes (mirrors, post, UI, probes).

use std::mem::size_of;

use super::pass::{MeshRenderPass, OverlayRenderPass, RenderGraph, RenderGraphContext};
use super::view::ViewParams;
use super::target::RenderTarget;
use super::SpaceDrawBatch;
use crate::gpu::{GpuState, PipelineManager};
use crate::session::Session;

/// Number of timestamp slots (beginning and end of mesh pass).
const TIMESTAMP_QUERY_COUNT: u32 = 2;

/// Interval (frames) between GPU timestamp readbacks for bottleneck diagnosis.
const GPU_READBACK_INTERVAL: u32 = 60;

/// Encapsulates the render frame logic.
pub struct RenderLoop {
    pipeline_manager: PipelineManager,
    graph: RenderGraph,
    /// Query set for mesh pass GPU timestamps. Used when TIMESTAMP_QUERY is supported.
    timestamp_query_set: wgpu::QuerySet,
    /// Buffer to resolve timestamps into. QUERY_RESOLVE | COPY_SRC.
    timestamp_resolve_buffer: wgpu::Buffer,
    /// Staging buffer for readback. COPY_DST | MAP_READ.
    timestamp_staging_buffer: wgpu::Buffer,
    /// Frame count for throttling readback.
    frame_count: u32,
    /// Last measured mesh pass GPU time in milliseconds. Updated every GPU_READBACK_INTERVAL frames.
    last_gpu_mesh_pass_ms: Option<f64>,
}

impl RenderLoop {
    /// Creates a new render loop with pipelines for the given device and config.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let mut graph = RenderGraph::new();
        graph.add_pass(Box::new(MeshRenderPass::new()));
        graph.add_pass(Box::new(OverlayRenderPass::new()));

        let timestamp_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("mesh pass timestamp query set"),
            count: TIMESTAMP_QUERY_COUNT,
            ty: wgpu::QueryType::Timestamp,
        });
        let timestamp_resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve buffer"),
            size: (size_of::<u64>() as u64) * TIMESTAMP_QUERY_COUNT as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let timestamp_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp staging buffer"),
            size: (size_of::<u64>() as u64) * TIMESTAMP_QUERY_COUNT as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            pipeline_manager: PipelineManager::new(device, config),
            graph,
            timestamp_query_set,
            timestamp_resolve_buffer,
            timestamp_staging_buffer,
            frame_count: 0,
            last_gpu_mesh_pass_ms: None,
        }
    }

    /// Returns the last measured mesh pass GPU time in milliseconds, if available.
    pub fn last_gpu_mesh_pass_ms(&self) -> Option<f64> {
        self.last_gpu_mesh_pass_ms
    }

    /// Renders one frame: clear, draw batches. Caller must present the returned texture.
    ///
    /// Uses [`RenderTarget::Surface`] for the main window. Depth texture dimensions
    /// must match the target; the caller ensures this via resize handling.
    pub fn render_frame(
        &mut self,
        gpu: &mut GpuState,
        session: &Session,
        draw_batches: &[SpaceDrawBatch],
    ) -> Result<RenderTarget, wgpu::SurfaceError> {
        let output = gpu.surface.get_current_texture()?;
        let target = RenderTarget::from_surface_texture(output);

        let (width, height) = target.dimensions();
        // Ensure depth texture dimensions match the target.
        let mut target_config = gpu.config.clone();
        target_config.width = width;
        target_config.height = height;
        if let Some(new_depth) =
            crate::gpu::ensure_depth_texture(&gpu.device, &target_config, gpu.depth_size)
        {
            gpu.depth_texture = Some(new_depth);
            gpu.depth_size = (width, height);
        }
        let depth_view = gpu
            .depth_texture
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        let aspect = width as f32 / height.max(1) as f32;
        let view_params = ViewParams::perspective_from_session(session, aspect);
        let proj = view_params.to_projection_matrix();

        let overlay_projection_override = session
            .primary_camera_task()
            .and_then(|t| t.parameters.as_ref())
            .filter(|p| p.projection == crate::shared::CameraProjection::orthographic)
            .map(|p| {
                ViewParams {
                    projection: super::view::ViewProjection::Orthographic {
                        half_size: p.orthographic_size,
                        aspect,
                    },
                    near_clip: p.near_clip.max(0.01),
                    far_clip: p.far_clip,
                }
            });

        let mut ctx = RenderGraphContext {
            gpu,
            session,
            draw_batches,
            pipeline_manager: &mut self.pipeline_manager,
            target: &target,
            depth_view_override: depth_view.as_ref(),
            viewport: (width, height),
            proj,
            overlay_projection_override,
            timestamp_query_set: Some(&self.timestamp_query_set),
            timestamp_resolve_buffer: Some(&self.timestamp_resolve_buffer),
            timestamp_staging_buffer: Some(&self.timestamp_staging_buffer),
        };

        self.graph.execute(&mut ctx).map_err(|e| match e {
            super::pass::RenderPassError::Surface(s) => s,
        })?;

        self.frame_count += 1;
        if self.frame_count >= GPU_READBACK_INTERVAL {
            self.frame_count = 0;
            if let Some(ms) = Self::readback_gpu_timestamps(
                &gpu.device,
                &gpu.queue,
                &self.timestamp_staging_buffer,
            ) {
                self.last_gpu_mesh_pass_ms = Some(ms);
            }
        }

        Ok(target)
    }

    /// Renders to an offscreen target (e.g. CameraRenderTask).
    ///
    /// Uses the target's own depth texture. No timestamp queries. Caller must copy
    /// texture to shared memory after this returns.
    pub fn render_to_target(
        &mut self,
        gpu: &mut GpuState,
        session: &Session,
        draw_batches: &[SpaceDrawBatch],
        target: &RenderTarget,
        proj: nalgebra::Matrix4<f32>,
    ) -> Result<(), super::pass::RenderPassError> {
        let (width, height) = target.dimensions();
        let mut ctx = super::pass::RenderGraphContext {
            gpu,
            session,
            draw_batches,
            pipeline_manager: &mut self.pipeline_manager,
            target,
            depth_view_override: None,
            viewport: (width, height),
            proj,
            overlay_projection_override: None,
            timestamp_query_set: None,
            timestamp_resolve_buffer: None,
            timestamp_staging_buffer: None,
        };
        self.graph.execute(&mut ctx)
    }

    /// Reads back GPU timestamps from the staging buffer. Returns mesh pass duration in ms.
    fn readback_gpu_timestamps(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        staging: &wgpu::Buffer,
    ) -> Option<f64> {
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        let poll_result = device.poll(wgpu::PollType::wait_indefinitely());
        if poll_result.is_err() {
            return None;
        }
        let view = staging.slice(..(size_of::<u64>() as u64 * TIMESTAMP_QUERY_COUNT as u64));
        let timestamps = {
            let mapped = view.get_mapped_range();
            bytemuck::pod_read_unaligned::<[u64; 2]>(&mapped)
        };
        staging.unmap();
        let period = queue.get_timestamp_period();
        let elapsed_ns = timestamps[1].saturating_sub(timestamps[0]) as f64 * period as f64;
        Some(elapsed_ns / 1_000_000.0)
    }
}
