//! Render loop: executes one frame via the render graph.
//!
//! Extension point for RenderGraph passes (mirrors, post, UI, probes).
//!
//! Offscreen camera tasks may enqueue [`PendingCameraTaskReadback`] entries; call
//! [`RenderLoop::drain_pending_camera_task_readbacks`] each tick so completed GPU copies are written
//! to shared memory without blocking the full queue between tasks.

use std::collections::VecDeque;
use std::mem::size_of;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use super::SpaceDrawBatch;
use super::pass::{
    PreCollectedFrameData, RenderGraph, RenderGraphContext, build_main_render_graph,
};
use super::target::RenderTarget;
use super::view::ViewParams;
use crate::gpu::{GpuState, PipelineManager};
use crate::session::Session;
use crate::shared::CameraRenderTask;

/// Number of timestamp slots (beginning and end of mesh pass).
const TIMESTAMP_QUERY_COUNT: u32 = 2;

/// Interval (frames) between GPU timestamp readbacks for bottleneck diagnosis.
const GPU_READBACK_INTERVAL: u32 = 60;

/// A camera render task whose readback buffer is mapping asynchronously.
pub struct PendingCameraTaskReadback {
    /// Staging buffer filled by `copy_texture_to_buffer` in the same submit as the render graph.
    pub buffer: wgpu::Buffer,
    /// Completes when host mapping is ready (after [`wgpu::Device::poll`]).
    pub rx: Receiver<Result<(), wgpu::BufferAsyncError>>,
    /// Host task descriptor for the shared-memory write.
    pub task: CameraRenderTask,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub row_bytes: u32,
}

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
    /// Whether RTAO diagnostic has been logged once at startup.
    rtao_diagnostic_logged: bool,
    /// Last built main graph (`RTAO MRT`, `RT shadow compute`, `fullscreen filter hook`); rebuild
    /// [`Self::graph`] when any changes.
    cached_main_graph_key: Option<(bool, bool, bool)>,
    /// Camera tasks awaiting `map_async` completion before writing [`CameraRenderTask::result_data`].
    pending_camera_task_readbacks: VecDeque<PendingCameraTaskReadback>,
}

impl RenderLoop {
    /// Creates a new render loop with pipelines for the given device and config.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let graph = build_main_render_graph(false, false, false)
            .expect("default render graph has no cycles");

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
            rtao_diagnostic_logged: false,
            cached_main_graph_key: Some((false, false, false)),
            pending_camera_task_readbacks: VecDeque::new(),
        }
    }

    /// Enqueues a camera task readback to be finalized after the GPU copy and map complete.
    pub fn enqueue_pending_camera_task_readback(&mut self, pending: PendingCameraTaskReadback) {
        self.pending_camera_task_readbacks.push_back(pending);
    }

    /// Polls the device and writes any ready readbacks into shared memory.
    ///
    /// Call once per application tick (e.g. before the main view render and after submitting camera
    /// tasks) so completions are flushed without indefinite blocking.
    pub fn drain_pending_camera_task_readbacks(
        &mut self,
        device: &wgpu::Device,
        session: &mut Session,
    ) {
        let Some(shm) = session.shared_memory_mut() else {
            self.pending_camera_task_readbacks.clear();
            return;
        };

        loop {
            let _ = device.poll(wgpu::PollType::Poll);
            let before_len = self.pending_camera_task_readbacks.len();
            let mut still_pending = VecDeque::with_capacity(before_len);
            let mut progressed = false;
            for p in self.pending_camera_task_readbacks.drain(..) {
                match p.rx.try_recv() {
                    Ok(Ok(())) => {
                        progressed = true;
                        let slice = p.buffer.slice(..);
                        let mapped = slice.get_mapped_range();
                        let dest_len = p.task.result_data.length as usize;
                        let row_len = (p.width as usize) * 4;
                        let bytes_per_row_u64 = p.bytes_per_row as u64;
                        if bytes_per_row_u64 == p.row_bytes as u64 {
                            shm.access_mut_bytes(&p.task.result_data, |dest: &mut [u8]| {
                                let copy_len = dest_len.min(mapped.len());
                                dest[..copy_len].copy_from_slice(&mapped[..copy_len]);
                            });
                        } else {
                            shm.access_mut_bytes(&p.task.result_data, |dest: &mut [u8]| {
                                let rows = (dest_len / row_len).min(p.height as usize);
                                for row in 0..rows {
                                    let src_start = (row as u64 * bytes_per_row_u64) as usize;
                                    let dst_start = row * row_len;
                                    let copy_len = row_len.min(dest_len.saturating_sub(dst_start));
                                    dest[dst_start..dst_start + copy_len]
                                        .copy_from_slice(&mapped[src_start..src_start + copy_len]);
                                }
                            });
                        }
                        drop(mapped);
                        p.buffer.unmap();
                    }
                    Ok(Err(e)) => {
                        logger::error!("Camera task readback map failed: {}", e);
                        progressed = true;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        logger::error!("Camera task readback channel disconnected");
                        progressed = true;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        still_pending.push_back(p);
                    }
                }
            }
            self.pending_camera_task_readbacks = still_pending;
            if self.pending_camera_task_readbacks.is_empty() {
                break;
            }
            if !progressed {
                break;
            }
        }
    }

    /// Returns the last measured mesh pass GPU time in milliseconds, if available.
    pub fn last_gpu_mesh_pass_ms(&self) -> Option<f64> {
        self.last_gpu_mesh_pass_ms
    }

    /// Number of offscreen camera readbacks waiting for GPU mapping completion.
    pub fn pending_camera_task_readback_count(&self) -> usize {
        self.pending_camera_task_readbacks.len()
    }

    /// Evicts pipelines for the given material. Call when a material is unloaded to avoid unbounded registry growth.
    pub fn evict_material(&mut self, material_id: i32) {
        self.pipeline_manager.evict_material(material_id);
    }

    /// Evicts cached shader-family pipelines for `shader_asset_id` at the given swapchain format.
    pub fn evict_shader_pipelines(&mut self, shader_asset_id: i32, format: wgpu::TextureFormat) {
        self.pipeline_manager
            .evict_shader_pipelines(shader_asset_id, format);
    }

    /// Renders one frame: clear, draw batches. Caller must acquire the swapchain with
    /// [`wgpu::Surface::get_current_texture`], wrap it in [`RenderTarget::from_surface_texture`],
    /// and present the inner [`wgpu::SurfaceTexture`] after a successful return.
    ///
    /// Depth texture dimensions are updated from [`RenderTarget::dimensions`] when they differ
    /// from the last frame.
    ///
    /// When `pre_collected` is `Some`, mesh draws and projection were built for the same
    /// target extent the caller passes here (typically by acquiring once, then running
    /// [`super::pass::prepare_mesh_draws_for_view`] with [`RenderTarget::dimensions`]).
    pub fn render_frame(
        &mut self,
        gpu: &mut GpuState,
        session: &Session,
        draw_batches: &[SpaceDrawBatch],
        target: RenderTarget,
        pre_collected: Option<&PreCollectedFrameData>,
    ) -> Result<RenderTarget, wgpu::SurfaceError> {
        let (width, height) = target.dimensions();
        gpu.config.width = width;
        gpu.config.height = height;
        if let Some(new_depth) =
            crate::gpu::ensure_depth_texture(&gpu.device, &gpu.config, gpu.depth_size)
        {
            gpu.depth_texture = Some(new_depth);
            gpu.depth_size = (width, height);
        }
        let depth_view = gpu
            .depth_texture
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        let (proj, overlay_projection_override) = match pre_collected {
            Some(pc) => (pc.proj, pc.overlay_projection_override.clone()),
            None => {
                let aspect = width as f32 / height.max(1) as f32;
                let view_params = ViewParams::perspective_from_session(session, aspect);
                let proj = view_params.to_projection_matrix();
                let overlay =
                    ViewParams::overlay_projection_for_frame(session, draw_batches, aspect);
                (proj, overlay)
            }
        };

        let rtao_mrt_graph = session.render_config().rtao_enabled && gpu.ray_tracing_available;
        let rt_shadow_compute = rtao_mrt_graph
            && session.render_config().ray_traced_shadows_use_compute
            && session.render_config().ray_traced_shadows_enabled
            && gpu.ray_tracing_available;
        if !self.rtao_diagnostic_logged {
            logger::info!(
                "RTAO diagnostic: rtao_enabled={} (config={} ray_tracing_available={})",
                rtao_mrt_graph,
                session.render_config().rtao_enabled,
                gpu.ray_tracing_available
            );
            self.rtao_diagnostic_logged = true;
        }

        let fullscreen_filter_hook =
            session.render_config().fullscreen_filter_hook && !rtao_mrt_graph;
        let graph_key = (rtao_mrt_graph, rt_shadow_compute, fullscreen_filter_hook);
        if self.cached_main_graph_key != Some(graph_key) {
            self.graph =
                build_main_render_graph(rtao_mrt_graph, rt_shadow_compute, fullscreen_filter_hook)
                    .expect("main render graph rebuild has no cycles");
            self.cached_main_graph_key = Some(graph_key);
        }

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
            enable_rtao_mrt: rtao_mrt_graph,
            pre_collected: pre_collected.map(|pc| &pc.cached_mesh_draws),
            before_submit: None,
        };

        self.graph.execute(&mut ctx).map_err(|e| match e {
            super::pass::RenderPassError::Surface(s) => s,
            super::pass::RenderPassError::MissingCachedMeshDraws
            | super::pass::RenderPassError::MissingMrtViews => {
                logger::error!("Render pass error: {:?}", e);
                wgpu::SurfaceError::Lost
            }
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
    /// Uses the target's own depth texture. No timestamp queries.
    ///
    /// When `before_submit` is `Some`, extra commands are recorded on the same encoder before the
    /// single queue submit (e.g. texture → readback buffer copy for camera tasks).
    pub fn render_to_target<'a>(
        &'a mut self,
        gpu: &'a mut GpuState,
        session: &'a Session,
        draw_batches: &'a [SpaceDrawBatch],
        target: &'a RenderTarget,
        proj: nalgebra::Matrix4<f32>,
        before_submit: Option<&'a mut dyn FnMut(&mut wgpu::CommandEncoder)>,
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
            enable_rtao_mrt: false,
            pre_collected: None,
            before_submit,
        };
        self.graph.execute(&mut ctx)
    }

    /// Reads back GPU timestamps from the staging buffer. Returns mesh pass duration in ms.
    fn readback_gpu_timestamps(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        staging: &wgpu::Buffer,
    ) -> Option<f64> {
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let poll_result = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        });
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
