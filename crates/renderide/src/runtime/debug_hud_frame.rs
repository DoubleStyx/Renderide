//! Per-tick wiring from [`super::RendererRuntime`] to the backend [`crate::backend::RenderBackend`] debug HUD.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::diagnostics::DebugHudEncodeError;
use crate::diagnostics::GpuAllocatorReportHud;
use crate::gpu::GpuContext;

use super::RendererRuntime;

/// How often [`wgpu::Device::generate_allocator_report`] replaces the **GPU memory** tab payload.
const GPU_ALLOCATOR_FULL_REPORT_INTERVAL: Duration = Duration::from_secs(2);

impl RendererRuntime {
    /// Copies [`crate::config::DebugSettings::debug_hud_enabled`] into the backend before the render graph runs.
    pub(super) fn sync_debug_hud_diagnostics_from_settings(&mut self) {
        let main = self
            .settings
            .read()
            .map(|s| s.debug.debug_hud_enabled)
            .unwrap_or(false);
        self.backend.set_debug_hud_main_enabled(main);
    }

    /// Updates debug HUD snapshots after [`crate::gpu::GpuContext::end_frame_timing`] for the winit tick.
    pub fn capture_debug_hud_after_frame_end(&mut self, gpu: &GpuContext) {
        let frame_timing = crate::diagnostics::FrameTimingHudSnapshot::capture(
            gpu,
            self.backend.debug_frame_time_ms(),
        );
        self.backend.set_debug_hud_frame_timing(frame_timing);

        let (main_hud, transforms_hud) = self
            .settings
            .read()
            .map(|s| (s.debug.debug_hud_enabled, s.debug.debug_hud_transforms))
            .unwrap_or((false, false));

        if main_hud {
            let host = self.host_hud.snapshot();
            let now = Instant::now();
            let should_refresh_allocator_report = self
                .allocator_report_last_refresh
                .map(|t| now.duration_since(t) >= GPU_ALLOCATOR_FULL_REPORT_INTERVAL)
                .unwrap_or(true);
            if should_refresh_allocator_report {
                self.allocator_report_last_refresh = Some(now);
                if let Some(rep) = gpu.device().generate_allocator_report() {
                    let mut order: Vec<usize> = (0..rep.allocations.len()).collect();
                    order.sort_by_key(|&i| std::cmp::Reverse(rep.allocations[i].size));
                    self.allocator_report_hud = Some(GpuAllocatorReportHud {
                        report: Arc::new(rep),
                        allocation_indices_by_size: order.into(),
                    });
                }
            }
            let next_refresh_in_secs = self
                .allocator_report_last_refresh
                .map(|t| {
                    let elapsed = now.saturating_duration_since(t);
                    GPU_ALLOCATOR_FULL_REPORT_INTERVAL
                        .saturating_sub(elapsed)
                        .as_secs_f32()
                })
                .unwrap_or(GPU_ALLOCATOR_FULL_REPORT_INTERVAL.as_secs_f32());
            let frame_diag = crate::diagnostics::FrameDiagnosticsSnapshot::capture(
                gpu,
                self.backend.debug_frame_time_ms(),
                host,
                self.last_submit_render_task_count,
                &self.backend,
                self.allocator_report_hud.clone(),
                next_refresh_in_secs,
            );
            let msaa_requested = self
                .settings
                .read()
                .map(|s| s.rendering.msaa.as_count())
                .unwrap_or(1);
            let snapshot = crate::diagnostics::RendererInfoSnapshot::capture(
                crate::diagnostics::RendererInfoSnapshotCapture {
                    ipc_connected: self.is_ipc_connected(),
                    init_state: self.init_state(),
                    last_frame_index: self.last_frame_index(),
                    adapter_info: gpu.adapter_info(),
                    gpu_limits: gpu.limits().as_ref(),
                    surface_format: gpu.config_format(),
                    viewport_px: gpu.surface_extent_px(),
                    present_mode: gpu.present_mode(),
                    frame_time_ms: self.backend.debug_frame_time_ms(),
                    scene: &self.scene,
                    backend: &self.backend,
                    gpu,
                    msaa_requested_samples: msaa_requested,
                },
            );
            self.backend.set_debug_hud_snapshot(snapshot);
            self.backend.set_debug_hud_frame_diagnostics(frame_diag);
        } else {
            self.backend.clear_debug_hud_stats_snapshots();
            self.allocator_report_hud = None;
            self.allocator_report_last_refresh = None;
        }

        if transforms_hud {
            let scene_transforms =
                crate::diagnostics::SceneTransformsSnapshot::capture(&self.scene);
            self.backend
                .set_debug_hud_scene_transforms_snapshot(scene_transforms);
        } else {
            self.backend.clear_debug_hud_scene_transforms_snapshot();
        }
    }

    /// Encodes the Dear ImGui debug overlay onto an acquired swapchain view (e.g. after the VR mirror blit).
    ///
    /// Uses the same composite path as the desktop render graph (`LoadOp::Load`). Caller must keep
    /// [`Self::set_debug_hud_frame_data`] in sync for this tick before encoding.
    pub(crate) fn encode_debug_hud_overlay_on_surface(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
    ) -> Result<(), DebugHudEncodeError> {
        let device = gpu.device().as_ref();
        let extent = gpu.surface_extent_px();
        let q = gpu
            .queue()
            .lock()
            .map_err(|_| DebugHudEncodeError::QueueMutexPoisoned)?;
        self.backend
            .encode_debug_hud_overlay(device, &q, encoder, backbuffer, extent)
    }
}
