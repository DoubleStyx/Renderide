//! Per-tick wiring from [`super::RendererRuntime`] to the backend [`crate::backend::RenderBackend`] debug HUD.

use std::time::Instant;

use crate::config::{DebugHudMainTab, RendererSettings};
use crate::diagnostics::{DebugHudEncodeError, DebugHudMetricInterest};
use crate::gpu::GpuContext;

use super::RendererRuntime;

fn debug_hud_metric_interest(settings: &RendererSettings) -> DebugHudMetricInterest {
    DebugHudMetricInterest::from_settings(settings)
}

impl RendererRuntime {
    /// Fills the active main-tab HUD snapshot when the main HUD has an active tab.
    fn capture_main_debug_hud_tab(
        &mut self,
        gpu: &GpuContext,
        now: Instant,
        interest: DebugHudMetricInterest,
        tab: DebugHudMainTab,
    ) {
        let backend_diag = match tab {
            DebugHudMainTab::Stats | DebugHudMainTab::ShaderRoutes | DebugHudMainTab::DrawState => {
                self.backend.snapshot_for_diagnostics(interest)
            }
            DebugHudMainTab::GpuMemory | DebugHudMainTab::GpuPasses => {
                crate::diagnostics::BackendDiagSnapshot::default()
            }
        };
        let host = if tab == DebugHudMainTab::Stats {
            self.diagnostics.host_hud.snapshot()
        } else {
            Default::default()
        };
        let next_refresh_in_secs = self.diagnostics.allocator_report_next_refresh_in_secs(now);
        let (ipc_pri_str, ipc_bg_str) = if tab == DebugHudMainTab::Stats {
            self.frontend.ipc_consecutive_outbound_drop_streaks()
        } else {
            (0, 0)
        };
        let frame_diag = crate::diagnostics::FrameDiagnosticsSnapshot::capture(
            crate::diagnostics::FrameDiagnosticsSnapshotCapture {
                main_tab: tab,
                host,
                last_submit_render_task_count: self.diagnostics.last_submit_render_task_count,
                pending_camera_readbacks: self.diagnostics.pending_camera_readbacks,
                completed_camera_readbacks: self.diagnostics.completed_camera_readbacks,
                failed_camera_readbacks: self.diagnostics.failed_camera_readbacks,
                backend: &backend_diag,
                ipc: crate::diagnostics::FrameDiagnosticsIpcQueues {
                    ipc_primary_outbound_drop_this_tick: self
                        .frontend
                        .ipc_outbound_primary_drop_this_tick(),
                    ipc_background_outbound_drop_this_tick: self
                        .frontend
                        .ipc_outbound_background_drop_this_tick(),
                    ipc_primary_consecutive_fail_streak: ipc_pri_str,
                    ipc_background_consecutive_fail_streak: ipc_bg_str,
                },
                xr: crate::diagnostics::XrRecoverableFailureCounts {
                    xr_wait_frame_failures: self.xr_stats.wait_frame_failures,
                    xr_locate_views_failures: self.xr_stats.locate_views_failures,
                },
                allocator: crate::diagnostics::GpuAllocatorHudRefresh {
                    gpu_allocator_totals: self.diagnostics.allocator_report_totals,
                    gpu_allocator_report: self.diagnostics.allocator_report_hud.clone(),
                    gpu_allocator_report_next_refresh_in_secs: next_refresh_in_secs,
                },
                frame_submit_apply_failures: self.diagnostics.frame_submit_apply_failures,
                unhandled_ipc_command_event_total: self.unhandled_ipc_command_event_total(),
            },
        );
        if tab == DebugHudMainTab::Stats {
            let msaa_requested = self
                .config
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
                    scene: &self.scene,
                    backend: &backend_diag,
                    gpu,
                    msaa_requested_samples: msaa_requested,
                },
            );
            self.backend.set_debug_hud_snapshot(snapshot);
        }
        self.backend.set_debug_hud_frame_diagnostics(frame_diag);
    }

    /// Copies debug HUD capture flags into the backend before the render graph runs.
    pub(super) fn sync_debug_hud_diagnostics_from_settings(&mut self) {
        let flags = self
            .config
            .settings
            .read()
            .map(|s| debug_hud_metric_interest(&s))
            .unwrap_or_default();
        self.backend
            .set_debug_hud_per_view_config(crate::diagnostics::PerViewHudConfig::from(flags));
        self.backend
            .clear_debug_hud_current_view_texture_2d_asset_ids();
    }

    /// Updates debug HUD snapshots after [`crate::gpu::GpuContext::end_frame_timing`] for the winit tick.
    pub fn capture_debug_hud_after_frame_end(&mut self, gpu: &GpuContext) {
        profiling::scope!("hud::capture_snapshot");
        let wall_ms = self.backend.debug_frame_time_ms();
        let flags = self
            .config
            .settings
            .read()
            .map(|s| debug_hud_metric_interest(&s))
            .unwrap_or_default();
        let now = Instant::now();
        if flags.wants_allocator_totals() {
            self.diagnostics
                .refresh_gpu_allocator_report_hud(gpu, now, flags.wants_gpu_memory());
            if !flags.wants_gpu_memory() {
                self.diagnostics.clear_allocator_report_detail();
            }
        } else {
            self.diagnostics.clear_allocator_report();
        }

        if flags.frame_timing {
            let host = self.diagnostics.host_hud.snapshot();
            let frame_timing = crate::diagnostics::FrameTimingHudSnapshot::capture(
                crate::diagnostics::FrameTimingHudCapture {
                    gpu,
                    wall_frame_time_ms: wall_ms,
                    host_frame_begin_to_submit: self.last_frame_begin_to_submit(),
                    host_hud: &host,
                    gpu_allocator: self.diagnostics.allocator_report_totals,
                    history: &mut self.diagnostics.frame_time_history,
                    ema: &mut self.diagnostics.frame_timing_ema,
                    now,
                },
            );
            self.backend.set_debug_hud_frame_timing(frame_timing);
        } else {
            self.backend.clear_debug_hud_frame_timing();
        }

        if flags.wants_gpu_passes() {
            self.backend.clear_debug_hud_stats_snapshots();
            self.diagnostics.clear_main_hud_snapshot_timer();
            let gpu_profiler_snapshot = gpu
                .latest_gpu_profiler_snapshot_handle()
                .lock()
                .map(|guard| guard.clone())
                .unwrap_or_default();
            self.backend
                .set_debug_hud_gpu_profiler_snapshot(gpu_profiler_snapshot);
        } else if flags.wants_main_debug() {
            self.backend.clear_debug_hud_gpu_profiler_snapshot();
            if let Some(tab) = flags.main_tab
                && self.diagnostics.should_refresh_main_hud_snapshot(now, tab)
            {
                self.capture_main_debug_hud_tab(gpu, now, flags, tab);
            }
        } else {
            self.backend.clear_debug_hud_stats_snapshots();
            self.diagnostics.clear_main_hud_snapshot_timer();
            if flags.frame_timing {
                self.diagnostics.clear_allocator_report_detail();
            } else {
                self.diagnostics.clear_allocator_report();
            }
        }

        if flags.scene_transforms
            && self
                .diagnostics
                .should_refresh_scene_transforms_snapshot(now)
        {
            profiling::scope!("hud::capture_scene_transforms_snapshot");
            let scene_transforms =
                crate::diagnostics::SceneTransformsSnapshot::capture(&self.scene);
            self.backend
                .set_debug_hud_scene_transforms_snapshot(scene_transforms);
        } else if !flags.scene_transforms {
            self.backend.clear_debug_hud_scene_transforms_snapshot();
            self.diagnostics.clear_scene_transforms_snapshot_timer();
        }

        if flags.textures && self.diagnostics.should_refresh_texture_debug_snapshot(now) {
            profiling::scope!("hud::capture_texture_debug_snapshot");
            let textures = crate::diagnostics::TextureDebugSnapshot::capture(
                self.backend.texture_pool(),
                self.backend.debug_hud_current_view_texture_2d_asset_ids(),
            );
            self.backend.set_debug_hud_texture_debug_snapshot(textures);
        } else if !flags.textures {
            self.backend.clear_debug_hud_texture_debug_snapshot();
            self.diagnostics.clear_texture_debug_snapshot_timer();
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
        if !self.backend.debug_hud_has_visible_content() {
            self.backend.clear_debug_hud_input_capture();
            return Ok(());
        }
        let device = gpu.device().as_ref();
        let extent = gpu.surface_extent_px();
        let q = gpu.queue().as_ref();
        self.backend.encode_debug_hud_overlay(
            device,
            q,
            encoder,
            backbuffer,
            extent,
            gpu.gpu_profiler(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::debug_hud_metric_interest;
    use crate::config::{DebugHudMainTab, RendererSettings};
    use crate::diagnostics::DebugHudMetricInterest;

    #[test]
    fn metric_interest_requires_visible_imgui() {
        let mut settings = RendererSettings::default();
        settings.debug.debug_hud_enabled = true;
        settings.debug.debug_hud_transforms = true;
        settings.debug.debug_hud_textures = true;
        settings.debug.hud.imgui_visible = false;

        assert_eq!(
            debug_hud_metric_interest(&settings),
            DebugHudMetricInterest::default()
        );
    }

    #[test]
    fn scene_transform_capture_ignores_retained_open_flag() {
        let mut settings = RendererSettings::default();
        settings.debug.debug_hud_transforms = true;
        settings.debug.hud.scene_transforms_open = false;

        assert!(debug_hud_metric_interest(&settings).scene_transforms);
    }

    #[test]
    fn texture_capture_ignores_retained_open_flag() {
        let mut settings = RendererSettings::default();
        settings.debug.debug_hud_textures = true;
        settings.debug.hud.texture_debug_open = false;

        assert!(debug_hud_metric_interest(&settings).textures);
    }

    #[test]
    fn main_tab_interest_follows_selected_open_tab_only() {
        let mut settings = RendererSettings::default();
        settings.debug.debug_hud_enabled = true;
        settings.debug.hud.main_tab = DebugHudMainTab::GpuPasses;

        let interest = debug_hud_metric_interest(&settings);

        assert_eq!(interest.main_tab, Some(DebugHudMainTab::GpuPasses));
        assert!(interest.wants_gpu_passes());
        assert!(!interest.wants_gpu_memory());
    }
}
