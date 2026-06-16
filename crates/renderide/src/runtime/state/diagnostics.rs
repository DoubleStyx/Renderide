//! Runtime-owned diagnostics accumulation state.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::DebugHudMainTab;
use crate::diagnostics::{GpuAllocatorHud, GpuAllocatorReportHud, HostHudGatherer};
use crate::gpu::GpuContext;

/// How often [`wgpu::Device::generate_allocator_report`] replaces the GPU memory tab payload.
const GPU_ALLOCATOR_FULL_REPORT_INTERVAL: Duration = Duration::from_secs(2);
/// Heavy debug HUD tables are informational; updating them every frame can dominate epilogue time.
const HUD_HEAVY_SNAPSHOT_INTERVAL: Duration = Duration::from_millis(250);

/// Diagnostics state that belongs to runtime orchestration rather than the backend HUD widget.
pub(in crate::runtime) struct RuntimeDiagnosticsState {
    /// Throttled host CPU/RAM sampling for the debug HUD.
    pub(in crate::runtime) host_hud: HostHudGatherer,
    /// Rolling per-frame wall time history that feeds the frame timing sparkline.
    pub(in crate::runtime) frame_time_history: crate::diagnostics::FrameTimeHistory,
    /// Persistent EMA state for frame timing scalar readouts.
    pub(in crate::runtime) frame_timing_ema: crate::diagnostics::FrameTimingEma,
    /// `FrameSubmitData::render_tasks` length from the last applied frame submit.
    pub(in crate::runtime) last_submit_render_task_count: usize,
    /// Camera readback tasks currently waiting to be drained before the next begin-frame send.
    pub(in crate::runtime) pending_camera_readbacks: usize,
    /// Cumulative camera readback tasks successfully written to host shared memory.
    pub(in crate::runtime) completed_camera_readbacks: u64,
    /// Cumulative camera readback tasks failed and zero-filled when possible.
    pub(in crate::runtime) failed_camera_readbacks: u64,
    /// Cached full allocator report for the GPU memory HUD tab.
    pub(in crate::runtime) allocator_report_hud: Option<GpuAllocatorReportHud>,
    /// Cached allocator totals from the same throttled report.
    pub(in crate::runtime) allocator_report_totals: GpuAllocatorHud,
    /// Wall clock when a GPU memory tab refresh was last attempted.
    pub(in crate::runtime) allocator_report_last_refresh: Option<Instant>,
    /// Wall clock when the main debug HUD table snapshots were last refreshed.
    main_hud_snapshot_last_refresh: Option<Instant>,
    /// Main debug tab whose snapshot timer is currently active.
    main_hud_snapshot_tab: Option<DebugHudMainTab>,
    /// Capture signature for the active main debug HUD snapshot timer.
    main_hud_snapshot_signature: Option<u32>,
    /// Wall clock when the scene-transform HUD snapshot was last refreshed.
    scene_transforms_snapshot_last_refresh: Option<Instant>,
    /// Wall clock when the texture-debug HUD snapshot was last refreshed.
    texture_debug_snapshot_last_refresh: Option<Instant>,
    /// Count of failed frame-submit apply or cache-flush operations after host submits.
    pub(in crate::runtime) frame_submit_apply_failures: u64,
    /// Whether the first successful frame submit has been logged.
    pub(in crate::runtime) logged_first_frame_submit: bool,
    /// Last logged render-space count after scene apply.
    pub(in crate::runtime) last_scene_render_space_count: usize,
    /// Last logged mesh-renderable count after scene apply.
    pub(in crate::runtime) last_scene_mesh_renderable_count: usize,
}

impl RuntimeDiagnosticsState {
    /// Creates empty runtime diagnostics state.
    pub(in crate::runtime) fn new() -> Self {
        Self {
            host_hud: HostHudGatherer::default(),
            frame_time_history: crate::diagnostics::FrameTimeHistory::new(),
            frame_timing_ema: crate::diagnostics::FrameTimingEma::default(),
            last_submit_render_task_count: 0,
            pending_camera_readbacks: 0,
            completed_camera_readbacks: 0,
            failed_camera_readbacks: 0,
            allocator_report_hud: None,
            allocator_report_totals: GpuAllocatorHud::default(),
            allocator_report_last_refresh: None,
            main_hud_snapshot_last_refresh: None,
            main_hud_snapshot_tab: None,
            main_hud_snapshot_signature: None,
            scene_transforms_snapshot_last_refresh: None,
            texture_debug_snapshot_last_refresh: None,
            frame_submit_apply_failures: 0,
            logged_first_frame_submit: false,
            last_scene_render_space_count: 0,
            last_scene_mesh_renderable_count: 0,
        }
    }

    pub(in crate::runtime) fn should_refresh_main_hud_snapshot(
        &mut self,
        now: Instant,
        tab: DebugHudMainTab,
        signature: u32,
    ) -> bool {
        if self.main_hud_snapshot_tab != Some(tab)
            || self.main_hud_snapshot_signature != Some(signature)
        {
            self.main_hud_snapshot_tab = Some(tab);
            self.main_hud_snapshot_signature = Some(signature);
            self.main_hud_snapshot_last_refresh = None;
        }
        should_refresh_snapshot(&mut self.main_hud_snapshot_last_refresh, now)
    }

    pub(in crate::runtime) fn should_refresh_scene_transforms_snapshot(
        &mut self,
        now: Instant,
    ) -> bool {
        should_refresh_snapshot(&mut self.scene_transforms_snapshot_last_refresh, now)
    }

    pub(in crate::runtime) fn should_refresh_texture_debug_snapshot(
        &mut self,
        now: Instant,
    ) -> bool {
        should_refresh_snapshot(&mut self.texture_debug_snapshot_last_refresh, now)
    }

    pub(in crate::runtime) fn clear_main_hud_snapshot_timer(&mut self) {
        self.main_hud_snapshot_last_refresh = None;
        self.main_hud_snapshot_tab = None;
        self.main_hud_snapshot_signature = None;
    }

    pub(in crate::runtime) fn clear_scene_transforms_snapshot_timer(&mut self) {
        self.scene_transforms_snapshot_last_refresh = None;
    }

    pub(in crate::runtime) fn clear_texture_debug_snapshot_timer(&mut self) {
        self.texture_debug_snapshot_last_refresh = None;
    }

    /// Updates the latest render-task count for the HUD.
    pub(in crate::runtime) fn set_last_submit_render_task_count(&mut self, n: usize) {
        self.last_submit_render_task_count = n;
    }

    /// Replaces the current pending camera readback count.
    pub(in crate::runtime) fn set_pending_camera_readbacks(&mut self, n: usize) {
        self.pending_camera_readbacks = n;
    }

    /// Adds completed and failed camera readback counts to the cumulative HUD counters.
    pub(in crate::runtime) fn note_camera_readback_results(&mut self, completed: u64, failed: u64) {
        self.completed_camera_readbacks = self.completed_camera_readbacks.saturating_add(completed);
        self.failed_camera_readbacks = self.failed_camera_readbacks.saturating_add(failed);
    }

    /// Increments the cumulative scene-apply failure counter.
    pub(in crate::runtime) fn note_frame_submit_apply_failure(&mut self) {
        self.frame_submit_apply_failures = self.frame_submit_apply_failures.saturating_add(1);
    }

    /// Refreshes GPU allocator totals when the interval elapses.
    ///
    /// The full sorted report is only retained when the main debug HUD needs the GPU memory tab;
    /// the frame-timing overlay uses the cheaper totals.
    pub(in crate::runtime) fn refresh_gpu_allocator_report_hud(
        &mut self,
        gpu: &GpuContext,
        now: Instant,
        retain_full_report: bool,
    ) {
        let needs_full_report = retain_full_report && self.allocator_report_hud.is_none();
        let should_refresh = needs_full_report
            || self
                .allocator_report_last_refresh
                .is_none_or(|t| now.duration_since(t) >= GPU_ALLOCATOR_FULL_REPORT_INTERVAL);
        if !should_refresh {
            return;
        }
        self.allocator_report_last_refresh = Some(now);
        if let Some(rep) = gpu.device().generate_allocator_report() {
            self.allocator_report_totals = GpuAllocatorHud {
                allocated_bytes: Some(rep.total_allocated_bytes),
                reserved_bytes: Some(rep.total_reserved_bytes),
            };
            if retain_full_report {
                let mut order: Vec<usize> = (0..rep.allocations.len()).collect();
                order.sort_by_key(|&i| std::cmp::Reverse(rep.allocations[i].size));
                self.allocator_report_hud = Some(GpuAllocatorReportHud {
                    report: Arc::new(rep),
                    allocation_indices_by_size: order.into(),
                });
            } else {
                self.allocator_report_hud = None;
            }
        }
    }

    /// Seconds until the next full allocator refresh should be attempted.
    pub(in crate::runtime) fn allocator_report_next_refresh_in_secs(&self, now: Instant) -> f32 {
        self.allocator_report_last_refresh.map_or(
            GPU_ALLOCATOR_FULL_REPORT_INTERVAL.as_secs_f32(),
            |t| {
                let elapsed = now.saturating_duration_since(t);
                GPU_ALLOCATOR_FULL_REPORT_INTERVAL
                    .saturating_sub(elapsed)
                    .as_secs_f32()
            },
        )
    }

    /// Clears main-HUD allocator report state when the main HUD is disabled.
    pub(in crate::runtime) fn clear_allocator_report(&mut self) {
        self.allocator_report_hud = None;
        self.allocator_report_totals = GpuAllocatorHud::default();
        self.allocator_report_last_refresh = None;
    }

    /// Clears the full GPU allocation table while keeping totals for the frame-timing overlay.
    pub(in crate::runtime) fn clear_allocator_report_detail(&mut self) {
        self.allocator_report_hud = None;
    }
}

fn should_refresh_snapshot(last_refresh: &mut Option<Instant>, now: Instant) -> bool {
    let should_refresh =
        last_refresh.is_none_or(|t| now.duration_since(t) >= HUD_HEAVY_SNAPSHOT_INTERVAL);
    if should_refresh {
        *last_refresh = Some(now);
    }
    should_refresh
}
