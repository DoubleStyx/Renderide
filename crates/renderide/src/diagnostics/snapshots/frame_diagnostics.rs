//! Per-frame diagnostics for the **Frame** debug HUD tab (CPU/GPU timing, allocator, draws)
//! and the **GPU memory** tab (throttled full [`wgpu::AllocatorReport`]).
//!
//! [`FrameDiagnosticsSnapshot`] composes seven independent fragments -- one per concern -- so each
//! HUD section can borrow exactly the data it consumes without threading the whole snapshot
//! through the call tree.

pub mod gpu_allocator;
pub mod host;
pub mod ipc_health;
pub mod mesh_draw;
pub mod shader_routes;
pub mod timing;
pub mod xr_health;

pub use gpu_allocator::{
    GpuAllocatorFragment, GpuAllocatorHud, GpuAllocatorHudRefresh, GpuAllocatorReportHud,
};
pub use host::HostCpuMemoryHud;
pub use ipc_health::{FrameDiagnosticsIpcQueues, IpcHealthFragment};
pub use mesh_draw::MeshDrawFragment;
pub use shader_routes::ShaderRoutesFragment;
pub use timing::FrameTimingFragment;
pub use xr_health::XrRecoverableFailureCounts;

use crate::diagnostics::BackendDiagSnapshot;
use crate::gpu::GpuContext;

/// Inputs for [`FrameDiagnosticsSnapshot::capture`], grouped like
/// [`crate::diagnostics::RendererInfoSnapshotCapture`].
pub struct FrameDiagnosticsSnapshotCapture<'a> {
    /// GPU timing state for this tick.
    pub gpu: &'a GpuContext,
    /// Wall-clock redraw interval (ms).
    pub wall_frame_time_ms: f64,
    /// Host CPU and memory HUD snapshot.
    pub host: HostCpuMemoryHud,
    /// Host [`crate::shared::FrameSubmitData::render_tasks`] count from the last applied submit.
    pub last_submit_render_task_count: usize,
    /// Plain-data backend snapshot capturing pools, draw stats, shader routes, and graph counts.
    pub backend: &'a BackendDiagSnapshot,
    /// Outbound IPC queue drops and streaks.
    pub ipc: FrameDiagnosticsIpcQueues,
    /// OpenXR recoverable failure counters.
    pub xr: XrRecoverableFailureCounts,
    /// Full allocator report refresh state.
    pub allocator: GpuAllocatorHudRefresh,
    /// Cumulative failed scene applies after host frame submit.
    pub frame_submit_apply_failures: u64,
    /// Cumulative unhandled renderer command observations.
    pub unhandled_ipc_command_event_total: u64,
}

/// Snapshot assembled after the winit frame tick ends (draw stats, timings, host metrics).
///
/// Each public field is a focused fragment whose `capture` orchestrates one concern. The HUD
/// layer borrows fragments individually so per-tab code never sees data it does not consume.
#[derive(Clone, Debug, Default)]
pub struct FrameDiagnosticsSnapshot {
    /// Wall-clock interval and CPU/GPU per-frame ms.
    pub timing: FrameTimingFragment,
    /// Host CPU model and memory usage.
    pub host: HostCpuMemoryHud,
    /// GPU allocator totals plus throttled full report.
    pub gpu_allocator: GpuAllocatorFragment,
    /// World mesh draw stats, draw-state rows, and resident pool counts.
    pub mesh_draw: MeshDrawFragment,
    /// Sorted host-shader -> pipeline routing rows.
    pub shader_routes: ShaderRoutesFragment,
    /// IPC outbound queue health plus host-command failure counters.
    pub ipc_health: IpcHealthFragment,
    /// Recoverable OpenXR error counts.
    pub xr_health: XrRecoverableFailureCounts,
}

impl FrameDiagnosticsSnapshot {
    /// Builds the snapshot after [`crate::gpu::GpuContext::end_frame_timing`] for the tick by
    /// composing each fragment's own capture.
    pub fn capture(capture: FrameDiagnosticsSnapshotCapture<'_>) -> Self {
        profiling::scope!("hud::build_diagnostics_snapshot");
        let FrameDiagnosticsSnapshotCapture {
            gpu,
            wall_frame_time_ms,
            host,
            last_submit_render_task_count,
            backend,
            ipc,
            xr,
            allocator,
            frame_submit_apply_failures,
            unhandled_ipc_command_event_total,
        } = capture;
        Self {
            timing: FrameTimingFragment::capture(gpu, wall_frame_time_ms),
            host,
            gpu_allocator: GpuAllocatorFragment::capture(allocator),
            mesh_draw: MeshDrawFragment::capture(backend, last_submit_render_task_count),
            shader_routes: ShaderRoutesFragment::capture(backend),
            ipc_health: IpcHealthFragment::capture(
                ipc,
                frame_submit_apply_failures,
                unhandled_ipc_command_event_total,
            ),
            xr_health: xr,
        }
    }

    /// FPS computed from the wall-clock interval between consecutive redraw events.
    pub fn fps_from_wall(&self) -> f64 {
        self.timing.fps_from_wall()
    }
}

#[cfg(test)]
mod tests {
    use super::FrameDiagnosticsSnapshot;
    use super::timing::FrameTimingFragment;

    #[test]
    fn fps_from_wall_delegates_to_timing_fragment() {
        let s = FrameDiagnosticsSnapshot {
            timing: FrameTimingFragment {
                wall_frame_time_ms: 16.0,
                cpu_frame_ms: Some(2.0),
                gpu_frame_ms: Some(1.0),
            },
            ..Default::default()
        };
        assert!((s.fps_from_wall() - 62.5).abs() < 0.01);
    }

    #[test]
    fn fps_from_wall_zero_interval() {
        let s = FrameDiagnosticsSnapshot::default();
        assert_eq!(s.fps_from_wall(), 0.0);
    }
}
