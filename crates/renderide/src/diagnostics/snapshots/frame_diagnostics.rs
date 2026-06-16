//! Per-frame diagnostics for the active **Renderide debug** top-level tab.
//!
//! [`FrameDiagnosticsSnapshot`] composes independent fragments -- one per concern -- and only
//! populates the fragments needed by the selected tab.

pub mod gpu_allocator;
pub mod host;
pub mod host_gatherer;
pub mod ipc_health;
pub mod mesh_draw;
pub mod shader_routes;
pub mod xr_health;

pub use gpu_allocator::{
    GpuAllocatorFragment, GpuAllocatorHud, GpuAllocatorHudRefresh, GpuAllocatorReportHud,
};
pub use host::HostCpuMemoryHud;
pub use host_gatherer::HostHudGatherer;
pub use ipc_health::{FrameDiagnosticsIpcQueues, IpcHealthFragment};
pub use mesh_draw::MeshDrawFragment;
pub use shader_routes::ShaderRoutesFragment;
pub use xr_health::XrRecoverableFailureCounts;

use crate::config::DebugHudMainTab;
use crate::diagnostics::{AssetDiagnosticsSnapshot, BackendDiagSnapshot, LightDiagnosticsSnapshot};
/// Inputs for [`FrameDiagnosticsSnapshot::capture`], grouped like
/// [`crate::diagnostics::RendererInfoSnapshotCapture`].
pub struct FrameDiagnosticsSnapshotCapture<'a> {
    /// Active main debug tab this snapshot was captured for.
    pub main_tab: DebugHudMainTab,
    /// Host CPU and memory HUD snapshot.
    pub host: HostCpuMemoryHud,
    /// Host [`crate::shared::FrameSubmitData::render_tasks`] count from the last applied submit.
    pub last_submit_render_task_count: usize,
    /// Camera readback tasks waiting for GPU processing before the next begin-frame send.
    pub pending_camera_readbacks: usize,
    /// Cumulative camera readback tasks successfully written to host shared memory.
    pub completed_camera_readbacks: u64,
    /// Cumulative camera readback tasks failed and zero-filled when possible.
    pub failed_camera_readbacks: u64,
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
    /// Active main debug tab this snapshot was captured for.
    pub main_tab: DebugHudMainTab,
    /// Host CPU model and memory usage.
    pub host: HostCpuMemoryHud,
    /// GPU allocator totals plus throttled full report.
    pub gpu_allocator: GpuAllocatorFragment,
    /// World mesh draw stats, draw-state rows, and resident pool counts.
    pub mesh_draw: MeshDrawFragment,
    /// Render-graph command recording and scheduling diagnostics.
    pub graph: crate::render_graph::CommandEncodingHudSnapshot,
    /// Asset streaming, deferred-work, and worker diagnostics.
    pub assets: AssetDiagnosticsSnapshot,
    /// Light packing and influence-volume culling diagnostics.
    pub lights: LightDiagnosticsSnapshot,
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
            main_tab,
            host,
            last_submit_render_task_count,
            pending_camera_readbacks,
            completed_camera_readbacks,
            failed_camera_readbacks,
            backend,
            ipc,
            xr,
            allocator,
            frame_submit_apply_failures,
            unhandled_ipc_command_event_total,
        } = capture;

        let mut snapshot = Self {
            main_tab,
            ..Self::default()
        };

        match main_tab {
            DebugHudMainTab::Stats => {
                snapshot.host = host;
                snapshot.gpu_allocator = GpuAllocatorFragment::capture(allocator);
                snapshot.mesh_draw = MeshDrawFragment::capture(
                    backend,
                    last_submit_render_task_count,
                    pending_camera_readbacks,
                    completed_camera_readbacks,
                    failed_camera_readbacks,
                );
                snapshot.ipc_health = IpcHealthFragment::capture(
                    ipc,
                    frame_submit_apply_failures,
                    unhandled_ipc_command_event_total,
                );
                snapshot.xr_health = xr;
                snapshot.lights = backend.lights;
                snapshot.graph = backend.command_encoding.clone();
                snapshot.assets = backend.assets.clone();
            }
            DebugHudMainTab::ShaderRoutes => {
                snapshot.shader_routes = ShaderRoutesFragment::capture(backend);
            }
            DebugHudMainTab::DrawState => {
                snapshot.mesh_draw = MeshDrawFragment::capture_draw_state_rows(backend);
            }
            DebugHudMainTab::GpuMemory => {
                snapshot.gpu_allocator = GpuAllocatorFragment::capture(allocator);
            }
            DebugHudMainTab::GpuPasses => {}
        }

        snapshot
    }
}
