//! Dear ImGui diagnostics: **Frame timing** ([`crate::config::DebugSettings::debug_hud_frame_timing`]),
//! **Feedback / Bug Report** ([`crate::config::DebugSettings::debug_hud_links`]),
//! **Renderide debug** ([`crate::config::DebugSettings::debug_hud_enabled`]: Stats / Visibility / Graph / Assets / Shader routes / Draw state / GPU memory / GPU passes),
//! **Scene transforms** ([`crate::config::DebugSettings::debug_hud_transforms`]),
//! and **Textures** ([`crate::config::DebugSettings::debug_hud_textures`]).
//!
//! Also hosts the cooperative renderer hang/hitch detector ([`Watchdog`]).

mod hud;
pub(crate) mod log_once;
mod per_view;
mod snapshots;
mod watchdog;

pub(crate) use hud::DebugHudOverlayContext;
pub use hud::{
    DebugHud, DebugHudInput, DebugHudMetricInterest, sanitize_input_state_for_imgui_host,
};
pub use snapshots::{
    AssetDiagnosticsSnapshot, BackendDiagSnapshot, FrameDiagnosticsIpcQueues,
    FrameDiagnosticsSnapshot, FrameDiagnosticsSnapshotCapture, FrameTimeHistory, FrameTimingEma,
    FrameTimingHudCapture, FrameTimingHudSnapshot, FrameTimingOnePercentStats,
    FrameUploadArenaSnapshot, GpuAllocatorHud, GpuAllocatorHudRefresh, GpuAllocatorReportHud,
    HostHudGatherer, LightDiagnosticsSnapshot, RendererInfoSnapshot, RendererInfoSnapshotCapture,
    SceneTransformsSnapshot, ShaderRouteSnapshot, TextureDebugSnapshot, XrRecoverableFailureCounts,
};
pub use watchdog::{Heartbeat, Watchdog, WatchdogPause};
