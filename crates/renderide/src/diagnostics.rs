//! Dear ImGui diagnostics: **Frame timing** ([`crate::config::DebugSettings::debug_hud_frame_timing`]),
//! **Renderide debug** ([`crate::config::DebugSettings::debug_hud_enabled`]: Stats / Shader routes / Draw state / GPU memory),
//! **Scene transforms** ([`crate::config::DebugSettings::debug_hud_transforms`]),
//! and **Textures** ([`crate::config::DebugSettings::debug_hud_textures`]).
//!
//! Also hosts the cooperative renderer hang/hitch detector ([`Watchdog`]).

pub(crate) mod crash_context;
pub(crate) mod gpu_flight_recorder;
mod hud;
pub(crate) mod log_once;
pub(crate) mod log_throttle;
pub(crate) mod per_view;
mod snapshots;
mod watchdog;

pub(crate) use hud::DebugHudOverlayContext;
pub use hud::{DebugHud, DebugHudEncodeError, DebugHudInput, sanitize_input_state_for_imgui_host};
pub use per_view::{PerViewHudConfig, PerViewHudOutputs, PerViewHudOutputsSlot};
pub use snapshots::{
    BackendDiagSnapshot, FrameDiagnosticsIpcQueues, FrameDiagnosticsSnapshot,
    FrameDiagnosticsSnapshotCapture, FrameTimeHistory, FrameTimingEma, FrameTimingHudCapture,
    FrameTimingHudSnapshot, FrameTimingOnePercentStats, FrameUploadArenaSnapshot, GpuAllocatorHud,
    GpuAllocatorHudRefresh, GpuAllocatorReportHud, HostHudGatherer, RendererInfoSnapshot,
    RendererInfoSnapshotCapture, SceneTransformsSnapshot, ShaderRouteSnapshot,
    TextureDebugSnapshot, XrRecoverableFailureCounts,
};
pub use watchdog::{Heartbeat, Watchdog, WatchdogPause};
