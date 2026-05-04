//! Dear ImGui diagnostics: **Frame timing** ([`crate::config::DebugSettings::debug_hud_frame_timing`]),
//! **Renderide debug** ([`crate::config::DebugSettings::debug_hud_enabled`]: Stats / Shader routes / Draw state / GPU memory),
//! **Scene transforms** ([`crate::config::DebugSettings::debug_hud_transforms`]),
//! and **Textures** ([`crate::config::DebugSettings::debug_hud_textures`]).
//!
//! Also hosts the cooperative renderer hang/hitch detector ([`Watchdog`]).

mod ema;
mod encode_error;
mod frame_history;
mod host_metrics;
mod hud;
mod input;
pub mod per_view;
mod snapshots;
mod watchdog;

pub use ema::{EMA_HISTORY_LEN, EmaScalar, FrameTimingEma};
pub use encode_error::DebugHudEncodeError;
pub use frame_history::{FRAME_TIME_HISTORY_LEN, FrameTimeHistory};
pub use host_metrics::HostHudGatherer;
pub use hud::DebugHud;
pub use input::{DebugHudInput, sanitize_input_state_for_imgui_host};
pub use per_view::{PerViewHudConfig, PerViewHudOutputs, PerViewHudOutputsSlot};
pub use snapshots::{
    BackendDiagSnapshot, FrameDiagnosticsIpcQueues, FrameDiagnosticsSnapshot,
    FrameDiagnosticsSnapshotCapture, FrameTimingHudSnapshot, GpuAllocatorHud,
    GpuAllocatorHudRefresh, GpuAllocatorReportHud, RenderSpaceTransformsSnapshot,
    RendererInfoSnapshot, RendererInfoSnapshotCapture, SceneTransformsSnapshot,
    ShaderRouteSnapshot, TextureDebugRow, TextureDebugSnapshot, TransformRow, WorldTransformSample,
    XrRecoverableFailureCounts,
};
pub use watchdog::{Heartbeat, Watchdog, WatchdogPause};
