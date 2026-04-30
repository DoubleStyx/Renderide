//! Dear ImGui diagnostics: **Frame timing** ([`crate::config::DebugSettings::debug_hud_frame_timing`]),
//! **Renderide debug** ([`crate::config::DebugSettings::debug_hud_enabled`]: Stats / Shader routes / Draw state / GPU memory),
//! **Scene transforms** ([`crate::config::DebugSettings::debug_hud_transforms`]),
//! and **Textures** ([`crate::config::DebugSettings::debug_hud_textures`]).
//!
//! Also hosts the cooperative renderer hang/hitch detector ([`Watchdog`]).

mod debug_hud;
mod debug_hud_encode_error;
mod frame_diagnostics_snapshot;
mod frame_timing_hud_snapshot;
mod host_hud;
mod hud_input;
pub mod per_view;
mod renderer_info_snapshot;
mod scene_transforms_snapshot;
mod texture_debug_snapshot;
mod watchdog;

pub use debug_hud::DebugHud;
pub use debug_hud_encode_error::DebugHudEncodeError;
pub use frame_diagnostics_snapshot::{
    FrameDiagnosticsIpcQueues, FrameDiagnosticsSnapshot, FrameDiagnosticsSnapshotCapture,
    GpuAllocatorHudRefresh, GpuAllocatorReportHud, XrRecoverableFailureCounts,
};
pub use frame_timing_hud_snapshot::{
    FRAME_TIME_HISTORY_LEN, FrameTimeHistory, FrameTimingHudSnapshot,
};
pub use host_hud::HostHudGatherer;
pub use hud_input::{DebugHudInput, sanitize_input_state_for_imgui_host};
pub use per_view::{PerViewHudConfig, PerViewHudOutputs, PerViewHudOutputsSlot};
pub use renderer_info_snapshot::{RendererInfoSnapshot, RendererInfoSnapshotCapture};
pub use scene_transforms_snapshot::{
    RenderSpaceTransformsSnapshot, SceneTransformsSnapshot, TransformRow, WorldTransformSample,
};
pub use texture_debug_snapshot::{TextureDebugRow, TextureDebugSnapshot};
pub use watchdog::{Heartbeat, Watchdog, WatchdogPause};
