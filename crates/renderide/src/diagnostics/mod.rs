//! Dear ImGui diagnostics: **Frame timing** ([`crate::config::DebugSettings::debug_hud_frame_timing`]),
//! **Renderide debug** ([`crate::config::DebugSettings::debug_hud_enabled`]; Stats / Shader routes / **GPU memory**),
//! **Scene transforms** ([`crate::config::DebugSettings::debug_hud_transforms`]).

mod debug_hud;
mod debug_hud_encode_error;
mod frame_diagnostics_snapshot;
mod frame_timing_hud_snapshot;
mod host_hud;
mod hud_input;
mod renderer_info_snapshot;
mod scene_transforms_snapshot;

pub use debug_hud::DebugHud;
pub use debug_hud_encode_error::DebugHudEncodeError;
pub use frame_diagnostics_snapshot::{FrameDiagnosticsSnapshot, GpuAllocatorReportHud};
pub use frame_timing_hud_snapshot::FrameTimingHudSnapshot;
pub use host_hud::HostHudGatherer;
pub use hud_input::{sanitize_input_state_for_imgui_host, DebugHudInput};
pub use renderer_info_snapshot::{RendererInfoSnapshot, RendererInfoSnapshotCapture};
pub use scene_transforms_snapshot::{
    RenderSpaceTransformsSnapshot, SceneTransformsSnapshot, TransformRow, WorldTransformSample,
};
