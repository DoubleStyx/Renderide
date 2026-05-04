//! Plain-data snapshots assembled once per HUD frame and consumed by the ImGui overlay.
//!
//! Snapshots are CPU-side, GPU-free, and shape-decoupled from `RenderBackend` and
//! [`crate::runtime::RendererRuntime`] internals so the HUD layer can evolve independently.

pub mod backend_diag;
pub mod frame_diagnostics;
pub mod frame_timing;
pub mod renderer_info;
pub mod scene_transforms;
pub mod texture_debug;

pub use backend_diag::{BackendDiagSnapshot, ShaderRouteSnapshot};
pub use frame_diagnostics::{
    FrameDiagnosticsIpcQueues, FrameDiagnosticsSnapshot, FrameDiagnosticsSnapshotCapture,
    GpuAllocatorHud, GpuAllocatorHudRefresh, GpuAllocatorReportHud, XrRecoverableFailureCounts,
};
pub use frame_timing::FrameTimingHudSnapshot;
pub use renderer_info::{RendererInfoSnapshot, RendererInfoSnapshotCapture};
pub use scene_transforms::{
    RenderSpaceTransformsSnapshot, SceneTransformsSnapshot, TransformRow, WorldTransformSample,
};
pub use texture_debug::{TextureDebugRow, TextureDebugSnapshot};
