//! Renderer facade: orchestrates **frontend** (IPC / shared memory / lock-step), **scene** (host
//! logical state), and **backend** (GPU pools, material store, uploads).
//!
//! [`RendererRuntime`] *composes* a [`RendererFrontend`], a [`SceneCoordinator`], and a
//! [`RenderBackend`]; it does **not** own IPC queue state, scene tables, or GPU resources directly.
//! Each layer keeps its state private; runtime code calls through the layer's API in a fixed
//! per-tick order. Adding new logic here usually means a new method on the right layer plus a
//! short call from the orchestration site, not a new field on [`RendererRuntime`].
//!
//! # Per-tick phase order
//!
//! The authoritative call site is the app driver's redraw tick; this
//! module's methods correspond to the named phases:
//!
//! 1. **Wall-clock prologue** -- [`RendererRuntime::tick_frame_wall_clock_begin`]; resets per-tick flags.
//! 2. **IPC poll** -- [`RendererRuntime::poll_ipc`]; drains incoming `RendererCommand`s before any work runs.
//! 3. **Asset integration** -- [`RendererRuntime::run_asset_integration`]; time-sliced cooperative
//!    mesh/texture/material uploads via [`crate::assets::asset_transfer_queue::drain_asset_tasks`].
//! 4. **Optional XR begin** -- `xr_begin_tick` in `app/`; OpenXR `wait_frame` / `locate_views` so the
//!    same view snapshot is visible to lock-step input.
//! 5. **Lock-step exchange** -- [`RendererRuntime::pre_frame`] emits
//!    [`FrameStartData`](crate::shared::FrameStartData) when allowed; the gating predicate
//!    [`RendererFrontend::should_send_begin_frame`] keeps the lock-step *state* in
//!    [`RendererFrontend`] (this module owns no lock-step counters).
//! 6. **Render** -- desktop multi-view or HMD path through [`crate::render_graph`].
//! 7. **Present + HUD** -- present surface, blit VR mirror, capture ImGui debug snapshots.
//!
//! Lock-step is driven by the `last_frame_index` field of [`FrameStartData`](crate::shared::FrameStartData)
//! on the **outgoing** `frame_start_data` the renderer sends from [`RendererRuntime::pre_frame`].
//! If the host sends [`RendererCommand::FrameStartData`](crate::shared::RendererCommand::FrameStartData),
//! optional payloads are trace-logged until consumers exist.
//!
//! `runtime/lockstep.rs` is a pure debug helper (duplicate-frame-index trace logging only); the
//! decision predicate and the counters live in [`crate::frontend`].
//!
//! # Submodule layout
//!
//! Per-tick logic is split by concern; every submodule extends [`RendererRuntime`] through its
//! own `impl` block:
//!
//! - [`accessors`] -- thin facade pass-throughs to the frontend, backend, scene, and settings.
//! - [`asset_integration`] -- cooperative asset-integration phase + once-per-tick gating.
//! - [`debug_hud_frame`] -- per-tick wiring for the diagnostics ImGui overlay.
//! - [`frame_extract`] -- immutable per-tick view extraction, draw collection, submit packet.
//! - [`frame_render`] -- render-mode dispatch, MSAA prep, frame-extract entry.
//! - [`frame_view_plan`] -- per-view CPU intent (target, clear, viewport, host camera).
//! - [`gpu_services`] -- GPU-facing helpers run once per tick (Hi-Z drain, async jobs, transient eviction).
//! - [`ipc_entry`] -- IPC poll + the `pub(crate)` shims invoked by `crate::frontend::dispatch`.
//! - [`lockstep`] -- diagnostic helper for duplicate frame indices.
//! - [`tick`] -- tick prologue, lock-step / output forwards, the two `tick_one_frame*` orchestrators.
//! - [`view_planning`] -- collection of HMD / secondary RT / main swapchain plans.
//! - [`xr_glue`] -- `XrHostCameraSync` and `XrFrameRenderer` impls for [`RendererRuntime`].
//!
//! IPC dispatch (RendererCommand routing, frame submit, lights/shader/material IPC, init handshake)
//! lives in `crate::frontend::dispatch`. Dispatch reaches into `RendererRuntime`'s `pub(crate)`
//! surface directly via the shims in [`ipc_entry`].

mod accessors;
mod asset_integration;
mod debug_hud_frame;
mod frame_extract;
pub(crate) mod frame_render;
mod frame_view_plan;
mod gpu_services;
mod ipc_entry;
mod lockstep;
mod tick;
mod view_planning;
mod xr_glue;

use hashbrown::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use crate::backend::RenderBackend;
use crate::camera::HostCameraFrame;
use crate::config::RendererSettingsHandle;
use crate::connection::ConnectionParams;
use crate::frontend::RendererFrontend;
use crate::render_graph::GraphExecuteError;
use crate::scene::{RenderSpaceId, SceneCoordinator};

pub use crate::frontend::InitState;

/// Result of one [`RendererRuntime::tick_one_frame`] call.
///
/// `shutdown_requested` lets the calling driver exit its event loop; `fatal_error` triggers a
/// non-zero process exit. `graph_error` carries any failure from [`RendererRuntime::render_frame`]
/// for the caller to decide whether to log + continue or escalate.
#[derive(Debug, Default)]
pub struct TickOutcome {
    /// Host requested an orderly shutdown via IPC during this tick.
    pub shutdown_requested: bool,
    /// IPC reported a fatal error during this tick (e.g. init dispatch protocol violation).
    pub fatal_error: bool,
    /// Render-graph execution error for this tick, if any.
    pub graph_error: Option<GraphExecuteError>,
}

/// Facade: [`RendererFrontend`] + [`SceneCoordinator`] + [`RenderBackend`] + ingestion helpers.
pub struct RendererRuntime {
    pub(crate) frontend: RendererFrontend,
    pub(crate) backend: RenderBackend,
    /// Render spaces and dense transform / mesh state from [`crate::shared::FrameSubmitData`].
    pub(crate) scene: SceneCoordinator,
    /// Last host clip / FOV / VR / ortho task state for [`crate::render_graph::GraphPassFrame`].
    pub(crate) host_camera: HostCameraFrame,
    /// Process-wide renderer settings (shared with the debug HUD and the frame loop).
    pub(crate) settings: RendererSettingsHandle,
    /// Target path for persisting [`Self::settings`] from the ImGui config window.
    pub(crate) config_save_path: PathBuf,
    /// Throttled host CPU/RAM sampling for the debug HUD.
    pub(super) host_hud: crate::diagnostics::HostHudGatherer,
    /// Rolling per-frame wall time history that feeds the Frame timing sparkline.
    pub(super) frame_time_history: crate::diagnostics::FrameTimeHistory,
    /// Persistent EMA state for the Frame / CPU / GPU scalar readouts in the Frame timing HUD.
    /// The sparkline still consumes raw samples from [`Self::frame_time_history`]; only the
    /// numeric overlay cells are smoothed.
    pub(super) frame_timing_ema: crate::diagnostics::FrameTimingEma,
    /// [`crate::shared::FrameSubmitData::render_tasks`] length from the last applied frame submit (HUD).
    pub(super) last_submit_render_task_count: usize,
    /// Cached full [`wgpu::AllocatorReport`] for the **GPU memory** HUD tab (refreshed on a timer).
    pub(super) allocator_report_hud: Option<crate::diagnostics::GpuAllocatorReportHud>,
    /// Cached allocator totals from the same throttled report as [`Self::allocator_report_hud`].
    pub(super) allocator_report_totals: crate::diagnostics::GpuAllocatorHud,
    /// Wall clock when a **GPU memory** tab refresh was last attempted (typically every 2s while the main debug HUD runs).
    pub(super) allocator_report_last_refresh: Option<Instant>,
    /// Set when [`Self::run_asset_integration`] completed for the current winit tick (cleared in [`Self::tick_frame_wall_clock_begin`]).
    pub(super) did_integrate_this_tick: bool,
    /// Count of failed [`SceneCoordinator::apply_frame_submit`] or [`SceneCoordinator::flush_world_caches`] after a host submit (HUD / drift).
    pub(super) frame_submit_apply_failures: u64,
    /// Count of OpenXR `wait_frame` errors since startup (recoverable).
    pub(crate) xr_wait_frame_failures: u64,
    /// Count of OpenXR `locate_views` errors when `should_render` was true (recoverable).
    pub(crate) xr_locate_views_failures: u64,
    /// Running counts of post-init [`crate::shared::RendererCommand`] variants seen without a running handler.
    pub(crate) unhandled_ipc_command_counts: HashMap<&'static str, u64>,
    /// When `true`, ImGui and [`crate::config::save_renderer_settings_from_load`] must not overwrite `config.toml`.
    pub(crate) suppress_renderer_config_disk_writes: bool,
    /// In-flight shader uploads whose [`crate::assets::resolve_shader_upload`] is running on the
    /// rayon pool; drained by [`Self::poll_ipc`] before this tick's IPC batch is dispatched.
    pub(crate) pending_shader_resolutions:
        Vec<crate::frontend::dispatch::shader_material_ipc::PendingShaderResolution>,
    /// Reusable per-frame scratch for secondary render-texture view collection. Holds
    /// `(render_space_id, camera_depth, camera_index)` tuples for sorting; cleared and refilled
    /// each tick so secondary-RT scenes don't allocate a fresh `Vec` per frame.
    pub(super) secondary_view_tasks_scratch: Vec<(RenderSpaceId, f32, usize)>,
}

impl RendererRuntime {
    /// Builds a runtime; does not open IPC yet (see [`Self::connect_ipc`]).
    pub fn new(
        params: Option<ConnectionParams>,
        settings: RendererSettingsHandle,
        config_save_path: PathBuf,
    ) -> Self {
        Self {
            frontend: RendererFrontend::new(params),
            backend: RenderBackend::new(),
            scene: SceneCoordinator::new(),
            host_camera: HostCameraFrame::default(),
            settings,
            config_save_path,
            host_hud: crate::diagnostics::HostHudGatherer::default(),
            frame_time_history: crate::diagnostics::FrameTimeHistory::new(),
            frame_timing_ema: crate::diagnostics::FrameTimingEma::default(),
            last_submit_render_task_count: 0,
            allocator_report_hud: None,
            allocator_report_totals: crate::diagnostics::GpuAllocatorHud::default(),
            allocator_report_last_refresh: None,
            did_integrate_this_tick: false,
            frame_submit_apply_failures: 0,
            xr_wait_frame_failures: 0,
            xr_locate_views_failures: 0,
            unhandled_ipc_command_counts: HashMap::new(),
            suppress_renderer_config_disk_writes: false,
            pending_shader_resolutions: Vec::new(),
            secondary_view_tasks_scratch: Vec::new(),
        }
    }
}

#[cfg(test)]
mod orchestration_tests;
