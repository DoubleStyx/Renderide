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
//!    mesh/texture/material uploads via [`crate::backend::RenderBackend::drain_asset_tasks`].
//! 4. **Offscreen readback tasks** -- [`RendererRuntime::drain_reflection_probe_render_tasks`]
//!    and [`RendererRuntime::drain_camera_render_tasks`] render host-requested captures and write
//!    the resulting bytes to shared memory.
//! 5. **Optional XR begin** -- `xr_begin_tick` in `app/`; VR waits for coupled host submits before
//!    this point so `xrBeginFrame` is not opened for a frame that cannot render.
//! 6. **Lock-step exchange** -- [`RendererRuntime::pre_frame`] emits
//!    [`FrameStartData`](crate::shared::FrameStartData) when allowed. Desktop sends before render;
//!    VR sends before XR only when it needs an initial host submit, then sends the next begin-frame
//!    after the HMD render attempt so the sampled pose is current. The gating predicate
//!    [`RendererFrontend::should_send_begin_frame`] keeps the lock-step *state* in
//!    [`RendererFrontend`] (this module owns no lock-step counters).
//! 7. **Render** -- desktop multi-view, HMD, and offscreen paths run the explicit CPU render
//!    schedule in [`frame::schedule`]: extract, asset prepare, view planning, draw queueing,
//!    sort, resource prepare, command record, cleanup.
//! 8. **Present + HUD** -- present surface, blit VR mirror, capture ImGui debug snapshots.
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
//! Per-tick logic is grouped by concern; every submodule that adds behavior to
//! [`RendererRuntime`] does so through its own `impl` block.
//!
//! - [`frame`] -- per-frame pipeline: [`frame::view_planning`] collects views,
//!   [`frame::view_plan`] holds per-view CPU intent, [`frame::extract`] turns that into the
//!   immutable submit packet, [`frame::render`] dispatches the render mode, and
//!   [`frame::submit`] applies host frame-submit payloads.
//! - [`ipc`] -- IPC ingestion: [`ipc::entry`] drains the queue, [`ipc::effects`] dispatches
//!   decoded effects to per-domain handlers under [`ipc::effects`]'s submodules, and
//!   [`ipc::shader_material`] / [`ipc::lights`] own shader, material-batch, and light-buffer
//!   submissions.
//! - [`offscreen_tasks`] -- host-requested offscreen rendering: [`offscreen_tasks::camera`] for
//!   camera capture, [`offscreen_tasks::reflection_probe`] for cubemap bake tasks, and
//!   [`offscreen_tasks::readback`] for the shared GPU buffer-mapping plumbing.
//! - [`state`] -- runtime-owned state holders aggregated as private fields on
//!   [`RendererRuntime`]: [`state::config`], [`state::diagnostics`], [`state::ipc`],
//!   [`state::tick`], [`state::xr`].
//! - [`tick`] -- the two `tick_one_frame*` orchestrators and the lock-step / output forwards the
//!   app driver calls inside one redraw iteration.
//! - [`accessors`] -- thin facade pass-throughs to the frontend, backend, scene, and settings.
//! - [`asset_integration`] -- cooperative asset-integration phase + once-per-tick gating.
//! - [`gpu_services`] -- GPU-facing helpers run once per tick (Hi-Z drain, async jobs,
//!   transient eviction).
//! - [`debug_hud_frame`] -- per-tick wiring for the diagnostics ImGui overlay.
//! - [`shutdown`] -- graceful shutdown signaling and the compact final-summary log.
//! - [`lockstep`] -- diagnostic helper for duplicate frame indices.
//! - [`xr_glue`] -- [`crate::xr::XrHostCameraSync`] and [`crate::xr::XrFrameRenderer`] impls
//!   for [`RendererRuntime`].
//!
//! IPC dispatch in [`crate::frontend::dispatch`] is decode-only. [`ipc::entry`] polls queue
//! commands, `frontend::dispatch` classifies them into domain effects, and [`ipc::effects`] is
//! the single runtime-owned application point for frontend, scene, backend, host camera,
//! settings, and IPC scratch mutations.

mod accessors;
mod asset_integration;
mod debug_hud_frame;
pub mod display;
mod frame;
mod gpu_services;
mod ipc;
mod lockstep;
mod offscreen_tasks;
mod shutdown;
mod state;
mod tick;
mod xr_glue;

use std::path::PathBuf;

use crate::backend::RenderBackend;
use crate::camera::HostCameraFrame;
use crate::config::RendererSettingsHandle;
use crate::connection::ConnectionParams;
use crate::frontend::RendererFrontend;
use crate::render_graph::GraphExecuteError;
use crate::scene::SceneCoordinator;

use self::display::DisplayBlitResources;
use state::{
    RuntimeConfigState, RuntimeDiagnosticsState, RuntimeIpcState, RuntimeTickState, RuntimeXrStats,
};

pub(crate) use state::DesktopFramePacingCaps;

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
    /// Whether this tick intentionally skipped rendering while waiting for host lockstep.
    pub render_skipped: bool,
}

/// Facade: [`RendererFrontend`] + [`SceneCoordinator`] + [`RenderBackend`] + ingestion helpers.
pub struct RendererRuntime {
    frontend: RendererFrontend,
    backend: RenderBackend,
    /// Render spaces and dense transform / mesh state from [`crate::shared::FrameSubmitData`].
    scene: SceneCoordinator,
    /// Last host clip / FOV / VR / ortho task state for [`crate::render_graph::GraphPassFrame`].
    host_camera: HostCameraFrame,
    /// Settings handle, config path, and disk-write suppression.
    config: RuntimeConfigState,
    /// Runtime-side diagnostics accumulation.
    diagnostics: RuntimeDiagnosticsState,
    /// IPC scratch and unhandled-command counters.
    ipc_state: RuntimeIpcState,
    /// Per-tick gates and reusable view-planning scratch.
    tick_state: RuntimeTickState,
    /// Cumulative recoverable OpenXR failure counts.
    xr_stats: RuntimeXrStats,
    /// Lazy GPU resources for the host `BlitToDisplay` desktop pass; created on first use.
    display_blit: DisplayBlitResources,
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
            config: RuntimeConfigState::new(settings, config_save_path),
            diagnostics: RuntimeDiagnosticsState::new(),
            ipc_state: RuntimeIpcState::new(),
            tick_state: RuntimeTickState::new(),
            xr_stats: RuntimeXrStats::default(),
            display_blit: DisplayBlitResources::new(),
        }
    }

    /// Disjoint mutable access to the display-blit cache and render backend so callers can run
    /// the desktop-blit pass with the HUD overlay callback against the backend without aliasing.
    pub fn display_blit_and_backend_mut(
        &mut self,
    ) -> (&mut DisplayBlitResources, &mut RenderBackend) {
        (&mut self.display_blit, &mut self.backend)
    }
}

#[cfg(test)]
mod orchestration_tests;
