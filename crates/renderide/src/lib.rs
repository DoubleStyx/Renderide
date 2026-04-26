//! Renderide: host–renderer IPC, window loop, and GPU presentation (skeleton).
//!
//! The library exposes [`run`] for the `renderide` binary. Shared IPC types live in [`shared`] and
//! are generated; regenerate from **`SharedTypeGenerator`** instead of editing by hand.
//!
//! ## Layering
//!
//! - **[`frontend`]** — IPC queues, shared memory accessor, init handshake, lock-step frame gating,
//!   and window [`input`](crate::frontend::input) (winit to [`InputState`](crate::shared::InputState)).
//! - **[`scene`]** — Render spaces, transforms, mesh renderables, host light cache (no wgpu).
//! - **[`backend`]** — GPU device usage, mesh/texture pools, material property store, uploads,
//!   [`MeshPreprocessPipelines`](crate::backend::mesh_deform::MeshPreprocessPipelines), and the compiled
//!   [`render_graph`](crate::render_graph).
//!
//! [`RendererRuntime`](crate::runtime::RendererRuntime) composes these three; prefer adding new
//! logic in the appropriate module rather than growing the façade.
//!
//! A future optional **`renderide-scene`** crate could hold [`scene`](crate::scene) types with **no
//! `wgpu` dependency**, enforcing the “no GPU in scene” rule via Cargo boundaries; the current
//! single crate keeps iteration cheaper until the scene API stabilizes.

mod process_io;
mod run_error;

pub(crate) use process_io::fatal_crash_log;
pub(crate) use process_io::native_stdio;

/// Winit-driven application: startup, frame loop / pacing, and the [`app::run`] entry point.
pub mod app;
/// Mesh / texture / material / shader asset integration (host IPC → GPU pools).
pub mod assets;
/// GPU resource pools, material tables, mesh/texture uploads, preprocess pipelines — **backend** layer.
pub mod backend;
/// `config.toml` loading and [`config::RendererSettings`] (process-wide defaults).
pub mod config;
/// Developer overlay: Dear ImGui frame snapshot + HUD ([`diagnostics::DebugHud`]).
pub mod diagnostics;
/// Host IPC, shared memory, init, lock-step — **frontend** layer.
pub mod frontend;
/// wgpu device + adapter init, instance/device limits, MSAA helpers, present + VR mirror.
pub mod gpu;

/// Surface acquire and swapchain clear helpers ([`gpu::present`]).
pub use gpu::present;

/// Composed WGSL targets from `build.rs` (`shaders/target/*.wgsl`).
#[doc(hidden)]
pub mod embedded_shaders {
    include!(concat!(env!("OUT_DIR"), "/embedded_shaders.rs"));
}

/// IPC queues, shared-memory accessor, init handshake, and dual-queue dispatch.
pub mod ipc;
/// CLI IPC queue parameters and queue name helpers ([`ipc::connection`]).
pub use ipc::connection;
/// Material registry, shader routing, pipeline cache, and naga-reflection-driven layout.
pub mod materials;
/// Host `HeadOutputDevice` → VR / OpenXR GPU path ([`output_device::head_output_device_wants_openxr`]).
pub mod output_device;
/// Pipeline-primitive vocabulary: shader permutations and the null/debug raster fallback.
pub mod pipelines;
/// Tracy profiling integration: CPU spans, frame marks, and optional GPU timestamp queries.
/// All items compile to nothing when the `tracy` Cargo feature is not active.
pub mod profiling;
/// Compiled render-graph IR, pass nodes, transient pool, parallel recording, and execution.
pub mod render_graph;
/// Embedded asset blobs (icons, fallback textures, etc.) bundled into the binary.
pub mod resources;
/// Per-tick orchestration façade ([`runtime::RendererRuntime`]) wiring frontend, scene, and backend.
pub mod runtime;
/// Transforms, render spaces, mesh renderables — **scene** layer (no wgpu).
pub mod scene;

/// Generated IPC structs and enums shared with the host (regenerate via `SharedTypeGenerator`).
pub mod shared;

/// OpenXR session bootstrap, swapchains, input, and per-frame integration.
pub mod xr;

/// Small set of types for embedding the renderer; import everything else via submodules
/// (for example `crate::materials::MaterialRegistry` in-tree, `renderide::materials::…` externally).
pub mod prelude {
    pub use crate::config::{MsaaSampleCount, RendererSettings, RendererSettingsHandle};
    pub use crate::render_graph::HostCameraFrame;
    pub use crate::runtime::{InitState, RendererRuntime};
    pub use crate::xr::{XrFrameRenderer, XrHostCameraSync};
}

pub use run_error::RunError;

/// Forwards native stdout/stderr into the active file logger after [`logger`] initialization.
///
/// Idempotent. The main [`run`] path installs this after logging starts; entry points that skip
/// [`run`] should call this after [`logger::init_for`] or [`logger::init`].
pub fn ensure_native_stdio_forwarded_to_logger() {
    native_stdio::ensure_stdio_forwarded_to_logger();
}

/// Runs the renderer process: logging, optional IPC, winit loop, and wgpu presentation.
///
/// Returns [`Ok`] with [`None`] when the event loop exits without a host-requested exit code,
/// [`Ok`] with [`Some`] when the host (or handler) sets a process exit code, and [`Err`] for
/// fatal failures during startup (logging, IPC, event loop creation, and similar).
pub fn run() -> Result<Option<i32>, RunError> {
    app::run()
}
