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

#![warn(missing_docs)]

mod process_io;
mod run_error;

pub(crate) use process_io::fatal_crash_log;
pub(crate) use process_io::native_stdio;

pub mod app;
pub mod assets;
/// GPU resource pools, material tables, mesh/texture uploads, preprocess pipelines — **backend** layer.
pub mod backend;
/// `config.toml` loading and [`config::RendererSettings`] (process-wide defaults).
pub mod config;
/// Developer overlay: Dear ImGui frame snapshot + HUD ([`diagnostics::DebugHud`]).
pub mod diagnostics;
/// Host IPC, shared memory, init, lock-step — **frontend** layer.
pub mod frontend;
pub mod gpu;

/// Surface acquire and swapchain clear helpers ([`gpu::present`]).
pub use gpu::present;

/// Composed WGSL targets from `build.rs` (`shaders/target/*.wgsl`).
#[doc(hidden)]
pub mod embedded_shaders {
    include!(concat!(env!("OUT_DIR"), "/embedded_shaders.rs"));
}

pub mod ipc;
/// CLI IPC queue parameters and queue name helpers ([`ipc::connection`]).
pub use ipc::connection;
pub mod materials;
/// Host `HeadOutputDevice` → VR / OpenXR GPU path ([`output_device::head_output_device_wants_openxr`]).
pub mod output_device;
pub mod pipelines;
pub mod render_graph;
pub mod resources;
pub mod runtime;
/// Transforms, render spaces, mesh renderables — **scene** layer (no wgpu).
pub mod scene;

pub mod shared;

pub mod xr;

/// Small set of types for embedding the renderer; import everything else via submodules
/// (for example `crate::materials::MaterialRegistry` in-tree, `renderide::materials::…` externally).
pub mod prelude {
    pub use crate::config::{MsaaSampleCount, RendererSettings, RendererSettingsHandle};
    pub use crate::render_graph::HostCameraFrame;
    pub use crate::runtime::{InitState, RendererRuntime};
    pub use crate::xr::{XrHostCameraSync, XrMultiviewFrameRenderer};
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
