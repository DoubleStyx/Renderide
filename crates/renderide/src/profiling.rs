//! Tracy profiling integration -- zero cost by default, enabled by the `tracy` Cargo feature.
//!
//! # How to enable
//!
//! Build with `--features tracy` to activate Tracy spans, frame marks, and GPU timestamp queries:
//!
//! ```bash
//! cargo build --profile dev-fast --features tracy
//! ```
//!
//! Then launch the [Tracy GUI](https://github.com/wolfpld/tracy) and connect on port **8086**.
//! Tracy uses `ondemand` mode, so CPU data is only streamed while a GUI is connected. The GPU
//! profiler bridge is rebuilt on clean frame boundaries as the GUI connects and disconnects, so
//! late attach starts with fresh GPU query ids instead of replaying disconnected GPU events.
//!
//! # Default builds (no `tracy` feature)
//!
//! Every macro and function in this module compiles to nothing. The `profiling` crate guarantees
//! this: when no backend feature is active, `profiling::scope!` and friends expand to `()`.
//! Verify with `cargo expand` if in doubt.
//!
//! # GPU profiling
//!
//! [`GpuProfilerHandle`] wraps [`wgpu_profiler::GpuProfiler`] (only compiled with `tracy`). It
//! connects to the currently attached Tracy client via
//! [`wgpu_profiler::GpuProfiler::new_with_tracy_client`], so pass-level GPU timestamps are
//! bridged into Tracy's GPU timeline. The bridge is disabled while Tracy is disconnected because
//! Tracy's serial GPU events are not gated by `ondemand`. Renderide keeps this as the
//! always-available timing spine: graph passes, manual compute/render passes, copy/readback
//! regions, and expensive bounded subpasses should all use stable labels so Tracy and vendor
//! captures line up.
//!
//! Pass-level timestamp writes (the preferred path) only require [`wgpu::Features::TIMESTAMP_QUERY`].
//! Encoder-level [`GpuProfilerHandle::begin_query`]/[`GpuProfilerHandle::end_query`] additionally
//! require [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`]; when the adapter is missing that
//! feature the handle is still created but encoder-level queries silently do nothing. When the
//! adapter is also missing [`wgpu::Features::TIMESTAMP_QUERY`], [`GpuProfilerHandle::try_new`]
//! returns [`None`] and a warning is logged; CPU spans still work.
//!
//! # Thread naming
//!
//! Call [`register_main_thread`] once at startup so the main thread appears by name in Tracy. It
//! also starts the Tracy client before any other profiling macro runs. Pass
//! [`rayon_thread_start_handler`] to `rayon::ThreadPoolBuilder::start_handler` so Rayon workers
//! are also named.

mod frame_marks;
mod gpu;
#[cfg(feature = "tracy")]
mod gpu_profiler_impl;
#[cfg(not(feature = "tracy"))]
mod gpu_profiler_stub;
mod gpu_scope;
mod plots;
mod resource_churn;
#[cfg(test)]
mod tests;

pub use profiling::scope;

pub use frame_marks::{
    emit_frame_mark, emit_render_submit_frame_mark, rayon_thread_start_handler,
    register_main_thread,
};
pub use gpu::{
    GpuPassEntry, GpuProfilerFrameStats, GpuProfilerSnapshot, PhaseQuery,
    compute_pass_timestamp_writes, render_pass_timestamp_writes,
    timestamp_query_features_if_supported,
};
#[cfg(feature = "tracy")]
pub use gpu_profiler_impl::GpuProfilerHandle;
#[cfg(not(feature = "tracy"))]
pub use gpu_profiler_stub::GpuProfilerHandle;
pub(crate) use gpu_scope::GpuEncoderScope;
pub use plots::{
    AssetIntegrationProfileSample, CommandEncodingProfileSample, FrameUploadArenaProfileSample,
    IpcPollProfileSample, MeshDeformProfileSample, RayonAdmissionProfileSample,
    plot_asset_integration, plot_command_encoding, plot_driver_submit_backlog,
    plot_event_loop_idle_ms, plot_event_loop_wait_ms, plot_fps_cap_active, plot_frame_upload_arena,
    plot_frame_upload_batch, plot_ipc_poll, plot_mesh_deform, plot_rayon_admission,
    plot_surface_acquire_outcome, plot_window_focused, plot_world_mesh_subpass,
};
pub(crate) use resource_churn::{
    ResourceChurnKind, ResourceChurnSite, flush_resource_churn_plots, note_resource_churn,
};
