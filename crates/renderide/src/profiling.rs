//! Tracy profiling integration -- zero cost by default, enabled by the `tracy` Cargo feature.
//!
//! # How to enable
//!
//! Build with `--features tracy` for CPU spans, frame marks, and plots:
//!
//! ```bash
//! cargo build --release --features tracy
//! ```
//!
//! Then launch the [Tracy GUI](https://github.com/wolfpld/tracy) and connect on port **8086**.
//! Tracy uses `ondemand` mode, so data is only streamed while a GUI is connected.
//!
//! # What a capture contains
//!
//! Two independent sources, and they answer different questions:
//!
//! 1. **Manual zones** from [`scope`], roughly a thousand of them across the renderer. These are
//!    the named rows in the timeline. They only cover what someone chose to instrument.
//! 2. **Automatic capture** from the Tracy client itself: periodic callstack sampling, plus
//!    context-switch tracing that marks when a thread is actually on core instead of blocked or
//!    preempted. This is what catches uninstrumented code, wgpu, the driver, and the allocator,
//!    and it is the only way to tell a slow zone apart from a zone that spent its time waiting.
//!    On by default through `tracy-client`'s own feature defaults, which `profiling` pulls in.
//!
//! On Windows the sampling and context-switch tracks come from ETW, so the renderer has to run
//! elevated to fill them in. Unelevated captures still contain every manual zone.
//!
//! Long-lived worker threads must call [`register_worker_thread`] at the top of their loop, or
//! their spans land under a bare thread id.
//!
//! # Default builds (no `tracy` feature)
//!
//! Every macro and function in this module compiles to nothing. The `profiling` crate guarantees
//! this: when no backend feature is active, `profiling::scope!` and friends expand to `()`.
//! Verify with `cargo expand` if in doubt.
//!
//! # GPU profiling (`tracy-gpu`, opt-in on top of `tracy`)
//!
//! GPU timestamp queries are a separate feature because they are not free on the CPU side: each
//! scope allocates a label `String` and a query slot on the recording thread, and every frame
//! pays a resolve plus a readback. That cost lands on `CommandEncoder::finish`, so a CPU-bound
//! capture taken with `tracy-gpu` on is measuring the profiler as much as the renderer. Profile
//! CPU with `tracy`; add `tracy-gpu` only when the question is which GPU pass is slow.
//!
//! With `tracy-gpu`, [`GpuProfilerHandle`] wraps [`wgpu_profiler::GpuProfiler`] and connects to
//! the attached Tracy client via [`wgpu_profiler::GpuProfiler::new_with_tracy_client`], bridging
//! pass-level GPU timestamps into Tracy's GPU timeline. The bridge is rebuilt at clean frame
//! boundaries as the GUI connects and disconnects, because Tracy's serial GPU events are not
//! gated by `ondemand` and would otherwise replay with stale query ids. Graph passes, manual
//! compute/render passes, copy/readback regions, and expensive bounded subpasses all use stable
//! labels so Tracy and vendor captures line up.
//!
//! Pass-level timestamp writes (the preferred path) only require [`wgpu::Features::TIMESTAMP_QUERY`].
//! Encoder-level [`GpuProfilerHandle::begin_query`]/[`GpuProfilerHandle::end_query`] additionally
//! require [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`]; when the adapter is missing that
//! feature the handle is still created but encoder-level queries silently do nothing. When the
//! adapter is also missing [`wgpu::Features::TIMESTAMP_QUERY`], [`GpuProfilerHandle::try_new`]
//! returns [`None`] and a warning is logged; CPU spans still work.
//!
//! The HUD's frame-bracket GPU time and the IPC `render_time` metric do **not** go through this
//! module's profiler; they use encoder-level timestamps requested unconditionally in
//! [`timestamp_query_features_if_supported`], so turning `tracy-gpu` off keeps them working.
//!
//! # Thread naming
//!
//! Call [`register_main_thread`] once at startup so the main thread appears by name in Tracy. It
//! also starts the Tracy client before any other profiling macro runs. Pass
//! [`rayon_thread_start_handler`] to `rayon::ThreadPoolBuilder::start_handler` so Rayon workers
//! are also named.

mod deferred_span;
mod frame_marks;
mod gpu;
#[cfg(feature = "tracy-gpu")]
mod gpu_profiler_impl;
#[cfg(not(feature = "tracy-gpu"))]
mod gpu_profiler_stub;
mod gpu_scope;
mod plots;
mod resource_churn;
#[cfg(test)]
mod tests;

pub use profiling::scope;

pub(crate) use deferred_span::DeferredCpuSpan;
pub use frame_marks::{
    emit_frame_mark, emit_render_submit_frame_mark, rayon_thread_start_handler,
    register_main_thread, register_worker_thread,
};
pub use gpu::{
    GpuPassEntry, GpuProfilerFrameStats, GpuProfilerSnapshot, PhaseQuery,
    compute_pass_timestamp_writes, render_pass_timestamp_writes,
    timestamp_query_features_if_supported,
};
#[cfg(feature = "tracy-gpu")]
pub use gpu_profiler_impl::GpuProfilerHandle;
#[cfg(not(feature = "tracy-gpu"))]
pub use gpu_profiler_stub::GpuProfilerHandle;
pub(crate) use gpu_scope::GpuEncoderScope;
pub use plots::{
    AssetIntegrationProfileSample, CommandEncodingProfileSample, FrameUploadArenaProfileSample,
    IpcPollProfileSample, LockstepPipelineProfileSample, MeshDeformProfileSample,
    RayonAdmissionProfileSample, RenderWorldMaintenanceProfileSample, ShadowCacheProfileSample,
    WorldMeshDrawPlanCacheProfileSample, WorldMeshGeometryArenaProfileSample,
    WorldMeshGpuCullCacheProfileSample, WorldMeshStaticSourceReleaseProfileSample,
    plot_asset_integration, plot_command_encoding, plot_driver_submit_backlog,
    plot_event_loop_idle_ms, plot_event_loop_wait_ms, plot_fps_cap_active, plot_frame_global_split,
    plot_frame_upload_arena, plot_frame_upload_batch, plot_ipc_poll, plot_lockstep_pipeline,
    plot_mesh_deform, plot_rayon_admission, plot_render_world_maintenance, plot_shadow_atlas,
    plot_shadow_cache, plot_shadow_static_split, plot_surface_acquire_outcome,
    plot_surface_get_current_texture_ms,
    plot_surface_in_flight_count, plot_surface_previous_present_wait_ms, plot_window_focused,
    plot_world_mesh_draw_plan_cache, plot_world_mesh_geometry_arena,
    plot_world_mesh_gpu_cull_cache, plot_world_mesh_prepare, plot_world_mesh_static_source_release,
    plot_world_mesh_subpass,
};
pub(crate) use plots::{
    MeshUploadBatchProfileSample, plot_mesh_derived_stream_masks, plot_mesh_upload_batch,
};
#[cfg(feature = "tracy")]
pub(crate) use plots::{WorldMeshForwardIndirectProfileSample, plot_world_mesh_forward_indirect};
pub(crate) use resource_churn::{
    ResourceChurnKind, ResourceChurnSite, flush_resource_churn_plots, note_resource_churn,
};
