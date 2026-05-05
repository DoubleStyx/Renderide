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
//! Tracy uses `ondemand` mode, so data is only streamed while a GUI is connected.
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
//! connects to the running Tracy client via
//! [`wgpu_profiler::GpuProfiler::new_with_tracy_client`], so pass-level GPU timestamps are
//! bridged into Tracy's GPU timeline.
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

pub use profiling::finish_frame;
pub use profiling::function_scope;
pub use profiling::scope;

/// Starts the Tracy client (if the `tracy` feature is on) and registers the calling thread as
/// `"renderer-main"` in the active profiler.
///
/// Must be called exactly once, before any other `profiling::scope!` macro or
/// [`GpuProfilerHandle::try_new`] runs -- the `profiling` crate's tracy backend expects a running
/// `tracy_client::Client` on every span, so the client has to be live first.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn register_main_thread() {
    #[cfg(feature = "tracy")]
    {
        let _ = tracy_client::Client::start();
    }
    profiling::register_thread!("renderer-main");
}

/// Emits a frame mark to the active profiler, closing the current frame boundary.
///
/// Call exactly once per winit tick, at the very end of the app driver's redraw tick.
/// Without frame marks Tracy still records spans but the frame timeline and histogram are empty.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn emit_frame_mark() {
    profiling::finish_frame!();
}

/// Emits a secondary Tracy frame mark for command-buffer batches handed to the GPU driver thread.
///
/// The default Tracy frame remains the winit redraw tick. This secondary track marks actual GPU
/// submits so empty redraw ticks, swapchain acquire skips, and delayed GPU timestamp readback do
/// not make the pass timeline look like graph work vanished.
#[inline]
pub fn emit_render_submit_frame_mark() {
    #[cfg(feature = "tracy")]
    {
        if let Some(client) = tracy_client::Client::running() {
            client.secondary_frame_mark(tracy_client::frame_name!("render-submit"));
        }
    }
}

/// Records the FPS cap currently applied by
/// the app driver's `about_to_wait` handler -- either
/// [`crate::config::DisplaySettings::focused_fps_cap`] or
/// [`crate::config::DisplaySettings::unfocused_fps_cap`], whichever matches the current focus
/// state. Zero means uncapped (winit is told `ControlFlow::Poll`); a VR tick emits zero because
/// the XR runtime paces the session independently.
///
/// Call once per winit iteration so the Tracy plot sits adjacent to the frame-mark timeline and
/// the value-per-frame is an exact reading rather than an interpolation. Expands to nothing when
/// the `tracy` feature is off.
#[inline]
pub fn plot_fps_cap_active(cap: u32) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("fps_cap_active", f64::from(cap));
    #[cfg(not(feature = "tracy"))]
    let _ = cap;
}

/// Records window focus (`1.0` focused, `0.0` unfocused) as a Tracy plot so focus-driven cap
/// switches in the app driver's `about_to_wait` handler are visible at a glance.
///
/// Intended to be plotted next to [`plot_fps_cap_active`]: a drop from `1.0` to `0.0` should line
/// up with the cap changing from `focused_fps_cap` to `unfocused_fps_cap` (or vice versa), which
/// is the usual cause of a sudden frame-time change while profiling.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_window_focused(focused: bool) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("window_focused", if focused { 1.0 } else { 0.0 });
    #[cfg(not(feature = "tracy"))]
    let _ = focused;
}

/// Records, in milliseconds, how long
/// the app driver's `about_to_wait` handler asked winit to park before the next
/// `RedrawRequested`. Emit the [`std::time::Duration`] between `now` and the
/// [`winit::event_loop::ControlFlow::WaitUntil`] deadline when the capped branch is taken, and
/// `0.0` when the handler returns with [`winit::event_loop::ControlFlow::Poll`].
///
/// The gap between Tracy frames that no [`profiling::scope`] can cover (because the main thread
/// is parked inside winit) shows up on this plot as a non-zero value, attributing the idle time
/// to the CPU-side frame-pacing cap rather than missing instrumentation. Expands to nothing when
/// the `tracy` feature is off.
#[inline]
pub fn plot_event_loop_wait_ms(ms: f64) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("event_loop_wait_ms", ms);
    #[cfg(not(feature = "tracy"))]
    let _ = ms;
}

/// Records the driver-thread submit backlog (`submits_pushed - submits_done`) as a Tracy
/// plot.
///
/// Call once per tick from the frame epilogue. A steady-state value of `0` or `1` is
/// healthy (one frame in flight on the driver matches the ring's nominal pipelining
/// depth); a sustained value at the ring capacity means the producer is back-pressured
/// by the driver and CPU/GPU pacing is bound by submit throughput. Useful next to
/// [`plot_event_loop_idle_ms`] when diagnosing why the main thread is sleeping.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_driver_submit_backlog(count: u64) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("driver_submit_backlog", count as f64);
    #[cfg(not(feature = "tracy"))]
    let _ = count;
}

/// Records, in milliseconds, the wall-clock gap between the end of the previous
/// app-driver redraw tick and the start of the current one.
///
/// Complements [`plot_event_loop_wait_ms`] (the *requested* wait) by showing the *actual* slept
/// duration -- divergence between the two points at additional blocking outside the pacing cap
/// (for example compositor vsync via `surface.get_current_texture`, which is itself already
/// covered by a dedicated `gpu::get_current_texture` scope).
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_event_loop_idle_ms(ms: f64) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("event_loop_idle_ms", ms);
    #[cfg(not(feature = "tracy"))]
    let _ = ms;
}

/// Records the result of a swapchain acquire attempt as one-hot Tracy plots.
///
/// These samples explain CPU frames that have a frame mark but no render-graph GPU markers: a
/// timeout or occluded surface intentionally skips graph recording for that tick, while a
/// reconfigure means the graph will resume on a later acquire.
#[inline]
pub fn plot_surface_acquire_outcome(acquired: bool, skipped: bool, reconfigured: bool) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!(
            "surface_acquire::acquired",
            if acquired { 1.0 } else { 0.0 }
        );
        tracy_client::plot!("surface_acquire::skipped", if skipped { 1.0 } else { 0.0 });
        tracy_client::plot!(
            "surface_acquire::reconfigured",
            if reconfigured { 1.0 } else { 0.0 }
        );
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = (acquired, skipped, reconfigured);
    }
}

/// Records, per call to `crate::passes::world_mesh_forward::encode::draw_subset`,
/// how many instance batches and how many input draws were submitted in that subpass.
///
/// One sample lands on the Tracy timeline per opaque or intersection subpass record, so the
/// plot trace shows fragmentation visually: when batches ~= draws, the merge isn't compressing;
/// when batches << draws, instancing is collapsing same-mesh runs as intended. Pair with
/// [`crate::world_mesh::WorldMeshDrawStats::gpu_instances_emitted`] in the HUD for a
/// per-frame integral. Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_world_mesh_subpass(batches: usize, draws: usize) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!("world_mesh::subpass_batches", batches as f64);
        tracy_client::plot!("world_mesh::subpass_draws", draws as f64);
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = (batches, draws);
    }
}

/// Records deferred queue-write traffic for one frame.
#[inline]
pub fn plot_frame_upload_batch(writes: usize, bytes: usize) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!("frame_upload::writes", writes as f64);
        tracy_client::plot!("frame_upload::bytes", bytes as f64);
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = (writes, bytes);
    }
}

/// CPU timings and counts for one render-graph command-encoding slice.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommandEncodingProfileSample {
    /// Number of views encoded by the graph.
    pub view_count: usize,
    /// Number of command buffers submitted in the batch.
    pub command_buffers: usize,
    /// Frame-global pass count in the compiled schedule.
    pub frame_global_passes: usize,
    /// Per-view pass count in the compiled schedule.
    pub per_view_passes: usize,
    /// Declared transient texture handles in the compiled graph.
    pub transient_textures: usize,
    /// Physical transient texture slots after aliasing.
    pub transient_texture_slots: usize,
    /// Transient texture allocation misses during this frame.
    pub transient_texture_misses: usize,
    /// Transient buffer allocation misses during this frame.
    pub transient_buffer_misses: usize,
    /// Deferred upload writes drained before submit.
    pub upload_writes: usize,
    /// Deferred upload payload bytes drained before submit.
    pub upload_bytes: usize,
    /// CPU time spent resolving transient resources for all views.
    pub pre_resolve_ms: f64,
    /// CPU time spent preparing shared/per-view resources before recording.
    pub prepare_resources_ms: f64,
    /// CPU time spent encoding frame-global work before `CommandEncoder::finish`.
    pub frame_global_encode_ms: f64,
    /// CPU time spent inside frame-global `CommandEncoder::finish`.
    pub frame_global_finish_ms: f64,
    /// CPU time spent encoding per-view work before `CommandEncoder::finish`.
    pub per_view_encode_ms: f64,
    /// Total CPU time spent inside per-view `CommandEncoder::finish` calls.
    pub per_view_finish_ms: f64,
    /// CPU time spent draining deferred uploads.
    pub upload_drain_ms: f64,
    /// CPU time spent inside the upload encoder `CommandEncoder::finish`.
    pub upload_finish_ms: f64,
    /// CPU time spent allocating and assembling the final command-buffer batch.
    pub command_batch_assembly_ms: f64,
    /// CPU time spent enqueueing the submit batch to the GPU driver thread.
    pub submit_enqueue_ms: f64,
    /// Largest single encoder finish observed in this frame.
    pub max_encoder_finish_ms: f64,
    /// World-mesh draw items visible to the command recorder.
    pub world_mesh_draws: usize,
    /// World-mesh indexed draw groups emitted by the command recorder.
    pub world_mesh_instance_batches: usize,
    /// World-mesh pipeline-pass draw submissions after multi-pass material expansion.
    pub world_mesh_pipeline_pass_submits: usize,
}

/// Records command-encoding timings and pressure counters for the current frame.
#[inline]
pub fn plot_command_encoding(sample: CommandEncodingProfileSample) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!("command_encoding::views", sample.view_count as f64);
        tracy_client::plot!(
            "command_encoding::command_buffers",
            sample.command_buffers as f64
        );
        tracy_client::plot!(
            "command_encoding::frame_global_passes",
            sample.frame_global_passes as f64
        );
        tracy_client::plot!(
            "command_encoding::per_view_passes",
            sample.per_view_passes as f64
        );
        tracy_client::plot!(
            "command_encoding::transient_textures",
            sample.transient_textures as f64
        );
        tracy_client::plot!(
            "command_encoding::transient_texture_slots",
            sample.transient_texture_slots as f64
        );
        tracy_client::plot!(
            "command_encoding::transient_texture_misses",
            sample.transient_texture_misses as f64
        );
        tracy_client::plot!(
            "command_encoding::transient_buffer_misses",
            sample.transient_buffer_misses as f64
        );
        tracy_client::plot!(
            "command_encoding::upload_writes",
            sample.upload_writes as f64
        );
        tracy_client::plot!("command_encoding::upload_bytes", sample.upload_bytes as f64);
        tracy_client::plot!("command_encoding::pre_resolve_ms", sample.pre_resolve_ms);
        tracy_client::plot!(
            "command_encoding::prepare_resources_ms",
            sample.prepare_resources_ms
        );
        tracy_client::plot!(
            "command_encoding::frame_global_encode_ms",
            sample.frame_global_encode_ms
        );
        tracy_client::plot!(
            "command_encoding::frame_global_finish_ms",
            sample.frame_global_finish_ms
        );
        tracy_client::plot!(
            "command_encoding::per_view_encode_ms",
            sample.per_view_encode_ms
        );
        tracy_client::plot!(
            "command_encoding::per_view_finish_ms",
            sample.per_view_finish_ms
        );
        tracy_client::plot!("command_encoding::upload_drain_ms", sample.upload_drain_ms);
        tracy_client::plot!(
            "command_encoding::upload_finish_ms",
            sample.upload_finish_ms
        );
        tracy_client::plot!(
            "command_encoding::command_batch_assembly_ms",
            sample.command_batch_assembly_ms
        );
        tracy_client::plot!(
            "command_encoding::submit_enqueue_ms",
            sample.submit_enqueue_ms
        );
        tracy_client::plot!(
            "command_encoding::max_encoder_finish_ms",
            sample.max_encoder_finish_ms
        );
        tracy_client::plot!(
            "command_encoding::world_mesh_draws",
            sample.world_mesh_draws as f64
        );
        tracy_client::plot!(
            "command_encoding::world_mesh_instance_batches",
            sample.world_mesh_instance_batches as f64
        );
        tracy_client::plot!(
            "command_encoding::world_mesh_pipeline_pass_submits",
            sample.world_mesh_pipeline_pass_submits as f64
        );
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = sample;
    }
}

/// Asset-integration backlog and budget-exhaustion counters for one drain.
#[derive(Clone, Copy, Debug, Default)]
pub struct AssetIntegrationProfileSample {
    /// High-priority tasks still queued after the drain.
    pub high_priority_queued: usize,
    /// Normal-priority tasks still queued after the drain.
    pub normal_priority_queued: usize,
    /// Whether the high-priority emergency ceiling stopped the drain.
    pub high_priority_budget_exhausted: bool,
    /// Whether the normal-priority frame budget stopped the drain.
    pub normal_priority_budget_exhausted: bool,
}

/// Records asset-integration backlog and budget pressure for the current frame.
#[inline]
pub fn plot_asset_integration(sample: AssetIntegrationProfileSample) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!(
            "asset_integration::high_priority_queued",
            sample.high_priority_queued as f64
        );
        tracy_client::plot!(
            "asset_integration::normal_priority_queued",
            sample.normal_priority_queued as f64
        );
        tracy_client::plot!(
            "asset_integration::high_priority_budget_exhausted",
            if sample.high_priority_budget_exhausted {
                1.0
            } else {
                0.0
            }
        );
        tracy_client::plot!(
            "asset_integration::normal_priority_budget_exhausted",
            if sample.normal_priority_budget_exhausted {
                1.0
            } else {
                0.0
            }
        );
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = sample;
    }
}

/// Mesh-deform workload and cache pressure counters for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshDeformProfileSample {
    /// Deform work items collected for this frame.
    pub work_items: u64,
    /// Compute passes opened while recording mesh deformation.
    pub compute_passes: u64,
    /// Bind groups created while recording mesh deformation.
    pub bind_groups_created: u64,
    /// Encoder copy operations recorded by mesh deformation.
    pub copy_ops: u64,
    /// Sparse blendshape compute dispatches recorded.
    pub blend_dispatches: u64,
    /// Skinning compute dispatches recorded.
    pub skin_dispatches: u64,
    /// Scratch-buffer grow operations triggered by this frame.
    pub scratch_buffer_grows: u64,
    /// Work items skipped because the skin cache could not allocate safely.
    pub skipped_allocations: u64,
    /// Skin-cache entries reused.
    pub cache_reuses: u64,
    /// Skin-cache entries allocated.
    pub cache_allocations: u64,
    /// Skin-cache arena growth operations.
    pub cache_grows: u64,
    /// Prior-frame skin-cache entries evicted.
    pub cache_evictions: u64,
    /// Allocation attempts where all evictable entries were current-frame entries.
    pub cache_current_frame_eviction_refusals: u64,
}

/// Records mesh-deform workload and cache pressure counters for the current frame.
#[inline]
pub fn plot_mesh_deform(sample: MeshDeformProfileSample) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!("mesh_deform::work_items", sample.work_items as f64);
        tracy_client::plot!("mesh_deform::compute_passes", sample.compute_passes as f64);
        tracy_client::plot!(
            "mesh_deform::bind_groups_created",
            sample.bind_groups_created as f64
        );
        tracy_client::plot!("mesh_deform::copy_ops", sample.copy_ops as f64);
        tracy_client::plot!(
            "mesh_deform::blend_dispatches",
            sample.blend_dispatches as f64
        );
        tracy_client::plot!(
            "mesh_deform::skin_dispatches",
            sample.skin_dispatches as f64
        );
        tracy_client::plot!(
            "mesh_deform::scratch_buffer_grows",
            sample.scratch_buffer_grows as f64
        );
        tracy_client::plot!(
            "mesh_deform::skipped_allocations",
            sample.skipped_allocations as f64
        );
        tracy_client::plot!("mesh_deform::cache_reuses", sample.cache_reuses as f64);
        tracy_client::plot!(
            "mesh_deform::cache_allocations",
            sample.cache_allocations as f64
        );
        tracy_client::plot!("mesh_deform::cache_grows", sample.cache_grows as f64);
        tracy_client::plot!(
            "mesh_deform::cache_evictions",
            sample.cache_evictions as f64
        );
        tracy_client::plot!(
            "mesh_deform::cache_current_frame_eviction_refusals",
            sample.cache_current_frame_eviction_refusals as f64
        );
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = sample;
    }
}

/// Returns a closure suitable for [`rayon::ThreadPoolBuilder::start_handler`].
///
/// Each Rayon worker thread registers itself as `"rayon-worker-{index}"` with the active profiler,
/// so it appears by name on the Tracy thread timeline. When the `tracy` feature is off this
/// returns a no-op closure with zero overhead.
pub fn rayon_thread_start_handler() -> impl Fn(usize) + Send + Sync + 'static {
    move |_thread_index| {
        profiling::register_thread!(&format!("rayon-worker-{_thread_index}"));
    }
}

/// Requests the GPU features needed for timestamp-query-based profiling.
///
/// Returns the subset of `{TIMESTAMP_QUERY, TIMESTAMP_QUERY_INSIDE_ENCODERS}` that the adapter
/// actually supports. Always queries the adapter regardless of Cargo features so the debug HUD's
/// frame-bracket GPU timing can use real hardware timestamps even in non-Tracy builds; the
/// `tracy`-gated [`GpuProfilerHandle`] consumes the same features for its pass-level path.
///
/// Call this in [`crate::gpu::context`]'s feature-intersection helpers and OR the result into
/// the device's requested features. `TIMESTAMP_QUERY` alone is enough for pass-level profiling;
/// `TIMESTAMP_QUERY_INSIDE_ENCODERS` unlocks encoder-level queries on adapters that offer it,
/// which is what the frame-bracket writes use to surround the entire tick of work.
pub fn timestamp_query_features_if_supported(adapter: &wgpu::Adapter) -> wgpu::Features {
    let needed = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    adapter.features() & needed
}

// ---------------------------------------------------------------------------
// PhaseQuery -- GPU timestamp query token, with a no-op stub when `tracy` is off
// ---------------------------------------------------------------------------

/// GPU timestamp query token returned by [`GpuProfilerHandle::begin_query`] /
/// [`GpuProfilerHandle::begin_pass_query`].
///
/// When the `tracy` feature is on this is [`wgpu_profiler::GpuProfilerQuery`]; when it is off
/// this is a zero-sized placeholder so call sites compile identically under both states.
#[cfg(feature = "tracy")]
pub type PhaseQuery = wgpu_profiler::GpuProfilerQuery;

/// Zero-sized placeholder for [`wgpu_profiler::GpuProfilerQuery`] when the `tracy` feature is off.
#[cfg(not(feature = "tracy"))]
pub struct PhaseQuery;

/// One resolved GPU pass timing, flattened from the `wgpu-profiler` result tree.
///
/// Emitted once per frame by [`GpuProfilerHandle::process_finished_frame`] so consumers can
/// display the per-pass breakdown without depending on `wgpu_profiler`'s types or feature gates.
#[derive(Clone, Debug)]
pub struct GpuPassEntry {
    /// Pass label captured at `begin_query` / `begin_pass_query` time.
    pub name: String,
    /// Measured GPU time in milliseconds for this pass.
    pub ms: f32,
    /// Depth in the original query tree (0 for top-level scopes, >0 for nested ones).
    pub depth: u32,
}

/// Reads the render-pass timestamp writes reserved for a pass-level query.
///
/// Forwards to [`wgpu_profiler::GpuProfilerQuery::render_pass_timestamp_writes`] when the
/// `tracy` feature is on; returns [`None`] otherwise. Feed the result into
/// [`wgpu::RenderPassDescriptor::timestamp_writes`] when opening the pass, then pair the query
/// with [`GpuProfilerHandle::end_query`] after the pass drops.
#[inline]
pub fn render_pass_timestamp_writes(
    query: Option<&PhaseQuery>,
) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
    #[cfg(feature = "tracy")]
    {
        query.and_then(wgpu_profiler::GpuProfilerQuery::render_pass_timestamp_writes)
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = query;
        None
    }
}

/// Reads the compute-pass timestamp writes reserved for a pass-level query.
///
/// Forwards to [`wgpu_profiler::GpuProfilerQuery::compute_pass_timestamp_writes`] when the
/// `tracy` feature is on; returns [`None`] otherwise. Feed the result into
/// [`wgpu::ComputePassDescriptor::timestamp_writes`] when opening the pass, then pair the query
/// with [`GpuProfilerHandle::end_query`] after the pass drops.
#[inline]
pub fn compute_pass_timestamp_writes(
    query: Option<&PhaseQuery>,
) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
    #[cfg(feature = "tracy")]
    {
        query.and_then(wgpu_profiler::GpuProfilerQuery::compute_pass_timestamp_writes)
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = query;
        None
    }
}

// ---------------------------------------------------------------------------
// GPU profiler handle -- real implementation when `tracy` is on
// ---------------------------------------------------------------------------

#[cfg(feature = "tracy")]
mod gpu_profiler_impl {
    use std::sync::atomic::{AtomicBool, Ordering};

    use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

    use super::PhaseQuery;

    /// Number of GPU profiler frames allowed to wait for readback before `wgpu-profiler` starts
    /// dropping older timing data.
    const GPU_PROFILER_PENDING_FRAMES: usize = 8;

    /// Wraps [`GpuProfiler`] and provides a GPU timestamp query interface for render and
    /// compute passes, bridging results to the Tracy GPU timeline.
    ///
    /// Created via [`GpuProfilerHandle::try_new`]; only available when the `tracy` feature is on.
    pub struct GpuProfilerHandle {
        /// Underlying query allocator, resolver, readback processor, and Tracy bridge.
        inner: GpuProfiler,
        /// Whether any query was opened since the previous successful profiler frame boundary.
        queries_opened_since_frame_end: AtomicBool,
    }

    impl GpuProfilerHandle {
        /// Creates a new handle if the device supports [`wgpu::Features::TIMESTAMP_QUERY`].
        ///
        /// Connects to the running Tracy client so GPU timestamps appear on Tracy's GPU timeline;
        /// the client is expected to be started from
        /// [`super::register_main_thread`]. If the Tracy client is unavailable
        /// (e.g. test harness), falls back to a non-Tracy-bridged profiler -- spans still resolve
        /// but do not reach the Tracy GUI.
        ///
        /// Returns [`None`] when timestamp queries are unavailable; callers fall back to CPU-only
        /// spans without any GPU timeline data.
        pub fn try_new(
            adapter: &wgpu::Adapter,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
        ) -> Option<Self> {
            let features = device.features();
            if !features.contains(wgpu::Features::TIMESTAMP_QUERY) {
                return None;
            }
            let settings = GpuProfilerSettings {
                enable_timer_queries: true,
                enable_debug_groups: true,
                max_num_pending_frames: GPU_PROFILER_PENDING_FRAMES,
            };
            let backend = adapter.get_info().backend;
            let inner_result = if tracy_client::Client::running().is_some() {
                GpuProfiler::new_with_tracy_client(settings.clone(), backend, device, queue)
            } else {
                GpuProfiler::new(device, settings.clone())
            };
            let inner = match inner_result {
                Ok(inner) => inner,
                Err(e) => {
                    logger::warn!(
                        "GPU profiler (Tracy-bridged) creation failed: {e}; falling back to unbridged"
                    );
                    match GpuProfiler::new(device, settings) {
                        Ok(inner) => inner,
                        Err(e2) => {
                            logger::warn!(
                                "GPU profiler creation failed: {e2}; GPU timeline unavailable"
                            );
                            return None;
                        }
                    }
                }
            };
            Some(Self {
                inner,
                queries_opened_since_frame_end: AtomicBool::new(false),
            })
        }

        /// Marks the active profiler frame as non-empty.
        #[inline]
        fn note_query_opened(&self) {
            self.queries_opened_since_frame_end
                .store(true, Ordering::Release);
        }

        /// Returns whether the current profiler frame has opened any GPU queries.
        #[inline]
        pub fn has_queries_opened_since_frame_end(&self) -> bool {
            self.queries_opened_since_frame_end.load(Ordering::Acquire)
        }

        /// Opens an encoder-level GPU timestamp query.
        ///
        /// Writes `WriteTimestamp` commands into `encoder` -- requires
        /// [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`]. If the adapter lacks that
        /// feature the query is silently a no-op. Prefer [`Self::begin_pass_query`] for
        /// individual passes. The returned [`PhaseQuery`] must be closed via [`Self::end_query`]
        /// before [`Self::resolve_queries`] is called.
        #[inline]
        pub fn begin_query(
            &self,
            label: impl Into<String>,
            encoder: &mut wgpu::CommandEncoder,
        ) -> PhaseQuery {
            self.note_query_opened();
            self.inner.begin_query(label, encoder)
        }

        /// Reserves a pass-level timestamp query for a single render or compute pass.
        ///
        /// The returned [`PhaseQuery`] carries `timestamp_writes` the caller must inject into the
        /// [`wgpu::RenderPassDescriptor`] / [`wgpu::ComputePassDescriptor`] via
        /// [`super::render_pass_timestamp_writes`] or [`super::compute_pass_timestamp_writes`].
        /// After the pass drops, close the query with [`Self::end_query`]. Requires only
        /// [`wgpu::Features::TIMESTAMP_QUERY`].
        #[inline]
        pub fn begin_pass_query(
            &self,
            label: impl Into<String>,
            encoder: &mut wgpu::CommandEncoder,
        ) -> PhaseQuery {
            self.note_query_opened();
            self.inner.begin_pass_query(label, encoder)
        }

        /// Closes a query previously opened with [`Self::begin_query`] or
        /// [`Self::begin_pass_query`].
        #[inline]
        pub fn end_query(&self, encoder: &mut wgpu::CommandEncoder, query: PhaseQuery) {
            self.inner.end_query(encoder, query);
        }

        /// Inserts query-resolve commands into `encoder` for all unresolved queries this frame.
        ///
        /// Call once per encoder just before [`wgpu::CommandEncoder::finish`]. The encoder used
        /// for resolution must be submitted **after** all encoders that opened queries in this
        /// profiling frame.
        #[inline]
        pub fn resolve_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
            self.inner.resolve_queries(encoder);
        }

        /// Marks the end of the current profiling frame only if at least one query was opened.
        ///
        /// Call once per render tick after all command encoders for this frame have been submitted.
        /// Empty CPU ticks are intentionally ignored so `wgpu-profiler` does not enqueue empty GPU
        /// frames that later appear as missing markers in Tracy.
        #[inline]
        pub fn end_frame_if_queries_opened(&mut self) -> bool {
            let had_queries = self
                .queries_opened_since_frame_end
                .swap(false, Ordering::AcqRel);
            if had_queries && let Err(e) = self.inner.end_frame() {
                logger::warn!("GPU profiler end_frame failed: {e}");
            }
            had_queries
        }

        /// Drains results from the oldest completed profiling frame into Tracy and returns a
        /// flattened list of per-pass timings.
        ///
        /// Call once per render tick after [`Self::end_frame_if_queries_opened`]. Results are
        /// available 1-2 frames after recording because the GPU needs to finish executing before
        /// the timestamps are readable. `timestamp_period` is from
        /// [`wgpu::Queue::get_timestamp_period`].
        ///
        /// Returns [`None`] when no frame has completed yet or when `wgpu_profiler` could not
        /// resolve the frame's timestamps. Otherwise returns a depth-annotated preorder traversal
        /// of the query tree so callers can render it as a flat table.
        #[inline]
        pub fn process_finished_frame(
            &mut self,
            timestamp_period: f32,
        ) -> Option<Vec<super::GpuPassEntry>> {
            let tree = self.inner.process_finished_frame(timestamp_period)?;
            let mut out = Vec::new();
            flatten_results(&tree, 0, &mut out);
            Some(out)
        }
    }

    /// Preorder-flattens a [`wgpu_profiler::GpuTimerQueryResult`] tree into
    /// [`super::GpuPassEntry`] rows. Skips entries with no timing data (queries that were never
    /// written, e.g. when timestamp writes were not consumed by a pass).
    fn flatten_results(
        nodes: &[wgpu_profiler::GpuTimerQueryResult],
        depth: u32,
        out: &mut Vec<super::GpuPassEntry>,
    ) {
        for node in nodes {
            if let Some(range) = node.time.as_ref() {
                let ms = ((range.end - range.start) * 1000.0) as f32;
                out.push(super::GpuPassEntry {
                    name: node.label.clone(),
                    ms,
                    depth,
                });
            }
            flatten_results(&node.nested_queries, depth + 1, out);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU profiler handle -- zero-sized stub when `tracy` is off
// ---------------------------------------------------------------------------

#[cfg(not(feature = "tracy"))]
mod gpu_profiler_stub {
    use super::PhaseQuery;

    /// Zero-sized stub that stands in for the real GPU profiler handle when the `tracy` feature
    /// is not enabled. All methods are no-ops inlined to nothing; the stub is never instantiated
    /// because [`GpuProfilerHandle::try_new`] always returns [`None`].
    pub struct GpuProfilerHandle;

    impl GpuProfilerHandle {
        /// Always returns [`None`]; GPU profiling is unavailable without the `tracy` feature.
        #[inline]
        pub fn try_new(
            _adapter: &wgpu::Adapter,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
        ) -> Option<Self> {
            None
        }

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn begin_query(
            &self,
            _label: impl Into<String>,
            _encoder: &mut wgpu::CommandEncoder,
        ) -> PhaseQuery {
            PhaseQuery
        }

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn begin_pass_query(
            &self,
            _label: impl Into<String>,
            _encoder: &mut wgpu::CommandEncoder,
        ) -> PhaseQuery {
            PhaseQuery
        }

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn end_query(&self, _encoder: &mut wgpu::CommandEncoder, _query: PhaseQuery) {}

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn resolve_queries(&mut self, _encoder: &mut wgpu::CommandEncoder) {}

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn has_queries_opened_since_frame_end(&self) -> bool {
            false
        }

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn end_frame_if_queries_opened(&mut self) -> bool {
            false
        }

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        ///
        /// Always returns [`None`] because the stub never opens queries.
        #[inline]
        pub fn process_finished_frame(
            &mut self,
            _timestamp_period: f32,
        ) -> Option<Vec<super::GpuPassEntry>> {
            None
        }
    }
}

#[cfg(feature = "tracy")]
pub use gpu_profiler_impl::GpuProfilerHandle;

#[cfg(not(feature = "tracy"))]
pub use gpu_profiler_stub::GpuProfilerHandle;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that `rayon_thread_start_handler` produces a valid closure that does not panic
    /// when called with arbitrary thread indices.
    #[test]
    fn rayon_start_handler_does_not_panic_for_any_index() {
        let handler = rayon_thread_start_handler();
        handler(0);
        handler(1);
        handler(usize::MAX);
    }

    /// Confirms that the public surface of this module compiles and is callable without the
    /// `tracy` feature active. All calls must be no-ops; the test itself is the compile check.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn stubs_are_accessible_without_tracy_feature() {
        register_main_thread();
        emit_frame_mark();
        emit_render_submit_frame_mark();
        plot_fps_cap_active(240);
        plot_window_focused(true);
        plot_surface_acquire_outcome(true, false, false);
        plot_event_loop_wait_ms(11.0);
        plot_event_loop_idle_ms(11.0);
        let mut profiler = GpuProfilerHandle;
        assert!(!profiler.has_queries_opened_since_frame_end());
        assert!(!profiler.end_frame_if_queries_opened());
        let _ = rayon_thread_start_handler();
    }

    /// Verifies that `timestamp_query_features_if_supported` has the correct function signature
    /// and can be referenced as a function pointer when the `tracy` feature is off.
    ///
    /// The `cfg(not(feature = "tracy"))` branch returns `wgpu::Features::empty()` without ever
    /// calling `adapter.features()`, so no real wgpu instance is required.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn timestamp_features_fn_signature_compiles_without_tracy() {
        let _: fn(&wgpu::Adapter) -> wgpu::Features = timestamp_query_features_if_supported;
    }

    /// `register_main_thread` and `emit_frame_mark` must be safely callable more than once per
    /// process; calling them repeatedly should never panic under any feature configuration.
    #[test]
    fn thread_registration_and_frame_mark_are_idempotent() {
        register_main_thread();
        register_main_thread();
        emit_frame_mark();
        emit_frame_mark();
    }

    /// The no-tracy [`PhaseQuery`] placeholder is zero-sized so its presence in per-phase structs
    /// cannot regress memory layout when profiling is disabled.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn phase_query_stub_is_zero_sized() {
        assert_eq!(size_of::<PhaseQuery>(), 0);
    }

    /// The no-tracy [`GpuProfilerHandle`] stub is also zero-sized; construction is unreachable via
    /// [`GpuProfilerHandle::try_new`] (always returns [`None`]), so the placeholder must stay free.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn gpu_profiler_handle_stub_is_zero_sized() {
        assert_eq!(size_of::<GpuProfilerHandle>(), 0);
    }

    /// The no-tracy `render_pass_timestamp_writes` helper must always return `None` regardless
    /// of what `query` is -- the `PhaseQuery` placeholder carries no data to reserve writes from.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn render_pass_timestamp_writes_is_none_without_tracy() {
        let q = PhaseQuery;
        assert!(render_pass_timestamp_writes(Some(&q)).is_none());
        assert!(render_pass_timestamp_writes(None).is_none());
    }

    /// The no-tracy `compute_pass_timestamp_writes` helper must always return `None`.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn compute_pass_timestamp_writes_is_none_without_tracy() {
        let q = PhaseQuery;
        assert!(compute_pass_timestamp_writes(Some(&q)).is_none());
        assert!(compute_pass_timestamp_writes(None).is_none());
    }
}
