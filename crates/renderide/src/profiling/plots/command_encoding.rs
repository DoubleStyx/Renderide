//! Tracy plots for per-frame render-graph command-encoding timings and pressure counters.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;

/// CPU timings and counts for one render-graph command-encoding slice.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommandEncodingProfileSample {
    /// Number of views encoded by the graph.
    pub view_count: usize,
    /// Number of command buffers submitted in the batch.
    pub command_buffers: usize,
    /// Command recording path selected by the graph executor.
    pub recording_path: u64,
    /// Command recording strategy selected by the graph executor.
    pub recording_strategy: u64,
    /// Requested command recording mode from renderer config.
    pub requested_recording_mode: u64,
    /// Estimated draw count visible to per-view command recording.
    pub estimated_per_view_draw_count: usize,
    /// Estimated draw-equivalent work visible to per-view command recording diagnostics.
    pub estimated_per_view_record_work: usize,
    /// Whether automatic per-view recording would have used Rayon.
    pub auto_per_view_record_admitted: u64,
    /// Whether the effective per-view recording plan uses Rayon.
    pub per_view_record_admitted: u64,
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
    /// Transient texture-view cache hits during this frame.
    pub transient_texture_view_hits: usize,
    /// Transient texture-view cache misses during this frame.
    pub transient_texture_view_misses: usize,
    /// Transient buffer allocation misses during this frame.
    pub transient_buffer_misses: usize,
    /// Deferred upload writes drained before submit.
    pub upload_writes: usize,
    /// Deferred upload payload bytes drained before submit.
    pub upload_bytes: usize,
    /// Upload bytes staged through persistent arena slots.
    pub upload_persistent_staging_bytes: u64,
    /// Persistent arena slot reuse count.
    pub upload_persistent_slot_reuses: usize,
    /// Persistent arena slot allocation or growth count.
    pub upload_persistent_slot_grows: usize,
    /// Upload bytes staged through temporary fallback buffers.
    pub upload_temporary_staging_bytes: u64,
    /// Temporary staging fallback count caused by all persistent slots being unavailable.
    pub upload_temporary_staging_fallbacks: usize,
    /// Staged writes replayed through queue writes because no staging buffer fit.
    pub upload_oversized_queue_fallback_writes: usize,
    /// Bytes allocated across persistent upload arena slots.
    pub upload_arena_capacity_bytes: u64,
    /// Persistent upload arena slots mapped and available for writes.
    pub upload_arena_free_slots: usize,
    /// Persistent upload arena slots currently in flight.
    pub upload_arena_in_flight_slots: usize,
    /// Persistent upload arena slots waiting for remap completion.
    pub upload_arena_remapping_slots: usize,
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
    /// CPU time spent encoding the single-swapchain graph command path.
    pub single_swapchain_encode_ms: f64,
    /// CPU time spent inside the single-swapchain graph command encoder `finish`.
    pub single_swapchain_finish_ms: f64,
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
    /// Runtime graph passes skipped by `should_record`.
    pub graph_runtime_skipped_passes: usize,
    /// Runtime logical raster passes recorded.
    pub graph_recorded_raster_passes: usize,
    /// Runtime logical compute passes recorded.
    pub graph_recorded_compute_passes: usize,
    /// Runtime logical encoder passes recorded.
    pub graph_recorded_encoder_passes: usize,
    /// WGPU render-pass encoders opened by the graph.
    pub graph_opened_render_passes: usize,
    /// Explicit texture copies recorded by the graph.
    pub graph_copy_count: usize,
    /// Explicit texture copies skipped by the graph.
    pub graph_skipped_copy_count: usize,
    /// Manual or attachment resolves recorded by the graph.
    pub graph_resolve_count: usize,
    /// Manual or attachment resolves skipped by the graph.
    pub graph_skipped_resolve_count: usize,
    /// Estimated render-graph bandwidth in bytes.
    pub graph_estimated_bandwidth_bytes: u64,
}

/// Records command-encoding timings and pressure counters for the current frame.
pub fn plot_command_encoding(sample: &CommandEncodingProfileSample) {
    plot_pass_counts(sample);
    plot_upload_traffic(sample);
    plot_encoding_timings(sample);
    plot_world_mesh_stats(sample);
    plot_render_graph_stats(sample);
}

fn plot_pass_counts(sample: &CommandEncodingProfileSample) {
    tracy_plot!("command_encoding::views", sample.view_count as f64);
    tracy_plot!(
        "command_encoding::command_buffers",
        sample.command_buffers as f64
    );
    tracy_plot!(
        "command_encoding::recording_path",
        sample.recording_path as f64
    );
    tracy_plot!(
        "command_encoding::recording_strategy",
        sample.recording_strategy as f64
    );
    tracy_plot!(
        "command_encoding::requested_recording_mode",
        sample.requested_recording_mode as f64
    );
    tracy_plot!(
        "command_encoding::estimated_per_view_draw_count",
        sample.estimated_per_view_draw_count as f64
    );
    tracy_plot!(
        "command_encoding::estimated_per_view_record_work",
        sample.estimated_per_view_record_work as f64
    );
    tracy_plot!(
        "command_encoding::auto_per_view_record_admitted",
        sample.auto_per_view_record_admitted as f64
    );
    tracy_plot!(
        "command_encoding::per_view_record_admitted",
        sample.per_view_record_admitted as f64
    );
    tracy_plot!(
        "command_encoding::frame_global_passes",
        sample.frame_global_passes as f64
    );
    tracy_plot!(
        "command_encoding::per_view_passes",
        sample.per_view_passes as f64
    );
    tracy_plot!(
        "command_encoding::transient_textures",
        sample.transient_textures as f64
    );
    tracy_plot!(
        "command_encoding::transient_texture_slots",
        sample.transient_texture_slots as f64
    );
    tracy_plot!(
        "command_encoding::transient_texture_misses",
        sample.transient_texture_misses as f64
    );
    tracy_plot!(
        "command_encoding::transient_texture_view_hits",
        sample.transient_texture_view_hits as f64
    );
    tracy_plot!(
        "command_encoding::transient_texture_view_misses",
        sample.transient_texture_view_misses as f64
    );
    tracy_plot!(
        "command_encoding::transient_buffer_misses",
        sample.transient_buffer_misses as f64
    );
}

fn plot_upload_traffic(sample: &CommandEncodingProfileSample) {
    tracy_plot!(
        "command_encoding::upload_writes",
        sample.upload_writes as f64
    );
    tracy_plot!("command_encoding::upload_bytes", sample.upload_bytes as f64);
    tracy_plot!(
        "command_encoding::upload_persistent_staging_bytes",
        sample.upload_persistent_staging_bytes as f64
    );
    tracy_plot!(
        "command_encoding::upload_persistent_slot_reuses",
        sample.upload_persistent_slot_reuses as f64
    );
    tracy_plot!(
        "command_encoding::upload_persistent_slot_grows",
        sample.upload_persistent_slot_grows as f64
    );
    tracy_plot!(
        "command_encoding::upload_temporary_staging_bytes",
        sample.upload_temporary_staging_bytes as f64
    );
    tracy_plot!(
        "command_encoding::upload_temporary_staging_fallbacks",
        sample.upload_temporary_staging_fallbacks as f64
    );
    tracy_plot!(
        "command_encoding::upload_oversized_queue_fallback_writes",
        sample.upload_oversized_queue_fallback_writes as f64
    );
    tracy_plot!(
        "command_encoding::upload_arena_capacity_bytes",
        sample.upload_arena_capacity_bytes as f64
    );
    tracy_plot!(
        "command_encoding::upload_arena_free_slots",
        sample.upload_arena_free_slots as f64
    );
    tracy_plot!(
        "command_encoding::upload_arena_in_flight_slots",
        sample.upload_arena_in_flight_slots as f64
    );
    tracy_plot!(
        "command_encoding::upload_arena_remapping_slots",
        sample.upload_arena_remapping_slots as f64
    );
}

fn plot_encoding_timings(sample: &CommandEncodingProfileSample) {
    tracy_plot!("command_encoding::pre_resolve_ms", sample.pre_resolve_ms);
    tracy_plot!(
        "command_encoding::prepare_resources_ms",
        sample.prepare_resources_ms
    );
    tracy_plot!(
        "command_encoding::frame_global_encode_ms",
        sample.frame_global_encode_ms
    );
    tracy_plot!(
        "command_encoding::frame_global_finish_ms",
        sample.frame_global_finish_ms
    );
    tracy_plot!(
        "command_encoding::per_view_encode_ms",
        sample.per_view_encode_ms
    );
    tracy_plot!(
        "command_encoding::per_view_finish_ms",
        sample.per_view_finish_ms
    );
    tracy_plot!("command_encoding::upload_drain_ms", sample.upload_drain_ms);
    tracy_plot!(
        "command_encoding::upload_finish_ms",
        sample.upload_finish_ms
    );
    tracy_plot!(
        "command_encoding::single_swapchain_encode_ms",
        sample.single_swapchain_encode_ms
    );
    tracy_plot!(
        "command_encoding::single_swapchain_finish_ms",
        sample.single_swapchain_finish_ms
    );
    tracy_plot!(
        "command_encoding::command_batch_assembly_ms",
        sample.command_batch_assembly_ms
    );
    tracy_plot!(
        "command_encoding::submit_enqueue_ms",
        sample.submit_enqueue_ms
    );
    tracy_plot!(
        "command_encoding::max_encoder_finish_ms",
        sample.max_encoder_finish_ms
    );
}

fn plot_world_mesh_stats(sample: &CommandEncodingProfileSample) {
    tracy_plot!(
        "command_encoding::world_mesh_draws",
        sample.world_mesh_draws as f64
    );
    tracy_plot!(
        "command_encoding::world_mesh_instance_batches",
        sample.world_mesh_instance_batches as f64
    );
    tracy_plot!(
        "command_encoding::world_mesh_pipeline_pass_submits",
        sample.world_mesh_pipeline_pass_submits as f64
    );
}

fn plot_render_graph_stats(sample: &CommandEncodingProfileSample) {
    tracy_plot!(
        "command_encoding::graph_runtime_skipped_passes",
        sample.graph_runtime_skipped_passes as f64
    );
    tracy_plot!(
        "command_encoding::graph_recorded_raster_passes",
        sample.graph_recorded_raster_passes as f64
    );
    tracy_plot!(
        "command_encoding::graph_recorded_compute_passes",
        sample.graph_recorded_compute_passes as f64
    );
    tracy_plot!(
        "command_encoding::graph_recorded_encoder_passes",
        sample.graph_recorded_encoder_passes as f64
    );
    tracy_plot!(
        "command_encoding::graph_opened_render_passes",
        sample.graph_opened_render_passes as f64
    );
    tracy_plot!(
        "command_encoding::graph_copy_count",
        sample.graph_copy_count as f64
    );
    tracy_plot!(
        "command_encoding::graph_skipped_copy_count",
        sample.graph_skipped_copy_count as f64
    );
    tracy_plot!(
        "command_encoding::graph_resolve_count",
        sample.graph_resolve_count as f64
    );
    tracy_plot!(
        "command_encoding::graph_skipped_resolve_count",
        sample.graph_skipped_resolve_count as f64
    );
    tracy_plot!(
        "command_encoding::graph_estimated_bandwidth_bytes",
        sample.graph_estimated_bandwidth_bytes as f64
    );
}
