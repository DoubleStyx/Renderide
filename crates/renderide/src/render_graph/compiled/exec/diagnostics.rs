//! Command encoding timing diagnostics for compiled render graph execution.

use std::sync::atomic::{AtomicU64, Ordering};

use super::super::super::frame_upload_batch::FrameUploadBatchStats;
use super::super::super::pool::TransientPoolMetrics;
use super::{CompiledRenderGraph, RecordedPerViewBatch, SubmitFrameBatchStats, TimedCommandBuffer};
use crate::diagnostics::gpu_flight_recorder::GpuFlightEventKind;
use crate::gpu::GpuContext;
use crate::render_graph::blackboard::GraphCommandStats;

const SLOW_ENCODER_FINISH_WARN_MS: f64 = 2.0;
static COMMAND_ENCODING_SLOW_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Per-frame graph command-encoding diagnostics.
#[derive(Clone, Copy, Debug)]
pub(super) struct CommandEncodingDiagnostics {
    pub(super) view_count: usize,
    pub(super) target_is_swapchain: bool,
    pub(super) command_buffers: usize,
    pub(super) frame_global_passes: usize,
    pub(super) per_view_passes: usize,
    pub(super) scheduler_passes: usize,
    pub(super) scheduler_waves: usize,
    pub(super) scheduler_largest_wave: usize,
    pub(super) scheduler_submit_steps: usize,
    pub(super) scheduler_upload_phases: usize,
    pub(super) scheduler_resource_events: usize,
    pub(super) scheduler_import_final_accesses: usize,
    pub(super) scheduler_merge_groups: usize,
    pub(super) transient_texture_count: usize,
    pub(super) transient_texture_slots: usize,
    pub(super) pre_resolve_ms: f64,
    pub(super) prepare_resources_ms: f64,
    pub(super) frame_global_encode_ms: f64,
    pub(super) frame_global_finish_ms: f64,
    pub(super) per_view_encode_ms: f64,
    pub(super) per_view_finish_ms: f64,
    pub(super) per_view_max_finish_ms: f64,
    pub(super) upload_drain_ms: f64,
    pub(super) upload_finish_ms: f64,
    pub(super) command_batch_assembly_ms: f64,
    pub(super) submit_enqueue_ms: f64,
    pub(super) transient_delta: TransientPoolMetricsDelta,
    pub(super) upload_stats: FrameUploadBatchStats,
    pub(super) command_stats: GraphCommandStats,
}

impl CommandEncodingDiagnostics {
    pub(super) fn new(graph: &CompiledRenderGraph, view_count: usize) -> Self {
        Self {
            view_count,
            target_is_swapchain: false,
            command_buffers: 0,
            frame_global_passes: graph.schedule_hud.frame_global_count,
            per_view_passes: graph.schedule_hud.per_view_count,
            scheduler_passes: graph.schedule_hud.pass_count,
            scheduler_waves: graph.schedule_hud.wave_count,
            scheduler_largest_wave: graph
                .schedule_hud
                .passes_per_wave
                .iter()
                .copied()
                .max()
                .unwrap_or(0),
            scheduler_submit_steps: graph.schedule.submit_steps.len(),
            scheduler_upload_phases: graph.schedule.upload_phases.len(),
            scheduler_resource_events: graph.schedule.resource_events.len(),
            scheduler_import_final_accesses: graph.schedule.imported_final_accesses.len(),
            scheduler_merge_groups: graph.schedule.render_pass_merge_groups.len(),
            transient_texture_count: graph.compile_stats.transient_texture_count,
            transient_texture_slots: graph.compile_stats.transient_texture_slots,
            pre_resolve_ms: 0.0,
            prepare_resources_ms: 0.0,
            frame_global_encode_ms: 0.0,
            frame_global_finish_ms: 0.0,
            per_view_encode_ms: 0.0,
            per_view_finish_ms: 0.0,
            per_view_max_finish_ms: 0.0,
            upload_drain_ms: 0.0,
            upload_finish_ms: 0.0,
            command_batch_assembly_ms: 0.0,
            submit_enqueue_ms: 0.0,
            transient_delta: TransientPoolMetricsDelta::default(),
            upload_stats: FrameUploadBatchStats::default(),
            command_stats: GraphCommandStats::default(),
        }
    }

    pub(super) fn apply_frame_global(&mut self, command: &TimedCommandBuffer) {
        self.frame_global_encode_ms = command.encode_ms;
        self.frame_global_finish_ms = command.finish_ms;
    }

    pub(super) fn apply_per_view(&mut self, batch: &RecordedPerViewBatch) {
        self.per_view_encode_ms = batch.encode_ms;
        self.per_view_finish_ms = batch.finish_ms;
        self.per_view_max_finish_ms = batch.max_finish_ms;
        self.command_stats = batch.command_stats;
    }

    pub(super) fn apply_submit(&mut self, submit: SubmitFrameBatchStats) {
        self.command_buffers = submit.command_buffer_count;
        self.target_is_swapchain = submit.target_is_swapchain;
        self.upload_drain_ms = submit.upload_drain_ms;
        self.upload_finish_ms = submit.upload_finish_ms;
        self.command_batch_assembly_ms = submit.command_batch_assembly_ms;
        self.submit_enqueue_ms = submit.submit_enqueue_ms;
        self.upload_stats = submit.upload_stats;
    }

    pub(super) fn max_encoder_finish_ms(&self) -> f64 {
        self.frame_global_finish_ms
            .max(self.per_view_max_finish_ms)
            .max(self.upload_finish_ms)
    }

    pub(super) fn plot(&self) {
        let sample = crate::profiling::CommandEncodingProfileSample {
            view_count: self.view_count,
            command_buffers: self.command_buffers,
            frame_global_passes: self.frame_global_passes,
            per_view_passes: self.per_view_passes,
            transient_textures: self.transient_texture_count,
            transient_texture_slots: self.transient_texture_slots,
            transient_texture_misses: self.transient_delta.texture_misses,
            transient_buffer_misses: self.transient_delta.buffer_misses,
            upload_writes: self.upload_stats.writes,
            upload_bytes: self.upload_stats.bytes,
            upload_persistent_staging_bytes: self.upload_stats.persistent_staging_bytes,
            upload_persistent_slot_reuses: self.upload_stats.persistent_slot_reuses,
            upload_persistent_slot_grows: self.upload_stats.persistent_slot_grows,
            upload_temporary_staging_bytes: self.upload_stats.temporary_staging_bytes,
            upload_temporary_staging_fallbacks: self.upload_stats.temporary_staging_fallbacks,
            upload_oversized_queue_fallback_writes: self
                .upload_stats
                .oversized_queue_fallback_writes,
            upload_arena_capacity_bytes: self.upload_stats.arena_capacity_bytes,
            upload_arena_free_slots: self.upload_stats.arena_free_slots,
            upload_arena_in_flight_slots: self.upload_stats.arena_in_flight_slots,
            upload_arena_remapping_slots: self.upload_stats.arena_remapping_slots,
            pre_resolve_ms: self.pre_resolve_ms,
            prepare_resources_ms: self.prepare_resources_ms,
            frame_global_encode_ms: self.frame_global_encode_ms,
            frame_global_finish_ms: self.frame_global_finish_ms,
            per_view_encode_ms: self.per_view_encode_ms,
            per_view_finish_ms: self.per_view_finish_ms,
            upload_drain_ms: self.upload_drain_ms,
            upload_finish_ms: self.upload_finish_ms,
            command_batch_assembly_ms: self.command_batch_assembly_ms,
            submit_enqueue_ms: self.submit_enqueue_ms,
            max_encoder_finish_ms: self.max_encoder_finish_ms(),
            world_mesh_draws: self.command_stats.draw_items,
            world_mesh_instance_batches: self.command_stats.instance_batches,
            world_mesh_pipeline_pass_submits: self.command_stats.pipeline_pass_submits,
        };
        crate::profiling::plot_command_encoding(&sample);
    }

    pub(super) fn record_flight_event(&self, gpu: &GpuContext) {
        gpu.record_gpu_flight_event(GpuFlightEventKind::RenderGraphSubmit {
            swapchain: self.target_is_swapchain,
            views: self.view_count,
            command_buffers: self.command_buffers,
            draw_items: self.command_stats.draw_items,
            pipeline_pass_submits: self.command_stats.pipeline_pass_submits,
            upload_bytes: self.upload_stats.bytes,
            transient_texture_misses: self.transient_delta.texture_misses,
            transient_buffer_misses: self.transient_delta.buffer_misses,
        });
    }

    pub(super) fn log_if_slow(&self) {
        let max_finish_ms = self.max_encoder_finish_ms();
        if max_finish_ms < SLOW_ENCODER_FINISH_WARN_MS {
            return;
        }
        let count = COMMAND_ENCODING_SLOW_LOG_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        if count > 5 && !count.is_multiple_of(120) {
            return;
        }
        logger::warn!(
            "slow command encoder finish: max_finish_ms={:.3} frame_global_finish_ms={:.3} per_view_max_finish_ms={:.3} upload_finish_ms={:.3} views={} command_buffers={} passes(frame_global/per_view)={}/{} scheduler(passes/waves/largest_wave/submit_steps/upload_phases/resource_events/import_finals/merge_groups)={}/{}/{}/{}/{}/{}/{}/{} transients(textures/slots)={}/{} transient_misses(tex/buf)={}/{} uploads(writes/bytes/staged/fallback)={}/{}/{}/{} upload_arena(persistent_bytes/temp_bytes/reuses/grows/temp_fallbacks/oversized_queue/capacity/free/inflight/remapping)={}/{}/{}/{}/{}/{}/{}/{}/{}/{} timings_ms(pre_resolve/prepare/frame_global_encode/per_view_encode/upload_drain/assemble/submit)={:.3}/{:.3}/{:.3}/{:.3}/{:.3}/{:.3}/{:.3} commands(draws/instance_batches/pipeline_pass_submits)={}/{}/{}",
            max_finish_ms,
            self.frame_global_finish_ms,
            self.per_view_max_finish_ms,
            self.upload_finish_ms,
            self.view_count,
            self.command_buffers,
            self.frame_global_passes,
            self.per_view_passes,
            self.scheduler_passes,
            self.scheduler_waves,
            self.scheduler_largest_wave,
            self.scheduler_submit_steps,
            self.scheduler_upload_phases,
            self.scheduler_resource_events,
            self.scheduler_import_final_accesses,
            self.scheduler_merge_groups,
            self.transient_texture_count,
            self.transient_texture_slots,
            self.transient_delta.texture_misses,
            self.transient_delta.buffer_misses,
            self.upload_stats.writes,
            self.upload_stats.bytes,
            self.upload_stats.staged_writes,
            self.upload_stats.fallback_writes,
            self.upload_stats.persistent_staging_bytes,
            self.upload_stats.temporary_staging_bytes,
            self.upload_stats.persistent_slot_reuses,
            self.upload_stats.persistent_slot_grows,
            self.upload_stats.temporary_staging_fallbacks,
            self.upload_stats.oversized_queue_fallback_writes,
            self.upload_stats.arena_capacity_bytes,
            self.upload_stats.arena_free_slots,
            self.upload_stats.arena_in_flight_slots,
            self.upload_stats.arena_remapping_slots,
            self.pre_resolve_ms,
            self.prepare_resources_ms,
            self.frame_global_encode_ms,
            self.per_view_encode_ms,
            self.upload_drain_ms,
            self.command_batch_assembly_ms,
            self.submit_enqueue_ms,
            self.command_stats.draw_items,
            self.command_stats.instance_batches,
            self.command_stats.pipeline_pass_submits,
        );
    }
}

/// Transient-pool hit/miss deltas for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TransientPoolMetricsDelta {
    pub(super) texture_misses: usize,
    pub(super) buffer_misses: usize,
}

impl TransientPoolMetricsDelta {
    pub(super) fn from_metrics(before: TransientPoolMetrics, after: TransientPoolMetrics) -> Self {
        let texture_delta = after.texture_cache.delta_since(before.texture_cache);
        let buffer_delta = after.buffer_cache.delta_since(before.buffer_cache);
        Self {
            texture_misses: saturating_usize(texture_delta.misses),
            buffer_misses: saturating_usize(buffer_delta.misses),
        }
    }
}

fn saturating_usize(value: u64) -> usize {
    if value > usize::MAX as u64 {
        usize::MAX
    } else {
        value as usize
    }
}
