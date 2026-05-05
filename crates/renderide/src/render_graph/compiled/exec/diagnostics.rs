//! Command encoding timing diagnostics for compiled render graph execution.

use std::sync::atomic::{AtomicU64, Ordering};

use super::super::super::frame_upload_batch::FrameUploadBatchStats;
use super::super::super::pool::TransientPoolMetrics;
use super::{
    CompiledRenderGraph, RecordedPerViewBatch, SubmitFrameBatchStats, TimedCommandBuffer,
    WorldMeshCommandStats,
};

const SLOW_ENCODER_FINISH_WARN_MS: f64 = 2.0;
static COMMAND_ENCODING_SLOW_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Per-frame graph command-encoding diagnostics.
#[derive(Clone, Copy, Debug)]
pub(super) struct CommandEncodingDiagnostics {
    pub(super) view_count: usize,
    pub(super) command_buffers: usize,
    pub(super) frame_global_passes: usize,
    pub(super) per_view_passes: usize,
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
    pub(super) world_mesh: WorldMeshCommandStats,
}

impl CommandEncodingDiagnostics {
    pub(super) fn new(graph: &CompiledRenderGraph, view_count: usize) -> Self {
        Self {
            view_count,
            command_buffers: 0,
            frame_global_passes: graph.schedule.frame_global_pass_indices().len(),
            per_view_passes: graph.schedule.per_view_pass_indices().len(),
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
            world_mesh: WorldMeshCommandStats::default(),
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
        self.world_mesh = batch.world_mesh;
    }

    pub(super) fn apply_submit(&mut self, submit: SubmitFrameBatchStats) {
        self.command_buffers = submit.command_buffer_count;
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
        crate::profiling::plot_command_encoding(crate::profiling::CommandEncodingProfileSample {
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
            world_mesh_draws: self.world_mesh.draws,
            world_mesh_instance_batches: self.world_mesh.instance_batches,
            world_mesh_pipeline_pass_submits: self.world_mesh.pipeline_pass_submits,
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
            "slow command encoder finish: max_finish_ms={:.3} frame_global_finish_ms={:.3} per_view_max_finish_ms={:.3} upload_finish_ms={:.3} views={} command_buffers={} passes(frame_global/per_view)={}/{} transients(textures/slots)={}/{} transient_misses(tex/buf)={}/{} uploads(writes/bytes/staged/fallback)={}/{}/{}/{} timings_ms(pre_resolve/prepare/frame_global_encode/per_view_encode/upload_drain/assemble/submit)={:.3}/{:.3}/{:.3}/{:.3}/{:.3}/{:.3}/{:.3} world_mesh(draws/instance_batches/pipeline_pass_submits)={}/{}/{}",
            max_finish_ms,
            self.frame_global_finish_ms,
            self.per_view_max_finish_ms,
            self.upload_finish_ms,
            self.view_count,
            self.command_buffers,
            self.frame_global_passes,
            self.per_view_passes,
            self.transient_texture_count,
            self.transient_texture_slots,
            self.transient_delta.texture_misses,
            self.transient_delta.buffer_misses,
            self.upload_stats.writes,
            self.upload_stats.bytes,
            self.upload_stats.staged_writes,
            self.upload_stats.fallback_writes,
            self.pre_resolve_ms,
            self.prepare_resources_ms,
            self.frame_global_encode_ms,
            self.per_view_encode_ms,
            self.upload_drain_ms,
            self.command_batch_assembly_ms,
            self.submit_enqueue_ms,
            self.world_mesh.draws,
            self.world_mesh.instance_batches,
            self.world_mesh.pipeline_pass_submits,
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
        Self {
            texture_misses: after.texture_misses.saturating_sub(before.texture_misses),
            buffer_misses: after.buffer_misses.saturating_sub(before.buffer_misses),
        }
    }
}
