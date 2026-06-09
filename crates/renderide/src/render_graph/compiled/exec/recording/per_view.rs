//! Per-view command-buffer recording.

mod batch_plan;
mod frame_params;
mod offscreen_copy;

use hashbrown::HashMap;
use std::ops::Range;
use std::time::Instant;

use crate::camera::HostCameraFrame;
use crate::diagnostics::PerViewHudOutputsSlot;
use crate::graph_inputs::FrameViewClear;
use crate::render_graph::blackboard::{Blackboard, GraphCommandStatsSlot};
use crate::render_graph::context::GraphResolvedResources;
use crate::render_graph::error::GraphExecuteError;
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::render_graph::pass::PassPhase;
use crate::render_graph::schedule::{
    RecordingBatch, RecordingBatchKind, RecordingUnit, RenderPassMaterializationGroup,
};
use crate::shared::RenderingContext;

use super::super::super::{CompiledRenderGraph, ResolvedView, ViewPostProcessing};
use super::super::recording_path::GraphCommandRecordingStrategy;
use super::super::{
    GraphResolveKey, PerViewEncodeOutput, PerViewRecordShared, PerViewWorkItem,
    PreparedPerViewFrameInput, ResolvedOffscreenColorCopy, elapsed_ms,
};
use super::{PassExecution, PassGpuInputs, PassRecordTargets, PassViewInputs, PhaseRecordingScope};

use batch_plan::{next_phase_batch_index, serial_batch_run_end};
use frame_params::build_per_view_frame_params;
use offscreen_copy::{record_offscreen_color_copy, record_offscreen_color_copy_command};

struct PerViewUnitEncodeOutput {
    command_buffer: wgpu::CommandBuffer,
    encode_ms: f64,
    finish_ms: f64,
    command_stats: crate::render_graph::blackboard::GraphCommandStats,
}

struct PerViewInlineEncodeOutput {
    encode_ms: f64,
    command_stats: crate::render_graph::blackboard::GraphCommandStats,
    recorded_gpu_query: bool,
}

#[derive(Clone, Copy)]
struct PerViewRuntimeInputs<'a> {
    host_camera: &'a HostCameraFrame,
    render_context: RenderingContext,
    frame_time_seconds: f32,
    clear: FrameViewClear,
    post_processing: ViewPostProcessing,
}

#[derive(Clone, Copy)]
struct PerViewRecordingScope<'a> {
    shared: &'a PerViewRecordShared<'a>,
    view_idx: usize,
    view: PassViewInputs<'a>,
}

#[derive(Clone, Copy)]
struct PerViewFrameReuse<'a> {
    frame_input: &'a PreparedPerViewFrameInput,
    runtime: PerViewRuntimeInputs<'a>,
}

struct PerViewLiveState<'a, 'frame> {
    frame_params: &'frame mut crate::graph_inputs::GraphPassFrame<'a>,
    blackboard: &'frame mut Blackboard,
}

struct PerViewSchedulerInputs<'a> {
    upload_batch: &'a FrameUploadBatch,
    allow_parallel_batches: bool,
    profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

impl<'a> PerViewLiveState<'a, '_> {
    fn reborrow(&mut self) -> PerViewLiveState<'a, '_> {
        PerViewLiveState {
            frame_params: &mut *self.frame_params,
            blackboard: &mut *self.blackboard,
        }
    }
}

impl CompiledRenderGraph {
    /// Records the per-view pass phase into one command buffer for `work_item`.
    pub(in crate::render_graph::compiled::exec) fn record_one_view(
        &self,
        shared: &PerViewRecordShared<'_>,
        work_item: PerViewWorkItem,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        strategy: GraphCommandRecordingStrategy,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<PerViewEncodeOutput, GraphExecuteError> {
        profiling::scope!("graph::per_view");
        let encode_start = Instant::now();
        let PerViewWorkItem {
            view_idx,
            host_camera,
            render_context,
            frame_time_seconds,
            clear,
            post_processing,
            initial_blackboard,
            resolved,
            frame_input,
            ..
        } = work_item;

        let resolved_view = resolved.as_resolved();
        let resolved_resources =
            self.resolve_per_view_graph_resources(shared, &resolved_view, transient_by_key)?;
        let graph_resources: &GraphResolvedResources = &resolved_resources;
        let scope = PerViewRecordingScope {
            shared,
            view_idx,
            view: PassViewInputs {
                resolved: &resolved_view,
                graph_resources,
            },
        };
        let runtime = PerViewRuntimeInputs {
            host_camera: &host_camera,
            render_context,
            frame_time_seconds,
            clear,
            post_processing,
        };

        let mut frame_params =
            build_per_view_frame_params(shared, &frame_input, &resolved_view, runtime);
        let mut view_blackboard =
            self.build_per_view_blackboard(&frame_params, graph_resources, initial_blackboard);
        let state = PerViewLiveState {
            frame_params: &mut frame_params,
            blackboard: &mut view_blackboard,
        };

        let has_scheduler_parallel_batches = self
            .schedule
            .recording_plan
            .phase_has_parallel_batches(PassPhase::PerView);
        let (command_buffers, command_stats, encode_ms, finish_ms) =
            if has_scheduler_parallel_batches {
                self.record_one_view_scheduler(
                    scope,
                    PerViewFrameReuse {
                        frame_input: &frame_input,
                        runtime,
                    },
                    state,
                    resolved.offscreen_color_copy.as_ref(),
                    PerViewSchedulerInputs {
                        upload_batch,
                        allow_parallel_batches: strategy
                            == GraphCommandRecordingStrategy::InViewParallel,
                        profiler,
                    },
                )?
            } else {
                self.record_one_view_flat(
                    scope,
                    state,
                    resolved.offscreen_color_copy.as_ref(),
                    upload_batch,
                    profiler,
                )?
            };
        let hud_outputs = view_blackboard.take::<PerViewHudOutputsSlot>();
        let encode_ms = encode_ms.max(elapsed_ms(encode_start));
        Ok(PerViewEncodeOutput {
            command_buffers,
            hud_outputs,
            encode_ms,
            finish_ms,
            command_stats,
        })
    }

    /// Records one serial per-view work item into a caller-owned command encoder.
    pub(in crate::render_graph::compiled::exec) fn record_one_view_into_encoder(
        &self,
        shared: &PerViewRecordShared<'_>,
        work_item: PerViewWorkItem,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        encoder: &mut wgpu::CommandEncoder,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<PerViewEncodeOutput, GraphExecuteError> {
        profiling::scope!("graph::per_view::single_encoder");
        debug_assert!(
            !self
                .schedule
                .recording_plan
                .phase_has_parallel_batches(PassPhase::PerView)
        );
        let PerViewWorkItem {
            view_idx,
            host_camera,
            render_context,
            frame_time_seconds,
            clear,
            post_processing,
            initial_blackboard,
            resolved,
            frame_input,
            ..
        } = work_item;

        let resolved_view = resolved.as_resolved();
        let resolved_resources =
            self.resolve_per_view_graph_resources(shared, &resolved_view, transient_by_key)?;
        let graph_resources: &GraphResolvedResources = &resolved_resources;
        let scope = PerViewRecordingScope {
            shared,
            view_idx,
            view: PassViewInputs {
                resolved: &resolved_view,
                graph_resources,
            },
        };
        let mut frame_params = build_per_view_frame_params(
            shared,
            &frame_input,
            &resolved_view,
            PerViewRuntimeInputs {
                host_camera: &host_camera,
                render_context,
                frame_time_seconds,
                clear,
                post_processing,
            },
        );
        let mut view_blackboard =
            self.build_per_view_blackboard(&frame_params, graph_resources, initial_blackboard);
        let output = self.record_one_view_flat_into_encoder(
            scope,
            PassRecordTargets {
                frame_params: &mut frame_params,
                blackboard: &mut view_blackboard,
                encoder,
            },
            resolved.offscreen_color_copy.as_ref(),
            upload_batch,
            profiler,
        )?;
        let hud_outputs = view_blackboard.take::<PerViewHudOutputsSlot>();
        Ok(PerViewEncodeOutput {
            command_buffers: Vec::new(),
            hud_outputs,
            encode_ms: output.encode_ms,
            finish_ms: 0.0,
            command_stats: output.command_stats,
        })
    }

    fn record_one_view_flat<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        state: PerViewLiveState<'a, '_>,
        offscreen_color_copy: Option<&ResolvedOffscreenColorCopy>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<
        (
            Vec<wgpu::CommandBuffer>,
            crate::render_graph::blackboard::GraphCommandStats,
            f64,
            f64,
        ),
        GraphExecuteError,
    > {
        let device = scope.shared.device;
        let encode_start = Instant::now();
        let mut encoder = {
            profiling::scope!("graph::per_view::create_encoder");
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-graph-per-view"),
            })
        };
        let output = self.record_one_view_flat_into_encoder(
            scope,
            PassRecordTargets {
                frame_params: &mut *state.frame_params,
                blackboard: &mut *state.blackboard,
                encoder: &mut encoder,
            },
            offscreen_color_copy,
            upload_batch,
            profiler,
        )?;
        let encode_ms = output.encode_ms.max(elapsed_ms(encode_start));
        if !output.command_stats.has_recorded_work() && !output.recorded_gpu_query {
            return Ok((Vec::new(), output.command_stats, encode_ms, 0.0));
        }
        let (command_buffer, finish_ms) = {
            profiling::scope!("CommandEncoder::finish::graph_per_view");
            let finish_start = Instant::now();
            let command_buffer = encoder.finish();
            let finish_ms = elapsed_ms(finish_start);
            (command_buffer, finish_ms)
        };
        Ok((
            vec![command_buffer],
            output.command_stats,
            encode_ms,
            finish_ms,
        ))
    }

    fn record_one_view_flat_into_encoder<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        mut targets: PassRecordTargets<'a, '_, '_>,
        offscreen_color_copy: Option<&ResolvedOffscreenColorCopy>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<PerViewInlineEncodeOutput, GraphExecuteError> {
        let encode_start = Instant::now();
        let gpu_query = profiler.map(|p| p.begin_query("graph::per_view", &mut *targets.encoder));
        {
            profiling::scope!("graph::per_view::pass_loop");
            self.record_phase_steps(
                PhaseRecordingScope {
                    phase: PassPhase::PerView,
                    view_idx: Some(scope.view_idx),
                },
                scope.view,
                targets.reborrow(),
                PassGpuInputs {
                    device: scope.shared.device,
                    gpu_limits: scope.shared.gpu_limits,
                    profiler,
                },
                upload_batch,
            )?;
        }
        let offscreen_copy_recorded =
            record_offscreen_color_copy(&mut *targets.encoder, offscreen_color_copy, profiler);
        let recorded_gpu_query = gpu_query.is_some();
        if let Some(query) = gpu_query
            && let Some(prof) = profiler
        {
            prof.end_query(&mut *targets.encoder, query);
        }
        let mut command_stats = targets
            .blackboard
            .get_untracked::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        if offscreen_color_copy.is_some() {
            command_stats.record_copy_result(offscreen_copy_recorded);
        }
        Ok(PerViewInlineEncodeOutput {
            encode_ms: elapsed_ms(encode_start),
            command_stats,
            recorded_gpu_query,
        })
    }

    fn record_one_view_scheduler<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        frame_reuse: PerViewFrameReuse<'a>,
        mut state: PerViewLiveState<'a, '_>,
        offscreen_color_copy: Option<&ResolvedOffscreenColorCopy>,
        scheduler: PerViewSchedulerInputs<'a>,
    ) -> Result<
        (
            Vec<wgpu::CommandBuffer>,
            crate::render_graph::blackboard::GraphCommandStats,
            f64,
            f64,
        ),
        GraphExecuteError,
    > {
        profiling::scope!("graph::per_view::scheduler");
        let mut command_buffers = Vec::new();
        let mut parallel_stats = crate::render_graph::blackboard::GraphCommandStats::default();
        let mut encode_ms = 0.0;
        let mut finish_ms = 0.0;
        let batches = self.schedule.recording_plan.batches.as_slice();
        let mut batch_index = next_phase_batch_index(batches, 0, PassPhase::PerView);
        while let Some(current_batch_index) = batch_index {
            let batch = batches[current_batch_index];
            match batch.kind {
                RecordingBatchKind::Serial => {
                    let (next_batch_index, end_unit) =
                        serial_batch_run_end(batches, current_batch_index);
                    let output = self.record_serial_unit_range(
                        scope,
                        state.reborrow(),
                        batch.start_unit..end_unit,
                        scheduler.upload_batch,
                        scheduler.profiler,
                    )?;
                    encode_ms += output.encode_ms;
                    finish_ms += output.finish_ms;
                    command_buffers.push(output.command_buffer);
                    batch_index =
                        next_phase_batch_index(batches, next_batch_index, PassPhase::PerView);
                }
                RecordingBatchKind::Parallel => {
                    let outputs = if scheduler.allow_parallel_batches {
                        self.record_parallel_batch(
                            scope,
                            frame_reuse,
                            &*state.blackboard,
                            batch,
                            scheduler.upload_batch,
                            scheduler.profiler,
                        )?
                    } else {
                        vec![self.record_serial_unit_range(
                            scope,
                            state.reborrow(),
                            batch.start_unit..batch.end_unit,
                            scheduler.upload_batch,
                            scheduler.profiler,
                        )?]
                    };
                    for output in outputs {
                        encode_ms += output.encode_ms;
                        finish_ms += output.finish_ms;
                        parallel_stats.add(output.command_stats);
                        command_buffers.push(output.command_buffer);
                    }
                    batch_index = next_phase_batch_index(
                        batches,
                        current_batch_index + 1,
                        PassPhase::PerView,
                    );
                }
            }
        }
        if let Some(copy_output) = record_offscreen_color_copy_command(
            scope.shared.device,
            offscreen_color_copy,
            scheduler.profiler,
        ) {
            let (command_buffer, recorded, copy_encode_ms, copy_finish_ms) = copy_output;
            encode_ms += copy_encode_ms;
            finish_ms += copy_finish_ms;
            let mut stats = crate::render_graph::blackboard::GraphCommandStats::default();
            stats.record_copy_result(recorded);
            parallel_stats.add(stats);
            command_buffers.push(command_buffer);
        } else if offscreen_color_copy.is_some() {
            let mut stats = crate::render_graph::blackboard::GraphCommandStats::default();
            stats.record_copy_result(false);
            parallel_stats.add(stats);
        }
        let mut command_stats = state
            .blackboard
            .get_untracked::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        command_stats.add(parallel_stats);
        Ok((command_buffers, command_stats, encode_ms, finish_ms))
    }

    fn record_serial_unit_range<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        state: PerViewLiveState<'a, '_>,
        unit_range: Range<usize>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<PerViewUnitEncodeOutput, GraphExecuteError> {
        let encode_start = Instant::now();
        let mut encoder =
            scope
                .shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render-graph-per-view-serial-units"),
                });
        for unit_idx in unit_range {
            let unit = self.schedule.recording_plan.units[unit_idx];
            let query_label = self.recording_unit_label(unit);
            let gpu_query = profiler.map(|p| p.begin_query(query_label.as_str(), &mut encoder));
            self.record_unit_into_encoder(
                scope,
                PassRecordTargets {
                    frame_params: &mut *state.frame_params,
                    blackboard: &mut *state.blackboard,
                    encoder: &mut encoder,
                },
                upload_batch,
                profiler,
                unit,
            )?;
            if let Some(query) = gpu_query
                && let Some(prof) = profiler
            {
                prof.end_query(&mut encoder, query);
            }
        }
        let command_stats = state
            .blackboard
            .get_untracked::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        let encode_ms = elapsed_ms(encode_start);
        let (command_buffer, finish_ms) = {
            profiling::scope!("CommandEncoder::finish::graph_per_view_serial_batch");
            let finish_start = Instant::now();
            let command_buffer = encoder.finish();
            (command_buffer, elapsed_ms(finish_start))
        };
        Ok(PerViewUnitEncodeOutput {
            command_buffer,
            encode_ms,
            finish_ms,
            command_stats,
        })
    }

    fn record_parallel_batch<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        frame_reuse: PerViewFrameReuse<'a>,
        view_blackboard: &Blackboard,
        batch: RecordingBatch,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<Vec<PerViewUnitEncodeOutput>, GraphExecuteError> {
        profiling::scope!("graph::per_view::scheduler::parallel_batch");
        use rayon::prelude::*;
        let mut outputs = (batch.start_unit..batch.end_unit)
            .into_par_iter()
            .map(|unit_idx| {
                let unit = self.schedule.recording_plan.units[unit_idx];
                let mut frame_params = build_per_view_frame_params(
                    scope.shared,
                    frame_reuse.frame_input,
                    scope.view.resolved,
                    frame_reuse.runtime,
                );
                let mut local_blackboard = view_blackboard.clone_read_only();
                local_blackboard.insert_untracked::<GraphCommandStatsSlot>(
                    crate::render_graph::blackboard::GraphCommandStats::default(),
                );
                let output = self.record_unit_command_buffer(
                    scope,
                    PerViewLiveState {
                        frame_params: &mut frame_params,
                        blackboard: &mut local_blackboard,
                    },
                    unit,
                    upload_batch,
                    profiler,
                    "render-graph-per-view-parallel-unit",
                )?;
                Ok((unit_idx, output))
            })
            .collect::<Result<Vec<_>, GraphExecuteError>>()?;
        outputs.sort_unstable_by_key(|(unit_idx, _)| *unit_idx);
        Ok(outputs.into_iter().map(|(_, output)| output).collect())
    }

    fn record_unit_command_buffer<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        state: PerViewLiveState<'a, '_>,
        unit: RecordingUnit,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
        encoder_label: &'static str,
    ) -> Result<PerViewUnitEncodeOutput, GraphExecuteError> {
        let encode_start = Instant::now();
        let mut encoder =
            scope
                .shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(encoder_label),
                });
        let query_label = self.recording_unit_label(unit);
        let gpu_query = profiler.map(|p| p.begin_query(query_label.as_str(), &mut encoder));
        self.record_unit_into_encoder(
            scope,
            PassRecordTargets {
                frame_params: &mut *state.frame_params,
                blackboard: &mut *state.blackboard,
                encoder: &mut encoder,
            },
            upload_batch,
            profiler,
            unit,
        )?;
        if let Some(query) = gpu_query
            && let Some(prof) = profiler
        {
            prof.end_query(&mut encoder, query);
        }
        let command_stats = state
            .blackboard
            .get_untracked::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        let encode_ms = elapsed_ms(encode_start);
        let (command_buffer, finish_ms) = {
            profiling::scope!("CommandEncoder::finish::graph_per_view_unit");
            let finish_start = Instant::now();
            let command_buffer = encoder.finish();
            (command_buffer, elapsed_ms(finish_start))
        };
        Ok(PerViewUnitEncodeOutput {
            command_buffer,
            encode_ms,
            finish_ms,
            command_stats,
        })
    }

    fn record_unit_into_encoder<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        mut targets: PassRecordTargets<'a, '_, '_>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
        unit: RecordingUnit,
    ) -> Result<(), GraphExecuteError> {
        if unit.is_materialized_group()
            && self.try_execute_raster_materialization_group(
                RenderPassMaterializationGroup {
                    start_step: unit.start_step,
                    end_step: unit.end_step,
                },
                PhaseRecordingScope {
                    phase: PassPhase::PerView,
                    view_idx: Some(scope.view_idx),
                },
                scope.view,
                targets.reborrow(),
                PassGpuInputs {
                    device: scope.shared.device,
                    gpu_limits: scope.shared.gpu_limits,
                    profiler,
                },
                upload_batch,
            )?
        {
            return Ok(());
        }
        for step in &self.schedule.steps[unit.start_step..unit.end_step] {
            self.execute_pass_node(
                PassExecution {
                    pass_idx: step.pass_idx,
                    upload_scope: step.frame_upload_scope(Some(scope.view_idx)),
                },
                scope.view,
                targets.reborrow(),
                PassGpuInputs {
                    device: scope.shared.device,
                    gpu_limits: scope.shared.gpu_limits,
                    profiler,
                },
                upload_batch,
            )?;
        }
        Ok(())
    }

    fn recording_unit_label(&self, unit: RecordingUnit) -> String {
        let mut label = String::from("graph::per_view::unit(");
        for (idx, step) in self.schedule.steps[unit.start_step..unit.end_step]
            .iter()
            .enumerate()
        {
            if idx != 0 {
                label.push_str(" + ");
            }
            label.push_str(self.passes[step.pass_idx].profiling_label().as_ref());
        }
        label.push(')');
        label
    }

    /// Resolves this view's transient/imported graph resources from pre-record shared state.
    fn resolve_per_view_graph_resources(
        &self,
        shared: &PerViewRecordShared<'_>,
        resolved: &ResolvedView<'_>,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
    ) -> Result<GraphResolvedResources, GraphExecuteError> {
        profiling::scope!("graph::per_view::resolve_transients");
        let key = GraphResolveKey::from_resolved(resolved);
        let mut resolved_resources = transient_by_key.get(&key).cloned().ok_or_else(|| {
            logger::warn!("pre-resolve: missing transient resources for view key {key:?}");
            GraphExecuteError::MissingTransientResources
        })?;
        self.resolve_imported_textures(resolved, shared.history, &mut resolved_resources)?;
        self.resolve_imported_buffers(
            shared.frame_resources,
            shared.history,
            resolved,
            &mut resolved_resources,
        )?;
        Ok(resolved_resources)
    }
}
