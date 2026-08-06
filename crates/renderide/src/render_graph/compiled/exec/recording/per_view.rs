//! Per-view command-buffer recording.

mod batch_plan;
mod frame_params;
mod offscreen_copy;

use hashbrown::HashMap;
use std::ops::Range;
use std::time::Instant;

use crate::camera::HostCameraFrame;
use crate::frame_contract::FrameViewClear;
use crate::frame_upload_batch::FrameUploadBatch;
use crate::gpu::GpuRetainedResources;
use crate::graph_inputs::PerViewHudOutputsSlot;
use crate::render_graph::blackboard::{Blackboard, GraphCommandStatsSlot};
use crate::render_graph::context::GraphResolvedResources;
use crate::render_graph::error::GraphExecuteError;
use crate::render_graph::pass::PassPhase;
use crate::render_graph::schedule::{
    RecordingBatchKind, RecordingExecutionRun, RecordingUnit, RenderPassMaterializationGroup,
};
use crate::shared::RenderingContext;

use super::super::super::{CompiledRenderGraph, ResolvedView, ViewPostProcessing};
use super::super::recording_path::GraphCommandRecordingStrategy;
use super::super::{
    GraphResolveKey, PerViewEncodeOutput, PerViewRecordShared, PerViewWorkItem,
    PreparedPerViewFrameInput, ResolvedOffscreenColorCopy, elapsed_ms,
};
use super::{PassExecution, PassGpuInputs, PassRecordTargets, PassViewInputs, PhaseRecordingScope};

use batch_plan::per_view_recording_runs;
use frame_params::build_per_view_frame_params;
use offscreen_copy::{record_offscreen_color_copy, record_offscreen_color_copy_command};

struct PerViewUnitEncodeOutput {
    command_buffer: Option<wgpu::CommandBuffer>,
    encode_ms: f64,
    finish_ms: f64,
    command_stats: crate::render_graph::blackboard::GraphCommandStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PerViewUnitRecordingOutcome {
    admitted_pass: bool,
    recorded_gpu_query: bool,
    materialized_group: bool,
}

impl PerViewUnitRecordingOutcome {
    fn add(&mut self, other: Self) {
        self.admitted_pass |= other.admitted_pass;
        self.recorded_gpu_query |= other.recorded_gpu_query;
        self.materialized_group |= other.materialized_group;
    }

    fn retains_command_buffer(self) -> bool {
        self.admitted_pass || self.recorded_gpu_query || self.materialized_group
    }
}

fn finish_unit_command_buffer_if_needed<T>(
    outcome: PerViewUnitRecordingOutcome,
    finish: impl FnOnce() -> T,
) -> Option<T> {
    outcome.retains_command_buffer().then(finish)
}

fn append_unit_command_buffer<T>(command_buffers: &mut Vec<T>, command_buffer: Option<T>) {
    if let Some(command_buffer) = command_buffer {
        command_buffers.push(command_buffer);
    }
}

struct PerViewCommandEncodeBatch {
    command_buffers: Vec<wgpu::CommandBuffer>,
    command_stats: crate::render_graph::blackboard::GraphCommandStats,
    encode_ms: f64,
    finish_ms: f64,
    max_finish_ms: f64,
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

struct PerViewUnitCommandLabels<'a> {
    query: &'a str,
    encoder: &'static str,
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
            view_id,
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

        let use_scheduler = strategy.uses_in_view_scheduler()
            && !self
                .schedule
                .recording_plan
                .per_view_execution_runs()
                .is_empty();
        let encoded = if use_scheduler {
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
                    allow_parallel_batches: strategy.allows_in_view_parallel_batches(),
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
        let encode_ms = encoded.encode_ms.max(elapsed_ms(encode_start));
        let mut retained_resources = GpuRetainedResources::new();
        resolved_resources.retain_submit_resources(&mut retained_resources);
        retained_resources.append(crate::passes::take_gpu_cull_submit_resources(
            &mut view_blackboard,
        ));
        self.recycle_view_blackboard(view_id, view_blackboard);
        Ok(PerViewEncodeOutput {
            command_buffers: encoded.command_buffers,
            hud_outputs,
            encode_ms,
            finish_ms: encoded.finish_ms,
            max_finish_ms: encoded.max_finish_ms,
            command_stats: encoded.command_stats,
            retained_resources,
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
        let PerViewWorkItem {
            view_idx,
            host_camera,
            render_context,
            frame_time_seconds,
            view_id,
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
        let mut retained_resources = GpuRetainedResources::new();
        resolved_resources.retain_submit_resources(&mut retained_resources);
        retained_resources.append(crate::passes::take_gpu_cull_submit_resources(
            &mut view_blackboard,
        ));
        self.recycle_view_blackboard(view_id, view_blackboard);
        Ok(PerViewEncodeOutput {
            command_buffers: Vec::new(),
            hud_outputs,
            encode_ms: output.encode_ms,
            finish_ms: 0.0,
            max_finish_ms: 0.0,
            command_stats: output.command_stats,
            retained_resources,
        })
    }

    fn record_one_view_flat<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        state: PerViewLiveState<'a, '_>,
        offscreen_color_copy: Option<&ResolvedOffscreenColorCopy>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<PerViewCommandEncodeBatch, GraphExecuteError> {
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
            return Ok(PerViewCommandEncodeBatch {
                command_buffers: Vec::new(),
                command_stats: output.command_stats,
                encode_ms,
                finish_ms: 0.0,
                max_finish_ms: 0.0,
            });
        }
        let (command_buffer, finish_ms) = {
            profiling::scope!("CommandEncoder::finish::graph_per_view");
            let finish_start = Instant::now();
            let command_buffer = encoder.finish();
            let finish_ms = elapsed_ms(finish_start);
            (command_buffer, finish_ms)
        };
        Ok(PerViewCommandEncodeBatch {
            command_buffers: vec![command_buffer],
            command_stats: output.command_stats,
            encode_ms,
            finish_ms,
            max_finish_ms: finish_ms,
        })
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
    ) -> Result<PerViewCommandEncodeBatch, GraphExecuteError> {
        profiling::scope!("graph::per_view::scheduler");
        let mut command_buffers = Vec::new();
        let mut parallel_stats = crate::render_graph::blackboard::GraphCommandStats::default();
        let mut encode_ms = 0.0;
        let mut finish_ms = 0.0;
        let mut max_finish_ms = 0.0;
        let runs = per_view_recording_runs(&self.schedule.recording_plan);
        for run in runs.iter().copied() {
            match run.kind {
                RecordingBatchKind::Serial => {
                    let output = self.record_serial_unit_range(
                        scope,
                        state.reborrow(),
                        run.start_unit..run.end_unit,
                        scheduler.upload_batch,
                        scheduler.profiler,
                    )?;
                    encode_ms += output.encode_ms;
                    finish_ms += output.finish_ms;
                    max_finish_ms = f64::max(max_finish_ms, output.finish_ms);
                    append_unit_command_buffer(&mut command_buffers, output.command_buffer);
                }
                RecordingBatchKind::Parallel => {
                    let outputs = if scheduler.allow_parallel_batches {
                        self.record_parallel_batch(
                            scope,
                            frame_reuse,
                            &*state.blackboard,
                            run,
                            scheduler.upload_batch,
                            scheduler.profiler,
                        )?
                    } else {
                        vec![self.record_serial_unit_range(
                            scope,
                            state.reborrow(),
                            run.start_unit..run.end_unit,
                            scheduler.upload_batch,
                            scheduler.profiler,
                        )?]
                    };
                    for output in outputs {
                        encode_ms += output.encode_ms;
                        finish_ms += output.finish_ms;
                        max_finish_ms = f64::max(max_finish_ms, output.finish_ms);
                        parallel_stats.add(output.command_stats);
                        append_unit_command_buffer(&mut command_buffers, output.command_buffer);
                    }
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
            max_finish_ms = f64::max(max_finish_ms, copy_finish_ms);
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
        Ok(PerViewCommandEncodeBatch {
            command_buffers,
            command_stats,
            encode_ms,
            finish_ms,
            max_finish_ms,
        })
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
        let mut recording_outcome = PerViewUnitRecordingOutcome::default();
        for unit_idx in unit_range {
            let unit = self.schedule.recording_plan.units[unit_idx];
            let query_label = self.schedule.recording_plan.unit_label(unit_idx);
            recording_outcome.add(self.record_profiled_unit_into_encoder(
                scope,
                PassRecordTargets {
                    frame_params: &mut *state.frame_params,
                    blackboard: &mut *state.blackboard,
                    encoder: &mut encoder,
                },
                upload_batch,
                profiler,
                unit,
                query_label,
            )?);
        }
        let command_stats = state
            .blackboard
            .get_untracked::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        let encode_ms = elapsed_ms(encode_start);
        let finish_start = recording_outcome
            .retains_command_buffer()
            .then(Instant::now);
        let command_buffer = finish_unit_command_buffer_if_needed(recording_outcome, || {
            profiling::scope!("CommandEncoder::finish::graph_per_view_serial_batch");
            encoder.finish()
        });
        let finish_ms = finish_start.map_or(0.0, elapsed_ms);
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
        run: RecordingExecutionRun,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<Vec<PerViewUnitEncodeOutput>, GraphExecuteError> {
        profiling::scope!("graph::per_view::scheduler::parallel_batch");
        use rayon::prelude::*;
        let mut outputs = (run.start_unit..run.end_unit)
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
                    PerViewUnitCommandLabels {
                        query: self.schedule.recording_plan.unit_label(unit_idx),
                        encoder: "render-graph-per-view-parallel-unit",
                    },
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
        labels: PerViewUnitCommandLabels<'_>,
    ) -> Result<PerViewUnitEncodeOutput, GraphExecuteError> {
        let encode_start = Instant::now();
        let mut encoder =
            scope
                .shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(labels.encoder),
                });
        let recording_outcome = self.record_profiled_unit_into_encoder(
            scope,
            PassRecordTargets {
                frame_params: &mut *state.frame_params,
                blackboard: &mut *state.blackboard,
                encoder: &mut encoder,
            },
            upload_batch,
            profiler,
            unit,
            labels.query,
        )?;
        let command_stats = state
            .blackboard
            .get_untracked::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        let encode_ms = elapsed_ms(encode_start);
        let finish_start = recording_outcome
            .retains_command_buffer()
            .then(Instant::now);
        let command_buffer = finish_unit_command_buffer_if_needed(recording_outcome, || {
            profiling::scope!("CommandEncoder::finish::graph_per_view_unit");
            encoder.finish()
        });
        let finish_ms = finish_start.map_or(0.0, elapsed_ms);
        Ok(PerViewUnitEncodeOutput {
            command_buffer,
            encode_ms,
            finish_ms,
            command_stats,
        })
    }

    fn record_profiled_unit_into_encoder<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        mut targets: PassRecordTargets<'a, '_, '_>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
        unit: RecordingUnit,
        query_label: &str,
    ) -> Result<PerViewUnitRecordingOutcome, GraphExecuteError> {
        if !recording_unit_query_starts_on_pass_admission(unit) {
            let query = profiler.map(|profiler| profiler.begin_query(query_label, targets.encoder));
            let result = self.record_unit_into_encoder(
                scope,
                targets.reborrow(),
                upload_batch,
                profiler,
                unit,
                None,
            );
            let recorded_gpu_query = query.is_some();
            if let (Some(profiler), Some(query)) = (profiler, query) {
                profiler.end_query(targets.encoder, query);
            }
            result?;
            return Ok(PerViewUnitRecordingOutcome {
                admitted_pass: false,
                recorded_gpu_query,
                materialized_group: true,
            });
        }

        let mut query = None;
        let mut admitted_pass = false;
        let mut begin_query = |encoder: &mut wgpu::CommandEncoder| {
            admitted_pass = true;
            if let Some(profiler) = profiler {
                open_recording_unit_query_if_admitted(&mut query, true, || {
                    profiler.begin_query(query_label, encoder)
                });
            }
        };
        let result = self.record_unit_into_encoder(
            scope,
            targets.reborrow(),
            upload_batch,
            profiler,
            unit,
            Some(&mut begin_query),
        );
        drop(begin_query);
        let recorded_gpu_query = query.is_some();
        if let (Some(profiler), Some(query)) = (profiler, query) {
            profiler.end_query(targets.encoder, query);
        }
        result?;
        Ok(PerViewUnitRecordingOutcome {
            admitted_pass,
            recorded_gpu_query,
            materialized_group: false,
        })
    }

    fn record_unit_into_encoder<'a>(
        &self,
        scope: PerViewRecordingScope<'a>,
        mut targets: PassRecordTargets<'a, '_, '_>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
        unit: RecordingUnit,
        mut before_record: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder)>,
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
                before_record.take(),
            )?;
        }
        Ok(())
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

#[inline]
const fn recording_unit_query_starts_on_pass_admission(unit: RecordingUnit) -> bool {
    !unit.is_materialized_group()
}

#[inline]
fn open_recording_unit_query_if_admitted<T>(
    query: &mut Option<T>,
    admitted: bool,
    open: impl FnOnce() -> T,
) {
    if admitted && query.is_none() {
        *query = Some(open());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::pass::PassPhase;
    use crate::render_graph::schedule::RecordingSerialReason;
    use std::cell::Cell;

    fn recording_unit(start_step: usize, end_step: usize) -> RecordingUnit {
        RecordingUnit {
            start_step,
            end_step,
            phase: PassPhase::PerView,
            wave_idx: 0,
            parallel_safe: false,
            serial_reason: RecordingSerialReason::NeverParallel,
        }
    }

    #[test]
    fn single_pass_unit_defers_gpu_query_until_pass_admission() {
        assert!(recording_unit_query_starts_on_pass_admission(
            recording_unit(3, 4)
        ));
    }

    #[test]
    fn materialized_unit_keeps_eager_gpu_query_boundary() {
        assert!(!recording_unit_query_starts_on_pass_admission(
            recording_unit(3, 5)
        ));
    }

    #[test]
    fn skipped_single_pass_opens_no_deferred_query() {
        let opens = Cell::new(0);
        let mut query = None;
        open_recording_unit_query_if_admitted(&mut query, false, || {
            opens.set(opens.get() + 1);
            7
        });

        assert!(query.is_none());
        assert_eq!(opens.get(), 0);
    }

    #[test]
    fn admitted_single_pass_opens_one_deferred_query() {
        let opens = Cell::new(0);
        let mut query = None;
        open_recording_unit_query_if_admitted(&mut query, true, || {
            opens.set(opens.get() + 1);
            7
        });
        open_recording_unit_query_if_admitted(&mut query, true, || {
            opens.set(opens.get() + 1);
            8
        });

        assert_eq!(query, Some(7));
        assert_eq!(opens.get(), 1);
    }

    #[test]
    fn skipped_single_pass_unit_returns_no_command_buffer_without_finishing() {
        let finishes = Cell::new(0);
        let command_buffer =
            finish_unit_command_buffer_if_needed(PerViewUnitRecordingOutcome::default(), || {
                finishes.set(finishes.get() + 1);
                7
            });

        assert_eq!(command_buffer, None);
        assert_eq!(finishes.get(), 0);
    }

    #[test]
    fn admitted_single_pass_unit_retains_its_command_buffer() {
        let finishes = Cell::new(0);
        let command_buffer = finish_unit_command_buffer_if_needed(
            PerViewUnitRecordingOutcome {
                admitted_pass: true,
                ..Default::default()
            },
            || {
                finishes.set(finishes.get() + 1);
                7
            },
        );

        assert_eq!(command_buffer, Some(7));
        assert_eq!(finishes.get(), 1);
    }

    #[test]
    fn recorded_gpu_query_retains_its_command_buffer() {
        let finishes = Cell::new(0);
        let command_buffer = finish_unit_command_buffer_if_needed(
            PerViewUnitRecordingOutcome {
                recorded_gpu_query: true,
                ..Default::default()
            },
            || {
                finishes.set(finishes.get() + 1);
                7
            },
        );

        assert_eq!(command_buffer, Some(7));
        assert_eq!(finishes.get(), 1);
    }

    #[test]
    fn all_skipped_serial_range_returns_no_command_buffer() {
        let finishes = Cell::new(0);
        let mut range_outcome = PerViewUnitRecordingOutcome::default();
        range_outcome.add(PerViewUnitRecordingOutcome::default());
        range_outcome.add(PerViewUnitRecordingOutcome::default());
        let command_buffer = finish_unit_command_buffer_if_needed(range_outcome, || {
            finishes.set(finishes.get() + 1);
            7
        });

        assert_eq!(command_buffer, None);
        assert_eq!(finishes.get(), 0);
    }

    #[test]
    fn materialized_group_retains_its_command_buffer() {
        let finishes = Cell::new(0);
        let command_buffer = finish_unit_command_buffer_if_needed(
            PerViewUnitRecordingOutcome {
                materialized_group: true,
                ..Default::default()
            },
            || {
                finishes.set(finishes.get() + 1);
                7
            },
        );

        assert_eq!(command_buffer, Some(7));
        assert_eq!(finishes.get(), 1);
    }

    #[test]
    fn parallel_unit_filter_omits_empty_buffers_without_reordering() {
        let mut command_buffers = Vec::new();
        append_unit_command_buffer(&mut command_buffers, Some(3));
        append_unit_command_buffer(&mut command_buffers, None);
        append_unit_command_buffer(&mut command_buffers, Some(7));

        assert_eq!(command_buffers, vec![3, 7]);
    }
}
