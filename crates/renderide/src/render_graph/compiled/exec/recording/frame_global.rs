//! Frame-global command-buffer recording.

use hashbrown::HashMap;
use hashbrown::hash_map::Entry;
use rayon::prelude::*;
use std::time::Instant;

use crate::cpu_parallelism::{ParallelAdmission, record_parallel_admission};
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::context::GraphResolvedResources;
use crate::render_graph::error::GraphExecuteError;
use crate::render_graph::execution_backend::{
    FrameGlobalPassSplitWorkload, FrameGlobalSplitPassEncodeParams, GraphExecutionBackend,
    GraphFrameResources,
};
use crate::render_graph::frame_upload_batch::{FrameUploadBatch, GraphUploadSink};
use crate::render_graph::pass::PassPhase;

use super::super::super::helpers;
use super::super::super::{
    CompiledRenderGraph, FrameGlobalView, FrameView, MultiViewExecutionContext, ResolvedView,
};
use super::super::{
    FrameGlobalPassRecordInputs, GraphResolveKey, TimedCommandBuffer,
    TransientTextureResolveSurfaceParams, elapsed_ms,
};
use super::{PassGpuInputs, PassRecordTargets, PassViewInputs, PhaseRecordingScope};

/// Mutable state needed to replay frame-global schedule steps.
struct FrameGlobalPassLoop<'record, 'frame> {
    /// Compiled graph being recorded.
    graph: &'record CompiledRenderGraph,
    /// Resolved view used by frame-global passes.
    resolved: &'frame ResolvedView<'frame>,
    /// Graph resource table resolved for the frame-global view.
    graph_resources: &'frame GraphResolvedResources,
    /// Mutable per-pass frame parameters.
    frame_params: &'record mut crate::graph_inputs::GraphPassFrame<'frame>,
    /// Blackboard shared across frame-global passes.
    frame_blackboard: &'record mut Blackboard,
    /// Command encoder receiving frame-global work.
    encoder: &'record mut wgpu::CommandEncoder,
    /// GPU device.
    device: &'frame wgpu::Device,
    /// GPU limits for pass contexts.
    gpu_limits: &'frame crate::gpu::GpuLimits,
    /// Deferred upload batch for scoped graph uploads.
    upload_batch: &'record FrameUploadBatch,
    /// Optional profiler handle for pass GPU scopes.
    pass_profiler: Option<&'frame crate::profiling::GpuProfilerHandle>,
}

#[derive(Clone, Copy)]
pub(in crate::render_graph::compiled::exec) struct FrameGlobalSplitCandidate {
    step_idx: usize,
    pass_idx: usize,
    workload: FrameGlobalPassSplitWorkload,
}

struct FrameGlobalSplitRecordState<'record, 'frame> {
    resolved: &'frame ResolvedView<'frame>,
    graph_resources: &'frame GraphResolvedResources,
    frame_params: &'record mut crate::graph_inputs::GraphPassFrame<'frame>,
    frame_blackboard: &'record mut Blackboard,
    device: &'frame wgpu::Device,
    gpu_limits: &'frame crate::gpu::GpuLimits,
    upload_batch: &'record FrameUploadBatch,
}

struct FrameGlobalRangeCommand<'record, 'frame> {
    label: &'record str,
    step_range: std::ops::Range<usize>,
    resolved: &'frame ResolvedView<'frame>,
    graph_resources: &'frame GraphResolvedResources,
    frame_params: &'record mut crate::graph_inputs::GraphPassFrame<'frame>,
    frame_blackboard: &'record mut Blackboard,
    device: &'frame wgpu::Device,
    gpu_limits: &'frame crate::gpu::GpuLimits,
    upload_batch: &'record FrameUploadBatch,
    pass_profiler: Option<&'frame crate::profiling::GpuProfilerHandle>,
}

impl FrameGlobalPassLoop<'_, '_> {
    /// Records every frame-global pass in scheduler wave order.
    fn record(self) -> Result<(), GraphExecuteError> {
        let step_count = self.graph.schedule.steps.len();
        self.record_range(0..step_count)
    }

    /// Records frame-global passes inside a schedule step range.
    fn record_range(self, step_range: std::ops::Range<usize>) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::frame_global::pass_loop");
        self.graph.record_phase_steps_range(
            PhaseRecordingScope {
                phase: PassPhase::FrameGlobal,
                view_idx: None,
            },
            step_range,
            PassViewInputs {
                resolved: self.resolved,
                graph_resources: self.graph_resources,
            },
            PassRecordTargets {
                frame_params: self.frame_params,
                blackboard: self.frame_blackboard,
                encoder: self.encoder,
            },
            PassGpuInputs {
                device: self.device,
                gpu_limits: self.gpu_limits,
                profiler: self.pass_profiler,
            },
            self.upload_batch,
        )
    }
}

impl CompiledRenderGraph {
    /// Encodes [`crate::render_graph::pass::PassPhase::FrameGlobal`] passes into command buffers.
    ///
    /// Returns an empty vector when there are no frame-global passes to submit for this phase.
    /// The caller is responsible for preserving the returned order in the single-submit batch.
    pub(in crate::render_graph::compiled::exec) fn encode_frame_global_passes(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
        frame_global: &FrameGlobalView,
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
    ) -> Result<Vec<TimedCommandBuffer>, GraphExecuteError> {
        profiling::scope!("graph::frame_global");
        if self.frame_global_passes_are_inactive(&*mv_ctx.backend) {
            return Ok(Vec::new());
        }
        if let Some(candidate) = self.frame_global_split_candidate(&*mv_ctx.backend) {
            return self.encode_frame_global_passes_split(
                mv_ctx,
                views,
                frame_global,
                transient_by_key,
                upload_batch,
                candidate,
            );
        }
        let encode_start = Instant::now();
        let mut encoder = {
            let device = mv_ctx.device;
            profiling::scope!("graph::frame_global::create_encoder");
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-graph-frame-global"),
            })
        };
        let gpu_query = mv_ctx
            .gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::frame_global", &mut encoder));
        let pass_profiler = mv_ctx.gpu.take_gpu_profiler();

        let record_result = self.record_frame_global_passes_into_encoder(
            mv_ctx,
            FrameGlobalPassRecordInputs {
                views,
                frame_global,
                transient_by_key,
                encoder: &mut encoder,
                upload_batch,
                pass_profiler: pass_profiler.as_ref(),
            },
        );
        mv_ctx.gpu.restore_gpu_profiler(pass_profiler);
        record_result?;
        let encode_ms = elapsed_ms(encode_start);
        let command_buffer =
            Self::finish_frame_global_encoder(mv_ctx.gpu, encoder, gpu_query, encode_ms);
        Ok(vec![command_buffer])
    }

    pub(in crate::render_graph::compiled::exec) fn frame_global_passes_are_inactive(
        &self,
        backend: &dyn GraphExecutionBackend,
    ) -> bool {
        let frame_resources = backend.frame_resources();
        for pass_idx in self.schedule.frame_global_pass_indices().iter().copied() {
            if !frame_resources.frame_global_pass_is_inactive(self.passes[pass_idx].name()) {
                return false;
            }
        }
        true
    }

    pub(in crate::render_graph::compiled::exec) fn frame_global_has_split_workload(
        &self,
        backend: &dyn GraphExecutionBackend,
    ) -> bool {
        self.frame_global_split_candidate(backend).is_some()
    }

    pub(in crate::render_graph::compiled::exec) fn frame_global_split_candidate(
        &self,
        backend: &dyn GraphExecutionBackend,
    ) -> Option<FrameGlobalSplitCandidate> {
        let frame_resources = backend.frame_resources();
        self.schedule
            .steps
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, step)| step.phase == PassPhase::FrameGlobal)
            .find_map(|(step_idx, step)| {
                let pass = &self.passes[step.pass_idx];
                if frame_resources.frame_global_pass_is_inactive(pass.name()) {
                    return None;
                }
                let workload = frame_resources.frame_global_pass_split_workload(pass.name())?;
                if workload.unit_count < 2 || workload.chunk_size == 0 {
                    return None;
                }
                Some(FrameGlobalSplitCandidate {
                    step_idx,
                    pass_idx: step.pass_idx,
                    workload,
                })
            })
    }

    fn frame_global_range_has_active_passes(
        &self,
        frame_resources: &dyn GraphFrameResources,
        step_range: std::ops::Range<usize>,
    ) -> bool {
        let end_step = step_range.end.min(self.schedule.steps.len());
        let start_step = step_range.start.min(end_step);
        for step_idx in start_step..end_step {
            let step = self.schedule.steps[step_idx];
            if step.phase != PassPhase::FrameGlobal {
                continue;
            }
            if !frame_resources.frame_global_pass_is_inactive(self.passes[step.pass_idx].name()) {
                return true;
            }
        }
        false
    }

    pub(in crate::render_graph::compiled::exec) fn record_frame_global_passes_into_encoder(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        inputs: FrameGlobalPassRecordInputs<'_, '_>,
    ) -> Result<(), GraphExecuteError> {
        let FrameGlobalPassRecordInputs {
            views,
            frame_global,
            transient_by_key,
            encoder,
            upload_batch,
            pass_profiler,
        } = inputs;
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            backbuffer_view_holder,
        } = mv_ctx;

        let anchor = views
            .iter()
            .find(|view| view.view_id() == frame_global.view_id)
            .or_else(|| views.first())
            .ok_or(GraphExecuteError::NoViewsInBatch)?;
        let resolved = {
            profiling::scope!("graph::frame_global::resolve_target");
            Self::resolve_view_from_target(
                anchor.view_id(),
                anchor.view_winding,
                anchor.profile,
                &anchor.host_camera,
                &anchor.target,
                gpu,
                backbuffer_view_holder.as_ref(),
            )
        }?;
        let resolved_resources = {
            profiling::scope!("graph::frame_global::resolve_transients");
            self.resolve_frame_global_transients(
                &resolved,
                transient_by_key,
                device,
                &mut **backend,
                gpu_limits,
            )
        }?;
        {
            profiling::scope!("graph::frame_global::resolve_imported_resources");
            self.resolve_imported_textures(
                &resolved,
                backend.history_registry(),
                resolved_resources,
            )?;
            self.resolve_imported_buffers(
                backend.frame_resources(),
                backend.history_registry(),
                &resolved,
                resolved_resources,
            )?;
        }
        let graph_resources: &GraphResolvedResources = &*resolved_resources;

        let mut frame_params = {
            profiling::scope!("graph::frame_global::build_frame_params");
            helpers::frame_render_params_from_resolved(
                *scene,
                &mut **backend,
                helpers::ResolvedFrameRenderParamsInputs {
                    resolved: &resolved,
                    host_camera: &frame_global.host_camera,
                    render_context: frame_global.render_context,
                    frame_time_seconds: frame_global.frame_time_seconds,
                    clear: frame_global.clear,
                    post_processing: frame_global.post_processing,
                },
            )
        };
        let mut frame_blackboard = Self::build_frame_global_blackboard();

        FrameGlobalPassLoop {
            graph: self,
            resolved: &resolved,
            graph_resources,
            frame_params: &mut frame_params,
            frame_blackboard: &mut frame_blackboard,
            encoder,
            device,
            gpu_limits,
            upload_batch,
            pass_profiler,
        }
        .record()
    }

    fn encode_frame_global_passes_split(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
        frame_global: &FrameGlobalView,
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        candidate: FrameGlobalSplitCandidate,
    ) -> Result<Vec<TimedCommandBuffer>, GraphExecuteError> {
        profiling::scope!("graph::frame_global::split");
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            backbuffer_view_holder,
        } = mv_ctx;
        let anchor = views
            .iter()
            .find(|view| view.view_id() == frame_global.view_id)
            .or_else(|| views.first())
            .ok_or(GraphExecuteError::NoViewsInBatch)?;
        let resolved = {
            profiling::scope!("graph::frame_global::split::resolve_target");
            Self::resolve_view_from_target(
                anchor.view_id(),
                anchor.view_winding,
                anchor.profile,
                &anchor.host_camera,
                &anchor.target,
                gpu,
                backbuffer_view_holder.as_ref(),
            )
        }?;
        let resolved_resources = {
            profiling::scope!("graph::frame_global::split::resolve_transients");
            self.resolve_frame_global_transients(
                &resolved,
                transient_by_key,
                device,
                &mut **backend,
                gpu_limits,
            )
        }?;
        {
            profiling::scope!("graph::frame_global::split::resolve_imported_resources");
            self.resolve_imported_textures(
                &resolved,
                backend.history_registry(),
                resolved_resources,
            )?;
            self.resolve_imported_buffers(
                backend.frame_resources(),
                backend.history_registry(),
                &resolved,
                resolved_resources,
            )?;
        }
        let graph_resources: &GraphResolvedResources = &*resolved_resources;

        let mut frame_params = {
            profiling::scope!("graph::frame_global::split::build_frame_params");
            helpers::frame_render_params_from_resolved(
                scene,
                &mut **backend,
                helpers::ResolvedFrameRenderParamsInputs {
                    resolved: &resolved,
                    host_camera: &frame_global.host_camera,
                    render_context: frame_global.render_context,
                    frame_time_seconds: frame_global.frame_time_seconds,
                    clear: frame_global.clear,
                    post_processing: frame_global.post_processing,
                },
            )
        };
        let mut frame_blackboard = Self::build_frame_global_blackboard();
        self.record_frame_global_split_commands_ordered(
            candidate,
            FrameGlobalSplitRecordState {
                resolved: &resolved,
                graph_resources,
                frame_params: &mut frame_params,
                frame_blackboard: &mut frame_blackboard,
                device,
                gpu_limits,
                upload_batch,
            },
        )
    }

    fn record_frame_global_split_commands_ordered(
        &self,
        candidate: FrameGlobalSplitCandidate,
        state: FrameGlobalSplitRecordState<'_, '_>,
    ) -> Result<Vec<TimedCommandBuffer>, GraphExecuteError> {
        let mut commands = Vec::new();
        let before_range = 0..candidate.step_idx;
        if self.frame_global_range_has_active_passes(
            state.frame_params.shared.frame_resources,
            before_range.clone(),
        ) {
            commands.push(
                self.record_frame_global_range_command(FrameGlobalRangeCommand {
                    label: "render-graph-frame-global-before-split",
                    step_range: before_range,
                    resolved: state.resolved,
                    graph_resources: state.graph_resources,
                    frame_params: &mut *state.frame_params,
                    frame_blackboard: &mut *state.frame_blackboard,
                    device: state.device,
                    gpu_limits: state.gpu_limits,
                    upload_batch: state.upload_batch,
                    pass_profiler: None,
                })?,
            );
        }

        let split_pass_name = self.passes[candidate.pass_idx].name();
        let split_step = self.schedule.steps[candidate.step_idx];
        let split_uploads =
            GraphUploadSink::new(state.upload_batch, split_step.frame_upload_scope(None));
        let prepared = state
            .frame_params
            .shared
            .frame_resources
            .prepare_frame_global_split_pass(split_pass_name, state.gpu_limits, split_uploads);
        if prepared {
            let mut split_commands = self.record_frame_global_split_pass_commands(
                split_pass_name,
                candidate.workload,
                state.frame_params,
                state.device,
                state.gpu_limits,
                None,
            )?;
            commands.append(&mut split_commands);
        } else {
            commands.push(
                self.record_frame_global_range_command(FrameGlobalRangeCommand {
                    label: "render-graph-frame-global-split-fallback",
                    step_range: candidate.step_idx..candidate.step_idx + 1,
                    resolved: state.resolved,
                    graph_resources: state.graph_resources,
                    frame_params: &mut *state.frame_params,
                    frame_blackboard: &mut *state.frame_blackboard,
                    device: state.device,
                    gpu_limits: state.gpu_limits,
                    upload_batch: state.upload_batch,
                    pass_profiler: None,
                })?,
            );
        }

        let after_start = candidate.step_idx.saturating_add(1);
        let after_range = after_start..self.schedule.steps.len();
        if self.frame_global_range_has_active_passes(
            state.frame_params.shared.frame_resources,
            after_range.clone(),
        ) {
            commands.push(
                self.record_frame_global_range_command(FrameGlobalRangeCommand {
                    label: "render-graph-frame-global-after-split",
                    step_range: after_range,
                    resolved: state.resolved,
                    graph_resources: state.graph_resources,
                    frame_params: &mut *state.frame_params,
                    frame_blackboard: &mut *state.frame_blackboard,
                    device: state.device,
                    gpu_limits: state.gpu_limits,
                    upload_batch: state.upload_batch,
                    pass_profiler: None,
                })?,
            );
        }
        Ok(commands)
    }

    fn record_frame_global_range_command(
        &self,
        command: FrameGlobalRangeCommand<'_, '_>,
    ) -> Result<TimedCommandBuffer, GraphExecuteError> {
        profiling::scope!("graph::frame_global::record_serial_range");
        let FrameGlobalRangeCommand {
            label,
            step_range,
            resolved,
            graph_resources,
            frame_params,
            frame_blackboard,
            device,
            gpu_limits,
            upload_batch,
            pass_profiler,
        } = command;
        let encode_start = Instant::now();
        let mut encoder = {
            profiling::scope!("graph::frame_global::split::create_serial_encoder");
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
        };
        FrameGlobalPassLoop {
            graph: self,
            resolved,
            graph_resources,
            frame_params,
            frame_blackboard,
            encoder: &mut encoder,
            device,
            gpu_limits,
            upload_batch,
            pass_profiler,
        }
        .record_range(step_range)?;
        let encode_ms = elapsed_ms(encode_start);
        Ok(Self::finish_frame_global_split_encoder(encoder, encode_ms))
    }

    fn record_frame_global_split_pass_commands(
        &self,
        pass_name: &str,
        workload: FrameGlobalPassSplitWorkload,
        frame_params: &crate::graph_inputs::GraphPassFrame<'_>,
        device: &wgpu::Device,
        gpu_limits: &crate::gpu::GpuLimits,
        pass_profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<Vec<TimedCommandBuffer>, GraphExecuteError> {
        profiling::scope!("graph::frame_global::record_split_pass");
        let chunk_size = workload.chunk_size.max(1);
        let ranges = (0..workload.unit_count)
            .step_by(chunk_size)
            .map(|start| {
                let end = start.saturating_add(chunk_size).min(workload.unit_count);
                start..end
            })
            .collect::<Vec<_>>();
        record_parallel_admission(
            "graph_frame_global_split_pass",
            workload.estimated_work,
            workload.unit_count,
            ParallelAdmission::Parallel { chunk_size },
        );
        crate::profiling::plot_frame_global_split(workload.unit_count, ranges.len(), chunk_size);

        let frame_resources = frame_params.shared.frame_resources;
        let materials = frame_params.shared.materials;
        let asset_resources = frame_params.shared.asset_resources;
        let skin_cache = frame_params
            .shared
            .mesh_deform_skin_cache
            .as_deref()
            .or(frame_params.shared.skin_cache);

        ranges
            .into_par_iter()
            .enumerate()
            .map(|(chunk_idx, unit_range)| {
                profiling::scope!("graph::frame_global::split_worker");
                let encode_start = Instant::now();
                let label = format!("render-graph-frame-global-split-{chunk_idx}");
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(label.as_str()),
                });
                let encoded = frame_resources.encode_frame_global_split_pass(
                    pass_name,
                    unit_range,
                    FrameGlobalSplitPassEncodeParams {
                        device,
                        encoder: &mut encoder,
                        materials,
                        asset_resources,
                        skin_cache,
                        gpu_limits,
                        profiler: pass_profiler,
                    },
                );
                if !encoded {
                    return Err(GraphExecuteError::NoViewsInBatch);
                }
                let encode_ms = elapsed_ms(encode_start);
                Ok(Self::finish_frame_global_split_encoder(encoder, encode_ms))
            })
            .collect()
    }

    fn finish_frame_global_split_encoder(
        encoder: wgpu::CommandEncoder,
        encode_ms: f64,
    ) -> TimedCommandBuffer {
        profiling::scope!("CommandEncoder::finish::graph_frame_global_split");
        let finish_start = Instant::now();
        let command_buffer = encoder.finish();
        let finish_ms = elapsed_ms(finish_start);
        TimedCommandBuffer {
            command_buffer,
            encode_ms,
            finish_ms,
        }
    }

    fn finish_frame_global_encoder(
        gpu: &mut crate::gpu::GpuContext,
        mut encoder: wgpu::CommandEncoder,
        gpu_query: Option<crate::profiling::PhaseQuery>,
        encode_ms: f64,
    ) -> TimedCommandBuffer {
        if let Some(query) = gpu_query
            && let Some(prof) = gpu.gpu_profiler_mut()
        {
            profiling::scope!("graph::frame_global::profiler_resolve");
            prof.end_query(&mut encoder, query);
            prof.resolve_queries(&mut encoder);
        }
        {
            profiling::scope!("CommandEncoder::finish::graph_frame_global");
            let finish_start = Instant::now();
            let command_buffer = encoder.finish();
            let finish_ms = elapsed_ms(finish_start);
            TimedCommandBuffer {
                command_buffer,
                encode_ms,
                finish_ms,
            }
        }
    }

    /// Resolves (or reuses) transient textures and buffers for the frame-global view layout.
    ///
    /// On a cache miss, runs transient resolution under the `render::transient_resolve` scope and
    /// inserts the result into `transient_by_key`; otherwise returns the cached entry in place.
    fn resolve_frame_global_transients<'t>(
        &self,
        resolved: &ResolvedView<'_>,
        transient_by_key: &'t mut HashMap<GraphResolveKey, GraphResolvedResources>,
        device: &wgpu::Device,
        backend: &mut dyn GraphExecutionBackend,
        gpu_limits: &crate::gpu::GpuLimits,
    ) -> Result<&'t mut GraphResolvedResources, GraphExecuteError> {
        let key = GraphResolveKey::from_resolved(resolved);
        match transient_by_key.entry(key) {
            Entry::Vacant(v) => {
                profiling::scope!("render::transient_resolve");
                let mut resources = GraphResolvedResources::with_capacity(
                    self.transient_textures.len(),
                    self.transient_buffers.len(),
                    self.imported_textures.len(),
                    self.imported_buffers.len(),
                    self.subresources.len(),
                );
                let alloc_viewport = helpers::clamp_viewport_for_transient_alloc(
                    resolved.viewport_px,
                    gpu_limits.max_texture_dimension_2d(),
                );
                let scene_color_format = backend.scene_color_format_wgpu();
                self.resolve_transient_textures(
                    device,
                    gpu_limits,
                    backend.transient_pool_mut(),
                    TransientTextureResolveSurfaceParams {
                        viewport_px: alloc_viewport,
                        surface_format: resolved.surface_format,
                        depth_stencil_format: resolved.depth_texture.format(),
                        scene_color_format,
                        sample_count: resolved.sample_count,
                        multiview_stereo: resolved.multiview_stereo,
                    },
                    &mut resources,
                )?;
                self.resolve_transient_buffers(
                    device,
                    gpu_limits,
                    backend.transient_pool_mut(),
                    alloc_viewport,
                    &mut resources,
                )?;
                self.resolve_subresource_views(&mut resources);
                Ok(v.insert(resources))
            }
            Entry::Occupied(o) => Ok(o.into_mut()),
        }
    }
}
