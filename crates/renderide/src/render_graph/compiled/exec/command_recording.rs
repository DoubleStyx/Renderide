//! High-level command-buffer recording orchestration for compiled graph execution.

use hashbrown::HashMap;
use std::time::Instant;

use crate::camera::{HostCameraFrame, ViewId};
use crate::cpu_parallelism::record_parallel_admission;
use crate::gpu::GpuRetainedResources;
use crate::hud_contract::PerViewHudOutputs;
use crate::render_graph::blackboard::GraphCommandStats;

use super::recording_path::GraphCommandRecordingPlan;
use super::recording_path::GraphCommandRecordingStrategy;
use super::{
    CommandEncodingDiagnostics, CompiledRenderGraph, FrameGlobalPassRecordInputs, FrameGlobalView,
    FrameUploadBatch, GraphCommandRecordingPath, GraphExecuteError, GraphResolveKey,
    GraphResolvedResources, MultiViewExecutionContext, PerViewRecordInputs, PerViewRecordOutput,
    PerViewRecordShared, PerViewWorkItem, RecordedPerViewBatch, TimedCommandBuffer, elapsed_ms,
};

struct SingleSwapchainGraphRecord {
    command: Option<TimedCommandBuffer>,
    per_view_batch: RecordedPerViewBatch,
    encode_ms: f64,
    finish_ms: f64,
}

/// Inputs needed to record frame-global and per-view graph commands.
pub(in crate::render_graph::compiled::exec) struct GraphCommandRecordingInputs<'a, 'view> {
    /// Views being recorded in frame order.
    pub(in crate::render_graph::compiled::exec) views: &'a [super::FrameView<'view>],
    /// Frame-global graph view.
    pub(in crate::render_graph::compiled::exec) frame_global: &'a FrameGlobalView,
    /// Owned per-view work packets prepared before recording.
    pub(in crate::render_graph::compiled::exec) per_view_work_items: Vec<PerViewWorkItem>,
    /// Pre-resolved transient resources keyed by compatible view layouts.
    pub(in crate::render_graph::compiled::exec) transient_by_key:
        &'a mut HashMap<GraphResolveKey, GraphResolvedResources>,
    /// Deferred upload batch shared by frame-global and per-view recorders.
    pub(in crate::render_graph::compiled::exec) upload_batch: &'a FrameUploadBatch,
    /// Selected recording strategy and its parallelism metadata.
    pub(in crate::render_graph::compiled::exec) plan: GraphCommandRecordingPlan,
    /// Mutable command diagnostics for the current frame.
    pub(in crate::render_graph::compiled::exec) command_diagnostics:
        &'a mut CommandEncodingDiagnostics,
}

fn take_single_per_view_work_item(
    per_view_work_items: Vec<PerViewWorkItem>,
) -> Result<PerViewWorkItem, GraphExecuteError> {
    let mut work_items = per_view_work_items.into_iter();
    let work_item = work_items.next().ok_or(GraphExecuteError::NoViewsInBatch)?;
    if work_items.next().is_some() {
        return Err(GraphExecuteError::NoViewsInBatch);
    }
    Ok(work_item)
}

impl CompiledRenderGraph {
    /// Records one view work item and wraps the encoded command buffer with submit-order metadata.
    pub(in crate::render_graph::compiled::exec) fn record_per_view_work_item_output(
        &self,
        work_item: PerViewWorkItem,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        per_view_shared: &PerViewRecordShared<'_>,
        strategy: GraphCommandRecordingStrategy,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<(usize, PerViewRecordOutput), GraphExecuteError> {
        let view_idx = work_item.view_idx;
        let view_id = work_item.view_id;
        let host_camera = work_item.host_camera;
        let encoded = self.record_one_view(
            per_view_shared,
            work_item,
            transient_by_key,
            upload_batch,
            strategy,
            profiler,
        )?;
        Ok((
            view_idx,
            PerViewRecordOutput {
                view_id,
                host_camera,
                command_buffers: encoded.command_buffers,
                hud_outputs: encoded.hud_outputs,
                encode_ms: encoded.encode_ms,
                finish_ms: encoded.finish_ms,
                command_stats: encoded.command_stats,
                retained_resources: encoded.retained_resources,
            },
        ))
    }

    /// Records graph command buffers through the selected command-recording path.
    pub(in crate::render_graph::compiled::exec) fn record_graph_commands(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        inputs: GraphCommandRecordingInputs<'_, '_>,
    ) -> Result<(Vec<wgpu::CommandBuffer>, RecordedPerViewBatch), GraphExecuteError> {
        let GraphCommandRecordingInputs {
            views,
            frame_global,
            per_view_work_items,
            transient_by_key,
            upload_batch,
            plan,
            command_diagnostics,
        } = inputs;
        let path = if plan.path == GraphCommandRecordingPath::SingleSwapchainEncoder
            && self.frame_global_has_split_workload(&*mv_ctx.backend)
        {
            command_diagnostics.recording_path = GraphCommandRecordingPath::StandardCommandBuffers;
            GraphCommandRecordingPath::StandardCommandBuffers
        } else {
            plan.path
        };
        match path {
            GraphCommandRecordingPath::StandardCommandBuffers => {
                let frame_global_cmds = self.encode_frame_global_command(
                    mv_ctx,
                    views,
                    frame_global,
                    transient_by_key,
                    upload_batch,
                    command_diagnostics,
                )?;
                let batch = {
                    profiling::scope!("graph::record_per_view_batch");
                    let batch = self.record_per_view_batch(
                        mv_ctx,
                        per_view_work_items,
                        transient_by_key,
                        upload_batch,
                        plan,
                    )?;
                    command_diagnostics.apply_per_view(&batch);
                    batch
                };
                Ok((frame_global_cmds, batch))
            }
            GraphCommandRecordingPath::SingleSwapchainEncoder => {
                record_parallel_admission(
                    "graph_record_per_view",
                    plan.estimated_per_view_record_work,
                    views.len(),
                    plan.per_view_record_admission,
                );
                let single = self.record_single_swapchain_graph_command(
                    mv_ctx,
                    views,
                    frame_global,
                    per_view_work_items,
                    transient_by_key,
                    upload_batch,
                )?;
                command_diagnostics.apply_single_swapchain(single.encode_ms, single.finish_ms);
                command_diagnostics.apply_per_view(&single.per_view_batch);
                let frame_global_cmds = single
                    .command
                    .map(|command| vec![command.command_buffer])
                    .unwrap_or_default();
                Ok((frame_global_cmds, single.per_view_batch))
            }
        }
    }

    /// Records optional frame-global graph work and folds its diagnostics into the frame report.
    fn encode_frame_global_command(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[super::FrameView<'_>],
        frame_global: &FrameGlobalView,
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        command_diagnostics: &mut CommandEncodingDiagnostics,
    ) -> Result<Vec<wgpu::CommandBuffer>, GraphExecuteError> {
        let frame_global_cmds = {
            profiling::scope!("graph::encode_frame_global_batch");
            self.encode_frame_global_passes(
                mv_ctx,
                views,
                frame_global,
                transient_by_key,
                upload_batch,
            )?
        };
        command_diagnostics.apply_frame_global(&frame_global_cmds);
        Ok(frame_global_cmds
            .into_iter()
            .map(|command| command.command_buffer)
            .collect())
    }

    /// Records per-view command buffers and resolves per-view profiler queries.
    fn record_per_view_batch(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        per_view_work_items: Vec<PerViewWorkItem>,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        plan: GraphCommandRecordingPlan,
    ) -> Result<RecordedPerViewBatch, GraphExecuteError> {
        let n_views = per_view_work_items.len();
        let device = mv_ctx.device;
        let per_view_shared = PerViewRecordShared {
            scene: mv_ctx.scene,
            device,
            gpu_limits: mv_ctx.gpu_limits,
            occlusion: mv_ctx.backend.occlusion(),
            frame_resources: mv_ctx.backend.frame_resources(),
            history: mv_ctx.backend.history_registry(),
            materials: mv_ctx.backend.materials(),
            asset_resources: mv_ctx.backend.asset_resources(),
            mesh_preprocess: mv_ctx.backend.mesh_preprocess(),
            skin_cache: mv_ctx.backend.skin_cache(),
            skin_weight_mode: mv_ctx.backend.skin_weight_mode(),
            debug_hud: mv_ctx.backend.per_view_hud_config(),
            scene_color_format: mv_ctx.backend.scene_color_format_wgpu(),
        };
        let mut per_view_profiler = mv_ctx.gpu.take_gpu_profiler();
        let record_result = (|| -> Result<RecordedPerViewBatch, GraphExecuteError> {
            let per_view_outputs = self.record_per_view_outputs(
                per_view_work_items,
                PerViewRecordInputs {
                    transient_by_key,
                    upload_batch,
                    per_view_shared: &per_view_shared,
                    strategy: plan.strategy,
                    profiler: per_view_profiler.as_ref(),
                },
                n_views,
                plan.estimated_per_view_record_work,
                plan.per_view_record_admission,
            )?;
            let mut per_view_cmds: Vec<wgpu::CommandBuffer> = Vec::with_capacity(n_views);
            let mut per_view_occlusion_info: Vec<(ViewId, HostCameraFrame)> =
                Vec::with_capacity(n_views);
            let mut per_view_hud_outputs: Vec<Option<PerViewHudOutputs>> =
                Vec::with_capacity(n_views);
            let mut encode_ms = 0.0;
            let mut finish_ms = 0.0;
            let mut max_finish_ms = 0.0;
            let mut command_stats = GraphCommandStats::default();
            let mut retained_resources = GpuRetainedResources::new();
            for output in per_view_outputs {
                encode_ms += output.encode_ms;
                finish_ms += output.finish_ms;
                max_finish_ms = f64::max(max_finish_ms, output.finish_ms);
                command_stats.add(output.command_stats);
                retained_resources.append(output.retained_resources);
                per_view_cmds.extend(output.command_buffers);
                per_view_occlusion_info.push((output.view_id, output.host_camera));
                per_view_hud_outputs.push(output.hud_outputs);
            }
            let per_view_profiler_cmd = per_view_profiler.as_mut().map(|profiler| {
                let mut profiler_encoder = {
                    profiling::scope!("graph::per_view_profiler::create_encoder");
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("render-graph-per-view-profiler-resolve"),
                    })
                };
                profiler.resolve_queries(&mut profiler_encoder);
                {
                    profiling::scope!("CommandEncoder::finish::graph_per_view_profiler");
                    profiler_encoder.finish()
                }
            });
            Ok(RecordedPerViewBatch {
                per_view_cmds,
                per_view_occlusion_info,
                per_view_hud_outputs,
                per_view_profiler_cmd,
                encode_ms,
                finish_ms,
                max_finish_ms,
                command_stats,
                retained_resources,
            })
        })();
        mv_ctx.gpu.restore_gpu_profiler(per_view_profiler);

        record_result
    }

    /// Records frame-global work and one serial swapchain view into a single command encoder.
    fn record_single_swapchain_graph_command(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[super::FrameView<'_>],
        frame_global: &FrameGlobalView,
        per_view_work_items: Vec<PerViewWorkItem>,
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
    ) -> Result<SingleSwapchainGraphRecord, GraphExecuteError> {
        profiling::scope!("graph::single_swapchain_encoder");
        let encode_start = Instant::now();
        let device = mv_ctx.device;
        let mut encoder = {
            profiling::scope!("graph::single_swapchain_encoder::create_encoder");
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-graph-single-swapchain"),
            })
        };
        let mut graph_profiler = mv_ctx.gpu.take_gpu_profiler();
        let record_result = (|| -> Result<SingleSwapchainGraphRecord, GraphExecuteError> {
            let frame_global_active = self.record_single_swapchain_frame_global(
                mv_ctx,
                FrameGlobalPassRecordInputs {
                    views,
                    frame_global,
                    transient_by_key,
                    encoder: &mut encoder,
                    upload_batch,
                    pass_profiler: graph_profiler.as_ref(),
                },
            )?;

            let per_view_shared = PerViewRecordShared {
                scene: mv_ctx.scene,
                device,
                gpu_limits: mv_ctx.gpu_limits,
                occlusion: mv_ctx.backend.occlusion(),
                frame_resources: mv_ctx.backend.frame_resources(),
                history: mv_ctx.backend.history_registry(),
                materials: mv_ctx.backend.materials(),
                asset_resources: mv_ctx.backend.asset_resources(),
                mesh_preprocess: mv_ctx.backend.mesh_preprocess(),
                skin_cache: mv_ctx.backend.skin_cache(),
                skin_weight_mode: mv_ctx.backend.skin_weight_mode(),
                debug_hud: mv_ctx.backend.per_view_hud_config(),
                scene_color_format: mv_ctx.backend.scene_color_format_wgpu(),
            };
            let work_item = take_single_per_view_work_item(per_view_work_items)?;
            let view_id = work_item.view_id;
            let host_camera = work_item.host_camera;
            let per_view_output = self.record_one_view_into_encoder(
                &per_view_shared,
                work_item,
                transient_by_key,
                &mut encoder,
                upload_batch,
                graph_profiler.as_ref(),
            )?;
            if let Some(profiler) = graph_profiler.as_mut() {
                profiling::scope!("graph::single_swapchain_encoder::profiler_resolve");
                profiler.resolve_queries(&mut encoder);
            }

            let command_stats = per_view_output.command_stats;
            let encode_ms = elapsed_ms(encode_start);
            let has_encoder_work = frame_global_active
                || command_stats.has_recorded_work()
                || graph_profiler.is_some();
            let (command, finish_ms) = if has_encoder_work {
                profiling::scope!("CommandEncoder::finish::graph_single_swapchain");
                let finish_start = Instant::now();
                let command_buffer = encoder.finish();
                let finish_ms = elapsed_ms(finish_start);
                (
                    Some(TimedCommandBuffer {
                        command_buffer,
                        encode_ms,
                        finish_ms,
                    }),
                    finish_ms,
                )
            } else {
                (None, 0.0)
            };
            Ok(SingleSwapchainGraphRecord {
                command,
                per_view_batch: RecordedPerViewBatch {
                    per_view_cmds: Vec::new(),
                    per_view_occlusion_info: vec![(view_id, host_camera)],
                    per_view_hud_outputs: vec![per_view_output.hud_outputs],
                    per_view_profiler_cmd: None,
                    encode_ms: per_view_output.encode_ms,
                    finish_ms: 0.0,
                    max_finish_ms: 0.0,
                    command_stats,
                    retained_resources: per_view_output.retained_resources,
                },
                encode_ms,
                finish_ms,
            })
        })();
        mv_ctx.gpu.restore_gpu_profiler(graph_profiler);
        record_result
    }

    /// Records frame-global graph work into the single swapchain encoder when any is active.
    fn record_single_swapchain_frame_global(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        inputs: FrameGlobalPassRecordInputs<'_, '_>,
    ) -> Result<bool, GraphExecuteError> {
        if self.frame_global_passes_are_inactive(&*mv_ctx.backend) {
            return Ok(false);
        }
        let FrameGlobalPassRecordInputs {
            views,
            frame_global,
            transient_by_key,
            encoder,
            upload_batch,
            pass_profiler,
        } = inputs;
        profiling::scope!("graph::single_swapchain_encoder::frame_global");
        let frame_global_query =
            pass_profiler.map(|p| p.begin_query("graph::frame_global", encoder));
        self.record_frame_global_passes_into_encoder(
            mv_ctx,
            FrameGlobalPassRecordInputs {
                views,
                frame_global,
                transient_by_key,
                encoder: &mut *encoder,
                upload_batch,
                pass_profiler,
            },
        )?;
        if let Some(query) = frame_global_query
            && let Some(profiler) = pass_profiler
        {
            profiler.end_query(encoder, query);
        }
        Ok(true)
    }
}
