//! [`CompiledRenderGraph`] execution: multi-view scheduling, resource resolution, and submits.
//!
//! ## Submit model
//!
//! Multi-view execution records optional frame-global work plus per-view graph work, then submits
//! the whole batch through a single [`wgpu::Queue::submit`] call. The standard path records
//! phase-specific command buffers; the one-view serial swapchain path can record frame-global and
//! per-view work into one command encoder to avoid a second finish. Per-view graph upload writes
//! (per-draw slab, frame uniforms, cluster params) are drained before submit, so each view's GPU
//! commands see coherent buffer contents. Each view owns its own per-draw slab buffer, so views
//! never compete for per-draw storage capacity. World-mesh slab/frame-uniform uploads are prepared
//! before pass-node recording begins.
//!
//! ## Pass dispatch
//!
//! Each retained pass is a [`super::super::pass::PassNode`] enum. The executor matches on the
//! variant to call the correct record method:
//! - `Raster` -> graph opens `wgpu::RenderPass` from template; calls `record_raster`.
//! - `Compute` -> passes receive raw encoder; calls `record_compute`.

use hashbrown::HashMap;
use std::time::Instant;

use crate::cpu_parallelism::{
    FrameCpuWorkload, FrameParallelPolicy, ParallelAdmission, record_parallel_admission,
};
use crate::diagnostics::{PerViewHudConfig, PerViewHudOutputs};
use crate::gpu::{GpuContext, GpuLimits};
use crate::render_graph::GraphExecutionBackend;
use crate::render_graph::blackboard::GraphCommandStats;
use crate::render_graph::execution_backend::{
    GraphAssetResources, GraphFrameResources, GraphViewBlackboardPreparer,
};
use crate::scene::SceneCoordinator;

use super::super::context::{GraphResolvedResources, PostSubmitContext};
use super::super::error::GraphExecuteError;
use super::super::frame_upload_batch::{FrameUploadBatch, GraphUploadSink};
use super::{CompiledRenderGraph, FrameView, FrameViewTarget, MultiViewExecutionContext};
use crate::camera::{HostCameraFrame, ViewId};
use crate::graph_inputs::{FrameSystemsShared, PerViewFramePlan};
use crate::materials::MaterialSystem;
use crate::mesh_deform::{GpuSkinCache, MeshPreprocessPipelines};
use crate::occlusion::OcclusionGraphHook;

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
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

/// Per-view pre-record work items assigned to one blackboard-preparation worker.
const PRE_RECORD_VIEW_PREP_PARALLEL_CHUNK_VIEWS: usize = 1;

struct SingleSwapchainGraphRecord {
    command: Option<TimedCommandBuffer>,
    per_view_batch: RecordedPerViewBatch,
    encode_ms: f64,
    finish_ms: f64,
}

struct GraphCommandRecordingPlan {
    path: GraphCommandRecordingPath,
    estimated_per_view_record_work: usize,
    per_view_record_admission: ParallelAdmission,
}

struct GraphCommandRecordingInputs<'a, 'view> {
    views: &'a [FrameView<'view>],
    per_view_work_items: Vec<PerViewWorkItem>,
    transient_by_key: &'a mut HashMap<GraphResolveKey, GraphResolvedResources>,
    upload_batch: &'a FrameUploadBatch,
    plan: GraphCommandRecordingPlan,
    command_diagnostics: &'a mut CommandEncodingDiagnostics,
}

struct ViewBlackboardPrepareShared<'a> {
    scene: &'a SceneCoordinator,
    device: &'a wgpu::Device,
    gpu_limits: &'a GpuLimits,
    upload_batch: &'a FrameUploadBatch,
    preparer: &'a dyn GraphViewBlackboardPreparer,
    occlusion: &'a dyn OcclusionGraphHook,
    frame_resources: &'a dyn GraphFrameResources,
    materials: &'a MaterialSystem,
    asset_resources: &'a dyn GraphAssetResources,
    mesh_preprocess: Option<&'a MeshPreprocessPipelines>,
    skin_cache: Option<&'a GpuSkinCache>,
    debug_hud: PerViewHudConfig,
    scene_color_format: wgpu::TextureFormat,
}

mod diagnostics;
mod per_view;
mod submit;
mod types;

use diagnostics::{CommandEncodingDiagnostics, TransientPoolMetricsDelta};
use submit::release_transients_and_gc;
use types::{
    DrainedUploadCommand, GraphCommandRecordingPath, GraphResolveKey, OwnedResolvedView,
    PerViewEncodeOutput, PerViewRecordInputs, PerViewRecordOutput, PerViewRecordShared,
    PerViewWorkItem, PreparedPerViewFrameInput, RecordedPerViewBatch, ResolvedOffscreenColorCopy,
    SubmitFrameBatchStats, SubmitFrameInputs, TimedCommandBuffer,
    TransientTextureResolveSurfaceParams,
};

impl CompiledRenderGraph {
    /// Ordered pass count.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Whether this graph targets the swapchain this frame.
    #[cfg(test)]
    pub fn needs_surface_acquire(&self) -> bool {
        self.needs_surface_acquire
    }

    /// Records one view work item and wraps the encoded command buffer with submit-order metadata.
    fn record_per_view_work_item_output(
        &self,
        work_item: PerViewWorkItem,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        per_view_shared: &PerViewRecordShared<'_>,
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
            },
        ))
    }

    /// Records all graph views and submits them in a single [`wgpu::Queue::submit`] call.
    ///
    /// ## Per-view write ordering
    ///
    /// Per-view graph upload writes (per-draw slab, frame uniforms, cluster params) happen
    /// during pre-record world-mesh frame planning. Since all writes are drained before the single
    /// submit, wgpu guarantees they are visible to every GPU command in that submit. Each view
    /// owns its own per-draw slab buffer (keyed by [`ViewId`]), so views never compete for buffer
    /// space.
    ///
    /// ## Per-view frame plan
    ///
    /// A [`crate::graph_inputs::PerViewFramePlanSlot`] is inserted into each view's
    /// per-view blackboard carrying the per-view `@group(0)` frame bind group and uniform buffer.
    pub(crate) fn execute_multi_view(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut dyn GraphExecutionBackend,
        views: &mut [FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::execute_multi_view");
        if views.is_empty() {
            return Ok(());
        }

        let Some((mut swapchain_scope, backbuffer_view_holder)) = ({
            profiling::scope!("graph::enter_swapchain_scope");
            self.enter_swapchain_scope_for_views(gpu, views)?
        }) else {
            return Ok(());
        };

        let device_arc = gpu.device().clone();
        let queue_arc = gpu.queue().clone();
        let limits_arc = gpu.limits().clone();
        let device = device_arc.as_ref();
        let gpu_limits = limits_arc.as_ref();

        let mut command_diagnostics = CommandEncodingDiagnostics::new(self, views.len());

        let mut mv_ctx = MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            backbuffer_view_holder: &backbuffer_view_holder,
        };

        let mut transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources> = HashMap::new();
        self.begin_generation_and_pre_resolve_transients(
            &mut mv_ctx,
            views,
            &mut transient_by_key,
            &mut command_diagnostics,
        )?;

        // Deferred graph upload sink shared by pre-record, frame-global, and per-view paths.
        // Drained onto the main thread after all recording completes and before submit.
        let upload_batch = FrameUploadBatch::new();

        let (per_view_work_items, prepare_resources_ms) =
            self.prepare_resources_and_work_items(&mut mv_ctx, views, &upload_batch)?;
        command_diagnostics.prepare_resources_ms = prepare_resources_ms;
        let recording_plan = self.graph_command_recording_plan(views, &per_view_work_items);
        command_diagnostics.recording_path = recording_plan.path;

        // -- Graph command recording ----------------------------------------------------------
        let (
            frame_global_cmd,
            RecordedPerViewBatch {
                per_view_cmds,
                per_view_occlusion_info,
                per_view_hud_outputs,
                per_view_profiler_cmd,
                ..
            },
        ) = self.record_graph_commands(
            &mut mv_ctx,
            GraphCommandRecordingInputs {
                views,
                per_view_work_items,
                transient_by_key: &mut transient_by_key,
                upload_batch: &upload_batch,
                plan: recording_plan,
                command_diagnostics: &mut command_diagnostics,
            },
        )?;

        let submit_stats = {
            profiling::scope!("graph::submit_frame_batch");
            self.submit_frame_batch(
                &mut mv_ctx,
                SubmitFrameInputs {
                    views,
                    frame_global_cmd,
                    per_view_cmds,
                    per_view_profiler_cmd,
                    per_view_hud_outputs,
                    per_view_occlusion_info: &per_view_occlusion_info,
                    swapchain_scope: &mut swapchain_scope,
                    backbuffer_view_holder: &backbuffer_view_holder,
                    upload_batch: &upload_batch,
                    queue_arc: &queue_arc,
                },
            )?
        };
        command_diagnostics.apply_submit(submit_stats);
        {
            profiling::scope!("graph::command_diagnostics");
            command_diagnostics.record_flight_event(mv_ctx.gpu);
            command_diagnostics.plot();
            command_diagnostics.log_if_slow();
        }

        {
            profiling::scope!("graph::run_post_submit_passes");
            self.run_post_submit_passes(&mut mv_ctx, views, device, &per_view_occlusion_info)?;
        }

        {
            profiling::scope!("graph::release_transients_and_gc");
            release_transients_and_gc(&mut mv_ctx, transient_by_key);
        }

        Ok(())
    }

    /// Selects the command-recording path and captures its admission metrics.
    fn graph_command_recording_plan(
        &self,
        views: &[FrameView<'_>],
        per_view_work_items: &[PerViewWorkItem],
    ) -> GraphCommandRecordingPlan {
        let (estimated_per_view_record_work, per_view_record_admission) =
            self.per_view_record_admission_for_work_items(per_view_work_items, views.len());
        GraphCommandRecordingPlan {
            path: select_graph_command_recording_path(
                views.len(),
                single_view_targets_swapchain(views),
                per_view_record_admission,
                self.schedule
                    .recording_plan
                    .phase_has_parallel_batches(crate::render_graph::pass::PassPhase::PerView),
            ),
            estimated_per_view_record_work,
            per_view_record_admission,
        }
    }

    /// Begins a transient generation and pre-resolves shared graph resources for every view key.
    fn begin_generation_and_pre_resolve_transients(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &mut [FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        command_diagnostics: &mut CommandEncodingDiagnostics,
    ) -> Result<(), GraphExecuteError> {
        let transient_metrics_before = mv_ctx.backend.transient_pool_mut().metrics();
        mv_ctx.backend.transient_pool_mut().begin_generation();
        let pre_resolve_start = Instant::now();
        {
            profiling::scope!("graph::pre_resolve_transients_for_views");
            self.pre_resolve_transients_for_views(mv_ctx, views, transient_by_key)?;
        }
        command_diagnostics.pre_resolve_ms = elapsed_ms(pre_resolve_start);
        command_diagnostics.transient_delta = TransientPoolMetricsDelta::from_metrics(
            transient_metrics_before,
            mv_ctx.backend.transient_pool_mut().metrics(),
        );
        Ok(())
    }

    /// Records graph command buffers through the selected command-recording path.
    fn record_graph_commands(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        inputs: GraphCommandRecordingInputs<'_, '_>,
    ) -> Result<(Option<wgpu::CommandBuffer>, RecordedPerViewBatch), GraphExecuteError> {
        let GraphCommandRecordingInputs {
            views,
            per_view_work_items,
            transient_by_key,
            upload_batch,
            plan,
            command_diagnostics,
        } = inputs;
        match plan.path {
            GraphCommandRecordingPath::StandardCommandBuffers => {
                let frame_global_cmd = self.encode_frame_global_command(
                    mv_ctx,
                    views,
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
                        plan.estimated_per_view_record_work,
                        plan.per_view_record_admission,
                    )?;
                    command_diagnostics.apply_per_view(&batch);
                    batch
                };
                Ok((frame_global_cmd, batch))
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
                    per_view_work_items,
                    transient_by_key,
                    upload_batch,
                )?;
                command_diagnostics.apply_single_swapchain(single.encode_ms, single.finish_ms);
                command_diagnostics.apply_per_view(&single.per_view_batch);
                let frame_global_cmd = single.command.map(|command| command.command_buffer);
                Ok((frame_global_cmd, single.per_view_batch))
            }
        }
    }

    /// Prepares shared frame resources and owned per-view work packets before recording.
    fn prepare_resources_and_work_items(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &mut [FrameView<'_>],
        upload_batch: &FrameUploadBatch,
    ) -> Result<(Vec<PerViewWorkItem>, f64), GraphExecuteError> {
        let prepare_resources_start = Instant::now();
        {
            profiling::scope!("graph::prepare_resources_for_views");
            Self::prepare_view_resources_for_views(mv_ctx, views, upload_batch)?;
        }
        let mut per_view_work_items = {
            profiling::scope!("graph::prepare_work_items");
            self.prepare_per_view_work_items(mv_ctx, views)?
        };
        {
            profiling::scope!("graph::prepare_view_blackboards_for_work_items");
            self.prepare_view_blackboards_for_work_items(
                mv_ctx,
                &mut per_view_work_items,
                upload_batch,
            );
        }
        Ok((per_view_work_items, elapsed_ms(prepare_resources_start)))
    }

    /// Records optional frame-global graph work and folds its diagnostics into the frame report.
    fn encode_frame_global_command(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        command_diagnostics: &mut CommandEncodingDiagnostics,
    ) -> Result<Option<wgpu::CommandBuffer>, GraphExecuteError> {
        let frame_global_cmd = {
            profiling::scope!("graph::encode_frame_global_batch");
            self.encode_frame_global_passes(mv_ctx, views, transient_by_key, upload_batch)?
        };
        Ok(frame_global_cmd.map(|command| {
            command_diagnostics.apply_frame_global(&command);
            command.command_buffer
        }))
    }

    /// Enters [`SwapchainScope`] for `views` if any target the swapchain; `Ok(None)` signals a frame skip.
    ///
    /// The scope holds the [`wgpu::SurfaceTexture`] for the entire frame. After all encoders
    /// are finished, the texture is taken out of the scope via
    /// [`super::super::swapchain_scope::SwapchainScope::take_surface_texture`] and handed to
    /// the driver thread for `Queue::submit` + `SurfaceTexture::present`. On any early return
    /// before the handoff, the scope still presents on drop so the wgpu Vulkan acquire
    /// semaphore is returned to the pool.
    fn enter_swapchain_scope_for_views(
        &self,
        gpu: &mut GpuContext,
        views: &[FrameView<'_>],
    ) -> Result<
        Option<(
            super::super::swapchain_scope::SwapchainScope,
            Option<wgpu::TextureView>,
        )>,
        GraphExecuteError,
    > {
        let needs_swapchain = views
            .iter()
            .any(|v| matches!(v.target, FrameViewTarget::Swapchain));
        match super::super::swapchain_scope::SwapchainScope::enter(
            needs_swapchain,
            self.needs_surface_acquire,
            gpu,
        )? {
            super::super::swapchain_scope::SwapchainEnterOutcome::NotNeeded => Ok(Some((
                super::super::swapchain_scope::SwapchainScope::none(),
                None,
            ))),
            super::super::swapchain_scope::SwapchainEnterOutcome::SkipFrame => Ok(None),
            super::super::swapchain_scope::SwapchainEnterOutcome::Acquired(scope) => {
                let bb = scope.backbuffer_view().cloned();
                Ok(Some((scope, bb)))
            }
        }
    }

    /// Records per-view command buffers and resolves per-view profiler queries.
    ///
    /// Builds [`PerViewRecordShared`] from `mv_ctx`, drives [`Self::record_per_view_outputs`], and
    /// splits the owned outputs into the parallel vectors consumed by [`SubmitFrameInputs`].
    fn record_per_view_batch(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        per_view_work_items: Vec<PerViewWorkItem>,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        estimated_record_work: usize,
        admission: ParallelAdmission,
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
                    profiler: per_view_profiler.as_ref(),
                },
                n_views,
                estimated_record_work,
                admission,
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
            for output in per_view_outputs {
                encode_ms += output.encode_ms;
                finish_ms += output.finish_ms;
                max_finish_ms = f64::max(max_finish_ms, output.finish_ms);
                command_stats.add(output.command_stats);
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
            })
        })();
        mv_ctx.gpu.restore_gpu_profiler(per_view_profiler);

        record_result
    }

    /// Records frame-global work and one serial swapchain view into a single command encoder.
    fn record_single_swapchain_graph_command(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
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
                views,
                transient_by_key,
                &mut encoder,
                upload_batch,
                graph_profiler.as_ref(),
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
        views: &[FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        encoder: &mut wgpu::CommandEncoder,
        upload_batch: &FrameUploadBatch,
        graph_profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<bool, GraphExecuteError> {
        if self.frame_global_passes_are_inactive(&*mv_ctx.backend) {
            return Ok(false);
        }
        profiling::scope!("graph::single_swapchain_encoder::frame_global");
        let frame_global_query =
            graph_profiler.map(|p| p.begin_query("graph::frame_global", encoder));
        self.record_frame_global_passes_into_encoder(
            mv_ctx,
            views,
            transient_by_key,
            encoder,
            upload_batch,
            graph_profiler,
        )?;
        if let Some(query) = frame_global_query
            && let Some(profiler) = graph_profiler
        {
            profiler.end_query(encoder, query);
        }
        Ok(true)
    }

    /// Runs frame-global and per-view `post_submit` hooks on every pass in schedule order.
    fn run_post_submit_passes(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
        device: &wgpu::Device,
        per_view_occlusion_info: &[(ViewId, HostCameraFrame)],
    ) -> Result<(), GraphExecuteError> {
        // Frame-global post-submit (uses first view's occlusion slot).
        if let Some((first_occlusion, first_hc)) = per_view_occlusion_info.first().copied() {
            profiling::scope!("graph::post_submit_frame_global");
            let mut post_ctx = PostSubmitContext {
                _device: device,
                _occlusion: mv_ctx.backend.occlusion_mut(),
                _view_id: first_occlusion,
                _host_camera: first_hc,
            };
            for &pass_idx in self.schedule.frame_global_pass_indices() {
                self.passes[pass_idx]
                    .post_submit(&mut post_ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }

        // Per-view post-submit.
        {
            profiling::scope!("graph::post_submit_per_view");
            for (view, (view_id, host_camera)) in views.iter().zip(per_view_occlusion_info.iter()) {
                let _ = view;
                let mut post_ctx = PostSubmitContext {
                    _device: device,
                    _occlusion: mv_ctx.backend.occlusion_mut(),
                    _view_id: *view_id,
                    _host_camera: *host_camera,
                };
                for &pass_idx in self.schedule.per_view_pass_indices() {
                    self.passes[pass_idx]
                        .post_submit(&mut post_ctx)
                        .map_err(GraphExecuteError::Pass)?;
                }
            }
        }
        Ok(())
    }

    /// Lets backend-specific systems enrich per-view blackboards before graph pass recording.
    fn prepare_view_blackboards_for_work_items(
        &self,
        mv_ctx: &MultiViewExecutionContext<'_>,
        work_items: &mut [PerViewWorkItem],
        upload_batch: &FrameUploadBatch,
    ) {
        profiling::scope!("graph::prepare_view_blackboards");
        let backend = &*mv_ctx.backend;
        let total_draw_count = work_items
            .iter()
            .map(|work_item| work_item.estimated_draw_count)
            .sum::<usize>();
        let preparer = backend.view_blackboard_preparer();
        let shared = ViewBlackboardPrepareShared {
            scene: mv_ctx.scene,
            device: mv_ctx.device,
            gpu_limits: mv_ctx.gpu_limits,
            upload_batch,
            preparer: preparer.as_ref(),
            occlusion: backend.occlusion(),
            frame_resources: backend.frame_resources(),
            materials: backend.materials(),
            asset_resources: backend.asset_resources(),
            mesh_preprocess: backend.mesh_preprocess(),
            skin_cache: backend.skin_cache(),
            debug_hud: backend.per_view_hud_config(),
            scene_color_format: backend.scene_color_format_wgpu(),
        };
        let admission = view_blackboard_prepare_admission(
            FrameParallelPolicy::for_current_thread_pool(),
            work_items.len(),
            total_draw_count,
        );
        if admission.is_parallel() {
            profiling::scope!("graph::prepare_view_blackboards::parallel");
            use rayon::prelude::*;
            work_items
                .par_iter_mut()
                .with_min_len(admission.chunk_size().unwrap_or(1))
                .for_each(|work_item| {
                    self.prepare_one_view_blackboard(&shared, work_item);
                });
        } else {
            profiling::scope!("graph::prepare_view_blackboards::serial");
            for work_item in work_items.iter_mut() {
                self.prepare_one_view_blackboard(&shared, work_item);
            }
        }
    }

    /// Prepares one view's blackboard before command recording.
    fn prepare_one_view_blackboard(
        &self,
        shared: &ViewBlackboardPrepareShared<'_>,
        work_item: &mut PerViewWorkItem,
    ) {
        profiling::scope!("graph::prepare_view_blackboard");
        let resolved = work_item.resolved.as_resolved();
        let frame_params = work_item.frame_input.frame_params(
            FrameSystemsShared {
                scene: shared.scene,
                occlusion: shared.occlusion,
                frame_resources: shared.frame_resources,
                materials: shared.materials,
                asset_resources: shared.asset_resources,
                mesh_preprocess: shared.mesh_preprocess,
                mesh_deform_scratch: None,
                mesh_deform_skin_cache: None,
                skin_cache: shared.skin_cache,
                debug_hud: shared.debug_hud,
            },
            &resolved,
            shared.scene_color_format,
            &work_item.host_camera,
            work_item.render_context,
            work_item.frame_time_seconds,
            work_item.clear,
            work_item.post_processing,
        );
        shared.preparer.prepare_view_blackboard(
            shared.device,
            GraphUploadSink::pre_record_view(shared.upload_batch, work_item.view_idx),
            shared.gpu_limits,
            &frame_params,
            &work_item.frame_input.frame_plan,
            &mut work_item.initial_blackboard,
        );
    }

    /// Prepares owned per-view work items on the main thread before serial or parallel recording.
    fn prepare_per_view_work_items(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &mut [FrameView<'_>],
    ) -> Result<Vec<PerViewWorkItem>, GraphExecuteError> {
        profiling::scope!("graph::prepare_per_view_work_items");
        let mut work_items = Vec::with_capacity(views.len());
        for (view_idx, view) in views.iter_mut().enumerate() {
            let view_id = view.view_id();
            let host_camera = view.host_camera;
            let render_context = view.render_context;
            let frame_time_seconds = view.frame_time_seconds;
            let post_processing = view.post_processing();
            let resolved = Self::resolve_owned_view_from_target(
                view_id,
                view.profile,
                &view.target,
                mv_ctx.gpu,
                mv_ctx.backbuffer_view_holder.as_ref(),
            )?;
            let Some(per_view_frame_bg_and_buf) = mv_ctx
                .backend
                .frame_resources()
                .per_view_frame_bind_group_and_buffer(view_id)
            else {
                logger::warn!(
                    "graph prepare: missing per-view frame resources for view {view_id:?}"
                );
                return Err(GraphExecuteError::MissingPerViewResources {
                    view_id,
                    resource: "frame",
                });
            };
            let (frame_bind_group, frame_uniform_buffer) = per_view_frame_bg_and_buf;
            let frame_plan = PerViewFramePlan {
                frame_bind_group,
                frame_uniform_buffer,
                view_idx,
            };
            let hi_z_slot = mv_ctx.backend.occlusion().ensure_hi_z_state(view_id);
            let frame_input = {
                let resolved_view = resolved.as_resolved();
                PreparedPerViewFrameInput::from_resolved(
                    &resolved_view,
                    frame_plan,
                    mv_ctx.backend.gpu_limits().cloned(),
                    mv_ctx.backend.msaa_depth_resolve(),
                    hi_z_slot,
                )
            };
            let estimated_draw_count = mv_ctx
                .backend
                .estimate_view_blackboard_prepare_draw_count(&view.initial_blackboard);
            work_items.push(PerViewWorkItem {
                view_idx,
                host_camera,
                render_context,
                frame_time_seconds,
                view_id,
                clear: view.clear,
                post_processing,
                initial_blackboard: std::mem::take(&mut view.initial_blackboard),
                resolved,
                frame_input,
                estimated_draw_count,
            });
        }
        Ok(work_items)
    }
}

/// Returns the Rayon admission decision for per-view blackboard preparation.
fn view_blackboard_prepare_admission(
    policy: FrameParallelPolicy,
    view_count: usize,
    total_draw_count: usize,
) -> ParallelAdmission {
    policy.admit_draw_heavy_views(
        FrameCpuWorkload::view_draws(view_count, total_draw_count),
        PRE_RECORD_VIEW_PREP_PARALLEL_CHUNK_VIEWS,
    )
}

fn single_view_targets_swapchain(views: &[FrameView<'_>]) -> bool {
    views.len() == 1 && matches!(&views[0].target, FrameViewTarget::Swapchain)
}

fn select_graph_command_recording_path(
    view_count: usize,
    single_view_targets_swapchain: bool,
    per_view_admission: ParallelAdmission,
    has_parallel_per_view_batches: bool,
) -> GraphCommandRecordingPath {
    profiling::scope!("graph::recording_path_selection");
    if view_count == 1
        && single_view_targets_swapchain
        && !per_view_admission.is_parallel()
        && !has_parallel_per_view_batches
    {
        GraphCommandRecordingPath::SingleSwapchainEncoder
    } else {
        GraphCommandRecordingPath::StandardCommandBuffers
    }
}

mod pre_warm;
mod recording;
mod resolve;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_recording_path_selects_single_swapchain_encoder_for_serial_swapchain_view() {
        assert_eq!(
            select_graph_command_recording_path(1, true, ParallelAdmission::Serial, false),
            GraphCommandRecordingPath::SingleSwapchainEncoder
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_multi_view() {
        assert_eq!(
            select_graph_command_recording_path(2, false, ParallelAdmission::Serial, false),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_non_swapchain_view() {
        assert_eq!(
            select_graph_command_recording_path(1, false, ParallelAdmission::Serial, false),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_rayon_admitted_work() {
        assert_eq!(
            select_graph_command_recording_path(
                1,
                true,
                ParallelAdmission::Parallel { chunk_size: 1 },
                false
            ),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_scheduler_parallel_work() {
        assert_eq!(
            select_graph_command_recording_path(1, true, ParallelAdmission::Serial, true),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }
}
