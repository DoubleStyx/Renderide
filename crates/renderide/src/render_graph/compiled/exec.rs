//! [`CompiledRenderGraph`] execution: multi-view scheduling, resource resolution, and submits.
//!
//! ## Submit model
//!
//! Multi-view execution records optional frame-global work plus one command buffer per view, then
//! submits the whole batch through a single [`wgpu::Queue::submit`] call. Per-view graph upload
//! writes (per-draw slab, frame uniforms, cluster params) are drained
//! before submit, so each view's GPU commands see coherent buffer contents. Each view owns its
//! own per-draw slab buffer, so views never compete for per-draw storage capacity. World-mesh
//! slab/frame-uniform uploads are prepared before pass-node recording begins.
//!
//! ## Pass dispatch
//!
//! Each retained pass is a [`super::super::pass::PassNode`] enum. The executor matches on the
//! variant to call the correct record method:
//! - `Raster` -> graph opens `wgpu::RenderPass` from template; calls `record_raster`.
//! - `Compute` -> passes receive raw encoder; calls `record_compute`.

use hashbrown::HashMap;
use std::time::Instant;

use crate::diagnostics::PerViewHudOutputs;
use crate::gpu::GpuContext;
use crate::render_graph::GraphExecutionBackend;
use crate::render_graph::blackboard::GraphCommandStats;
use crate::scene::SceneCoordinator;

use super::super::context::{GraphResolvedResources, PostSubmitContext};
use super::super::error::GraphExecuteError;
use super::super::frame_params::{FrameSystemsShared, PerViewFramePlan};
use super::super::frame_upload_batch::{FrameUploadBatch, GraphUploadSink};
use super::{CompiledRenderGraph, FrameView, FrameViewTarget, MultiViewExecutionContext, helpers};
use crate::camera::{HostCameraFrame, ViewId};

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

mod diagnostics;
mod per_view;
mod submit;
mod types;

use diagnostics::{CommandEncodingDiagnostics, TransientPoolMetricsDelta};
use submit::release_transients_and_gc;
use types::{
    DrainedUploadCommand, GraphResolveKey, OwnedResolvedView, PerViewEncodeOutput,
    PerViewRecordInputs, PerViewRecordOutput, PerViewRecordShared, PerViewWorkItem,
    RecordedPerViewBatch, SubmitFrameBatchStats, SubmitFrameInputs, TimedCommandBuffer,
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
                command_buffer: encoded.command_buffer,
                hud_outputs: encoded.hud_outputs,
                encode_ms: encoded.encode_ms,
                finish_ms: encoded.finish_ms,
                command_stats: encoded.command_stats,
            },
        ))
    }

    /// Records all views into separate command encoders and submits them in a single
    /// [`wgpu::Queue::submit`] call alongside the frame-global encoder.
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
    /// A [`super::super::frame_params::PerViewFramePlanSlot`] is inserted into each view's
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

        let transient_metrics_before = backend.transient_pool_mut().metrics();
        backend.transient_pool_mut().begin_generation();
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

        // Pre-resolve transient textures and buffers for every unique view key before any
        // per-view recording begins. The record loop then reads `transient_by_key` without
        // touching the shared transient pool.
        let pre_resolve_start = Instant::now();
        {
            profiling::scope!("graph::pre_resolve_transients_for_views");
            self.pre_resolve_transients_for_views(&mut mv_ctx, views, &mut transient_by_key)?;
        }
        command_diagnostics.pre_resolve_ms = elapsed_ms(pre_resolve_start);
        command_diagnostics.transient_delta = TransientPoolMetricsDelta::from_metrics(
            transient_metrics_before,
            mv_ctx.backend.transient_pool_mut().metrics(),
        );

        // Deferred graph upload sink shared by pre-record, frame-global, and per-view paths.
        // Drained onto the main thread after all recording completes and before submit.
        let upload_batch = FrameUploadBatch::new();

        let (per_view_work_items, prepare_resources_ms) =
            self.prepare_resources_and_work_items(&mut mv_ctx, views, &upload_batch)?;
        command_diagnostics.prepare_resources_ms = prepare_resources_ms;

        // -- Frame-global pass (optional) -----------------------------------------------------
        let frame_global_cmd = self.encode_frame_global_command(
            &mut mv_ctx,
            views,
            &mut transient_by_key,
            &upload_batch,
            &mut command_diagnostics,
        )?;

        // -- Per-view recording (no submit per view) ------------------------------------------
        let RecordedPerViewBatch {
            per_view_cmds,
            per_view_occlusion_info,
            per_view_hud_outputs,
            per_view_profiler_cmd,
            ..
        } = {
            profiling::scope!("graph::record_per_view_batch");
            let batch = self.record_per_view_batch(
                &mut mv_ctx,
                per_view_work_items,
                &transient_by_key,
                &upload_batch,
            )?;
            command_diagnostics.apply_per_view(&batch);
            batch
        };

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
            gpu_limits_arc: mv_ctx.backend.gpu_limits().cloned(),
            msaa_depth_resolve: mv_ctx.backend.msaa_depth_resolve(),
            live_gtao_settings: mv_ctx.backend.live_gtao_settings(),
            live_bloom_settings: mv_ctx.backend.live_bloom_settings(),
            live_motion_blur_settings: mv_ctx.backend.live_motion_blur_settings(),
            live_auto_exposure_settings: mv_ctx.backend.live_auto_exposure_settings(),
            wall_frame_delta_seconds: mv_ctx.backend.wall_frame_delta_seconds(),
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
                per_view_cmds.push(output.command_buffer);
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
        for work_item in work_items.iter_mut() {
            profiling::scope!("graph::prepare_view_blackboard");
            let resolved = work_item.resolved.as_resolved();
            let hi_z_slot = mv_ctx
                .backend
                .occlusion()
                .ensure_hi_z_state(resolved.view_id);
            let frame_params = helpers::frame_render_params_from_shared(
                FrameSystemsShared {
                    scene: mv_ctx.scene,
                    occlusion: mv_ctx.backend.occlusion(),
                    frame_resources: mv_ctx.backend.frame_resources(),
                    materials: mv_ctx.backend.materials(),
                    asset_resources: mv_ctx.backend.asset_resources(),
                    mesh_preprocess: mv_ctx.backend.mesh_preprocess(),
                    mesh_deform_scratch: None,
                    mesh_deform_skin_cache: None,
                    skin_cache: mv_ctx.backend.skin_cache(),
                    debug_hud: mv_ctx.backend.per_view_hud_config(),
                },
                helpers::GraphPassFrameViewInputs {
                    resolved: &resolved,
                    scene_color_format: mv_ctx.backend.scene_color_format_wgpu(),
                    host_camera: &work_item.host_camera,
                    render_context: work_item.render_context,
                    clear: work_item.clear,
                    post_processing: work_item.post_processing,
                    gpu_limits: mv_ctx.backend.gpu_limits().cloned(),
                    msaa_depth_resolve: mv_ctx.backend.msaa_depth_resolve(),
                    hi_z_slot,
                },
            );
            let (frame_bg, frame_buf) = &work_item.per_view_frame_bg_and_buf;
            let frame_plan = PerViewFramePlan {
                frame_bind_group: std::sync::Arc::clone(frame_bg),
                frame_uniform_buffer: frame_buf.clone(),
                view_idx: work_item.view_idx,
            };
            mv_ctx.backend.prepare_view_blackboard(
                mv_ctx.device,
                GraphUploadSink::pre_record(upload_batch),
                mv_ctx.gpu_limits,
                &frame_params,
                &frame_plan,
                &mut work_item.initial_blackboard,
            );
        }
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
            let resolved = Self::resolve_owned_view_from_target(
                view_id,
                view.post_processing,
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
            work_items.push(PerViewWorkItem {
                view_idx,
                host_camera,
                render_context,
                view_id,
                clear: view.clear,
                post_processing: view.post_processing,
                initial_blackboard: std::mem::take(&mut view.initial_blackboard),
                resolved,
                per_view_frame_bg_and_buf,
            });
        }
        Ok(work_items)
    }
}

mod pre_warm;
mod recording;
mod resolve;
