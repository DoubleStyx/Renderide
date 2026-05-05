//! [`CompiledRenderGraph`] execution: multi-view scheduling, resource resolution, and submits.
//!
//! ## Submit model
//!
//! Multi-view execution records optional frame-global work plus one command buffer per view, then
//! submits the whole batch through a single [`wgpu::Queue::submit`] call. Per-view
//! `Queue::write_buffer` uploads (per-draw slab, frame uniforms, cluster params) are drained
//! before submit, so each view's GPU commands see coherent buffer contents. Each view owns its
//! own per-draw slab buffer, so views never compete for per-draw storage capacity.
//!
//! ## Pass dispatch
//!
//! Each retained pass is a [`super::super::pass::PassNode`] enum. The executor matches on the
//! variant to call the correct record method:
//! - `Raster` -> graph opens `wgpu::RenderPass` from template; calls `record_raster`.
//! - `Compute` -> passes receive raw encoder; calls `record_compute`.
//! - `Copy` -> same as compute; calls `record_copy`.
//! - `Callback` -> no encoder; calls `run_callback`.

use hashbrown::HashMap;
use std::time::Instant;

use crate::backend::BackendGraphAccess;
use crate::diagnostics::PerViewHudOutputs;
use crate::gpu::GpuContext;
use crate::render_graph::WorldMeshDrawPlan;
use crate::scene::SceneCoordinator;

use super::super::context::{GraphResolvedResources, PostSubmitContext};
use super::super::error::GraphExecuteError;
use super::super::frame_upload_batch::FrameUploadBatch;
use super::{CompiledRenderGraph, FrameView, FrameViewTarget, MultiViewExecutionContext};
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
    TransientTextureResolveSurfaceParams, WorldMeshCommandStats,
};

impl CompiledRenderGraph {
    /// Ordered pass count.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Whether this graph targets the swapchain this frame.
    pub fn needs_surface_acquire(&self) -> bool {
        self.needs_surface_acquire
    }

    /// Returns a CPU-side snapshot of the compiled schedule for the debug HUD.
    ///
    /// Captures pass count, wave count, phase distribution, and per-wave pass counts.
    pub fn schedule_hud_snapshot(&self) -> super::super::schedule::ScheduleHudSnapshot {
        super::super::schedule::ScheduleHudSnapshot::from_schedule(&self.schedule)
    }

    /// Validates the compiled schedule for structural invariants
    /// (frame-global before per-view, monotonic waves, wave ranges cover steps).
    ///
    /// Called by tests; production code can use this to surface graph build failures early.
    pub fn validate_schedule(&self) -> Result<(), super::super::schedule::ScheduleValidationError> {
        self.schedule.validate()
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
                world_mesh: encoded.world_mesh,
            },
        ))
    }

    /// Records all views into separate command encoders and submits them in a single
    /// [`wgpu::Queue::submit`] call alongside the frame-global encoder.
    ///
    /// ## Per-view write ordering
    ///
    /// Per-view `Queue::write_buffer` calls (per-draw slab, frame uniforms, cluster params) happen
    /// during per-view callback passes. Since all writes are issued BEFORE the single submit, wgpu
    /// guarantees they are visible to every GPU command in that submit. Each view owns its own
    /// per-draw slab buffer (keyed by [`ViewId`]), so views never compete for buffer
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
        backend: &mut BackendGraphAccess<'_>,
        views: &mut [FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::execute_multi_view");
        if views.is_empty() {
            return Ok(());
        }

        let Some((mut swapchain_scope, backbuffer_view_holder)) =
            self.enter_swapchain_scope_for_views(gpu, views)?
        else {
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
            queue_arc: &queue_arc,
            backbuffer_view_holder: &backbuffer_view_holder,
        };

        let mut transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources> = HashMap::new();

        // Pre-resolve transient textures and buffers for every unique view key before any
        // per-view recording begins. The record loop then reads `transient_by_key` without
        // touching the shared transient pool.
        let pre_resolve_start = Instant::now();
        self.pre_resolve_transients_for_views(&mut mv_ctx, views, &mut transient_by_key)?;
        command_diagnostics.pre_resolve_ms = elapsed_ms(pre_resolve_start);
        command_diagnostics.transient_delta = TransientPoolMetricsDelta::from_metrics(
            transient_metrics_before,
            mv_ctx.backend.transient_pool_mut().metrics(),
        );

        // Deferred `queue.write_buffer` sink shared by frame-global and per-view record paths.
        // Drained onto the main thread after all recording completes and before submit.
        let upload_batch = FrameUploadBatch::new();

        // Shared frame resources, per-view slots, mesh extended streams, and material pipelines
        // are warmed up front so later per-view recording can run with read-only shared state
        // plus per-view interior mutability.
        let prepare_resources_start = Instant::now();
        Self::prepare_view_resources_for_views(&mut mv_ctx, views)?;
        command_diagnostics.prepare_resources_ms = elapsed_ms(prepare_resources_start);

        // -- Frame-global pass (optional) -----------------------------------------------------
        let frame_global_cmd = self.encode_frame_global_passes(
            &mut mv_ctx,
            views,
            &mut transient_by_key,
            &upload_batch,
        )?;
        let frame_global_cmd = frame_global_cmd.map(|command| {
            command_diagnostics.apply_frame_global(&command);
            command.command_buffer
        });
        let per_view_work_items = self.prepare_per_view_work_items(&mut mv_ctx, views)?;

        // -- Per-view recording (no submit per view) ------------------------------------------
        let RecordedPerViewBatch {
            per_view_cmds,
            per_view_occlusion_info,
            per_view_hud_outputs,
            per_view_profiler_cmd,
            ..
        } = {
            let batch = self.record_per_view_batch(
                &mut mv_ctx,
                per_view_work_items,
                &transient_by_key,
                &upload_batch,
            )?;
            command_diagnostics.apply_per_view(&batch);
            batch
        };

        let submit_stats = self.submit_frame_batch(
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
        )?;
        command_diagnostics.apply_submit(submit_stats);
        command_diagnostics.plot();
        command_diagnostics.log_if_slow();

        self.run_post_submit_passes(&mut mv_ctx, views, device, &per_view_occlusion_info)?;

        release_transients_and_gc(&mut mv_ctx, transient_by_key);

        Ok(())
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
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        per_view_work_items: Vec<PerViewWorkItem>,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
    ) -> Result<RecordedPerViewBatch, GraphExecuteError> {
        let n_views = per_view_work_items.len();
        let device = mv_ctx.device;
        let record_parallelism = mv_ctx.backend.record_parallelism;
        let per_view_shared = PerViewRecordShared {
            scene: mv_ctx.scene,
            device,
            gpu_limits: mv_ctx.gpu_limits,
            queue_arc: mv_ctx.queue_arc,
            occlusion: mv_ctx.backend.occlusion(),
            frame_resources: mv_ctx.backend.frame_resources(),
            history: mv_ctx.backend.history_registry(),
            materials: mv_ctx.backend.materials(),
            asset_transfers: mv_ctx.backend.asset_transfers(),
            mesh_preprocess: mv_ctx.backend.mesh_preprocess(),
            skin_cache: mv_ctx.backend.skin_cache(),
            debug_hud: mv_ctx.backend.per_view_hud_config(),
            scene_color_format: mv_ctx.backend.scene_color_format_wgpu(),
            gpu_limits_arc: mv_ctx.backend.gpu_limits().cloned(),
            msaa_depth_resolve: mv_ctx.backend.msaa_depth_resolve(),
            live_gtao_settings: mv_ctx.backend.live_gtao_settings(),
            live_bloom_settings: mv_ctx.backend.live_bloom_settings(),
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
                record_parallelism,
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
            let mut world_mesh = WorldMeshCommandStats::default();
            for output in per_view_outputs {
                encode_ms += output.encode_ms;
                finish_ms += output.finish_ms;
                max_finish_ms = f64::max(max_finish_ms, output.finish_ms);
                world_mesh.add(output.world_mesh);
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
                world_mesh,
            })
        })();
        mv_ctx.gpu.restore_gpu_profiler(per_view_profiler);

        record_result
    }

    /// Runs frame-global and per-view `post_submit` hooks on every pass in schedule order.
    fn run_post_submit_passes(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        views: &[FrameView<'_>],
        device: &wgpu::Device,
        per_view_occlusion_info: &[(ViewId, HostCameraFrame)],
    ) -> Result<(), GraphExecuteError> {
        // Frame-global post-submit (uses first view's occlusion slot).
        if let Some((first_occlusion, first_hc)) = per_view_occlusion_info.first().copied() {
            profiling::scope!("graph::post_submit_frame_global");
            let mut post_ctx = PostSubmitContext {
                device,
                occlusion: mv_ctx.backend.occlusion_mut(),
                view_id: first_occlusion,
                host_camera: first_hc,
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
                    device,
                    occlusion: mv_ctx.backend.occlusion_mut(),
                    view_id: *view_id,
                    host_camera: *host_camera,
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

    /// Prepares owned per-view work items on the main thread before serial or parallel recording.
    fn prepare_per_view_work_items(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        views: &mut [FrameView<'_>],
    ) -> Result<Vec<PerViewWorkItem>, GraphExecuteError> {
        profiling::scope!("graph::prepare_per_view_work_items");
        let mut work_items = Vec::with_capacity(views.len());
        for (view_idx, view) in views.iter_mut().enumerate() {
            let view_id = view.view_id();
            let host_camera = view.host_camera;
            let per_view_frame_bg_and_buf = mv_ctx
                .backend
                .frame_resources
                .per_view_frame(view_id)
                .map(|state| {
                    (
                        state.frame_bind_group.clone(),
                        state.frame_uniform_buffer.clone(),
                    )
                });
            work_items.push(PerViewWorkItem {
                view_idx,
                host_camera,
                view_id,
                draw_filter: view.draw_filter.clone(),
                clear: view.clear,
                world_mesh_draw_plan: std::mem::replace(
                    &mut view.world_mesh_draw_plan,
                    WorldMeshDrawPlan::Empty,
                ),
                resolved: Self::resolve_owned_view_from_target(
                    view_id,
                    &view.target,
                    mv_ctx.gpu,
                    mv_ctx.backbuffer_view_holder.as_ref(),
                )?,
                per_view_frame_bg_and_buf,
            });
        }
        Ok(work_items)
    }
}

mod pre_warm;
mod recording;
mod resolve;
