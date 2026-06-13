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
use std::sync::Arc;
use std::time::Instant;

use crate::gpu::{GpuContext, GpuLimits};
use crate::graph_inputs::GraphSceneView;
use crate::render_graph::GraphExecutionBackend;
use crate::render_graph::swapchain_scope::SwapchainScope;

use super::super::context::{GraphResolvedResources, PostSubmitContext};
use super::super::error::GraphExecuteError;
use super::super::frame_upload_batch::FrameUploadBatch;
use super::{
    CompiledRenderGraph, FrameGlobalView, FrameView, FrameViewTarget, MultiViewExecutionContext,
};
use crate::camera::{HostCameraFrame, ViewId};

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

struct RecordSubmitFinalizeInputs<'a, 'view> {
    views: &'a [FrameView<'view>],
    frame_global: &'a FrameGlobalView,
    transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources>,
    upload_batch: &'a FrameUploadBatch,
    per_view_work_items: Vec<PerViewWorkItem>,
    recording_plan: GraphCommandRecordingPlan,
    swapchain_scope: &'a mut SwapchainScope,
    backbuffer_view_holder: &'a Option<wgpu::TextureView>,
    queue_arc: &'a Arc<wgpu::Queue>,
    device: &'a wgpu::Device,
}

mod command_recording;
mod diagnostics;
mod per_view;
mod prepare;
mod recording_path;
mod submit;
mod swapchain;
mod types;

use command_recording::GraphCommandRecordingInputs;
use diagnostics::{CommandEncodingDiagnostics, TransientPoolMetricsDelta};
use recording_path::GraphCommandRecordingPlan;
use submit::release_transients_and_gc;
use types::{
    DrainedUploadCommand, FrameGlobalPassRecordInputs, GraphCommandRecordingPath, GraphResolveKey,
    OwnedResolvedView, PerViewEncodeOutput, PerViewRecordInputs, PerViewRecordOutput,
    PerViewRecordShared, PerViewWorkItem, PreparedPerViewFrameInput, PreparedPerViewFrameParams,
    RecordedPerViewBatch, ResolvedOffscreenColorCopy, SubmitFrameBatchStats, SubmitFrameInputs,
    TimedCommandBuffer, TransientTextureResolveSurfaceParams,
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
        scene: GraphSceneView<'_>,
        backend: &mut dyn GraphExecutionBackend,
        frame_global: &FrameGlobalView,
        views: &mut [FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::execute_multi_view");
        if views.is_empty() {
            return Ok(());
        }

        let device_arc = gpu.device().clone();
        let queue_arc = gpu.queue().clone();
        let limits_arc = gpu.limits().clone();
        let device = device_arc.as_ref();
        let gpu_limits = limits_arc.as_ref();

        let mut command_diagnostics = CommandEncodingDiagnostics::new(self, views.len());

        let mut transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources> = HashMap::new();

        // Deferred graph upload sink shared by pre-record, frame-global, and per-view paths.
        // Drained onto the main thread after all recording completes and before submit.
        let upload_batch = FrameUploadBatch::new();

        let backbuffer_view_holder = None;
        let mut per_view_work_items = {
            let mut mv_ctx = MultiViewExecutionContext {
                gpu,
                scene,
                backend,
                device,
                gpu_limits,
                backbuffer_view_holder: &backbuffer_view_holder,
            };
            self.begin_generation_and_pre_resolve_transients(
                &mut mv_ctx,
                views,
                &mut transient_by_key,
                &mut command_diagnostics,
            )?;
            let (per_view_work_items, prepare_resources_ms) =
                self.prepare_resources_and_work_items(&mut mv_ctx, views, &upload_batch)?;
            command_diagnostics.prepare_resources_ms = prepare_resources_ms;
            per_view_work_items
        };
        let command_recording_mode = backend.command_recording_mode();
        let recording_plan =
            self.graph_command_recording_plan(views, &per_view_work_items, command_recording_mode);
        command_diagnostics.recording_path = recording_plan.path;
        command_diagnostics.recording_strategy = recording_plan.strategy;
        command_diagnostics.requested_recording_mode = recording_plan.requested_mode;
        command_diagnostics.estimated_per_view_draw_count =
            recording_plan.estimated_per_view_draw_count;
        command_diagnostics.estimated_per_view_record_work =
            recording_plan.estimated_per_view_record_work;
        command_diagnostics.auto_per_view_record_admitted =
            recording_plan.auto_per_view_record_admission.is_parallel();
        command_diagnostics.per_view_record_admitted =
            recording_plan.per_view_record_admission.is_parallel();

        let (mut swapchain_scope, backbuffer_view_holder) = match self
            .late_acquire_swapchain_for_prepared_views(gpu, views, &mut per_view_work_items)
        {
            Ok(Some(acquired)) => acquired,
            Ok(None) => {
                Self::release_transients_after_early_exit(
                    gpu,
                    scene,
                    backend,
                    device,
                    gpu_limits,
                    transient_by_key,
                );
                return Ok(());
            }
            Err(err) => {
                Self::release_transients_after_early_exit(
                    gpu,
                    scene,
                    backend,
                    device,
                    gpu_limits,
                    transient_by_key,
                );
                return Err(err);
            }
        };

        let mut mv_ctx = MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            backbuffer_view_holder: &backbuffer_view_holder,
        };

        self.record_submit_and_finalize_multi_view(
            &mut mv_ctx,
            RecordSubmitFinalizeInputs {
                views,
                frame_global,
                transient_by_key,
                upload_batch: &upload_batch,
                per_view_work_items,
                recording_plan,
                swapchain_scope: &mut swapchain_scope,
                backbuffer_view_holder: &backbuffer_view_holder,
                queue_arc: &queue_arc,
                device,
            },
            &mut command_diagnostics,
        )
    }

    fn record_submit_and_finalize_multi_view(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        inputs: RecordSubmitFinalizeInputs<'_, '_>,
        command_diagnostics: &mut CommandEncodingDiagnostics,
    ) -> Result<(), GraphExecuteError> {
        let RecordSubmitFinalizeInputs {
            views,
            frame_global,
            mut transient_by_key,
            upload_batch,
            per_view_work_items,
            recording_plan,
            swapchain_scope,
            backbuffer_view_holder,
            queue_arc,
            device,
        } = inputs;

        // -- Graph command recording ----------------------------------------------------------
        let (
            frame_global_cmd,
            RecordedPerViewBatch {
                per_view_cmds,
                per_view_occlusion_info,
                per_view_hud_outputs,
                per_view_profiler_cmd,
                retained_resources: per_view_retained_resources,
                ..
            },
        ) = self.record_graph_commands(
            mv_ctx,
            GraphCommandRecordingInputs {
                views,
                frame_global,
                per_view_work_items,
                transient_by_key: &mut transient_by_key,
                upload_batch,
                plan: recording_plan,
                command_diagnostics,
            },
        )?;

        let submit_stats = {
            profiling::scope!("graph::submit_frame_batch");
            self.submit_frame_batch(
                mv_ctx,
                SubmitFrameInputs {
                    views,
                    transient_by_key,
                    frame_global_cmd,
                    per_view_cmds,
                    per_view_profiler_cmd,
                    per_view_retained_resources,
                    per_view_hud_outputs,
                    per_view_occlusion_info: &per_view_occlusion_info,
                    swapchain_scope,
                    backbuffer_view_holder,
                    upload_batch,
                    queue_arc,
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
            self.run_post_submit_passes(mv_ctx, views, device, &per_view_occlusion_info)?;
        }

        Ok(())
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

        // Pre-resolve transient textures and buffers for every unique view key before any
        // per-view recording begins. The record loop then reads `transient_by_key` without
        // touching the shared transient pool. Swapchain views resolve layout and depth here
        // without acquiring a desktop surface image.
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

    /// Releases pre-acquired transient leases when late swapchain acquire skips or fails.
    fn release_transients_after_early_exit(
        gpu: &mut GpuContext,
        scene: GraphSceneView<'_>,
        backend: &mut dyn GraphExecutionBackend,
        device: &wgpu::Device,
        gpu_limits: &GpuLimits,
        transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources>,
    ) {
        let backbuffer_view_holder = None;
        let mut mv_ctx = MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            backbuffer_view_holder: &backbuffer_view_holder,
        };
        release_transients_and_gc(&mut mv_ctx, transient_by_key);
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
}

mod pre_warm;
mod recording;
mod resolve;
