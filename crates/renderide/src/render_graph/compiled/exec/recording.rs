//! Per-view and frame-global command encoding paths plus the single `execute_pass_node` dispatch.

use hashbrown::HashMap;
use hashbrown::hash_map::Entry;
use std::time::Instant;

use super::super::super::blackboard::{Blackboard, GraphCommandStatsSlot};
use super::super::super::context::{
    ComputePassCtx, EncoderPassCtx, GraphResolvedResources, RasterPassCtx,
};
use super::super::super::error::GraphExecuteError;
use super::super::super::frame_params::{
    FrameSystemsShared, MsaaViewsSlot, PerViewFramePlan, PerViewFramePlanSlot,
};
use super::super::super::frame_upload_batch::{
    FrameUploadBatch, FrameUploadScope, GraphUploadSink,
};
use super::super::super::pass::{PassKind, PassPhase};
use super::super::super::schedule::{RenderPassMaterializationGroup, ScheduleStep};
use super::super::helpers;
use super::super::{CompiledRenderGraph, FrameView, MultiViewExecutionContext, ResolvedView};
use super::{
    GraphResolveKey, PerViewEncodeOutput, PerViewRecordShared, PerViewWorkItem,
    ResolvedOffscreenColorCopy, TimedCommandBuffer, TransientTextureResolveSurfaceParams,
    elapsed_ms,
};
use crate::diagnostics::PerViewHudOutputsSlot;
use crate::render_graph::GraphExecutionBackend;
use crate::render_graph::post_process_settings::{
    AutoExposureSettingsSlot, AutoExposureSettingsValue, BloomSettingsSlot, BloomSettingsValue,
    GtaoSettingsSlot, GtaoSettingsValue, MotionBlurSettingsSlot, MotionBlurSettingsValue,
};

/// Mutable state needed to replay frame-global schedule steps.
struct FrameGlobalPassLoop<'record, 'frame> {
    /// Compiled graph being recorded.
    graph: &'record CompiledRenderGraph,
    /// Resolved view used by frame-global passes.
    resolved: &'frame ResolvedView<'frame>,
    /// Graph resource table resolved for the frame-global view.
    graph_resources: &'frame GraphResolvedResources,
    /// Mutable per-pass frame parameters.
    frame_params: &'record mut crate::render_graph::frame_params::GraphPassFrame<'frame>,
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

impl FrameGlobalPassLoop<'_, '_> {
    /// Records every frame-global pass in scheduler wave order.
    fn record(self) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::frame_global::pass_loop");
        self.graph.record_phase_steps(
            PassPhase::FrameGlobal,
            None,
            self.resolved,
            self.graph_resources,
            self.frame_params,
            self.frame_blackboard,
            self.encoder,
            self.device,
            self.gpu_limits,
            self.upload_batch,
            self.pass_profiler,
        )
    }
}

impl CompiledRenderGraph {
    /// Finds a materialization group that begins at `step_idx`.
    fn materialization_group_starting_at(
        &self,
        step_idx: usize,
    ) -> Option<RenderPassMaterializationGroup> {
        self.schedule
            .render_pass_materialization_plan
            .groups
            .iter()
            .copied()
            .find(|group| group.start_step == step_idx)
    }

    /// Collects raster templates for a materialization candidate.
    fn materialization_templates(
        &self,
        steps: &[ScheduleStep],
    ) -> Result<Option<Vec<crate::render_graph::pass::RenderPassTemplate>>, GraphExecuteError> {
        if !steps
            .iter()
            .all(|step| self.passes[step.pass_idx].kind() == PassKind::Raster)
        {
            return Ok(None);
        }
        let mut templates = Vec::with_capacity(steps.len());
        for step in steps {
            templates.push(helpers::pass_info_raster_template(
                &self.pass_info,
                step.pass_idx,
            )?);
        }
        Ok(Some(templates))
    }

    /// Records all schedule steps for one pass phase in flat topological order.
    #[expect(
        clippy::too_many_arguments,
        reason = "recording loop borrows must stay independent"
    )]
    fn record_phase_steps<'a>(
        &self,
        phase: PassPhase,
        view_idx: Option<usize>,
        resolved: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        frame_params: &mut crate::render_graph::frame_params::GraphPassFrame<'a>,
        blackboard: &mut Blackboard,
        encoder: &mut wgpu::CommandEncoder,
        device: &'a wgpu::Device,
        gpu_limits: &'a crate::gpu::GpuLimits,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<(), GraphExecuteError> {
        let mut step_idx = 0usize;
        while step_idx < self.schedule.steps.len() {
            let step = self.schedule.steps[step_idx];
            if step.phase != phase {
                step_idx += 1;
                continue;
            }
            if let Some(group) = self.materialization_group_starting_at(step_idx)
                && self.try_execute_raster_materialization_group(
                    group,
                    phase,
                    view_idx,
                    graph_resources,
                    frame_params,
                    blackboard,
                    encoder,
                    device,
                    upload_batch,
                    profiler,
                )?
            {
                step_idx = group.end_step;
                continue;
            }
            self.execute_pass_node(
                step.pass_idx,
                step.frame_upload_scope(view_idx),
                resolved,
                graph_resources,
                frame_params,
                blackboard,
                encoder,
                device,
                gpu_limits,
                upload_batch,
                profiler,
            )?;
            step_idx += 1;
        }
        Ok(())
    }

    /// Attempts to record one compatible raster group inside a single `wgpu::RenderPass`.
    #[expect(
        clippy::too_many_arguments,
        reason = "merged raster recording shares the same pass context borrows"
    )]
    fn try_execute_raster_materialization_group<'a>(
        &self,
        group: RenderPassMaterializationGroup,
        phase: PassPhase,
        view_idx: Option<usize>,
        graph_resources: &'a GraphResolvedResources,
        frame_params: &mut crate::render_graph::frame_params::GraphPassFrame<'a>,
        blackboard: &mut Blackboard,
        encoder: &mut wgpu::CommandEncoder,
        device: &'a wgpu::Device,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<bool, GraphExecuteError> {
        let steps = &self.schedule.steps[group.start_step..group.end_step];
        if steps.len() < 2 || !steps.iter().all(|step| step.phase == phase) {
            return Ok(false);
        }
        let Some(first_step) = steps.first().copied() else {
            return Ok(false);
        };
        let Some(templates) = self.materialization_templates(steps)? else {
            return Ok(false);
        };
        let Some(merged_template) = helpers::coalesce_render_pass_template(&templates) else {
            return Ok(false);
        };

        let uploads = GraphUploadSink::new(upload_batch, first_step.frame_upload_scope(view_idx));
        let mut ctx = RasterPassCtx {
            device,
            pass_frame: frame_params,
            uploads,
            graph_resources,
            blackboard,
            profiler,
        };
        if !self.materialized_group_dynamic_state_matches(steps, &templates, &ctx) {
            return Ok(false);
        }
        let first_pass = &self.passes[first_step.pass_idx];
        let first_should_record = first_pass
            .should_record_raster(&ctx)
            .map_err(GraphExecuteError::Pass)?;
        if !first_should_record {
            return Ok(false);
        }

        let sample_count = helpers::frame_sample_count_from_raster_ctx(&ctx);
        let color_attachments = helpers::resolve_color_attachments(
            "render-graph-raster-merged",
            &merged_template,
            graph_resources,
            sample_count,
        )?;
        let stencil_ops = merged_template
            .depth_stencil_attachment
            .as_ref()
            .and_then(|depth| first_pass.stencil_ops_override(&ctx, depth));
        let depth_stencil_attachment = helpers::resolve_depth_attachment_with_stencil(
            "render-graph-raster-merged",
            &merged_template,
            graph_resources,
            sample_count,
            stencil_ops,
        )?;
        let multiview_mask = first_pass.multiview_mask_override(&ctx, &merged_template);
        let pass_query = ctx.profiler.map(|p| {
            p.begin_pass_query(
                format!(
                    "graph::raster_merge[{}..{}]",
                    group.start_step, group.end_step
                ),
                encoder,
            )
        });
        let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render-graph-raster-merged"),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes,
            multiview_mask,
        });
        for (idx, step) in steps.iter().copied().enumerate() {
            let pass = &self.passes[step.pass_idx];
            let scope = step.frame_upload_scope(view_idx);
            let _upload_scope = upload_batch.enter_scope(scope);
            ctx.uploads = GraphUploadSink::new(upload_batch, scope);
            let should_record = if idx == 0 {
                true
            } else {
                pass.should_record_raster(&ctx)
                    .map_err(GraphExecuteError::Pass)?
            };
            if !should_record {
                continue;
            }
            self.validate_blackboard_inputs(step.pass_idx, pass.name(), ctx.blackboard)?;
            pass.record_raster(&mut ctx, &mut rpass)
                .map_err(GraphExecuteError::Pass)?;
        }
        drop(rpass);
        if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
            p.end_query(encoder, q);
        }
        Ok(true)
    }

    /// Returns whether dynamic multiview and stencil state match across a merge candidate.
    fn materialized_group_dynamic_state_matches(
        &self,
        steps: &[ScheduleStep],
        templates: &[crate::render_graph::pass::RenderPassTemplate],
        ctx: &RasterPassCtx<'_, '_>,
    ) -> bool {
        let Some((first_step, rest_steps)) = steps.split_first() else {
            return false;
        };
        let Some((first_template, rest_templates)) = templates.split_first() else {
            return false;
        };
        let first_pass = &self.passes[first_step.pass_idx];
        let multiview = first_pass.multiview_mask_override(ctx, first_template);
        let stencil = first_template
            .depth_stencil_attachment
            .as_ref()
            .and_then(|depth| first_pass.stencil_ops_override(ctx, depth));
        rest_steps
            .iter()
            .zip(rest_templates)
            .all(|(step, template)| {
                let pass = &self.passes[step.pass_idx];
                pass.multiview_mask_override(ctx, template) == multiview
                    && template
                        .depth_stencil_attachment
                        .as_ref()
                        .and_then(|depth| pass.stencil_ops_override(ctx, depth))
                        == stencil
            })
    }

    /// Validates declared blackboard inputs immediately before one pass records.
    fn validate_blackboard_inputs(
        &self,
        pass_idx: usize,
        pass_name: &str,
        blackboard: &Blackboard,
    ) -> Result<(), GraphExecuteError> {
        if !self.validation_mode.enabled() {
            return Ok(());
        }
        let Some(info) = self.pass_info.get(pass_idx) else {
            return Ok(());
        };
        for access in &info.blackboard_accesses {
            if !access.kind.requires_value() || blackboard.contains_type_id(access.slot.type_id) {
                continue;
            }
            if self.validation_mode.is_strict() {
                return Err(GraphExecuteError::MissingBlackboardSlot {
                    pass: pass_name.to_owned(),
                    slot: access.slot.type_name,
                });
            }
            logger::warn!(
                "render graph validation: pass `{pass_name}` requires blackboard slot `{}` but it was not present",
                access.slot.type_name
            );
        }
        Ok(())
    }

    /// Records the per-view pass phase into one command buffer for `work_item`.
    pub(super) fn record_one_view(
        &self,
        shared: &PerViewRecordShared<'_>,
        work_item: PerViewWorkItem,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<PerViewEncodeOutput, GraphExecuteError> {
        profiling::scope!("graph::per_view");
        let encode_start = Instant::now();
        let device = shared.device;
        let PerViewWorkItem {
            view_idx,
            host_camera,
            render_context,
            clear,
            initial_blackboard,
            resolved,
            per_view_frame_bg_and_buf,
            ..
        } = work_item;

        let mut encoder = {
            profiling::scope!("graph::per_view::create_encoder");
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-graph-per-view"),
            })
        };
        let gpu_query = profiler.map(|p| p.begin_query("graph::per_view", &mut encoder));

        let resolved_view = resolved.as_resolved();
        let resolved_resources =
            self.resolve_per_view_graph_resources(shared, &resolved_view, transient_by_key)?;
        let graph_resources: &GraphResolvedResources = &resolved_resources;

        let mut frame_params = Self::build_per_view_frame_params(
            shared,
            &resolved_view,
            &host_camera,
            render_context,
            clear,
        );
        let mut view_blackboard = self.build_per_view_blackboard(
            &frame_params,
            graph_resources,
            initial_blackboard,
            per_view_frame_bg_and_buf,
            view_idx,
        );
        Self::seed_live_post_process_settings(&mut view_blackboard, shared, resolved.view_id);

        {
            profiling::scope!("graph::per_view::pass_loop");
            self.record_phase_steps(
                PassPhase::PerView,
                Some(view_idx),
                &resolved_view,
                graph_resources,
                &mut frame_params,
                &mut view_blackboard,
                &mut encoder,
                shared.device,
                shared.gpu_limits,
                upload_batch,
                profiler,
            )?;
        }
        Self::record_offscreen_color_copy(
            &mut encoder,
            resolved.offscreen_color_copy.as_ref(),
            profiler,
        );
        if let Some(query) = gpu_query
            && let Some(prof) = profiler
        {
            prof.end_query(&mut encoder, query);
        }
        let command_stats = view_blackboard
            .get::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        let hud_outputs = view_blackboard.take::<PerViewHudOutputsSlot>();
        let encode_ms = elapsed_ms(encode_start);
        let (command_buffer, finish_ms) = {
            profiling::scope!("CommandEncoder::finish::graph_per_view");
            let finish_start = Instant::now();
            let command_buffer = encoder.finish();
            let finish_ms = elapsed_ms(finish_start);
            (command_buffer, finish_ms)
        };
        Ok(PerViewEncodeOutput {
            command_buffer,
            hud_outputs,
            encode_ms,
            finish_ms,
            command_stats,
        })
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

    /// Records the final scratch-to-render-texture copy for a partial offscreen viewport.
    fn record_offscreen_color_copy(
        encoder: &mut wgpu::CommandEncoder,
        copy: Option<&ResolvedOffscreenColorCopy>,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) {
        let Some(copy) = copy else {
            return;
        };
        if copy.extent_px.0 == 0 || copy.extent_px.1 == 0 {
            return;
        }
        profiling::scope!("graph::per_view::offscreen_color_copy");
        let copy_query =
            profiler.map(|p| p.begin_query("graph::per_view::offscreen_color_copy", encoder));
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &copy.source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &copy.destination_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: copy.destination_origin_px.0,
                    y: copy.destination_origin_px.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: copy.extent_px.0,
                height: copy.extent_px.1,
                depth_or_array_layers: 1,
            },
        );
        if let Some(query) = copy_query
            && let Some(profiler) = profiler
        {
            profiler.end_query(encoder, query);
        }
    }

    /// Builds [`GraphPassFrame`](crate::render_graph::frame_params::GraphPassFrame) for one per-view pass batch.
    fn build_per_view_frame_params<'a>(
        shared: &'a PerViewRecordShared<'a>,
        resolved: &'a ResolvedView<'a>,
        host_camera: &crate::camera::HostCameraFrame,
        render_context: crate::shared::RenderingContext,
        clear: super::super::super::frame_params::FrameViewClear,
    ) -> crate::render_graph::frame_params::GraphPassFrame<'a> {
        profiling::scope!("graph::per_view::build_frame_params");
        let hi_z_slot = shared.occlusion.ensure_hi_z_state(resolved.view_id);
        helpers::frame_render_params_from_shared(
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
            helpers::GraphPassFrameViewInputs {
                resolved,
                scene_color_format: shared.scene_color_format,
                host_camera,
                render_context,
                clear,
                post_processing: resolved.post_processing,
                gpu_limits: shared.gpu_limits_arc.clone(),
                msaa_depth_resolve: shared.msaa_depth_resolve.clone(),
                hi_z_slot,
            },
        )
    }

    /// Builds the per-view [`Blackboard`] seeded with MSAA views and preplanned frame data.
    fn build_per_view_blackboard(
        &self,
        frame_params: &crate::render_graph::frame_params::GraphPassFrame<'_>,
        graph_resources: &GraphResolvedResources,
        initial_blackboard: Blackboard,
        per_view_frame_bg_and_buf: (std::sync::Arc<wgpu::BindGroup>, wgpu::Buffer),
        view_idx: usize,
    ) -> Blackboard {
        profiling::scope!("graph::per_view::build_blackboard");
        let mut view_blackboard = initial_blackboard;
        let mut graph_blackboard = Blackboard::new();
        if let Some(msaa_views) = helpers::resolve_forward_msaa_views_from_graph_resources(
            frame_params,
            graph_resources,
            self.main_graph_msaa_transient_handles,
        ) {
            graph_blackboard.insert::<MsaaViewsSlot>(msaa_views);
        }
        let (frame_bg, frame_buf) = per_view_frame_bg_and_buf;
        // Seed per-view frame plan so backend world-mesh planning can write frame uniforms to the
        // correct per-view buffer and bind the right @group(0) bind group.
        graph_blackboard.insert::<PerViewFramePlanSlot>(PerViewFramePlan {
            frame_bind_group: frame_bg,
            frame_uniform_buffer: frame_buf,
            view_idx,
        });
        view_blackboard.extend(graph_blackboard);
        view_blackboard
    }

    fn seed_live_post_process_settings(
        blackboard: &mut Blackboard,
        shared: &PerViewRecordShared<'_>,
        view_id: crate::camera::ViewId,
    ) {
        // Propagate the live GTAO settings so the GTAO chain (`GtaoMainPass` / `GtaoDenoisePass`
        // / `GtaoApplyPass`) reads the current slider values every frame without rebuilding the
        // compiled render graph. Topology fields (`enabled`, `denoise_passes`) are tracked by
        // the chain signature; non-topology slider edits flow only through this slot.
        blackboard.insert::<GtaoSettingsSlot>(GtaoSettingsValue(shared.live_gtao_settings));
        // Same pattern for bloom: the first downsample reads `BloomSettingsSlot` to build its
        // params UBO and the upsamples use it to compute per-mip blend constants + pick
        // EnergyConserving vs Additive pipeline variants, so slider edits propagate next frame.
        blackboard.insert::<BloomSettingsSlot>(BloomSettingsValue(shared.live_bloom_settings));
        blackboard.insert::<MotionBlurSettingsSlot>(MotionBlurSettingsValue(
            shared.live_motion_blur_settings,
        ));
        blackboard.insert::<AutoExposureSettingsSlot>(AutoExposureSettingsValue::for_view(
            shared.live_auto_exposure_settings,
            shared.wall_frame_delta_seconds,
            view_id,
        ));
    }

    /// Encodes [`super::super::super::pass::PassPhase::FrameGlobal`] passes into a command buffer.
    ///
    /// Returns `None` when there are no frame-global passes (nothing to submit for this phase).
    /// The caller is responsible for including the returned buffer in the single-submit batch.
    pub(super) fn encode_frame_global_passes(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
    ) -> Result<Option<TimedCommandBuffer>, GraphExecuteError> {
        profiling::scope!("graph::frame_global");
        if self.frame_global_passes_are_inactive(&*mv_ctx.backend) {
            return Ok(None);
        }
        let encode_start = Instant::now();
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            backbuffer_view_holder,
        } = mv_ctx;

        let first = views.first().ok_or(GraphExecuteError::NoViewsInBatch)?;
        let mut encoder = {
            profiling::scope!("graph::frame_global::create_encoder");
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-graph-frame-global"),
            })
        };
        let gpu_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::frame_global", &mut encoder));
        let pass_profiler = gpu.take_gpu_profiler();

        let record_result = (|| -> Result<(), GraphExecuteError> {
            let resolved = {
                profiling::scope!("graph::frame_global::resolve_target");
                Self::resolve_view_from_target(
                    first.view_id(),
                    first.profile,
                    &first.target,
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
                    scene,
                    &mut **backend,
                    &resolved,
                    &first.host_camera,
                    first.render_context,
                    first.clear,
                    first.post_processing(),
                )
            };
            let mut frame_blackboard = Self::build_frame_global_blackboard();

            FrameGlobalPassLoop {
                graph: self,
                resolved: &resolved,
                graph_resources,
                frame_params: &mut frame_params,
                frame_blackboard: &mut frame_blackboard,
                encoder: &mut encoder,
                device,
                gpu_limits,
                upload_batch,
                pass_profiler: pass_profiler.as_ref(),
            }
            .record()?;
            Ok(())
        })();

        gpu.restore_gpu_profiler(pass_profiler);
        record_result?;
        let encode_ms = elapsed_ms(encode_start);
        let command_buffer = Self::finish_frame_global_encoder(gpu, encoder, gpu_query, encode_ms);
        Ok(Some(command_buffer))
    }

    fn frame_global_passes_are_inactive(&self, backend: &dyn GraphExecutionBackend) -> bool {
        let mut indices = self.schedule.frame_global_pass_indices().iter().copied();
        let Some(pass_idx) = indices.next() else {
            return true;
        };
        if indices.next().is_some() {
            return false;
        }
        self.passes[pass_idx].name() == "MeshDeform"
            && backend
                .frame_resources()
                .visible_mesh_deform_filter_is_empty()
    }

    fn build_frame_global_blackboard() -> Blackboard {
        profiling::scope!("graph::frame_global::build_blackboard");
        Blackboard::new()
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

    /// Dispatches one pass node to its correct execution path.
    ///
    /// - `Raster` -> opens `wgpu::RenderPass` from template, calls `record_raster`.
    /// - `Compute` -> calls `record_compute` with raw encoder.
    /// - `Encoder` -> calls `record_encoder` with raw encoder.
    ///
    /// Takes `&self` so per-view recording can be hoisted onto rayon workers without serialising
    /// on the [`CompiledRenderGraph`] handle. All pass `record_*` methods already require only
    /// `&self`, so the dispatch loop is structurally Send/Sync-safe at this layer.
    //
    // This function intentionally keeps independent parameters rather than bundling into a
    // context struct: `encoder` uses an anonymous `'_` lifetime so each call's mutable borrow
    // ends at the call boundary, and the other `&'a` references must all share the per-view
    // lifetime `'a` without being pulled into a single `'a`-bound struct that would couple
    // their borrow scopes.
    #[expect(
        clippy::too_many_arguments,
        reason = "borrow scopes forbid a single context struct"
    )]
    pub(super) fn execute_pass_node<'a>(
        &self,
        pass_idx: usize,
        upload_scope: FrameUploadScope,
        resolved: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        frame_params: &mut crate::render_graph::frame_params::GraphPassFrame<'a>,
        blackboard: &mut Blackboard,
        // `encoder` intentionally uses no named lifetime so each call's borrow
        // ends at the call boundary, avoiding cross-iteration borrow conflicts.
        encoder: &mut wgpu::CommandEncoder,
        device: &'a wgpu::Device,
        gpu_limits: &'a crate::gpu::GpuLimits,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<(), GraphExecuteError> {
        let _upload_scope = upload_batch.enter_scope(upload_scope);
        let uploads = GraphUploadSink::new(upload_batch, upload_scope);
        // Hoist the pass borrow once so the inner match arms do not re-index `self.passes` for
        // every dispatch. The Raster path still needs the explicit `&self.passes[pass_idx]`
        // because `helpers::execute_graph_raster_pass_node` takes a `&PassNode` and the borrow
        // matches `pass` exactly; this also keeps the inner record_* dispatches as pointer-cheap
        // direct calls.
        let pass = &self.passes[pass_idx];
        let _pass_label = pass.profiling_label();
        profiling::scope!("graph::execute_pass_node", _pass_label.as_ref());
        self.validate_blackboard_inputs(pass_idx, pass.name(), blackboard)?;
        match pass.kind() {
            PassKind::Raster => {
                profiling::scope!("graph::record_raster");
                let template = helpers::pass_info_raster_template(&self.pass_info, pass_idx)?;
                let mut ctx = RasterPassCtx {
                    device,
                    pass_frame: frame_params,
                    uploads,
                    graph_resources,
                    blackboard,
                    profiler,
                };
                helpers::execute_graph_raster_pass_node(
                    pass,
                    &template,
                    graph_resources,
                    encoder,
                    &mut ctx,
                )?;
            }
            PassKind::Compute => {
                profiling::scope!("graph::record_compute");
                // encoder is moved into ComputePassCtx; pass uses ctx.encoder.
                let mut ctx = {
                    profiling::scope!("graph::record_compute::build_context");
                    ComputePassCtx {
                        device,
                        gpu_limits,
                        encoder,
                        depth_view: Some(resolved.depth_view),
                        pass_frame: frame_params,
                        uploads,
                        graph_resources,
                        blackboard,
                        profiler,
                    }
                };
                let should_record = {
                    profiling::scope!("graph::record_compute::should_record");
                    pass.should_record_compute(&ctx)
                        .map_err(GraphExecuteError::Pass)?
                };
                if should_record {
                    let pass_query = ctx
                        .profiler
                        .map(|p| p.begin_query(pass.profiling_label(), ctx.encoder));
                    {
                        profiling::scope!("graph::record_compute::pass_record");
                        pass.record_compute(&mut ctx)
                            .map_err(GraphExecuteError::Pass)?;
                    }
                    if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
                        p.end_query(ctx.encoder, q);
                    }
                }
            }
            PassKind::Encoder => {
                profiling::scope!("graph::record_encoder");
                let mut ctx = {
                    profiling::scope!("graph::record_encoder::build_context");
                    EncoderPassCtx {
                        device,
                        encoder,
                        pass_frame: frame_params,
                        uploads,
                        graph_resources,
                        blackboard,
                        profiler,
                    }
                };
                let should_record = {
                    profiling::scope!("graph::record_encoder::should_record");
                    pass.should_record_encoder(&ctx)
                        .map_err(GraphExecuteError::Pass)?
                };
                if should_record {
                    let pass_query = ctx
                        .profiler
                        .map(|p| p.begin_query(pass.profiling_label(), ctx.encoder));
                    {
                        profiling::scope!("graph::record_encoder::pass_record");
                        pass.record_encoder(&mut ctx)
                            .map_err(GraphExecuteError::Pass)?;
                    }
                    if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
                        p.end_query(ctx.encoder, q);
                    }
                }
            }
        }
        Ok(())
    }
}
