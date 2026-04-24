//! Per-view and frame-global command encoding paths plus the single `execute_pass_node` dispatch.

use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

use super::super::super::blackboard::Blackboard;
use super::super::super::context::{
    CallbackCtx, ComputePassCtx, GraphResolvedResources, RasterPassCtx,
};
use super::super::super::error::GraphExecuteError;
use super::super::super::frame_params::{
    FrameSystemsShared, GtaoSettingsSlot, GtaoSettingsValue, MsaaViewsSlot, PerViewFramePlan,
    PerViewFramePlanSlot, PerViewHudOutputsSlot, PrefetchedWorldMeshDrawsSlot,
};
use super::super::super::pass::PassKind;
use super::super::helpers;
use super::super::{CompiledRenderGraph, FrameView, MultiViewExecutionContext, ResolvedView};
use super::{
    GraphResolveKey, PerViewEncodeOutput, PerViewRecordShared, PerViewWorkItem,
    TransientTextureResolveSurfaceParams,
};

impl CompiledRenderGraph {
    /// Records the per-view pass phase into one command buffer for `work_item`.
    pub(super) fn record_one_view(
        &self,
        shared: &PerViewRecordShared<'_>,
        work_item: PerViewWorkItem,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &super::super::super::frame_upload_batch::FrameUploadBatch,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<PerViewEncodeOutput, GraphExecuteError> {
        profiling::scope!("graph::per_view");
        let device = shared.device;
        let PerViewWorkItem {
            view_idx,
            host_camera,
            draw_filter,
            prefetched_world_mesh_draws,
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

        let resolved = resolved.as_resolved();
        let resolved_resources = {
            profiling::scope!("graph::per_view::resolve_transients");
            let key = GraphResolveKey::from_resolved(&resolved);
            // Transients were pre-resolved in `pre_resolve_transients_for_views` before the
            // per-view loop began, so a missing entry here is a bug.
            let mut resolved_resources = transient_by_key.get(&key).cloned().ok_or_else(|| {
                logger::warn!("pre-resolve: missing transient resources for view key {key:?}");
                GraphExecuteError::MissingTransientResources
            })?;
            self.resolve_imported_textures(&resolved, &mut resolved_resources);
            self.resolve_imported_buffers(
                shared.frame_resources,
                &resolved,
                &mut resolved_resources,
            );
            resolved_resources
        };
        let graph_resources: &GraphResolvedResources = &resolved_resources;

        let mut frame_params =
            Self::build_per_view_frame_params(shared, &resolved, &host_camera, draw_filter);
        let mut view_blackboard = self.build_per_view_blackboard(
            &frame_params,
            graph_resources,
            prefetched_world_mesh_draws,
            per_view_frame_bg_and_buf,
            view_idx,
        );
        // Propagate the live GTAO settings so `GtaoPass::record` reads the current slider values
        // every frame without rebuilding the compiled render graph (the chain signature only
        // tracks enable booleans, so parameter-only edits wouldn't otherwise reach the shader).
        view_blackboard.insert::<GtaoSettingsSlot>(GtaoSettingsValue(shared.live_gtao_settings));

        {
            profiling::scope!("graph::per_view::pass_loop");
            // Iterate the cached per-view `pass_idx` slice from `FrameSchedule` to avoid
            // rebuilding a scratch `Vec<usize>` every frame.
            for &pass_idx in self.schedule.per_view_pass_indices() {
                let pass_name = self.passes[pass_idx].name();

                // Open the GPU profiler query before calling execute_pass_node so we can
                // avoid capturing `encoder` in a closure while also passing it mutably.
                let pass_query = profiler.map(|p| p.begin_query(pass_name, &mut encoder));

                self.execute_pass_node(
                    pass_idx,
                    &resolved,
                    graph_resources,
                    &mut frame_params,
                    &mut view_blackboard,
                    &mut encoder,
                    shared.device,
                    shared.gpu_limits,
                    shared.queue_arc,
                    upload_batch,
                    profiler,
                )?;

                if let Some(q) = pass_query {
                    if let Some(p) = profiler {
                        p.end_query(&mut encoder, q);
                    }
                }
            }
        }
        if let Some(query) = gpu_query {
            if let Some(prof) = profiler {
                prof.end_query(&mut encoder, query);
            }
        }
        let hud_outputs = view_blackboard.take::<PerViewHudOutputsSlot>();
        Ok(PerViewEncodeOutput {
            command_buffer: encoder.finish(),
            hud_outputs,
        })
    }

    /// Builds [`FrameRenderParams`](crate::render_graph::frame_params::FrameRenderParams) for one per-view pass batch.
    fn build_per_view_frame_params<'a>(
        shared: &'a PerViewRecordShared<'a>,
        resolved: &'a ResolvedView<'a>,
        host_camera: &super::super::super::frame_params::HostCameraFrame,
        draw_filter: Option<crate::render_graph::world_mesh_draw_prep::CameraTransformDrawFilter>,
    ) -> crate::render_graph::frame_params::FrameRenderParams<'a> {
        profiling::scope!("graph::per_view::build_frame_params");
        let hi_z_slot = shared.occlusion.ensure_hi_z_state(resolved.occlusion_view);
        helpers::frame_render_params_from_shared(
            FrameSystemsShared {
                scene: shared.scene,
                occlusion: shared.occlusion,
                frame_resources: shared.frame_resources,
                materials: shared.materials,
                asset_transfers: shared.asset_transfers,
                mesh_preprocess: shared.mesh_preprocess,
                mesh_deform_scratch: None,
                mesh_deform_skin_cache: None,
                skin_cache: shared.skin_cache,
                debug_hud: shared.debug_hud,
            },
            helpers::FrameRenderParamsViewInputs {
                resolved,
                scene_color_format: shared.scene_color_format,
                host_camera: *host_camera,
                transform_draw_filter: draw_filter,
                gpu_limits: shared.gpu_limits_arc.clone(),
                msaa_depth_resolve: shared.msaa_depth_resolve.clone(),
                hi_z_slot,
            },
        )
    }

    /// Builds the per-view [`Blackboard`] seeded with MSAA views, prefetched draws, and the frame plan.
    fn build_per_view_blackboard(
        &self,
        frame_params: &crate::render_graph::frame_params::FrameRenderParams<'_>,
        graph_resources: &GraphResolvedResources,
        prefetched_world_mesh_draws: Option<
            crate::render_graph::world_mesh_draw_prep::WorldMeshDrawCollection,
        >,
        per_view_frame_bg_and_buf: Option<(std::sync::Arc<wgpu::BindGroup>, wgpu::Buffer)>,
        view_idx: usize,
    ) -> Blackboard {
        profiling::scope!("graph::per_view::build_blackboard");
        let mut view_blackboard = Blackboard::new();
        if let Some(msaa_views) = helpers::resolve_forward_msaa_views_from_graph_resources(
            frame_params,
            Some(graph_resources),
            self.main_graph_msaa_transient_handles,
        ) {
            view_blackboard.insert::<MsaaViewsSlot>(msaa_views);
        }
        if let Some(draws) = prefetched_world_mesh_draws {
            view_blackboard.insert::<PrefetchedWorldMeshDrawsSlot>(draws);
        }
        // Seed per-view frame plan so the prepare pass can write frame uniforms to the
        // correct per-view buffer and bind the right @group(0) bind group.
        if let Some((frame_bg, frame_buf)) = per_view_frame_bg_and_buf {
            view_blackboard.insert::<PerViewFramePlanSlot>(PerViewFramePlan {
                frame_bind_group: frame_bg,
                frame_uniform_buffer: frame_buf,
                view_idx,
            });
        }
        view_blackboard
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
        upload_batch: &super::super::super::frame_upload_batch::FrameUploadBatch,
    ) -> Result<Option<wgpu::CommandBuffer>, GraphExecuteError> {
        profiling::scope!("graph::frame_global");
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc,
            backbuffer_view_holder,
        } = mv_ctx;

        if self.schedule.frame_global_steps().next().is_none() {
            return Ok(None);
        }
        let first = views.first().ok_or(GraphExecuteError::NoViewsInBatch)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-frame-global"),
        });
        let gpu_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::frame_global", &mut encoder));
        let pass_profiler = gpu.take_gpu_profiler();

        {
            let resolved =
                Self::resolve_view_from_target(&first.target, gpu, backbuffer_view_holder)?;
            let resolved_resources = self.resolve_frame_global_transients(
                &resolved,
                transient_by_key,
                device,
                backend,
                gpu_limits,
            )?;
            self.resolve_imported_textures(&resolved, resolved_resources);
            self.resolve_imported_buffers(&backend.frame_resources, &resolved, resolved_resources);
            let graph_resources: &GraphResolvedResources = &*resolved_resources;

            let mut frame_params = helpers::frame_render_params_from_resolved(
                scene,
                backend,
                &resolved,
                first.host_camera,
                first.draw_filter.clone(),
            );
            // Frame-global blackboard (one per tick). MSAA views are per-view, not frame-global,
            // so no MSAA seed here. Frame-global passes (e.g. mesh deform) don't need MSAA views.
            let mut frame_blackboard = Blackboard::new();

            // Iterate the cached frame-global `pass_idx` slice from `FrameSchedule` to avoid
            // rebuilding a scratch `Vec<usize>` every frame.
            for &pass_idx in self.schedule.frame_global_pass_indices() {
                let pass_name = self.passes[pass_idx].name();

                let pass_query = pass_profiler
                    .as_ref()
                    .map(|p| p.begin_query(pass_name, &mut encoder));

                self.execute_pass_node(
                    pass_idx,
                    &resolved,
                    graph_resources,
                    &mut frame_params,
                    &mut frame_blackboard,
                    &mut encoder,
                    device,
                    gpu_limits,
                    queue_arc,
                    upload_batch,
                    pass_profiler.as_ref(),
                )?;

                if let Some(q) = pass_query {
                    if let Some(p) = pass_profiler.as_ref() {
                        p.end_query(&mut encoder, q);
                    }
                }
            }
        }

        gpu.restore_gpu_profiler(pass_profiler);
        if let Some(query) = gpu_query {
            if let Some(prof) = gpu.gpu_profiler_mut() {
                prof.end_query(&mut encoder, query);
                prof.resolve_queries(&mut encoder);
            }
        }
        // Return the encoded command buffer WITHOUT submitting; the caller handles single submit.
        Ok(Some(encoder.finish()))
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
        backend: &mut crate::backend::RenderBackend,
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
    /// - `Raster` → opens `wgpu::RenderPass` from template, calls `record_raster`.
    /// - `Compute` → calls `record_compute` with raw encoder.
    /// - `Copy` → calls `record_copy` with raw encoder.
    /// - `Callback` → calls `run_callback` (no encoder).
    ///
    /// Takes `&self` so per-view recording can be hoisted onto rayon workers without serialising
    /// on the [`CompiledRenderGraph`] handle. All pass `record_*` methods already require only
    /// `&self`, so the dispatch loop is structurally Send/Sync-safe at this layer.
    //
    // This function intentionally keeps ten independent parameters rather than bundling into a
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
        resolved: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        frame_params: &mut crate::render_graph::frame_params::FrameRenderParams<'a>,
        blackboard: &mut Blackboard,
        // `encoder` intentionally uses no named lifetime so each call's borrow
        // ends at the call boundary, avoiding cross-iteration borrow conflicts.
        encoder: &mut wgpu::CommandEncoder,
        device: &'a wgpu::Device,
        gpu_limits: &'a crate::gpu::GpuLimits,
        queue_arc: &'a std::sync::Arc<wgpu::Queue>,
        upload_batch: &super::super::super::frame_upload_batch::FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<(), GraphExecuteError> {
        let kind = self.passes[pass_idx].kind();
        match kind {
            PassKind::Raster => {
                profiling::scope!("graph::record_raster");
                let template = helpers::pass_info_raster_template(&self.pass_info, pass_idx)?;
                let mut ctx = RasterPassCtx {
                    device,
                    gpu_limits,
                    queue: queue_arc,
                    backbuffer: resolved.backbuffer,
                    depth_view: Some(resolved.depth_view),
                    frame: Some(frame_params),
                    frame_shared: None,
                    frame_view: None,
                    upload_batch,
                    graph_resources: Some(graph_resources),
                    blackboard,
                    profiler,
                };
                helpers::execute_graph_raster_pass_node(
                    &self.passes[pass_idx],
                    &template,
                    graph_resources,
                    encoder,
                    &mut ctx,
                )?;
            }
            PassKind::Compute => {
                profiling::scope!("graph::record_compute");
                // encoder is moved into ComputePassCtx; pass uses ctx.encoder.
                let mut ctx = ComputePassCtx {
                    device,
                    gpu_limits,
                    queue: queue_arc,
                    encoder,
                    depth_view: Some(resolved.depth_view),
                    frame: Some(frame_params),
                    frame_shared: None,
                    frame_view: None,
                    upload_batch,
                    graph_resources: Some(graph_resources),
                    blackboard,
                    profiler,
                };
                self.passes[pass_idx]
                    .record_compute(&mut ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
            PassKind::Copy => {
                profiling::scope!("graph::record_copy");
                let mut ctx = ComputePassCtx {
                    device,
                    gpu_limits,
                    queue: queue_arc,
                    encoder,
                    depth_view: Some(resolved.depth_view),
                    frame: Some(frame_params),
                    frame_shared: None,
                    frame_view: None,
                    upload_batch,
                    graph_resources: Some(graph_resources),
                    blackboard,
                    profiler,
                };
                self.passes[pass_idx]
                    .record_copy(&mut ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
            PassKind::Callback => {
                profiling::scope!("graph::record_callback");
                let mut ctx = CallbackCtx {
                    device,
                    gpu_limits,
                    queue: queue_arc,
                    frame: Some(frame_params),
                    frame_shared: None,
                    frame_view: None,
                    upload_batch,
                    graph_resources: Some(graph_resources),
                    blackboard,
                };
                self.passes[pass_idx]
                    .run_callback(&mut ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }
        Ok(())
    }
}
