//! Frame-global command-buffer recording.

use hashbrown::HashMap;
use hashbrown::hash_map::Entry;
use std::time::Instant;

use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::context::GraphResolvedResources;
use crate::render_graph::error::GraphExecuteError;
use crate::render_graph::execution_backend::GraphExecutionBackend;
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::render_graph::pass::PassPhase;

use super::super::super::helpers;
use super::super::super::{
    CompiledRenderGraph, FrameGlobalView, FrameView, MultiViewExecutionContext, ResolvedView,
};
use super::super::{
    FrameGlobalPassRecordInputs, GraphResolveKey, TimedCommandBuffer,
    TransientTextureResolveSurfaceParams, elapsed_ms,
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
    /// Encodes [`crate::render_graph::pass::PassPhase::FrameGlobal`] passes into a command buffer.
    ///
    /// Returns `None` when there are no frame-global passes (nothing to submit for this phase).
    /// The caller is responsible for including the returned buffer in the single-submit batch.
    pub(in crate::render_graph::compiled::exec) fn encode_frame_global_passes(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
        frame_global: &FrameGlobalView,
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
    ) -> Result<Option<TimedCommandBuffer>, GraphExecuteError> {
        profiling::scope!("graph::frame_global");
        if self.frame_global_passes_are_inactive(&*mv_ctx.backend) {
            return Ok(None);
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
        Ok(Some(command_buffer))
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

        let first = views.first().ok_or(GraphExecuteError::NoViewsInBatch)?;
        let resolved = {
            profiling::scope!("graph::frame_global::resolve_target");
            Self::resolve_view_from_target(
                first.view_id(),
                first.profile,
                &first.host_camera,
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
