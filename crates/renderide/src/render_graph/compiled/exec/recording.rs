//! Per-view and frame-global command encoding paths plus the single `execute_pass_node` dispatch.

mod blackboard;
mod frame_global;
mod materialization;
mod per_view;

use super::super::super::blackboard::Blackboard;
use super::super::super::context::{
    ComputePassCtx, EncoderPassCtx, GraphResolvedResources, RasterPassCtx,
};
use super::super::super::error::GraphExecuteError;
use super::super::super::frame_upload_batch::{
    FrameUploadBatch, FrameUploadScope, GraphUploadSink,
};
use super::super::super::pass::PassKind;
use super::super::helpers;
use super::super::{CompiledRenderGraph, ResolvedView};

impl CompiledRenderGraph {
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
        frame_params: &mut crate::graph_inputs::GraphPassFrame<'a>,
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
