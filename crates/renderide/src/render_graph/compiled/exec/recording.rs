//! Per-view and frame-global command encoding paths plus the single `execute_pass_node` dispatch.

mod blackboard;
mod frame_global;
mod materialization;
mod per_view;

use super::super::super::blackboard::{
    Blackboard, BlackboardRuntimeAccessViolation, GraphCommandStats, GraphCommandStatsSlot,
};
use super::super::super::context::{
    ComputePassCtx, EncoderPassCtx, GraphResolvedResources, RasterPassCtx,
};
use super::super::super::error::GraphExecuteError;
use super::super::super::frame_upload_batch::{
    FrameUploadBatch, FrameUploadScope, GraphUploadSink,
};
use super::super::super::pass::{PassKind, PassNode};
use super::super::helpers;
use super::super::{CompiledRenderGraph, ResolvedView};

fn update_command_stats(blackboard: &mut Blackboard, update: impl FnOnce(&mut GraphCommandStats)) {
    if blackboard
        .get_untracked::<GraphCommandStatsSlot>()
        .is_none()
    {
        blackboard.insert_untracked::<GraphCommandStatsSlot>(GraphCommandStats::default());
    }
    if let Some(stats) = blackboard.get_mut_untracked::<GraphCommandStatsSlot>() {
        update(stats);
    }
}

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
        encoder: &mut wgpu::CommandEncoder,
        device: &'a wgpu::Device,
        gpu_limits: &'a crate::gpu::GpuLimits,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<(), GraphExecuteError> {
        let _upload_scope = upload_batch.enter_scope(upload_scope);
        let uploads = GraphUploadSink::new(upload_batch, upload_scope);
        let pass = &self.passes[pass_idx];
        let _pass_label = pass.profiling_label();
        profiling::scope!("graph::execute_pass_node", _pass_label.as_ref());
        self.validate_blackboard_inputs(pass_idx, pass.name(), blackboard)?;
        self.begin_blackboard_access_validation(pass_idx, pass.name(), blackboard);
        let record_result = (|| -> Result<(), GraphExecuteError> {
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
                    )
                }
                PassKind::Compute => {
                    profiling::scope!("graph::record_compute");
                    let ctx = {
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
                    record_compute_pass(pass, ctx)
                }
                PassKind::Encoder => {
                    profiling::scope!("graph::record_encoder");
                    let ctx = {
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
                    record_encoder_pass(pass, ctx)
                }
            }
        })();
        let access_result = self.finish_blackboard_access_validation(blackboard);
        record_result?;
        access_result
    }

    /// Starts runtime blackboard access validation for one pass when validation is enabled.
    pub(super) fn begin_blackboard_access_validation(
        &self,
        pass_idx: usize,
        pass_name: &str,
        blackboard: &Blackboard,
    ) {
        if !self.validation_mode.enabled() {
            return;
        }
        let Some(info) = self.pass_info.get(pass_idx) else {
            return;
        };
        blackboard.begin_access_validation(pass_name, &info.blackboard_accesses);
    }

    /// Completes runtime blackboard access validation for one pass.
    pub(super) fn finish_blackboard_access_validation(
        &self,
        blackboard: &Blackboard,
    ) -> Result<(), GraphExecuteError> {
        if !self.validation_mode.enabled() {
            return Ok(());
        }
        let violations = blackboard.finish_access_validation();
        self.report_blackboard_access_violations(&violations)
    }

    fn report_blackboard_access_violations(
        &self,
        violations: &[BlackboardRuntimeAccessViolation],
    ) -> Result<(), GraphExecuteError> {
        for violation in violations {
            if self.validation_mode.is_strict() {
                return Err(GraphExecuteError::UndeclaredBlackboardAccess {
                    pass: violation.pass.clone(),
                    slot: violation.slot,
                    access: violation.access.label(),
                });
            }
            logger::warn!(
                "render graph validation: pass `{}` performed undeclared blackboard {} access to slot `{}`",
                violation.pass,
                violation.access.label(),
                violation.slot
            );
        }
        Ok(())
    }
}

fn record_compute_pass(
    pass: &PassNode,
    mut ctx: ComputePassCtx<'_, '_, '_>,
) -> Result<(), GraphExecuteError> {
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
        update_command_stats(ctx.blackboard, GraphCommandStats::record_compute_pass);
        if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
            p.end_query(ctx.encoder, q);
        }
    } else {
        update_command_stats(ctx.blackboard, GraphCommandStats::record_skipped_pass);
    }
    Ok(())
}

fn record_encoder_pass(
    pass: &PassNode,
    mut ctx: EncoderPassCtx<'_, '_, '_>,
) -> Result<(), GraphExecuteError> {
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
        update_command_stats(ctx.blackboard, GraphCommandStats::record_encoder_pass);
        if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
            p.end_query(ctx.encoder, q);
        }
    } else {
        update_command_stats(ctx.blackboard, GraphCommandStats::record_skipped_pass);
    }
    Ok(())
}
