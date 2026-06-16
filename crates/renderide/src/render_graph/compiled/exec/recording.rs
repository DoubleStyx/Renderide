//! Per-view and frame-global command encoding paths plus the single `execute_pass_node` dispatch.

mod blackboard;
mod frame_global;
mod materialization;
mod per_view;

use super::super::super::blackboard::{
    Blackboard, BlackboardRuntimeAccessViolation, GraphCommandStats, GraphCommandStatsSlot,
};
use super::super::super::context::{
    ComputePassCtx, EncoderPassCtx, GraphResolvedResources, PassFrameContext, RasterPassCtx,
};
use super::super::super::error::GraphExecuteError;
use super::super::super::pass::{PassKind, PassNode};
use super::super::helpers;
use super::super::{CompiledRenderGraph, ResolvedView};
use crate::frame_upload_batch::{FrameUploadBatch, FrameUploadScope, GraphUploadSink};

/// Pass phase and optional per-view index used to derive schedule and upload scopes.
#[derive(Clone, Copy)]
pub(super) struct PhaseRecordingScope {
    /// Schedule phase being recorded.
    pub(super) phase: super::super::super::pass::PassPhase,
    /// Per-view index for view-scoped passes.
    pub(super) view_idx: Option<usize>,
}

/// Concrete pass index and upload scope for one pass dispatch.
#[derive(Clone, Copy)]
pub(super) struct PassExecution {
    /// Pass index in the compiled pass list.
    pub(super) pass_idx: usize,
    /// Deferred upload scope used while this pass records.
    pub(super) upload_scope: FrameUploadScope,
}

/// Resolved view and resource table seen by one pass dispatch.
#[derive(Clone, Copy)]
pub(super) struct PassViewInputs<'a> {
    /// Resolved target, depth, and view metadata.
    pub(super) resolved: &'a ResolvedView<'a>,
    /// Typed graph resources resolved for this execution scope.
    pub(super) graph_resources: &'a GraphResolvedResources,
}

/// GPU handles shared by pass dispatch helpers.
#[derive(Clone, Copy)]
pub(super) struct PassGpuInputs<'a> {
    /// WGPU device used for pass-side resource creation.
    pub(super) device: &'a wgpu::Device,
    /// Effective device limits for this frame.
    pub(super) gpu_limits: &'a crate::gpu::GpuLimits,
    /// Optional GPU profiler handle for pass timestamp queries.
    pub(super) profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Mutable recording targets reborrowed for each pass or recording unit.
pub(super) struct PassRecordTargets<'a, 'frame, 'encoder> {
    /// Scene, backend system handles, and per-view frame state for this pass.
    pub(super) frame_params: &'frame mut crate::graph_inputs::GraphPassFrame<'a>,
    /// Per-scope typed blackboard populated before or during this scope.
    pub(super) blackboard: &'frame mut Blackboard,
    /// Active command encoder for this recording slice.
    pub(super) encoder: &'encoder mut wgpu::CommandEncoder,
}

impl<'a> PassRecordTargets<'a, '_, '_> {
    /// Reborrows the mutable targets for one nested pass dispatch.
    pub(super) fn reborrow(&mut self) -> PassRecordTargets<'a, '_, '_> {
        PassRecordTargets {
            frame_params: &mut *self.frame_params,
            blackboard: &mut *self.blackboard,
            encoder: &mut *self.encoder,
        }
    }
}

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
    pub(super) fn execute_pass_node<'a>(
        &self,
        execution: PassExecution,
        view: PassViewInputs<'a>,
        targets: PassRecordTargets<'a, '_, '_>,
        gpu: PassGpuInputs<'a>,
        upload_batch: &FrameUploadBatch,
    ) -> Result<(), GraphExecuteError> {
        let PassRecordTargets {
            frame_params,
            blackboard,
            encoder,
        } = targets;
        let _upload_scope = upload_batch.enter_scope(execution.upload_scope);
        let uploads = GraphUploadSink::new(upload_batch, execution.upload_scope);
        let pass = &self.passes[execution.pass_idx];
        let _pass_label = pass.profiling_label();
        profiling::scope!("graph::execute_pass_node", _pass_label.as_ref());
        self.validate_blackboard_inputs(execution.pass_idx, pass.name(), blackboard)?;
        self.begin_blackboard_access_validation(execution.pass_idx, pass.name(), blackboard);
        let record_result = (|| -> Result<(), GraphExecuteError> {
            match pass.kind() {
                PassKind::Raster => {
                    profiling::scope!("graph::record_raster");
                    let template =
                        helpers::pass_info_raster_template(&self.pass_info, execution.pass_idx)?;
                    let mut ctx = RasterPassCtx {
                        device: gpu.device,
                        frame: PassFrameContext::new(
                            &mut frame_params.shared,
                            &mut frame_params.view,
                        ),
                        uploads,
                        graph_resources: view.graph_resources,
                        blackboard: &mut *blackboard,
                        profiler: gpu.profiler,
                    };
                    helpers::execute_graph_raster_pass_node(
                        pass,
                        &template,
                        view.graph_resources,
                        encoder,
                        &mut ctx,
                    )
                }
                PassKind::Compute => {
                    profiling::scope!("graph::record_compute");
                    let ctx = {
                        profiling::scope!("graph::record_compute::build_context");
                        ComputePassCtx {
                            device: gpu.device,
                            gpu_limits: gpu.gpu_limits,
                            encoder,
                            depth_view: Some(view.resolved.depth_view),
                            frame: PassFrameContext::new(
                                &mut frame_params.shared,
                                &mut frame_params.view,
                            ),
                            uploads,
                            graph_resources: view.graph_resources,
                            blackboard: &mut *blackboard,
                            profiler: gpu.profiler,
                        }
                    };
                    record_compute_pass(pass, ctx)
                }
                PassKind::Encoder => {
                    profiling::scope!("graph::record_encoder");
                    let ctx = {
                        profiling::scope!("graph::record_encoder::build_context");
                        EncoderPassCtx {
                            device: gpu.device,
                            encoder,
                            frame: PassFrameContext::new(
                                &mut frame_params.shared,
                                &mut frame_params.view,
                            ),
                            uploads,
                            graph_resources: view.graph_resources,
                            blackboard: &mut *blackboard,
                            profiler: gpu.profiler,
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
