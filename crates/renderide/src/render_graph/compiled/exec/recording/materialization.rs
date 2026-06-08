//! Raster pass materialization and phase-step recording helpers.

use super::super::super::super::context::{PassFrameContext, RasterPassCtx};
use super::super::super::super::error::GraphExecuteError;
use super::super::super::super::frame_upload_batch::{FrameUploadBatch, GraphUploadSink};
use super::super::super::super::pass::{PassKind, PassNode};
use super::super::super::super::schedule::{RenderPassMaterializationGroup, ScheduleStep};
use super::super::super::CompiledRenderGraph;
use super::super::super::helpers;
use super::{
    PassExecution, PassGpuInputs, PassRecordTargets, PassViewInputs, PhaseRecordingScope,
    update_command_stats,
};
use crate::render_graph::blackboard::GraphCommandStats;

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
    pub(super) fn record_phase_steps<'a>(
        &self,
        scope: PhaseRecordingScope,
        view: PassViewInputs<'a>,
        mut targets: PassRecordTargets<'a, '_, '_>,
        gpu: PassGpuInputs<'a>,
        upload_batch: &FrameUploadBatch,
    ) -> Result<(), GraphExecuteError> {
        let mut step_idx = 0usize;
        while step_idx < self.schedule.steps.len() {
            let step = self.schedule.steps[step_idx];
            if step.phase != scope.phase {
                step_idx += 1;
                continue;
            }
            if let Some(group) = self.materialization_group_starting_at(step_idx)
                && self.try_execute_raster_materialization_group(
                    group,
                    scope,
                    view,
                    targets.reborrow(),
                    gpu,
                    upload_batch,
                )?
            {
                step_idx = group.end_step;
                continue;
            }
            self.execute_pass_node(
                PassExecution {
                    pass_idx: step.pass_idx,
                    upload_scope: step.frame_upload_scope(scope.view_idx),
                },
                view,
                targets.reborrow(),
                gpu,
                upload_batch,
            )?;
            step_idx += 1;
        }
        Ok(())
    }

    /// Attempts to record one compatible raster group inside a single `wgpu::RenderPass`.
    pub(super) fn try_execute_raster_materialization_group<'a>(
        &self,
        group: RenderPassMaterializationGroup,
        scope: PhaseRecordingScope,
        view: PassViewInputs<'a>,
        targets: PassRecordTargets<'a, '_, '_>,
        gpu: PassGpuInputs<'a>,
        upload_batch: &FrameUploadBatch,
    ) -> Result<bool, GraphExecuteError> {
        let steps = &self.schedule.steps[group.start_step..group.end_step];
        if steps.len() < 2 || !steps.iter().all(|step| step.phase == scope.phase) {
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

        let PassRecordTargets {
            frame_params,
            blackboard,
            encoder,
        } = targets;
        let uploads =
            GraphUploadSink::new(upload_batch, first_step.frame_upload_scope(scope.view_idx));
        let mut ctx = RasterPassCtx {
            device: gpu.device,
            frame: PassFrameContext::new(&mut frame_params.shared, &mut frame_params.view),
            uploads,
            graph_resources: view.graph_resources,
            blackboard,
            profiler: gpu.profiler,
        };
        if !self.materialized_group_dynamic_state_matches(steps, &templates, &ctx) {
            return Ok(false);
        }
        let first_pass = &self.passes[first_step.pass_idx];
        if !self.materialized_pass_should_record(first_step, first_pass, &ctx)? {
            update_command_stats(ctx.blackboard, GraphCommandStats::record_skipped_pass);
            return Ok(false);
        }

        let sample_count = helpers::frame_sample_count_from_raster_ctx(&ctx);
        let color_attachments = helpers::resolve_color_attachments(
            "render-graph-raster-merged",
            &merged_template,
            view.graph_resources,
            sample_count,
        )?;
        let stencil_ops = merged_template
            .depth_stencil_attachment
            .as_ref()
            .and_then(|depth| first_pass.stencil_ops_override(&ctx, depth));
        let depth_stencil_attachment = helpers::resolve_depth_attachment_with_stencil(
            "render-graph-raster-merged",
            &merged_template,
            view.graph_resources,
            sample_count,
            stencil_ops,
        )?;
        let multiview_mask = first_pass.multiview_mask_override(&ctx, &merged_template);
        let merged_profile_label = self.materialized_group_profile_label(steps);
        let pass_query = ctx
            .profiler
            .map(|p| p.begin_pass_query(merged_profile_label.as_str(), encoder));
        let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(merged_profile_label.as_str()),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes,
            multiview_mask,
        });
        for (idx, step) in steps.iter().copied().enumerate() {
            let pass = &self.passes[step.pass_idx];
            let upload_scope = step.frame_upload_scope(scope.view_idx);
            let _upload_scope = upload_batch.enter_scope(upload_scope);
            ctx.uploads = GraphUploadSink::new(upload_batch, upload_scope);
            let should_record = if idx == 0 {
                true
            } else {
                self.materialized_pass_should_record(step, pass, &ctx)?
            };
            if !should_record {
                update_command_stats(ctx.blackboard, GraphCommandStats::record_skipped_pass);
                continue;
            }
            self.record_materialized_raster_step(step, pass, &mut ctx, &mut rpass)?;
            update_command_stats(ctx.blackboard, GraphCommandStats::record_raster_pass);
        }
        update_command_stats(ctx.blackboard, GraphCommandStats::record_opened_render_pass);
        drop(rpass);
        if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
            p.end_query(encoder, q);
        }
        Ok(true)
    }

    /// Runs a materialized pass predicate with declared blackboard access validation.
    fn materialized_pass_should_record(
        &self,
        step: ScheduleStep,
        pass: &PassNode,
        ctx: &RasterPassCtx<'_, '_>,
    ) -> Result<bool, GraphExecuteError> {
        self.validate_blackboard_inputs(step.pass_idx, pass.name(), ctx.blackboard)?;
        self.begin_blackboard_access_validation(step.pass_idx, pass.name(), ctx.blackboard);
        let should_record = pass
            .should_record_raster(ctx)
            .map_err(GraphExecuteError::Pass);
        let access_result = self.finish_blackboard_access_validation(ctx.blackboard);
        let should_record = should_record?;
        access_result?;
        Ok(should_record)
    }

    /// Records one logical raster pass inside a materialized render pass.
    fn record_materialized_raster_step(
        &self,
        step: ScheduleStep,
        pass: &PassNode,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.validate_blackboard_inputs(step.pass_idx, pass.name(), ctx.blackboard)?;
        self.begin_blackboard_access_validation(step.pass_idx, pass.name(), ctx.blackboard);
        let record_result = pass
            .record_raster(ctx, rpass)
            .map_err(GraphExecuteError::Pass);
        let access_result = self.finish_blackboard_access_validation(ctx.blackboard);
        record_result?;
        access_result
    }

    /// Builds the GPU label for a merged raster group from its constituent pass labels.
    fn materialized_group_profile_label(&self, steps: &[ScheduleStep]) -> String {
        let mut label = String::from("graph::raster_merge(");
        for (idx, step) in steps.iter().enumerate() {
            if idx != 0 {
                label.push_str(" + ");
            }
            label.push_str(self.passes[step.pass_idx].profiling_label().as_ref());
        }
        label.push(')');
        label
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
}
