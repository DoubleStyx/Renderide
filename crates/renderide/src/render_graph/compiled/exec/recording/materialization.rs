//! Raster pass materialization and phase-step recording helpers.

use super::super::super::super::blackboard::Blackboard;
use super::super::super::super::context::{GraphResolvedResources, RasterPassCtx};
use super::super::super::super::error::GraphExecuteError;
use super::super::super::super::frame_upload_batch::{FrameUploadBatch, GraphUploadSink};
use super::super::super::super::pass::{PassKind, PassPhase};
use super::super::super::super::schedule::{RenderPassMaterializationGroup, ScheduleStep};
use super::super::super::helpers;
use super::super::super::{CompiledRenderGraph, ResolvedView};

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
    pub(super) fn record_phase_steps<'a>(
        &self,
        phase: PassPhase,
        view_idx: Option<usize>,
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
        frame_params: &mut crate::graph_inputs::GraphPassFrame<'a>,
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
}
