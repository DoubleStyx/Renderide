//! Raster pass attachment resolution and the wgpu render-pass lifecycle for graph passes.

use crate::render_graph::blackboard::{GraphCommandStats, GraphCommandStatsSlot};
use crate::render_graph::context::{GraphResolvedResources, RasterPassCtx};
use crate::render_graph::error::GraphExecuteError;
use crate::render_graph::pass::{PassNode, RenderPassTemplate};
use crate::render_graph::resources::{
    TextureAttachmentResolve, TextureAttachmentTarget, TextureResourceHandle,
};

use super::super::CompiledPassInfo;

fn update_command_stats(
    ctx: &mut RasterPassCtx<'_, '_>,
    update: impl FnOnce(&mut GraphCommandStats),
) {
    if ctx.blackboard.get::<GraphCommandStatsSlot>().is_none() {
        ctx.blackboard
            .insert::<GraphCommandStatsSlot>(GraphCommandStats::default());
    }
    if let Some(stats) = ctx.blackboard.get_mut::<GraphCommandStatsSlot>() {
        update(stats);
    }
}

pub(in crate::render_graph::compiled) fn pass_info_raster_template(
    pass_info: &[CompiledPassInfo],
    pass_idx: usize,
) -> Result<RenderPassTemplate, GraphExecuteError> {
    let Some(info) = pass_info.get(pass_idx) else {
        return Err(GraphExecuteError::MissingRasterTemplate {
            pass: format!("pass#{pass_idx}"),
        });
    };
    info.raster_template
        .clone()
        .ok_or_else(|| GraphExecuteError::MissingRasterTemplate {
            pass: info.name.clone(),
        })
}

pub(in crate::render_graph::compiled) fn frame_sample_count_from_raster_ctx(
    ctx: &RasterPassCtx<'_, '_>,
) -> u32 {
    ctx.pass_frame.view.sample_count.max(1)
}

pub(in crate::render_graph::compiled) fn resolve_attachment_target(
    target: TextureAttachmentTarget,
    sample_count: u32,
) -> TextureResourceHandle {
    match target {
        TextureAttachmentTarget::Resource(handle) => handle,
        TextureAttachmentTarget::FrameSampled {
            single_sample,
            multisampled,
        } => {
            if sample_count > 1 {
                multisampled
            } else {
                single_sample
            }
        }
    }
}

pub(in crate::render_graph::compiled) fn resolve_attachment_resolve_target(
    target: TextureAttachmentResolve,
    sample_count: u32,
) -> Option<TextureResourceHandle> {
    match target {
        TextureAttachmentResolve::Always(handle) => Some(handle),
        TextureAttachmentResolve::FrameMultisampled(handle) => (sample_count > 1).then_some(handle),
    }
}

/// Resolves the raster pass template's color attachments to live `wgpu::TextureView`s.
pub(in crate::render_graph::compiled) fn resolve_color_attachments<'a>(
    pass_name: &str,
    template: &RenderPassTemplate,
    graph_resources: &'a GraphResolvedResources,
    sample_count: u32,
) -> Result<Vec<Option<wgpu::RenderPassColorAttachment<'a>>>, GraphExecuteError> {
    profiling::scope!("graph::raster::resolve_color_attachments");
    let mut color_attachments = Vec::with_capacity(template.color_attachments.len());
    for color in &template.color_attachments {
        let target = resolve_attachment_target(color.target, sample_count);
        let view = graph_resources.texture_view(target).ok_or_else(|| {
            GraphExecuteError::MissingGraphAttachment {
                pass: pass_name.to_owned(),
                resource: format!("{target:?}"),
            }
        })?;
        let resolve_target = match color
            .resolve_to
            .and_then(|t| resolve_attachment_resolve_target(t, sample_count))
        {
            Some(target) => Some(graph_resources.texture_view(target).ok_or_else(|| {
                GraphExecuteError::MissingGraphAttachment {
                    pass: pass_name.to_owned(),
                    resource: format!("{target:?}"),
                }
            })?),
            None => None,
        };
        color_attachments.push(Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target,
            ops: wgpu::Operations {
                load: color.load,
                store: color.store.resolve(sample_count),
            },
            depth_slice: None,
        }));
    }
    Ok(color_attachments)
}

/// Resolves the raster pass template's depth/stencil attachment with caller-selected stencil ops.
pub(in crate::render_graph::compiled) fn resolve_depth_attachment_with_stencil<'a>(
    pass_name: &str,
    template: &RenderPassTemplate,
    graph_resources: &'a GraphResolvedResources,
    sample_count: u32,
    stencil_ops: Option<wgpu::Operations<u32>>,
) -> Result<Option<wgpu::RenderPassDepthStencilAttachment<'a>>, GraphExecuteError> {
    profiling::scope!("graph::raster::resolve_depth_attachment");
    let Some(depth) = &template.depth_stencil_attachment else {
        return Ok(None);
    };
    let target = resolve_attachment_target(depth.target, sample_count);
    let view = graph_resources.texture_view(target).ok_or_else(|| {
        GraphExecuteError::MissingGraphAttachment {
            pass: pass_name.to_owned(),
            resource: format!("{target:?}"),
        }
    })?;
    Ok(Some(wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(depth.depth.resolve(sample_count)),
        stencil_ops,
    }))
}

/// Resolves the raster pass template's depth/stencil attachment, if any, to a live view.
fn resolve_depth_attachment<'a>(
    pass: &PassNode,
    template: &RenderPassTemplate,
    graph_resources: &'a GraphResolvedResources,
    sample_count: u32,
    ctx: &RasterPassCtx<'_, '_>,
) -> Result<Option<wgpu::RenderPassDepthStencilAttachment<'a>>, GraphExecuteError> {
    let stencil_ops = template
        .depth_stencil_attachment
        .as_ref()
        .and_then(|depth| pass.stencil_ops_override(ctx, depth));
    resolve_depth_attachment_with_stencil(
        pass.name(),
        template,
        graph_resources,
        sample_count,
        stencil_ops,
    )
}

/// Builds the attachment template used when compatible raster passes are materialized together.
pub(in crate::render_graph::compiled) fn coalesce_render_pass_template(
    templates: &[RenderPassTemplate],
) -> Option<RenderPassTemplate> {
    let (first, rest) = templates.split_first()?;
    if rest.is_empty() {
        return None;
    }
    let last = templates.last()?;
    if templates.iter().any(|template| {
        template
            .color_attachments
            .iter()
            .any(|color| color.resolve_to.is_some())
    }) {
        return None;
    }
    if !rest.iter().all(|template| {
        template
            .color_attachments
            .iter()
            .all(|color| matches!(color.load, wgpu::LoadOp::Load))
    }) {
        return None;
    }
    if !templates[..templates.len() - 1].iter().all(|template| {
        template
            .color_attachments
            .iter()
            .all(|color| color.store.stores_for_all_targets())
    }) {
        return None;
    }
    if !depth_ops_can_be_coalesced(templates) {
        return None;
    }
    let mut merged = first.clone();
    for (merged_color, last_color) in merged
        .color_attachments
        .iter_mut()
        .zip(&last.color_attachments)
    {
        merged_color.store = last_color.store;
    }
    if let (Some(merged_depth), Some(last_depth)) = (
        merged.depth_stencil_attachment.as_mut(),
        last.depth_stencil_attachment.as_ref(),
    ) {
        merged_depth.depth.store = last_depth.depth.store;
        merged_depth.stencil = last_depth.stencil;
    }
    Some(merged)
}

/// Returns whether depth/stencil load/store ops can be represented by one merged pass.
fn depth_ops_can_be_coalesced(templates: &[RenderPassTemplate]) -> bool {
    if templates.iter().any(|template| {
        template
            .depth_stencil_attachment
            .as_ref()
            .is_some_and(|depth| depth.stencil.is_some())
    }) {
        return false;
    }
    let Some((_, rest)) = templates.split_first() else {
        return false;
    };
    rest.iter().all(|template| {
        template
            .depth_stencil_attachment
            .as_ref()
            .is_none_or(|depth| matches!(depth.depth.load, wgpu::LoadOp::Load))
    }) && templates[..templates.len() - 1].iter().all(|template| {
        template
            .depth_stencil_attachment
            .as_ref()
            .is_none_or(|depth| depth.depth.store.stores_for_all_targets())
    })
}

/// Opens a graph-managed raster render pass for a [`PassNode::Raster`] variant and calls
/// [`crate::render_graph::pass::PassNode::record_raster`].
///
/// This is the primary path for raster passes in the new pass-node system. Attachment resolution
/// is delegated to [`resolve_color_attachments`] and [`resolve_depth_attachment`]; the body here
/// owns only the wgpu render-pass lifecycle and the GPU-profiler query bracketing.
pub(in crate::render_graph::compiled) fn execute_graph_raster_pass_node(
    pass: &PassNode,
    template: &RenderPassTemplate,
    graph_resources: &GraphResolvedResources,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &mut RasterPassCtx<'_, '_>,
) -> Result<(), GraphExecuteError> {
    let should_record = {
        profiling::scope!("graph::raster::should_record");
        pass.should_record_raster(ctx)
            .map_err(GraphExecuteError::Pass)?
    };
    if !should_record {
        update_command_stats(ctx, GraphCommandStats::record_skipped_pass);
        return Ok(());
    }

    let sample_count = frame_sample_count_from_raster_ctx(ctx);
    let color_attachments =
        resolve_color_attachments(pass.name(), template, graph_resources, sample_count)?;
    let depth_stencil_attachment =
        resolve_depth_attachment(pass, template, graph_resources, sample_count, ctx)?;
    let multiview_mask = pass.multiview_mask_override(ctx, template);

    let pass_profile_label = pass.profiling_label();
    let pass_query = ctx
        .profiler
        .map(|p| p.begin_pass_query(pass_profile_label, encoder));
    let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
    let mut rpass = {
        profiling::scope!("graph::raster::begin_render_pass");
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render-graph-raster"),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes,
            multiview_mask,
        })
    };
    {
        profiling::scope!("graph::raster::record_draws");
        pass.record_raster(ctx, &mut rpass)
            .map_err(GraphExecuteError::Pass)?;
    }
    update_command_stats(ctx, |stats| {
        stats.record_raster_pass();
        stats.record_opened_render_pass();
    });
    {
        profiling::scope!("graph::raster::end_render_pass");
        drop(rpass);
    }
    if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
        p.end_query(encoder, q);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::resources::{ImportedTextureHandle, TextureHandle};

    #[test]
    fn attachment_target_selects_single_or_multisampled_resource() {
        let single = TextureResourceHandle::Transient(TextureHandle(1));
        let multi = TextureResourceHandle::Transient(TextureHandle(2));
        let imported = TextureResourceHandle::Imported(ImportedTextureHandle(3));

        assert_eq!(
            resolve_attachment_target(TextureAttachmentTarget::Resource(imported), 8),
            imported
        );
        assert_eq!(
            resolve_attachment_target(
                TextureAttachmentTarget::FrameSampled {
                    single_sample: single,
                    multisampled: multi,
                },
                1,
            ),
            single
        );
        assert_eq!(
            resolve_attachment_target(
                TextureAttachmentTarget::FrameSampled {
                    single_sample: single,
                    multisampled: multi,
                },
                4,
            ),
            multi
        );
    }

    #[test]
    fn attachment_resolve_target_is_sample_count_aware() {
        let target = TextureResourceHandle::Transient(TextureHandle(7));

        assert_eq!(
            resolve_attachment_resolve_target(TextureAttachmentResolve::Always(target), 1),
            Some(target)
        );
        assert_eq!(
            resolve_attachment_resolve_target(
                TextureAttachmentResolve::FrameMultisampled(target),
                1,
            ),
            None
        );
        assert_eq!(
            resolve_attachment_resolve_target(
                TextureAttachmentResolve::FrameMultisampled(target),
                2,
            ),
            Some(target)
        );
    }
}
