//! Render-pass recording helpers for transparent sequence draw groups.

use std::sync::Arc;

use crate::materials::SceneColorSnapshotMode;
use crate::render_graph::context::EncoderPassCtx;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::gpu_cache::stereo_mask_or_template;
use crate::world_mesh::DrawGroup;

use super::super::attachments::forward_draw_attachment_targets;
use super::super::raster_recording::{
    frame_bind_group_for_view, record_world_mesh_forward_groups_graph_raster_with_frame_bind_group,
    stencil_load_ops,
};
use super::super::{PreparedWorldMeshForwardFrame, WorldMeshForwardGraphResources};
use super::order::transparent_sequence_phase_pair;

/// Draws a slice of already sorted transparent sequence groups.
pub(super) fn draw_tail_groups(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
    groups: &[DrawGroup],
    frame_bind_group: &Arc<wgpu::BindGroup>,
) -> Result<bool, RenderPassError> {
    if groups.is_empty() {
        return Ok(true);
    }

    let frame = &ctx.frame;
    let sample_count = frame.view.sample_count.max(1);
    let Some(targets) = forward_draw_attachment_targets(resources, sample_count) else {
        return Err(RenderPassError::FrameParamsRequired {
            pass: "WorldMeshForwardTransparentSequence missing MSAA resources".to_string(),
        });
    };
    let Some(color_view) = ctx.graph_resources.texture_view(targets.color) else {
        return Err(RenderPassError::FrameParamsRequired {
            pass: format!(
                "WorldMeshForwardTransparentSequence missing color {:?}",
                targets.color
            ),
        });
    };
    let Some(depth_view) = ctx.graph_resources.texture_view(targets.depth) else {
        return Err(RenderPassError::FrameParamsRequired {
            pass: format!(
                "WorldMeshForwardTransparentSequence missing depth {:?}",
                targets.depth
            ),
        });
    };

    let color_attachments = [Some(wgpu::RenderPassColorAttachment {
        view: color_view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
        depth_slice: None,
    })];
    let depth_stencil_attachment = Some(wgpu::RenderPassDepthStencilAttachment {
        view: depth_view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: stencil_load_ops(prepared.pipeline.pass_desc.depth_stencil_format),
    });

    let pass_query = ctx
        .profiler
        .map(|p| p.begin_pass_query("WorldMeshForwardTransparentSequenceDraw", ctx.encoder));
    let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
    let recorded = {
        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("WorldMeshForwardTransparentSequenceDraw"),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes,
            multiview_mask: stereo_mask_or_template(prepared.pipeline.use_multiview, None),
        });
        #[cfg(feature = "tracy")]
        rpass.push_debug_group("world_mesh_forward::transparent_sequence_draw");
        let recorded = record_world_mesh_forward_groups_graph_raster_with_frame_bind_group(
            &mut rpass,
            frame,
            prepared,
            groups,
            frame_bind_group,
            frame.view.view_id,
        );
        #[cfg(feature = "tracy")]
        rpass.pop_debug_group();
        recorded
    };
    if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
        p.end_query(ctx.encoder, q);
    }
    if let Some(stats) = ctx
        .blackboard
        .get_mut::<crate::render_graph::blackboard::GraphCommandStatsSlot>()
    {
        stats.record_opened_render_pass();
    }
    Ok(recorded)
}

/// Flushes the pending transparent post range when one exists.
fn flush_post_groups(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
    start: Option<usize>,
    end: usize,
    frame_bind_group: &Arc<wgpu::BindGroup>,
) -> Result<bool, RenderPassError> {
    let Some(start) = start else {
        return Ok(true);
    };
    let (transparent_phase, _) = transparent_sequence_phase_pair();
    draw_tail_groups(
        ctx,
        prepared,
        resources,
        &prepared.plan.phase(transparent_phase)[start..end],
        frame_bind_group,
    )
}

/// Flushes an optional pending transparent-post run and reports whether it had any groups.
pub(super) fn flush_optional_post_groups(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
    start: Option<usize>,
    end: usize,
    frame_bind_group: &Arc<wgpu::BindGroup>,
) -> Result<Option<bool>, RenderPassError> {
    let flushed = start.is_some();
    if !flush_post_groups(ctx, prepared, resources, start, end, frame_bind_group)? {
        return Ok(None);
    }
    Ok(Some(flushed))
}

/// Returns default and named-scene-color frame bind groups for the current view.
pub(super) fn transparent_sequence_frame_bind_groups(
    ctx: &EncoderPassCtx<'_, '_, '_>,
) -> Option<(Arc<wgpu::BindGroup>, Arc<wgpu::BindGroup>)> {
    let default = frame_bind_group_for_view(&ctx.frame, ctx.blackboard)?;
    let named = ctx
        .frame
        .systems
        .frame_resources
        .per_view_named_scene_color_frame_bind_group(ctx.frame.view.view_id)?;
    Some((default, named))
}

/// Draws one grab-pass group with the frame bind group selected by its snapshot mode.
pub(super) fn draw_grab_group(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
    grab_group: &DrawGroup,
    snapshot_mode: SceneColorSnapshotMode,
    default_frame_bind_group: &Arc<wgpu::BindGroup>,
    named_frame_bind_group: &Arc<wgpu::BindGroup>,
) -> Result<bool, RenderPassError> {
    let grab_frame_bind_group = match snapshot_mode {
        SceneColorSnapshotMode::NamedBackgroundGrab => named_frame_bind_group,
        SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => {
            default_frame_bind_group
        }
    };
    draw_tail_groups(
        ctx,
        prepared,
        resources,
        std::slice::from_ref(grab_group),
        grab_frame_bind_group,
    )
}
