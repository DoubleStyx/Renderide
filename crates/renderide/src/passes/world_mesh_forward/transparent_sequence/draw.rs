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
use super::order::consecutive_named_grab_run_end;
use super::snapshot::scene_color_snapshot_mode_for_group;

/// Draws sorted transparent and grab ranges in one render pass.
pub(super) fn draw_transparent_sequence_ranges(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
    transparent_groups: &[DrawGroup],
    grab_groups: &[DrawGroup],
    default_frame_bind_group: &Arc<wgpu::BindGroup>,
    named_frame_bind_group: &Arc<wgpu::BindGroup>,
) -> Result<bool, RenderPassError> {
    if transparent_groups.is_empty() && grab_groups.is_empty() {
        return Ok(true);
    }

    let device = ctx.device;
    let frame = &ctx.frame;
    let geometry_arena_arc = frame.systems.frame_resources.shared_geometry_arena();
    let geometry_arena_guard = geometry_arena_arc.as_ref().map(|arena| arena.read());
    let geometry_arena = geometry_arena_guard
        .as_ref()
        .and_then(|guard| guard.as_ref());
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
        #[cfg(feature = "tracy-gpu")]
        rpass.push_debug_group("world_mesh_forward::transparent_sequence_draw");

        let mut post_idx = 0usize;
        let mut grab_idx = 0usize;
        let mut recorded = true;
        while post_idx < transparent_groups.len() || grab_idx < grab_groups.len() {
            let next_is_post = transparent_groups.get(post_idx).is_some_and(|post| {
                grab_groups
                    .get(grab_idx)
                    .is_none_or(|grab| post.representative_draw_idx <= grab.representative_draw_idx)
            });
            if next_is_post {
                let post_start = post_idx;
                let next_grab_draw_idx = grab_groups
                    .get(grab_idx)
                    .map(|grab| grab.representative_draw_idx)
                    .unwrap_or(usize::MAX);
                while transparent_groups
                    .get(post_idx)
                    .is_some_and(|post| post.representative_draw_idx <= next_grab_draw_idx)
                {
                    post_idx += 1;
                }
                recorded = record_world_mesh_forward_groups_graph_raster_with_frame_bind_group(
                    &mut rpass,
                    frame,
                    prepared,
                    &transparent_groups[post_start..post_idx],
                    default_frame_bind_group,
                    frame.view.view_id,
                    device,
                    ctx.uploads,
                    geometry_arena,
                    None,
                    None,
                );
            } else {
                let grab_end = consecutive_named_grab_run_end(
                    transparent_groups,
                    grab_groups,
                    post_idx,
                    grab_idx,
                    |_, group| scene_color_snapshot_mode_for_group(prepared, group),
                );
                let frame_bind_group =
                    match scene_color_snapshot_mode_for_group(prepared, &grab_groups[grab_idx]) {
                        SceneColorSnapshotMode::NamedBackgroundGrab => named_frame_bind_group,
                        SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => {
                            default_frame_bind_group
                        }
                    };
                recorded = record_world_mesh_forward_groups_graph_raster_with_frame_bind_group(
                    &mut rpass,
                    frame,
                    prepared,
                    &grab_groups[grab_idx..grab_end],
                    frame_bind_group,
                    frame.view.view_id,
                    device,
                    ctx.uploads,
                    geometry_arena,
                    None,
                    None,
                );
                grab_idx = grab_end;
            }
            if !recorded {
                break;
            }
        }

        #[cfg(feature = "tracy-gpu")]
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
