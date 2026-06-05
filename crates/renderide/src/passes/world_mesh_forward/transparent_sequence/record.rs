//! Ordered transparent sequence recording loop.

use crate::materials::SceneColorSnapshotMode;
use crate::render_graph::context::EncoderPassCtx;
use crate::render_graph::error::RenderPassError;

use super::super::{PreparedWorldMeshForwardFrame, WorldMeshForwardGraphResources};
use super::draw::{
    draw_grab_group, flush_optional_post_groups, transparent_sequence_frame_bind_groups,
};
use super::order::{advance_pending_post_run, transparent_sequence_phase_pair};
use super::snapshot::{
    copy_snapshot_for_mode, resolve_final_scene_color_for_skipped_tail,
    resolve_final_scene_color_if_needed, resolve_for_grab_snapshot,
    scene_color_snapshot_mode_for_group,
};

/// Records the ordered transparent tail and any scene-color snapshots needed by grab groups.
pub(super) fn record_transparent_sequence(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
) -> Result<bool, RenderPassError> {
    profiling::scope!("world_mesh_forward::transparent_sequence_record");
    let plan = &prepared.plan;
    let (transparent_phase, grab_phase) = transparent_sequence_phase_pair();
    let transparent_groups = plan.phase(transparent_phase);
    let grab_groups = plan.phase(grab_phase);
    let sample_count = ctx.frame.view.sample_count.max(1);
    let mut scene_color_resolved_current = false;
    if transparent_groups.is_empty() && grab_groups.is_empty() {
        return resolve_final_scene_color_for_skipped_tail(ctx, resources, sample_count);
    }
    let Some((default_frame_bind_group, named_frame_bind_group)) =
        transparent_sequence_frame_bind_groups(ctx)
    else {
        return resolve_final_scene_color_for_skipped_tail(ctx, resources, sample_count);
    };
    let mut post_idx = 0usize;
    let mut grab_idx = 0usize;
    let mut pending_post_start = None;
    let mut recorded_any = false;
    let mut named_background_snapshot_ready = false;

    while post_idx < transparent_groups.len() || grab_idx < grab_groups.len() {
        if advance_pending_post_run(plan, &mut post_idx, grab_idx, &mut pending_post_start) {
            continue;
        }

        let Some(flushed_post_groups) = flush_optional_post_groups(
            ctx,
            prepared,
            resources,
            pending_post_start.take(),
            post_idx,
            &default_frame_bind_group,
        )?
        else {
            return Ok(false);
        };
        recorded_any |= flushed_post_groups;
        if flushed_post_groups && sample_count > 1 {
            scene_color_resolved_current = false;
        }
        let grab_group = &grab_groups[grab_idx];
        let snapshot_mode = scene_color_snapshot_mode_for_group(prepared, grab_group);
        let needs_snapshot_copy = match snapshot_mode {
            SceneColorSnapshotMode::NamedBackgroundGrab => !named_background_snapshot_ready,
            SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => true,
        };
        if needs_snapshot_copy {
            scene_color_resolved_current |= resolve_for_grab_snapshot(ctx, resources)?;
            if !copy_snapshot_for_mode(
                ctx,
                prepared,
                resources,
                snapshot_mode,
                grab_idx,
                &mut named_background_snapshot_ready,
            ) {
                grab_idx += 1;
                continue;
            }
        }
        if !draw_grab_group(
            ctx,
            prepared,
            resources,
            grab_group,
            snapshot_mode,
            &default_frame_bind_group,
            &named_frame_bind_group,
        )? {
            return Ok(false);
        }
        recorded_any = true;
        if sample_count > 1 {
            scene_color_resolved_current = false;
        }
        grab_idx += 1;
    }

    let Some(flushed_post_tail) = flush_optional_post_groups(
        ctx,
        prepared,
        resources,
        pending_post_start,
        post_idx,
        &default_frame_bind_group,
    )?
    else {
        return Ok(false);
    };
    if flushed_post_tail {
        recorded_any = true;
        if sample_count > 1 {
            scene_color_resolved_current = false;
        }
    }
    resolve_final_scene_color_if_needed(
        ctx,
        resources,
        sample_count,
        scene_color_resolved_current,
    )?;
    Ok(recorded_any)
}
