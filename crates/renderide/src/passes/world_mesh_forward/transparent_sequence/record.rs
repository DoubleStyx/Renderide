//! Ordered transparent sequence recording loop.

use crate::materials::SceneColorSnapshotMode;
use crate::render_graph::context::EncoderPassCtx;
use crate::render_graph::error::RenderPassError;

use super::super::{PreparedWorldMeshForwardFrame, WorldMeshForwardGraphResources};
use super::draw::{draw_transparent_sequence_ranges, transparent_sequence_frame_bind_groups};
use super::order::{next_sequence_entry_is_post, transparent_sequence_phase_pair};
use super::snapshot::{
    refresh_snapshot_for_mode, resolve_final_scene_color_for_skipped_tail,
    resolve_final_scene_color_if_needed, scene_color_snapshot_mode_for_group,
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
    let mut pending_post_start = 0usize;
    let mut pending_grab_start = 0usize;
    let mut recorded_any = false;
    let mut named_background_snapshot_ready = false;

    while post_idx < transparent_groups.len() || grab_idx < grab_groups.len() {
        if next_sequence_entry_is_post(plan, post_idx, grab_idx) {
            post_idx += 1;
            continue;
        }

        let grab_group = &grab_groups[grab_idx];
        let snapshot_mode = scene_color_snapshot_mode_for_group(prepared, grab_group);
        let needs_snapshot_refresh = match snapshot_mode {
            SceneColorSnapshotMode::NamedBackgroundGrab => !named_background_snapshot_ready,
            SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => true,
        };
        if needs_snapshot_refresh {
            let pending_draws = pending_post_start < post_idx || pending_grab_start < grab_idx;
            if pending_draws {
                if !draw_transparent_sequence_ranges(
                    ctx,
                    prepared,
                    resources,
                    &transparent_groups[pending_post_start..post_idx],
                    &grab_groups[pending_grab_start..grab_idx],
                    &default_frame_bind_group,
                    &named_frame_bind_group,
                )? {
                    return Ok(false);
                }
                recorded_any = true;
            }
            pending_post_start = post_idx;
            pending_grab_start = grab_idx;

            if !refresh_snapshot_for_mode(
                ctx,
                prepared,
                resources,
                snapshot_mode,
                grab_idx,
                &mut named_background_snapshot_ready,
            )? {
                grab_idx += 1;
                pending_grab_start = grab_idx;
                continue;
            }
        }

        grab_idx += 1;
    }

    if pending_post_start < post_idx || pending_grab_start < grab_idx {
        if !draw_transparent_sequence_ranges(
            ctx,
            prepared,
            resources,
            &transparent_groups[pending_post_start..post_idx],
            &grab_groups[pending_grab_start..grab_idx],
            &default_frame_bind_group,
            &named_frame_bind_group,
        )? {
            return Ok(false);
        }
        recorded_any = true;
    }

    resolve_final_scene_color_if_needed(ctx, resources, sample_count, false)?;
    Ok(recorded_any)
}
