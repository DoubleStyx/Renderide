//! Scene-color snapshot and MSAA resolve helpers for grab-pass transparent draws.

use crate::materials::SceneColorSnapshotMode;
use crate::render_graph::context::EncoderPassCtx;
use crate::render_graph::error::RenderPassError;
use crate::world_mesh::DrawGroup;

use super::super::color_resolve::{
    WorldMeshForwardColorResolveEncodeContext, WorldMeshForwardColorResolveGraphResources,
    encode_world_mesh_forward_msaa_color_resolve,
};
use super::super::color_snapshot::encode_world_mesh_forward_color_snapshot;
use super::super::{PreparedWorldMeshForwardFrame, WorldMeshForwardGraphResources};
use super::final_scene_color_resolve_needed;

/// Returns the color resolve resources when the forward pass is multisampled.
fn color_resolve_resources(
    resources: WorldMeshForwardGraphResources,
) -> Option<WorldMeshForwardColorResolveGraphResources> {
    resources
        .msaa
        .map(|msaa| WorldMeshForwardColorResolveGraphResources {
            scene_color_hdr_msaa: msaa.scene_color_hdr,
            scene_color_hdr: resources.scene_color_hdr,
        })
}

fn record_color_resolve_stats(ctx: &mut EncoderPassCtx<'_, '_, '_>, resolved: bool) {
    if let Some(stats) = ctx
        .blackboard
        .get_mut::<crate::render_graph::blackboard::GraphCommandStatsSlot>()
    {
        stats.record_resolve_result(resolved);
        if resolved {
            stats.record_opened_render_pass();
        }
    }
}

fn encode_color_resolve(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    resources: WorldMeshForwardGraphResources,
    label: &'static str,
) -> Result<bool, RenderPassError> {
    let Some(resolve_resources) = color_resolve_resources(resources) else {
        return Ok(false);
    };
    let resolved =
        encode_world_mesh_forward_msaa_color_resolve(WorldMeshForwardColorResolveEncodeContext {
            device: ctx.device,
            graph_resources: ctx.graph_resources,
            encoder: ctx.encoder,
            frame: ctx.pass_frame,
            uploads: ctx.uploads,
            resources: resolve_resources,
            profiler: ctx.profiler,
            label,
        })?;
    record_color_resolve_stats(ctx, resolved);
    Ok(resolved)
}

/// Resolves the multisampled scene color before a grab snapshot copies the single-sample target.
pub(super) fn resolve_for_grab_snapshot(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    resources: WorldMeshForwardGraphResources,
) -> Result<bool, RenderPassError> {
    encode_color_resolve(
        ctx,
        resources,
        "WorldMeshForwardTransparentSequencePreGrabResolve",
    )
}

/// Copies the default per-object scene-color snapshot for a grab-pass group.
fn copy_grab_snapshot(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
) -> bool {
    let copied = encode_world_mesh_forward_color_snapshot(
        ctx.graph_resources,
        ctx.encoder,
        ctx.pass_frame,
        prepared,
        resources,
        ctx.profiler,
    );
    if let Some(stats) = ctx
        .blackboard
        .get_mut::<crate::render_graph::blackboard::GraphCommandStatsSlot>()
    {
        stats.record_copy_result(copied);
    }
    copied
}

/// Copies the reusable named background scene-color snapshot for the current view.
fn copy_named_grab_snapshot(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
) -> bool {
    profiling::scope!("world_mesh_forward::encode_named_color_snapshot");
    if !prepared.helper_needs.color_snapshot {
        logger::warn!(
            "world mesh named color snapshot copy: helper needs did not request a color snapshot"
        );
        return false;
    }
    if !ctx.pass_frame.shared.frame_resources.has_frame_gpu() {
        logger::warn!("world mesh named color snapshot copy: frame GPU resources are unavailable");
        return false;
    }
    let Some(source_color) = ctx
        .graph_resources
        .transient_texture(resources.scene_color_hdr)
    else {
        logger::warn!(
            "world mesh named color snapshot copy: resolved scene color source is unavailable"
        );
        return false;
    };
    let copy_query = ctx.profiler.map(|p| {
        p.begin_query(
            "world_mesh_forward::named_scene_color_snapshot_copy",
            ctx.encoder,
        )
    });
    let copied = ctx
        .pass_frame
        .shared
        .frame_resources
        .copy_named_scene_color_snapshot_for_view(
            ctx.pass_frame.view.view_id,
            ctx.encoder,
            &source_color.texture,
            ctx.pass_frame.view.viewport_px,
            prepared.pipeline.use_multiview,
        );
    if let (Some(profiler), Some(query)) = (ctx.profiler, copy_query) {
        profiler.end_query(ctx.encoder, query);
    }
    if let Some(stats) = ctx
        .blackboard
        .get_mut::<crate::render_graph::blackboard::GraphCommandStatsSlot>()
    {
        stats.record_copy_result(copied);
    }
    copied
}

/// Returns the scene-color snapshot refresh policy for a grab-pass draw group.
pub(super) fn scene_color_snapshot_mode_for_group(
    prepared: &PreparedWorldMeshForwardFrame,
    group: &DrawGroup,
) -> SceneColorSnapshotMode {
    prepared
        .draws
        .get(group.representative_draw_idx)
        .map(|draw| draw.batch_key.scene_color_snapshot_mode)
        .filter(|mode| mode.uses_scene_color())
        .unwrap_or(SceneColorSnapshotMode::PerObjectGrab)
}

/// Copies the scene-color snapshot required before drawing a grab-pass group.
pub(super) fn copy_snapshot_for_mode(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
    snapshot_mode: SceneColorSnapshotMode,
    grab_idx: usize,
    named_background_snapshot_ready: &mut bool,
) -> bool {
    let copied = match snapshot_mode {
        SceneColorSnapshotMode::NamedBackgroundGrab => {
            copy_named_grab_snapshot(ctx, prepared, resources)
        }
        SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => {
            copy_grab_snapshot(ctx, prepared, resources)
        }
    };
    if !copied {
        logger::warn!(
            "WorldMeshForwardTransparentSequence: skipping grab-pass filter group {} because scene-color snapshot copy failed",
            grab_idx
        );
        return false;
    }
    if snapshot_mode == SceneColorSnapshotMode::NamedBackgroundGrab {
        *named_background_snapshot_ready = true;
    }
    true
}

/// Resolves the multisampled forward color into the single-sample scene color consumed downstream.
fn resolve_final_scene_color(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    resources: WorldMeshForwardGraphResources,
) -> Result<(), RenderPassError> {
    encode_color_resolve(
        ctx,
        resources,
        "WorldMeshForwardTransparentSequenceFinalResolve",
    )?;
    Ok(())
}

/// Resolves final scene color when MSAA produced a newer multisampled source.
pub(super) fn resolve_final_scene_color_if_needed(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    resources: WorldMeshForwardGraphResources,
    sample_count: u32,
    scene_color_resolved_current: bool,
) -> Result<(), RenderPassError> {
    if final_scene_color_resolve_needed(true, resources.msaa_enabled(), sample_count)
        && !scene_color_resolved_current
    {
        resolve_final_scene_color(ctx, resources)?;
    }
    Ok(())
}

/// Resolves opaque-only MSAA color before skipping the transparent tail.
pub(super) fn resolve_final_scene_color_for_skipped_tail(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    resources: WorldMeshForwardGraphResources,
    sample_count: u32,
) -> Result<bool, RenderPassError> {
    resolve_final_scene_color_if_needed(ctx, resources, sample_count, false)?;
    Ok(false)
}
