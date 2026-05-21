//! Ordered transparent tail with per-grab and reusable named scene-color snapshots.

use std::sync::Arc;

use crate::graph_inputs::PerViewFramePlanSlot;
use crate::materials::SceneColorSnapshotMode;
use crate::render_graph::context::EncoderPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::stereo_mask_or_template;
use crate::render_graph::pass::{EncoderPass, PassBuilder};
use crate::render_graph::resources::{TextureAccess, TextureResourceHandle};
use crate::world_mesh::{DrawGroup, InstancePlan, MeshPassKind, WorldMeshPhase};

use super::color_resolve::{
    WorldMeshForwardColorResolveEncodeContext, WorldMeshForwardColorResolveGraphResources,
    encode_world_mesh_forward_msaa_color_resolve,
};
use super::color_snapshot::encode_world_mesh_forward_color_snapshot;
use super::raster_recording::{
    frame_bind_group_for_view, record_world_mesh_forward_groups_graph_raster_with_frame_bind_group,
    stencil_load_ops,
};
use super::{
    PreparedWorldMeshForwardFrame, WorldMeshForwardGraphResources, WorldMeshForwardPlanSlot,
    declare_forward_draw_reads,
};

/// Draws regular transparent groups and grab-pass groups in sorted order.
#[derive(Debug)]
pub struct WorldMeshForwardTransparentSequencePass {
    resources: WorldMeshForwardGraphResources,
}

impl WorldMeshForwardTransparentSequencePass {
    /// Creates the ordered transparent tail pass.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
    }
}

/// Returns whether the ordered transparent tail has view-local work to record.
pub(in crate::passes::world_mesh_forward) fn forward_transparent_sequence_needed(
    opaque_recorded: bool,
    plan: &InstancePlan,
) -> bool {
    let (transparent_phase, grab_phase) = transparent_sequence_phase_pair();
    opaque_recorded && (!plan.phase_is_empty(transparent_phase) || !plan.phase_is_empty(grab_phase))
}

fn transparent_sequence_phase_pair() -> (WorldMeshPhase, WorldMeshPhase) {
    match MeshPassKind::TransparentSequence.phases() {
        [transparent, grab] => (*transparent, *grab),
        _ => (WorldMeshPhase::Transparent, WorldMeshPhase::TransparentGrab),
    }
}

fn declare_transparent_sequence_accesses(
    b: &mut PassBuilder<'_>,
    resources: WorldMeshForwardGraphResources,
) {
    b.write_texture(
        resources.scene_color_hdr,
        TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
        },
    );
    b.read_texture(resources.scene_color_hdr, TextureAccess::CopySrc);
    b.write_texture(
        resources.scene_color_hdr_msaa,
        TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
        },
    );
    b.read_texture(
        resources.scene_color_hdr_msaa,
        TextureAccess::Sampled {
            stages: wgpu::ShaderStages::FRAGMENT,
        },
    );
    b.import_texture(
        resources.depth,
        TextureAccess::DepthAttachment {
            depth: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            stencil: None,
        },
    );
    b.write_texture(
        resources.msaa_depth,
        TextureAccess::DepthAttachment {
            depth: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            stencil: None,
        },
    );
    declare_forward_draw_reads(b, resources);
}

fn next_sequence_entry_is_post(plan: &InstancePlan, post_idx: usize, grab_idx: usize) -> bool {
    let (transparent_phase, grab_phase) = transparent_sequence_phase_pair();
    let Some(post) = plan.phase(transparent_phase).get(post_idx) else {
        return false;
    };
    let Some(grab) = plan.phase(grab_phase).get(grab_idx) else {
        return true;
    };
    post.representative_draw_idx <= grab.representative_draw_idx
}

fn color_resolve_resources(
    resources: WorldMeshForwardGraphResources,
) -> WorldMeshForwardColorResolveGraphResources {
    WorldMeshForwardColorResolveGraphResources {
        scene_color_hdr_msaa: resources.scene_color_hdr_msaa,
        scene_color_hdr: resources.scene_color_hdr,
    }
}

fn draw_tail_groups(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
    groups: &[DrawGroup],
    frame_bind_group: &Arc<wgpu::BindGroup>,
) -> Result<bool, RenderPassError> {
    if groups.is_empty() {
        return Ok(true);
    }

    let frame = &*ctx.pass_frame;
    let sample_count = frame.view.sample_count.max(1);
    let color_handle = if sample_count > 1 {
        TextureResourceHandle::Transient(resources.scene_color_hdr_msaa)
    } else {
        TextureResourceHandle::Transient(resources.scene_color_hdr)
    };
    let depth_handle = if sample_count > 1 {
        TextureResourceHandle::Transient(resources.msaa_depth)
    } else {
        TextureResourceHandle::Imported(resources.depth)
    };
    let Some(color_view) = ctx.graph_resources.texture_view(color_handle) else {
        return Err(RenderPassError::FrameParamsRequired {
            pass: format!("WorldMeshForwardTransparentSequence missing color {color_handle:?}"),
        });
    };
    let Some(depth_view) = ctx.graph_resources.texture_view(depth_handle) else {
        return Err(RenderPassError::FrameParamsRequired {
            pass: format!("WorldMeshForwardTransparentSequence missing depth {depth_handle:?}"),
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
        record_world_mesh_forward_groups_graph_raster_with_frame_bind_group(
            &mut rpass,
            frame,
            prepared,
            groups,
            frame_bind_group,
        )
    };
    if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
        p.end_query(ctx.encoder, q);
    }
    Ok(recorded)
}

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

fn resolve_for_grab_snapshot(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    resources: WorldMeshForwardGraphResources,
) -> Result<(), RenderPassError> {
    encode_world_mesh_forward_msaa_color_resolve(WorldMeshForwardColorResolveEncodeContext {
        device: ctx.device,
        graph_resources: ctx.graph_resources,
        encoder: ctx.encoder,
        frame: ctx.pass_frame,
        uploads: ctx.uploads,
        resources: color_resolve_resources(resources),
        profiler: ctx.profiler,
        label: "WorldMeshForwardTransparentSequencePreGrabResolve",
    })?;
    Ok(())
}

fn copy_grab_snapshot(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
) -> bool {
    encode_world_mesh_forward_color_snapshot(
        ctx.graph_resources,
        ctx.encoder,
        ctx.pass_frame,
        prepared,
        resources,
        ctx.profiler,
    )
}

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
    copied
}

fn transparent_sequence_frame_bind_groups(
    ctx: &EncoderPassCtx<'_, '_, '_>,
) -> Option<(Arc<wgpu::BindGroup>, Arc<wgpu::BindGroup>)> {
    let default = frame_bind_group_for_view(ctx.pass_frame, ctx.blackboard)?;
    let named = ctx
        .pass_frame
        .shared
        .frame_resources
        .per_view_named_scene_color_frame_bind_group(ctx.pass_frame.view.view_id)?;
    Some((default, named))
}

/// Returns the scene-color snapshot refresh policy for a grab-pass draw group.
fn scene_color_snapshot_mode_for_group(
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

fn final_resolve_after_tail(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    resources: WorldMeshForwardGraphResources,
) -> Result<(), RenderPassError> {
    encode_world_mesh_forward_msaa_color_resolve(WorldMeshForwardColorResolveEncodeContext {
        device: ctx.device,
        graph_resources: ctx.graph_resources,
        encoder: ctx.encoder,
        frame: ctx.pass_frame,
        uploads: ctx.uploads,
        resources: color_resolve_resources(resources),
        profiler: ctx.profiler,
        label: "WorldMeshForwardTransparentSequenceFinalResolve",
    })?;
    Ok(())
}

fn record_transparent_sequence(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
) -> Result<bool, RenderPassError> {
    profiling::scope!("world_mesh_forward::transparent_sequence_record");
    let plan = &prepared.plan;
    let (transparent_phase, grab_phase) = transparent_sequence_phase_pair();
    let transparent_groups = plan.phase(transparent_phase);
    let grab_groups = plan.phase(grab_phase);
    let Some((default_frame_bind_group, named_frame_bind_group)) =
        transparent_sequence_frame_bind_groups(ctx)
    else {
        return Ok(false);
    };
    let mut post_idx = 0usize;
    let mut grab_idx = 0usize;
    let mut pending_post_start = None;
    let mut recorded_any = false;
    let mut named_background_snapshot_ready = false;

    while post_idx < transparent_groups.len() || grab_idx < grab_groups.len() {
        if next_sequence_entry_is_post(plan, post_idx, grab_idx) {
            if pending_post_start.is_none() {
                pending_post_start = Some(post_idx);
            }
            post_idx += 1;
            continue;
        }

        let flushed_post_groups = pending_post_start.is_some();
        if !flush_post_groups(
            ctx,
            prepared,
            resources,
            pending_post_start.take(),
            post_idx,
            &default_frame_bind_group,
        )? {
            return Ok(false);
        }
        recorded_any |= flushed_post_groups;
        let grab_group = &grab_groups[grab_idx];
        let snapshot_mode = scene_color_snapshot_mode_for_group(prepared, grab_group);
        let needs_snapshot_copy = match snapshot_mode {
            SceneColorSnapshotMode::NamedBackgroundGrab => !named_background_snapshot_ready,
            SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => true,
        };
        if needs_snapshot_copy {
            resolve_for_grab_snapshot(ctx, resources)?;
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
                grab_idx += 1;
                continue;
            }
            if snapshot_mode == SceneColorSnapshotMode::NamedBackgroundGrab {
                named_background_snapshot_ready = true;
            }
        }
        let grab_frame_bind_group = match snapshot_mode {
            SceneColorSnapshotMode::NamedBackgroundGrab => &named_frame_bind_group,
            SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => {
                &default_frame_bind_group
            }
        };
        if !draw_tail_groups(
            ctx,
            prepared,
            resources,
            std::slice::from_ref(grab_group),
            grab_frame_bind_group,
        )? {
            return Ok(false);
        }
        recorded_any = true;
        grab_idx += 1;
    }

    if !flush_post_groups(
        ctx,
        prepared,
        resources,
        pending_post_start,
        post_idx,
        &default_frame_bind_group,
    )? {
        return Ok(false);
    }
    if post_idx > 0 {
        recorded_any = true;
    }
    if recorded_any {
        final_resolve_after_tail(ctx, resources)?;
    }
    Ok(recorded_any)
}

impl EncoderPass for WorldMeshForwardTransparentSequencePass {
    fn name(&self) -> &str {
        "WorldMeshForwardTransparentSequence"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.encoder();
        b.read_blackboard::<PerViewFramePlanSlot>();
        b.read_optional_blackboard::<WorldMeshForwardPlanSlot>();
        b.write_blackboard::<WorldMeshForwardPlanSlot>();
        declare_transparent_sequence_accesses(b, self.resources);
        Ok(())
    }

    fn should_record(&self, ctx: &EncoderPassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
        Ok(ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .is_some_and(|prepared| {
                forward_transparent_sequence_needed(prepared.opaque_recorded, &prepared.plan)
            }))
    }

    fn record(&self, ctx: &mut EncoderPassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(mut prepared) = ctx.blackboard.take::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        let recorded = if prepared.opaque_recorded {
            record_transparent_sequence(ctx, &prepared, self.resources)?
        } else {
            false
        };
        if recorded {
            prepared.tail_raster_recorded = true;
        }
        ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
enum TransparentSequenceTestOp {
    DrawPostRange(usize, usize),
    ResolveBeforeGrab(usize),
    SnapshotForGrab(usize),
    SnapshotForNamedGrab(usize),
    ReuseNamedGrabSnapshot(usize),
    DrawGrab(usize),
    SkipGrabMissingSnapshot(usize),
    FinalResolve,
}

#[cfg(test)]
fn collect_transparent_sequence_test_ops(
    plan: &InstancePlan,
    sample_count: u32,
) -> Vec<TransparentSequenceTestOp> {
    collect_transparent_sequence_test_ops_with_snapshot_result(plan, sample_count, true)
}

#[cfg(test)]
fn collect_transparent_sequence_test_ops_with_snapshot_result(
    plan: &InstancePlan,
    sample_count: u32,
    snapshot_copy_succeeds: bool,
) -> Vec<TransparentSequenceTestOp> {
    collect_transparent_sequence_test_ops_with_modes(
        plan,
        sample_count,
        snapshot_copy_succeeds,
        &[],
    )
}

#[cfg(test)]
fn collect_transparent_sequence_test_ops_with_modes(
    plan: &InstancePlan,
    sample_count: u32,
    snapshot_copy_succeeds: bool,
    grab_modes: &[SceneColorSnapshotMode],
) -> Vec<TransparentSequenceTestOp> {
    let mut ops = Vec::new();
    let mut post_idx = 0usize;
    let mut grab_idx = 0usize;
    let mut pending_post_start = None;
    let mut recorded_any = false;
    let mut named_background_snapshot_ready = false;

    let (transparent_phase, grab_phase) = transparent_sequence_phase_pair();
    let transparent_groups = plan.phase(transparent_phase);
    let grab_groups = plan.phase(grab_phase);

    while post_idx < transparent_groups.len() || grab_idx < grab_groups.len() {
        if next_sequence_entry_is_post(plan, post_idx, grab_idx) {
            if pending_post_start.is_none() {
                pending_post_start = Some(post_idx);
            }
            post_idx += 1;
            continue;
        }

        if let Some(start) = pending_post_start.take() {
            ops.push(TransparentSequenceTestOp::DrawPostRange(start, post_idx));
            recorded_any = true;
        }
        let mode = grab_modes
            .get(grab_idx)
            .copied()
            .unwrap_or(SceneColorSnapshotMode::PerObjectGrab);
        let needs_snapshot_copy = match mode {
            SceneColorSnapshotMode::NamedBackgroundGrab => !named_background_snapshot_ready,
            SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => true,
        };
        if needs_snapshot_copy {
            if sample_count > 1 {
                ops.push(TransparentSequenceTestOp::ResolveBeforeGrab(grab_idx));
            }
            match mode {
                SceneColorSnapshotMode::NamedBackgroundGrab => {
                    ops.push(TransparentSequenceTestOp::SnapshotForNamedGrab(grab_idx));
                }
                SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => {
                    ops.push(TransparentSequenceTestOp::SnapshotForGrab(grab_idx));
                }
            }
            if snapshot_copy_succeeds {
                if mode == SceneColorSnapshotMode::NamedBackgroundGrab {
                    named_background_snapshot_ready = true;
                }
            } else {
                ops.push(TransparentSequenceTestOp::SkipGrabMissingSnapshot(grab_idx));
                grab_idx += 1;
                continue;
            }
        } else {
            ops.push(TransparentSequenceTestOp::ReuseNamedGrabSnapshot(grab_idx));
        }
        ops.push(TransparentSequenceTestOp::DrawGrab(grab_idx));
        recorded_any = true;
        grab_idx += 1;
    }

    if let Some(start) = pending_post_start {
        ops.push(TransparentSequenceTestOp::DrawPostRange(start, post_idx));
        recorded_any = true;
    }
    if recorded_any && sample_count > 1 {
        ops.push(TransparentSequenceTestOp::FinalResolve);
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::{
        TransparentSequenceTestOp, collect_transparent_sequence_test_ops,
        collect_transparent_sequence_test_ops_with_modes,
        collect_transparent_sequence_test_ops_with_snapshot_result,
    };
    use crate::materials::SceneColorSnapshotMode;
    use crate::world_mesh::{DrawGroup, InstancePlan, WorldMeshPhase};

    fn group(representative_draw_idx: usize) -> DrawGroup {
        DrawGroup {
            representative_draw_idx,
            instance_range: 0..1,
            material_packet_idx: 0,
        }
    }

    fn plan_with_transparent_groups(
        non_grab: Vec<DrawGroup>,
        grab: Vec<DrawGroup>,
    ) -> InstancePlan {
        let mut plan = InstancePlan::default();
        plan.phase_mut(WorldMeshPhase::Transparent).extend(non_grab);
        plan.phase_mut(WorldMeshPhase::TransparentGrab).extend(grab);
        plan
    }

    #[test]
    fn non_grab_transparent_groups_stay_in_sorted_runs() {
        let plan = plan_with_transparent_groups(vec![group(2), group(4), group(8)], Vec::new());

        assert_eq!(
            collect_transparent_sequence_test_ops(&plan, 1),
            vec![TransparentSequenceTestOp::DrawPostRange(0, 3)]
        );
    }

    #[test]
    fn grab_groups_trigger_snapshot_immediately_before_draw() {
        let plan = plan_with_transparent_groups(vec![group(1), group(9)], vec![group(5)]);

        assert_eq!(
            collect_transparent_sequence_test_ops(&plan, 1),
            vec![
                TransparentSequenceTestOp::DrawPostRange(0, 1),
                TransparentSequenceTestOp::SnapshotForGrab(0),
                TransparentSequenceTestOp::DrawGrab(0),
                TransparentSequenceTestOp::DrawPostRange(1, 2),
            ]
        );
    }

    #[test]
    fn multiple_grab_groups_take_multiple_snapshots() {
        let plan = plan_with_transparent_groups(Vec::new(), vec![group(3), group(7)]);

        assert_eq!(
            collect_transparent_sequence_test_ops(&plan, 1),
            vec![
                TransparentSequenceTestOp::SnapshotForGrab(0),
                TransparentSequenceTestOp::DrawGrab(0),
                TransparentSequenceTestOp::SnapshotForGrab(1),
                TransparentSequenceTestOp::DrawGrab(1),
            ]
        );
    }

    #[test]
    fn named_grab_groups_reuse_the_first_named_snapshot() {
        let plan = plan_with_transparent_groups(Vec::new(), vec![group(3), group(7)]);

        assert_eq!(
            collect_transparent_sequence_test_ops_with_modes(
                &plan,
                1,
                true,
                &[
                    SceneColorSnapshotMode::NamedBackgroundGrab,
                    SceneColorSnapshotMode::NamedBackgroundGrab,
                ],
            ),
            vec![
                TransparentSequenceTestOp::SnapshotForNamedGrab(0),
                TransparentSequenceTestOp::DrawGrab(0),
                TransparentSequenceTestOp::ReuseNamedGrabSnapshot(1),
                TransparentSequenceTestOp::DrawGrab(1),
            ]
        );
    }

    #[test]
    fn per_object_grabs_still_copy_between_named_grab_reuse() {
        let plan = plan_with_transparent_groups(Vec::new(), vec![group(3), group(5), group(7)]);

        assert_eq!(
            collect_transparent_sequence_test_ops_with_modes(
                &plan,
                1,
                true,
                &[
                    SceneColorSnapshotMode::NamedBackgroundGrab,
                    SceneColorSnapshotMode::PerObjectGrab,
                    SceneColorSnapshotMode::NamedBackgroundGrab,
                ],
            ),
            vec![
                TransparentSequenceTestOp::SnapshotForNamedGrab(0),
                TransparentSequenceTestOp::DrawGrab(0),
                TransparentSequenceTestOp::SnapshotForGrab(1),
                TransparentSequenceTestOp::DrawGrab(1),
                TransparentSequenceTestOp::ReuseNamedGrabSnapshot(2),
                TransparentSequenceTestOp::DrawGrab(2),
            ]
        );
    }

    #[test]
    fn interleaved_named_grabs_reuse_background_after_per_object_copy() {
        let plan = plan_with_transparent_groups(
            vec![group(4), group(9)],
            vec![group(2), group(6), group(11)],
        );

        assert_eq!(
            collect_transparent_sequence_test_ops_with_modes(
                &plan,
                1,
                true,
                &[
                    SceneColorSnapshotMode::NamedBackgroundGrab,
                    SceneColorSnapshotMode::PerObjectGrab,
                    SceneColorSnapshotMode::NamedBackgroundGrab,
                ],
            ),
            vec![
                TransparentSequenceTestOp::SnapshotForNamedGrab(0),
                TransparentSequenceTestOp::DrawGrab(0),
                TransparentSequenceTestOp::DrawPostRange(0, 1),
                TransparentSequenceTestOp::SnapshotForGrab(1),
                TransparentSequenceTestOp::DrawGrab(1),
                TransparentSequenceTestOp::DrawPostRange(1, 2),
                TransparentSequenceTestOp::ReuseNamedGrabSnapshot(2),
                TransparentSequenceTestOp::DrawGrab(2),
            ]
        );
    }

    #[test]
    fn msaa_resolves_before_each_grab_and_after_tail() {
        let plan = plan_with_transparent_groups(vec![group(1)], vec![group(3), group(7)]);

        assert_eq!(
            collect_transparent_sequence_test_ops(&plan, 4),
            vec![
                TransparentSequenceTestOp::DrawPostRange(0, 1),
                TransparentSequenceTestOp::ResolveBeforeGrab(0),
                TransparentSequenceTestOp::SnapshotForGrab(0),
                TransparentSequenceTestOp::DrawGrab(0),
                TransparentSequenceTestOp::ResolveBeforeGrab(1),
                TransparentSequenceTestOp::SnapshotForGrab(1),
                TransparentSequenceTestOp::DrawGrab(1),
                TransparentSequenceTestOp::FinalResolve,
            ]
        );
    }

    #[test]
    fn failed_snapshot_copy_skips_grab_draw() {
        let plan = plan_with_transparent_groups(Vec::new(), vec![group(3)]);

        assert_eq!(
            collect_transparent_sequence_test_ops_with_snapshot_result(&plan, 1, false),
            vec![
                TransparentSequenceTestOp::SnapshotForGrab(0),
                TransparentSequenceTestOp::SkipGrabMissingSnapshot(0),
            ]
        );
    }

    #[test]
    fn post_groups_before_failed_grab_still_count_as_recorded_tail() {
        let plan = plan_with_transparent_groups(vec![group(1)], vec![group(3)]);

        assert_eq!(
            collect_transparent_sequence_test_ops_with_snapshot_result(&plan, 4, false),
            vec![
                TransparentSequenceTestOp::DrawPostRange(0, 1),
                TransparentSequenceTestOp::ResolveBeforeGrab(0),
                TransparentSequenceTestOp::SnapshotForGrab(0),
                TransparentSequenceTestOp::SkipGrabMissingSnapshot(0),
                TransparentSequenceTestOp::FinalResolve,
            ]
        );
    }
}
