//! Ordered transparent tail with per-grab scene-color snapshots.

use crate::render_graph::context::EncoderPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::stereo_mask_or_template;
use crate::render_graph::pass::{EncoderPass, PassBuilder};
use crate::render_graph::resources::{TextureAccess, TextureResourceHandle};
use crate::world_mesh::{DrawGroup, InstancePlan};

use super::color_resolve::{
    WorldMeshForwardColorResolveEncodeContext, WorldMeshForwardColorResolveGraphResources,
    encode_world_mesh_forward_msaa_color_resolve,
};
use super::color_snapshot::encode_world_mesh_forward_color_snapshot;
use super::raster_recording::{record_world_mesh_forward_groups_graph_raster, stencil_load_ops};
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
    opaque_recorded && (!plan.post_skybox_groups.is_empty() || !plan.transparent_groups.is_empty())
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
    let Some(post) = plan.post_skybox_groups.get(post_idx) else {
        return false;
    };
    let Some(grab) = plan.transparent_groups.get(grab_idx) else {
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
        record_world_mesh_forward_groups_graph_raster(
            &mut rpass,
            frame,
            &*ctx.blackboard,
            prepared,
            groups,
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
) -> Result<bool, RenderPassError> {
    let Some(start) = start else {
        return Ok(true);
    };
    draw_tail_groups(
        ctx,
        prepared,
        resources,
        &prepared.plan.post_skybox_groups[start..end],
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
) {
    encode_world_mesh_forward_color_snapshot(
        ctx.graph_resources,
        ctx.encoder,
        ctx.pass_frame,
        prepared,
        resources,
    );
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
    let mut post_idx = 0usize;
    let mut grab_idx = 0usize;
    let mut pending_post_start = None;
    let mut recorded_any = false;

    while post_idx < plan.post_skybox_groups.len() || grab_idx < plan.transparent_groups.len() {
        if next_sequence_entry_is_post(plan, post_idx, grab_idx) {
            if pending_post_start.is_none() {
                pending_post_start = Some(post_idx);
            }
            post_idx += 1;
            continue;
        }

        if !flush_post_groups(
            ctx,
            prepared,
            resources,
            pending_post_start.take(),
            post_idx,
        )? {
            return Ok(false);
        }
        resolve_for_grab_snapshot(ctx, resources)?;
        copy_grab_snapshot(ctx, prepared, resources);
        if !draw_tail_groups(
            ctx,
            prepared,
            resources,
            std::slice::from_ref(&plan.transparent_groups[grab_idx]),
        )? {
            return Ok(false);
        }
        recorded_any = true;
        grab_idx += 1;
    }

    if !flush_post_groups(ctx, prepared, resources, pending_post_start, post_idx)? {
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
    DrawGrab(usize),
    FinalResolve,
}

#[cfg(test)]
fn collect_transparent_sequence_test_ops(
    plan: &InstancePlan,
    sample_count: u32,
) -> Vec<TransparentSequenceTestOp> {
    let mut ops = Vec::new();
    let mut post_idx = 0usize;
    let mut grab_idx = 0usize;
    let mut pending_post_start = None;
    let mut recorded_any = false;

    while post_idx < plan.post_skybox_groups.len() || grab_idx < plan.transparent_groups.len() {
        if next_sequence_entry_is_post(plan, post_idx, grab_idx) {
            if pending_post_start.is_none() {
                pending_post_start = Some(post_idx);
            }
            post_idx += 1;
            continue;
        }

        if let Some(start) = pending_post_start.take() {
            ops.push(TransparentSequenceTestOp::DrawPostRange(start, post_idx));
        }
        if sample_count > 1 {
            ops.push(TransparentSequenceTestOp::ResolveBeforeGrab(grab_idx));
        }
        ops.push(TransparentSequenceTestOp::SnapshotForGrab(grab_idx));
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
    use super::{TransparentSequenceTestOp, collect_transparent_sequence_test_ops};
    use crate::world_mesh::{DrawGroup, InstancePlan};

    fn group(representative_draw_idx: usize) -> DrawGroup {
        DrawGroup {
            representative_draw_idx,
            instance_range: 0..1,
            material_packet_idx: 0,
        }
    }

    #[test]
    fn non_grab_transparent_groups_stay_in_sorted_runs() {
        let plan = InstancePlan {
            post_skybox_groups: vec![group(2), group(4), group(8)],
            ..Default::default()
        };

        assert_eq!(
            collect_transparent_sequence_test_ops(&plan, 1),
            vec![TransparentSequenceTestOp::DrawPostRange(0, 3)]
        );
    }

    #[test]
    fn grab_groups_trigger_snapshot_immediately_before_draw() {
        let plan = InstancePlan {
            post_skybox_groups: vec![group(1), group(9)],
            transparent_groups: vec![group(5)],
            ..Default::default()
        };

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
        let plan = InstancePlan {
            transparent_groups: vec![group(3), group(7)],
            ..Default::default()
        };

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
    fn msaa_resolves_before_each_grab_and_after_tail() {
        let plan = InstancePlan {
            post_skybox_groups: vec![group(1)],
            transparent_groups: vec![group(3), group(7)],
            ..Default::default()
        };

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
}
