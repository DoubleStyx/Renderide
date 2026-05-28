//! Ordered transparent tail with per-grab and reusable named scene-color snapshots.

mod draw;
mod order;
mod record;
mod snapshot;

#[cfg(test)]
mod tests;

use crate::graph_inputs::PerViewFramePlanSlot;
use crate::render_graph::context::EncoderPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{EncoderPass, PassBuilder};
use crate::render_graph::resources::TextureAccess;
use crate::world_mesh::InstancePlan;

use super::{WorldMeshForwardGraphResources, WorldMeshForwardPlanSlot, declare_forward_draw_reads};
use order::transparent_sequence_phase_pair;
use record::record_transparent_sequence;

/// Draws regular transparent groups and grab-pass groups in sorted order.
#[derive(Debug)]
pub struct WorldMeshForwardTransparentSequencePass {
    /// Graph resources shared by the ordered transparent tail.
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

/// Returns whether this pass must record transparent work or the final MSAA color resolve.
fn transparent_sequence_pass_needed(
    opaque_recorded: bool,
    plan: &InstancePlan,
    msaa_enabled: bool,
    sample_count: u32,
) -> bool {
    forward_transparent_sequence_needed(opaque_recorded, plan)
        || final_scene_color_resolve_needed(opaque_recorded, msaa_enabled, sample_count)
}

/// Returns whether the MSAA color attachment must be resolved for scene-color sampling.
fn final_scene_color_resolve_needed(
    opaque_recorded: bool,
    msaa_enabled: bool,
    sample_count: u32,
) -> bool {
    opaque_recorded && msaa_enabled && sample_count > 1
}

/// Declares graph texture and buffer accesses required by the transparent sequence pass.
fn declare_transparent_sequence_accesses(
    b: &mut PassBuilder<'_>,
    resources: WorldMeshForwardGraphResources,
) {
    b.read_texture(resources.scene_color_hdr, TextureAccess::CopySrc);
    b.write_texture(
        resources.scene_color_hdr,
        TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
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
    if let Some(msaa) = resources.msaa {
        b.write_texture(
            msaa.scene_color_hdr,
            TextureAccess::ColorAttachment {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
                resolve_to: None,
            },
        );
        b.read_texture(
            msaa.scene_color_hdr,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        b.write_texture(
            msaa.depth,
            TextureAccess::DepthAttachment {
                depth: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                stencil: None,
            },
        );
    }
    declare_forward_draw_reads(b, resources);
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
                transparent_sequence_pass_needed(
                    prepared.opaque_recorded,
                    &prepared.plan,
                    self.resources.msaa_enabled(),
                    ctx.pass_frame.view.sample_count.max(1),
                )
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
            if ctx.pass_frame.view.sample_count > 1 {
                prepared.depth_freshness.mark_dirty();
            }
        }
        ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }
}
