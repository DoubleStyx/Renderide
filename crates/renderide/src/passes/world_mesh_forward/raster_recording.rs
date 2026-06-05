//! Raster subpass recording helpers for world-mesh forward passes.

use std::sync::Arc;

use super::{MaterialBatchPacket, PreparedWorldMeshForwardFrame};
use crate::camera::ViewId;
use crate::gpu::GpuLimits;
use crate::graph_inputs::{GraphPassFrame, PerViewFramePlanSlot};
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::render_graph::blackboard::Blackboard;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;
use crate::world_mesh::{MeshPassKind, WorldMeshPhase};

use super::encode::{ForwardDrawBatch, NormalDrawBatch, draw_normals_subset, draw_subset};
use super::normal_pass::WorldMeshForwardNormalPipelineCache;

/// Returns stencil load/store ops when the active depth format has a stencil aspect.
pub(in crate::passes::world_mesh_forward) fn stencil_load_ops(
    depth_stencil_format: Option<wgpu::TextureFormat>,
) -> Option<wgpu::Operations<u32>> {
    depth_stencil_format
        .filter(wgpu::TextureFormat::has_stencil_aspect)
        .map(|_| wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        })
}

/// Bind groups shared across opaque and intersection forward subpasses.
struct ForwardPassBindGroups<'a> {
    /// Per-draw storage slab bind group (`@group(2)`).
    per_draw: &'a wgpu::BindGroup,
    /// Per-view frame globals bind group (`@group(0)`).
    frame: &'a Arc<wgpu::BindGroup>,
    /// Fallback material bind group (`@group(1)`) for unresolved embedded materials.
    empty_material: &'a Arc<wgpu::BindGroup>,
}

/// Pipeline and embedded-bind state for one opaque or nontransparent intersection subpass.
struct ForwardPassRasterConfig {
    /// Whether draw calls may use non-zero `first_instance`.
    supports_base_instance: bool,
    /// Overlay view-projection used by the per-draw UI scissor.
    overlay_view_proj: glam::Mat4,
    /// Active viewport extent in pixels.
    viewport_px: (u32, u32),
}

/// Draw state for a render pass that has already been opened.
struct ForwardSubpassDrawRecord<'a, 'c, 'd> {
    /// Device limits used for dynamic storage-buffer offsets.
    gpu_limits: &'a GpuLimits,
    /// Sorted draw list for the current view.
    draws: &'c [WorldMeshDrawItem],
    /// Instance groups for the selected forward subpass.
    groups: &'c [crate::world_mesh::DrawGroup],
    /// Pre-resolved material pipelines and bind groups.
    precomputed: &'c [MaterialBatchPacket],
    /// Mesh pool and skin cache ([`WorldMeshForwardEncodeRefs`]).
    encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
}

fn record_world_mesh_forward_subpass(
    rpass: &mut wgpu::RenderPass<'_>,
    sub: ForwardSubpassDrawRecord<'_, '_, '_>,
    bind_groups: &ForwardPassBindGroups<'_>,
    cfg: &ForwardPassRasterConfig,
) {
    profiling::scope!("world_mesh_forward::record_subpass");
    draw_subset(ForwardDrawBatch {
        rpass,
        groups: sub.groups,
        draws: sub.draws,
        precomputed: sub.precomputed,
        encode: sub.encode,
        gpu_limits: sub.gpu_limits,
        frame_bg: bind_groups.frame.as_ref(),
        empty_bg: bind_groups.empty_material.as_ref(),
        per_draw_bind_group: bind_groups.per_draw,
        supports_base_instance: cfg.supports_base_instance,
        overlay_view_proj: cfg.overlay_view_proj,
        viewport_px: cfg.viewport_px,
    });
}

/// Returns the per-view frame bind group captured before command recording.
pub(in crate::passes::world_mesh_forward) fn frame_bind_group_for_view(
    frame: &GraphPassFrame<'_>,
    blackboard: &Blackboard,
) -> Option<Arc<wgpu::BindGroup>> {
    blackboard
        .get::<PerViewFramePlanSlot>()
        .map(|plan| plan.frame_bind_group.clone())
        .or_else(|| {
            frame
                .shared
                .frame_resources
                .per_view_frame_bind_group_and_buffer(frame.view.view_id)
                .map(|(bind_group, _)| bind_group)
        })
}

/// Records one world-mesh forward subset into a render pass already opened by the graph.
fn record_world_mesh_forward_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &GraphPassFrame<'_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    mesh_pass: MeshPassKind,
) -> bool {
    for &phase in mesh_pass.phases() {
        if !record_world_mesh_forward_phase_graph_raster(rpass, frame, blackboard, prepared, phase)
        {
            return false;
        }
    }
    true
}

/// Records one named world-mesh phase into a render pass already opened by the caller.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_phase_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &GraphPassFrame<'_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    phase: WorldMeshPhase,
) -> bool {
    record_world_mesh_forward_phase_graph_raster_for_view(
        rpass,
        frame,
        blackboard,
        prepared,
        phase,
        frame.view.view_id,
    )
}

/// Records one named world-mesh phase using the supplied per-view resource identity.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_phase_graph_raster_for_view(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &GraphPassFrame<'_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    phase: WorldMeshPhase,
    resource_view_id: ViewId,
) -> bool {
    let groups = prepared.plan.phase(phase);
    #[cfg(feature = "tracy")]
    let debug_label = format!("world_mesh_forward::{phase:?}");
    #[cfg(feature = "tracy")]
    rpass.push_debug_group(debug_label.as_str());
    let recorded = record_world_mesh_forward_groups_graph_raster_for_view(
        rpass,
        frame,
        blackboard,
        prepared,
        groups,
        resource_view_id,
    );
    #[cfg(feature = "tracy")]
    rpass.pop_debug_group();
    recorded
}

/// Records an explicit draw-group slice using the supplied per-view resource identity.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_groups_graph_raster_for_view(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &GraphPassFrame<'_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    groups: &[crate::world_mesh::DrawGroup],
    resource_view_id: ViewId,
) -> bool {
    let Some(frame_bg_arc) = frame_bind_group_for_view(frame, blackboard) else {
        return false;
    };
    record_world_mesh_forward_groups_graph_raster_with_frame_bind_group(
        rpass,
        frame,
        prepared,
        groups,
        &frame_bg_arc,
        resource_view_id,
    )
}

/// Records an explicit draw-group slice with a caller-selected `@group(0)` bind group.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_groups_graph_raster_with_frame_bind_group(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &GraphPassFrame<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    groups: &[crate::world_mesh::DrawGroup],
    frame_bg_arc: &Arc<wgpu::BindGroup>,
    resource_view_id: ViewId,
) -> bool {
    if groups.is_empty() {
        return true;
    }

    let Some(per_draw_bg) = frame
        .shared
        .frame_resources
        .per_view_per_draw_bind_group(resource_view_id)
    else {
        return false;
    };
    let Some(empty_bg_arc) = frame.shared.frame_resources.empty_material_bind_group() else {
        return false;
    };

    let bind_groups = ForwardPassBindGroups {
        per_draw: per_draw_bg.as_ref(),
        frame: frame_bg_arc,
        empty_material: &empty_bg_arc,
    };

    let raster_cfg = ForwardPassRasterConfig {
        supports_base_instance: prepared.supports_base_instance,
        overlay_view_proj: prepared.overlay_view_proj,
        viewport_px: prepared.viewport_px,
    };

    let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
        return false;
    };
    let mut encode_refs = WorldMeshForwardEncodeRefs::from_frame(frame);
    record_world_mesh_forward_subpass(
        rpass,
        ForwardSubpassDrawRecord {
            gpu_limits: gpu_limits.as_ref(),
            draws: &prepared.draws,
            groups,
            precomputed: &prepared.precomputed_batches,
            encode: &mut encode_refs,
        },
        &bind_groups,
        &raster_cfg,
    );
    true
}

/// Records the opaque draw subset into a render pass already opened by the graph.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_opaque_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    _device: &wgpu::Device,
    frame: &GraphPassFrame<'_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    profiling::scope!("world_mesh_forward::record_opaque_graph_raster");
    record_world_mesh_forward_graph_raster(
        rpass,
        frame,
        blackboard,
        prepared,
        MeshPassKind::ForwardOpaque,
    )
}

/// Records the GTAO normal draw subset into a render pass already opened by the graph.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_normal_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    frame: &GraphPassFrame<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    pipelines: &WorldMeshForwardNormalPipelineCache,
) -> bool {
    profiling::scope!("world_mesh_forward::record_normal_graph_raster");
    let groups = prepared.plan.phase(MeshPassKind::ViewNormals.first_phase());
    if groups.is_empty() {
        return true;
    }

    let Some(per_draw_bg) = frame
        .shared
        .frame_resources
        .per_view_per_draw_bind_group(frame.view.view_id)
    else {
        return false;
    };
    let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
        return false;
    };
    let mut encode_refs = WorldMeshForwardEncodeRefs::from_frame(frame);
    #[cfg(feature = "tracy")]
    rpass.push_debug_group("world_mesh_forward::view_normals");
    draw_normals_subset(NormalDrawBatch {
        rpass,
        groups,
        draws: &prepared.draws,
        encode: &mut encode_refs,
        gpu_limits: gpu_limits.as_ref(),
        per_draw_bind_group: per_draw_bg.as_ref(),
        supports_base_instance: prepared.supports_base_instance,
        pipeline: &prepared.pipeline,
        device,
        normal_pipelines: pipelines,
    });
    #[cfg(feature = "tracy")]
    rpass.pop_debug_group();
    true
}

/// Records the nontransparent intersection draw subset into a render pass already opened by the graph.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_intersection_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    _device: &wgpu::Device,
    frame: &GraphPassFrame<'_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    profiling::scope!("world_mesh_forward::record_intersection_graph_raster");
    record_world_mesh_forward_graph_raster(
        rpass,
        frame,
        blackboard,
        prepared,
        MeshPassKind::Intersection,
    )
}
