//! Raster subpass recording helpers for world-mesh forward passes.

use std::sync::Arc;

use super::super::{MaterialBatchPacket, PreparedWorldMeshForwardFrame};
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::gpu::GpuLimits;
use crate::render_graph::frame_params::FrameRenderParams;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

use super::super::encode::{ForwardDrawBatch, draw_subset};

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

/// Pipeline and embedded-bind state for one opaque or intersection subpass.
struct ForwardPassRasterConfig {
    /// Whether draw calls may use non-zero `first_instance`.
    supports_base_instance: bool,
}

/// Draw state for a render pass that has already been opened.
struct ForwardSubpassDrawRecord<'a, 'c, 'd> {
    /// Device limits used for dynamic storage-buffer offsets.
    gpu_limits: &'a GpuLimits,
    /// Sorted draw list for the current view.
    draws: &'c [WorldMeshDrawItem],
    /// Instance groups for the selected forward subpass.
    groups: &'c [crate::world_mesh::draw_prep::DrawGroup],
    /// Pre-resolved material pipelines and bind groups.
    precomputed: &'c [MaterialBatchPacket],
    /// Mesh pool and skin cache ([`WorldMeshForwardEncodeRefs`]).
    encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
}

/// World-mesh forward subpass selection.
#[derive(Clone, Copy, Debug)]
enum ForwardSubpassKind {
    /// Opaque and alpha-cutout draws before scene-depth snapshotting.
    Opaque,
    /// Intersection material draws after the depth snapshot.
    Intersection,
    /// Transparent and grab-pass tail draws.
    Transparent,
}

impl ForwardSubpassKind {
    /// Returns the pre-built draw groups for this subpass.
    fn groups(
        self,
        plan: &crate::world_mesh::draw_prep::InstancePlan,
    ) -> &[crate::world_mesh::draw_prep::DrawGroup] {
        match self {
            Self::Opaque => &plan.regular_groups,
            Self::Intersection => &plan.intersect_groups,
            Self::Transparent => &plan.transparent_groups,
        }
    }
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
    });
}

/// Records one world-mesh forward subset into a render pass already opened by the graph.
fn record_world_mesh_forward_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    subpass: ForwardSubpassKind,
) -> bool {
    let groups = subpass.groups(&prepared.plan);
    if groups.is_empty() {
        return true;
    }

    let Some(per_draw_bg) = frame
        .shared
        .frame_resources
        .per_view_per_draw(frame.view.view_id)
        .map(|d| d.lock().bind_group.clone())
    else {
        return false;
    };
    let Some(frame_bg_arc) = frame
        .shared
        .frame_resources
        .per_view_frame(frame.view.view_id)
        .map(|s| s.frame_bind_group.clone())
    else {
        return false;
    };
    let Some(empty_bg_arc) = frame
        .shared
        .frame_resources
        .empty_material()
        .map(|e| e.bind_group.clone())
    else {
        return false;
    };

    let bind_groups = ForwardPassBindGroups {
        per_draw: per_draw_bg.as_ref(),
        frame: &frame_bg_arc,
        empty_material: &empty_bg_arc,
    };

    let raster_cfg = ForwardPassRasterConfig {
        supports_base_instance: prepared.supports_base_instance,
    };

    let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
        return false;
    };
    let mut encode_refs = frame.world_mesh_forward_encode_refs();
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
    _queue: &wgpu::Queue,
    frame: &FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    record_world_mesh_forward_graph_raster(rpass, frame, prepared, ForwardSubpassKind::Opaque)
}

/// Records the intersection draw subset into a render pass already opened by the graph.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_intersection_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    frame: &FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    record_world_mesh_forward_graph_raster(rpass, frame, prepared, ForwardSubpassKind::Intersection)
}

/// Records the grab-pass transparent draw subset into a render pass already opened by the graph.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_transparent_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    frame: &FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    record_world_mesh_forward_graph_raster(rpass, frame, prepared, ForwardSubpassKind::Transparent)
}
