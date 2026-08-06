//! Raster subpass recording helpers for world-mesh forward passes.

use std::sync::Arc;

use super::{MaterialBatchPacket, PreparedWorldMeshForwardFrame};
use crate::camera::ViewId;
use crate::frame_upload_batch::GraphUploadSink;
use crate::gpu::GpuLimits;
use crate::graph_inputs::PerViewFramePlanSlot;
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::context::PassFrameContext;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;
use crate::world_mesh::{MeshPassKind, WorldMeshPhase};

use super::encode::{
    ForwardDrawBatch, NormalDrawBatch, NormalPrepassIndirectDraw, collect_normal_prepass_indirect,
    draw_normals_subset, draw_subset, issue_normal_prepass_indirect,
};
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
    /// Device for building indirect command buffers.
    device: &'a wgpu::Device,
    /// Deferred upload sink used to update this phase's persistent indirect buffer.
    uploads: GraphUploadSink<'a>,
    /// Shared geometry mega-buffer; `Some` enables indirect static draws for this subpass.
    geometry_arena: Option<&'c crate::gpu_pools::geometry_arena::GeometryArena>,
    /// Persistent command buffer dedicated to this view and render phase.
    indirect_buffer: Option<&'a mut crate::gpu::indirect_buffer::IndirectDrawBuffer>,
    /// Compute-produced retained-static commands for this phase.
    gpu_cull: Option<(
        &'c super::gpu_cull::WorldMeshGpuCullResult,
        &'c [super::gpu_cull::GpuCulledForwardRun],
    )>,
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
        device: sub.device,
        uploads: sub.uploads,
        geometry_arena: sub.geometry_arena,
        indirect_buffer: sub.indirect_buffer,
        gpu_cull: sub.gpu_cull,
    });
}

/// Returns the per-view frame bind group captured before command recording.
pub(in crate::passes::world_mesh_forward) fn frame_bind_group_for_view(
    frame: &PassFrameContext<'_, '_>,
    blackboard: &Blackboard,
) -> Option<Arc<wgpu::BindGroup>> {
    blackboard
        .get::<PerViewFramePlanSlot>()
        .map(|plan| plan.frame_bind_group.clone())
        .or_else(|| {
            frame
                .systems
                .frame_resources
                .per_view_frame_bind_group_and_buffer(frame.view.view_id)
                .map(|(bind_group, _)| bind_group)
        })
}

/// Records one world-mesh forward subset into a render pass already opened by the graph.
fn record_world_mesh_forward_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &PassFrameContext<'_, '_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    mesh_pass: MeshPassKind,
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
) -> bool {
    for &phase in mesh_pass.phases() {
        if !record_world_mesh_forward_phase_graph_raster(
            rpass, frame, blackboard, prepared, phase, device, uploads,
        ) {
            return false;
        }
    }
    true
}

/// Records one named world-mesh phase into a render pass already opened by the caller.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_phase_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &PassFrameContext<'_, '_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    phase: WorldMeshPhase,
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
) -> bool {
    record_world_mesh_forward_phase_graph_raster_for_view(
        rpass,
        frame,
        blackboard,
        prepared,
        phase,
        frame.view.view_id,
        device,
        uploads,
    )
}

/// Records one named world-mesh phase using the supplied per-view resource identity.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_phase_graph_raster_for_view(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &PassFrameContext<'_, '_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    phase: WorldMeshPhase,
    resource_view_id: ViewId,
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
) -> bool {
    let groups = prepared.plan.phase(phase);
    #[cfg(feature = "tracy-gpu")]
    let debug_label = format!("world_mesh_forward::{phase:?}");
    #[cfg(feature = "tracy-gpu")]
    rpass.push_debug_group(debug_label.as_str());
    let recorded = record_world_mesh_forward_groups_graph_raster_for_view(
        rpass,
        frame,
        blackboard,
        prepared,
        groups,
        phase,
        resource_view_id,
        device,
        uploads,
    );
    #[cfg(feature = "tracy-gpu")]
    rpass.pop_debug_group();
    recorded
}

/// Records an explicit draw-group slice using the supplied per-view resource identity.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_groups_graph_raster_for_view(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &PassFrameContext<'_, '_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    groups: &[crate::world_mesh::DrawGroup],
    phase: WorldMeshPhase,
    resource_view_id: ViewId,
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
) -> bool {
    if groups.is_empty() {
        return true;
    }
    let Some(frame_bg_arc) = frame_bind_group_for_view(frame, blackboard) else {
        return false;
    };
    // Static opaque draws go indirect from the shared geometry arena. Read guards let views record
    // concurrently while the graph keeps population writes ordered before raster passes.
    let indirect_enabled = crate::world_mesh::world_mesh_render_path().uses_indirect_draws()
        && prepared.supports_base_instance
        && frame
            .view
            .gpu_limits
            .as_deref()
            .is_some_and(GpuLimits::supports_indirect_first_instance);
    let arena_arc = frame.systems.frame_resources.shared_geometry_arena();
    let arena_guard = arena_arc.as_ref().map(|arena| arena.read());
    let geometry_arena = arena_guard.as_ref().and_then(|guard| guard.as_ref());
    let gpu_cull = prepared.gpu_cull.as_ref().and_then(|result| {
        let runs = result.forward_runs(phase);
        (!runs.is_empty() && indirect_enabled && geometry_arena.is_some()).then_some((result, runs))
    });
    // gpu_hybrid also wants the buffer while gpu cull is running, to batch the groups no gpu run
    // covers. every other path keeps the original either/or.
    let cpu_fills_gaps = crate::world_mesh::world_mesh_render_path().fills_indirect_gaps_on_cpu()
        && gpu_cull.is_some();
    let forward_indirect_arc =
        (indirect_enabled && geometry_arena.is_some() && (gpu_cull.is_none() || cpu_fills_gaps))
            .then(|| frame.systems.frame_resources.forward_indirect())
            .flatten();
    let indirect_buffer_arc = forward_indirect_arc.as_ref().map(|buffers| {
        let mut buffers = buffers.lock();
        Arc::clone(buffers.entry((resource_view_id, phase)).or_insert_with(|| {
            let initial_commands = u32::try_from(groups.len()).unwrap_or(u32::MAX);
            Arc::new(parking_lot::Mutex::new(
                crate::gpu::indirect_buffer::IndirectDrawBuffer::new(device, initial_commands),
            ))
        }))
    });
    let mut indirect_buffer_guard = indirect_buffer_arc.as_ref().map(|buffer| buffer.lock());
    let indirect_buffer = indirect_buffer_guard.as_mut().map(|buffer| &mut **buffer);
    record_world_mesh_forward_groups_graph_raster_with_frame_bind_group(
        rpass,
        frame,
        prepared,
        groups,
        &frame_bg_arc,
        resource_view_id,
        device,
        uploads,
        geometry_arena,
        indirect_buffer,
        gpu_cull,
    )
}

/// Records an explicit draw-group slice with a caller-selected `@group(0)` bind group.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_groups_graph_raster_with_frame_bind_group(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &PassFrameContext<'_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    groups: &[crate::world_mesh::DrawGroup],
    frame_bg_arc: &Arc<wgpu::BindGroup>,
    resource_view_id: ViewId,
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
    geometry_arena: Option<&crate::gpu_pools::geometry_arena::GeometryArena>,
    indirect_buffer: Option<&mut crate::gpu::indirect_buffer::IndirectDrawBuffer>,
    gpu_cull: Option<(
        &super::gpu_cull::WorldMeshGpuCullResult,
        &[super::gpu_cull::GpuCulledForwardRun],
    )>,
) -> bool {
    if groups.is_empty() {
        return true;
    }

    let Some(per_draw_bg) = frame
        .systems
        .frame_resources
        .per_view_per_draw_bind_group(resource_view_id)
    else {
        return false;
    };
    let Some(empty_bg_arc) = frame.systems.frame_resources.empty_material_bind_group() else {
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
    let mut encode_refs = WorldMeshForwardEncodeRefs::from_pass_frame(frame);
    record_world_mesh_forward_subpass(
        rpass,
        ForwardSubpassDrawRecord {
            gpu_limits: gpu_limits.as_ref(),
            draws: &prepared.draws,
            groups,
            precomputed: &prepared.precomputed_batches,
            encode: &mut encode_refs,
            device,
            uploads,
            geometry_arena,
            indirect_buffer,
            gpu_cull,
        },
        &bind_groups,
        &raster_cfg,
    );
    true
}

/// Records the opaque draw subset into a render pass already opened by the graph.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_opaque_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    frame: &PassFrameContext<'_, '_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    uploads: GraphUploadSink<'_>,
) -> bool {
    profiling::scope!("world_mesh_forward::record_opaque_graph_raster");
    record_world_mesh_forward_graph_raster(
        rpass,
        frame,
        blackboard,
        prepared,
        MeshPassKind::ForwardOpaque,
        device,
        uploads,
    )
}

/// Records the GTAO normal draw subset into a render pass already opened by the graph.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_normal_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
    frame: &PassFrameContext<'_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    pipelines: &WorldMeshForwardNormalPipelineCache,
) -> bool {
    profiling::scope!("world_mesh_forward::record_normal_graph_raster");
    let groups = prepared.plan.phase(MeshPassKind::ViewNormals.first_phase());
    if groups.is_empty() {
        return true;
    }

    let Some(per_draw_bg) = frame
        .systems
        .frame_resources
        .per_view_per_draw_bind_group(frame.view.view_id)
    else {
        return false;
    };
    let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
        return false;
    };

    let indirect_enabled = crate::world_mesh::world_mesh_render_path().uses_indirect_draws()
        && prepared.supports_base_instance
        && gpu_limits.supports_indirect_first_instance();
    let arena_arc = frame.systems.frame_resources.shared_geometry_arena();
    let arena_guard = arena_arc.as_ref().map(|arena| arena.read());
    let arena = arena_guard.as_ref().and_then(|guard| guard.as_ref());
    let indirect_arena = indirect_enabled.then_some(arena).flatten();
    let gpu_normals = prepared
        .gpu_cull
        .as_ref()
        .filter(|result| !result.normal_runs.is_empty() && indirect_arena.is_some());

    let mut commands = Vec::new();
    let runs = indirect_arena
        .filter(|_| gpu_normals.is_none())
        .map(|arena| {
            collect_normal_prepass_indirect(
                groups,
                &prepared.draws,
                arena,
                &prepared.pipeline,
                &mut commands,
            )
        })
        .unwrap_or_default();

    let mut encode_refs = WorldMeshForwardEncodeRefs::from_pass_frame(frame);
    #[cfg(feature = "tracy-gpu")]
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
        geometry_arena: arena,
    });

    if let (Some(arena), Some(result)) = (indirect_arena, gpu_normals) {
        result.issue_normal_runs(rpass, device, per_draw_bg.as_ref(), arena, pipelines);
    } else if let Some(arena) = indirect_arena
        && !commands.is_empty()
        && let Some(indirect_arc) = frame.systems.frame_resources.normal_prepass_indirect()
    {
        let count = u32::try_from(commands.len()).unwrap_or(u32::MAX);
        let buffer_arc = {
            let mut buffers = indirect_arc.lock();
            Arc::clone(buffers.entry(frame.view.view_id).or_insert_with(|| {
                Arc::new(parking_lot::Mutex::new(
                    crate::gpu::indirect_buffer::IndirectDrawBuffer::new(device, count),
                ))
            }))
        };
        let mut buffer = buffer_arc.lock();
        buffer.prepare_len(device, count);
        uploads.write_buffer(buffer.buffer(), 0, bytemuck::cast_slice(&commands));
        issue_normal_prepass_indirect(NormalPrepassIndirectDraw {
            rpass,
            device,
            per_draw_bind_group: per_draw_bg.as_ref(),
            arena,
            commands: &buffer,
            runs: &runs,
            normal_pipelines: pipelines,
        });
    }
    #[cfg(feature = "tracy-gpu")]
    rpass.pop_debug_group();
    true
}

/// Records the nontransparent intersection draw subset into a render pass already opened by the graph.
pub(in crate::passes::world_mesh_forward) fn record_world_mesh_forward_intersection_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    frame: &PassFrameContext<'_, '_>,
    blackboard: &Blackboard,
    prepared: &PreparedWorldMeshForwardFrame,
    uploads: GraphUploadSink<'_>,
) -> bool {
    profiling::scope!("world_mesh_forward::record_intersection_graph_raster");
    record_world_mesh_forward_graph_raster(
        rpass,
        frame,
        blackboard,
        prepared,
        MeshPassKind::Intersection,
        device,
        uploads,
    )
}
