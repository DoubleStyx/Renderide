//! Encode indexed draws and material bind groups for graph-managed world-mesh forward passes.

use crate::assets::mesh::GpuMesh;
use crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::backend::mesh_deform::{GpuSkinCache, SkinCacheKey};
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::gpu::GpuLimits;
use crate::materials::MaterialPipelineSet;
use crate::render_graph::frame_params::MaterialBatchPacket;
use crate::render_graph::world_mesh_draw_prep::DrawGroup;
use crate::render_graph::WorldMeshDrawItem;
use crate::resources::MeshPool;

/// Embedded material vertex stream requirements for one draw (matches pipeline reflection flags).
#[derive(Clone, Copy)]
pub(crate) struct EmbeddedVertexStreamFlags {
    /// UV0 stream at `@location(2)`.
    pub embedded_uv: bool,
    /// Vertex color at `@location(3)`.
    pub embedded_color: bool,
    /// Extended streams (tangents, extra UVs) at `@location(4)` and above.
    pub embedded_extended_vertex_streams: bool,
}

/// GPU mesh pool and optional skin cache for [`draw_mesh_submesh_instanced`].
#[derive(Clone, Copy)]
pub(crate) struct WorldMeshDrawGpuRefs<'a> {
    /// Resident meshes and vertex buffers.
    pub mesh_pool: &'a MeshPool,
    /// Skin/deform cache when the draw uses deformed or blendshape streams.
    pub skin_cache: Option<&'a GpuSkinCache>,
}

/// Compact identity for a [`wgpu::Buffer`] sub-range used to skip redundant vertex / index binds.
///
/// `byte_len == None` encodes a full-buffer `.slice(..)` bind; `Some(n)` is a ranged bind
/// of `byte_offset..byte_offset + n`. Two `BufferBindId`s are equal when they refer to the
/// same buffer object, offset, and length — a sufficient condition for the bind to be a no-op.
///
/// Buffer identity is a raw pointer cast to `usize`; the pointer is stable for the lifetime
/// of the mesh pool / skin cache (both outlive any single render pass).
#[derive(Clone, Copy, PartialEq, Eq)]
struct BufferBindId {
    /// Stable buffer object identity for this render pass.
    ptr: usize,
    /// Byte offset for ranged binds, or zero for full-buffer binds.
    byte_offset: u64,
    /// Byte length for ranged binds, or [`None`] for full-buffer binds.
    byte_len: Option<u64>,
}

impl BufferBindId {
    /// Full-buffer bind (`buf.slice(..)`).
    fn full(buf: &wgpu::Buffer) -> Self {
        Self {
            ptr: core::ptr::from_ref(buf).addr(),
            byte_offset: 0,
            byte_len: None,
        }
    }

    /// Ranged bind (`buf.slice(byte_start..byte_end)`).
    fn ranged(buf: &wgpu::Buffer, byte_start: u64, byte_end: u64) -> Self {
        Self {
            ptr: core::ptr::from_ref(buf).addr(),
            byte_offset: byte_start,
            byte_len: Some(byte_end - byte_start),
        }
    }
}

/// Per-render-pass last-bound vertex and index buffer state for bind deduplication.
///
/// Tracks the last-submitted buffer identity for each of the 8 vertex slots and the index
/// buffer. Reset at every new render pass (i.e. at the start of [`draw_subset`]).
pub(crate) struct LastMeshBindState {
    /// Last bound buffer identity per vertex slot 0–7; `None` = never bound this pass.
    vertex: [Option<BufferBindId>; 8],
    /// Last bound index buffer (pointer-as-usize identity) and format; `None` = never bound.
    index: Option<(usize, wgpu::IndexFormat)>,
}

impl LastMeshBindState {
    /// Builds empty bind-state tracking for a fresh render pass.
    fn new() -> Self {
        Self {
            vertex: [None; 8],
            index: None,
        }
    }
}

/// Pre-grouped draws, bind groups, and precomputed-batch table for one mesh-forward raster subpass.
///
/// Pipelines and `@group(1)` bind groups are pre-resolved in
/// [`crate::render_graph::passes::world_mesh_forward::execute_helpers::precompute_material_resolve_batches`]
/// during the prepare pass, so this struct carries no material-system references and makes no
/// LRU cache lookups during recording.
pub(crate) struct ForwardDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Pre-built [`DrawGroup`]s for this subpass (opaque or intersect), in ascending
    /// `representative_draw_idx` order so the `precomputed` cursor stays monotonic.
    pub groups: &'c [DrawGroup],
    /// Full sorted world mesh draw list for the view (read by representative index).
    pub draws: &'c [WorldMeshDrawItem],
    /// Pre-resolved pipelines and bind groups; one entry per unique batch-key run in `draws`.
    pub precomputed: &'c [MaterialBatchPacket],
    /// Mesh pool and skin cache for vertex/index binding.
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Device limits snapshot (storage-offset alignment for `@group(2)`).
    pub gpu_limits: &'a GpuLimits,
    /// Frame globals at `@group(0)`.
    pub frame_bg: &'a wgpu::BindGroup,
    /// Fallback material bind group when a batch has no resolved `@group(1)`.
    pub empty_bg: &'a wgpu::BindGroup,
    /// Per-draw storage slab at `@group(2)` (dynamic offset; see [`Self::supports_base_instance`]).
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Whether `draw_indexed` may use non-zero `first_instance` / base instance. When
    /// false, every group carries `instance_range.len() == 1` and the per-draw slab is
    /// addressed via dynamic offset instead.
    pub supports_base_instance: bool,
}

/// Records one raster subpass by walking pre-built [`DrawGroup`]s.
///
/// Each group is one `draw_indexed` covering a contiguous slab range of identical instances.
/// The `precomputed` cursor advances on each group's `representative_draw_idx`, which is
/// monotonically increasing across the group list — O(1) amortised. Pipelines and `@group(1)`
/// bind groups are bound directly from the table; no cache lookups occur during recording.
pub(crate) fn draw_subset(batch: ForwardDrawBatch<'_, '_, '_, '_>) {
    profiling::scope!("world_mesh::draw_subset");
    let ForwardDrawBatch {
        rpass,
        groups,
        draws,
        precomputed,
        encode,
        gpu_limits,
        frame_bg,
        empty_bg,
        per_draw_bind_group,
        supports_base_instance,
    } = batch;

    let subpass_batch_count = groups.len();
    let subpass_input_draws: usize = groups
        .iter()
        .map(|g| (g.instance_range.end - g.instance_range.start) as usize)
        .sum();

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_stencil_ref: Option<u32> = None;
    // Cursor into `precomputed`; advances monotonically as group representatives increase.
    let mut batch_cursor: usize = 0;
    // Track which precomputed batch is currently bound to avoid redundant set_bind_group(1).
    let mut bound_batch_cursor: Option<usize> = None;
    // Track the last pipeline pointer to skip redundant set_pipeline across groups that share
    // the same pipeline (common when one precomputed batch covers many groups, or when
    // adjacent batches resolve to the same multi-pass pipeline set).
    let mut last_pipeline: Option<*const wgpu::RenderPipeline> = None;

    rpass.set_bind_group(0, frame_bg, &[]);

    for group in groups {
        let representative = group.representative_draw_idx;

        // Advance the cursor to the precomputed batch that covers `representative`.
        while batch_cursor + 1 < precomputed.len()
            && precomputed[batch_cursor].last_draw_idx < representative
        {
            batch_cursor += 1;
        }

        let pc = &precomputed[batch_cursor];
        debug_assert!(
            representative >= pc.first_draw_idx && representative <= pc.last_draw_idx,
            "precomputed batch [{}, {}] should cover representative draw index {}",
            pc.first_draw_idx,
            pc.last_draw_idx,
            representative,
        );
        debug_assert_eq!(
            pc.pipeline_key.shader_asset_id, draws[representative].batch_key.shader_asset_id,
            "material packet pipeline key must match the representative draw"
        );

        let Some(pipelines) = pc.pipelines.as_ref() else {
            continue; // pipeline unavailable for this batch — skip draws
        };

        // Bind @group(1) once per unique batch; skip when the cursor hasn't advanced.
        if bound_batch_cursor != Some(batch_cursor) {
            let material_bg = pc.bind_group.as_deref().unwrap_or(empty_bg);
            rpass.set_bind_group(1, material_bg, &[]);
            bound_batch_cursor = Some(batch_cursor);
        }

        let slab_first_instance = group.instance_range.start as usize;
        let instance_count = group.instance_range.end - group.instance_range.start;
        bind_per_draw_slab_if_changed(
            rpass,
            per_draw_bind_group,
            gpu_limits,
            slab_first_instance,
            instance_count,
            supports_base_instance,
            &mut last_per_draw_dyn_offset,
        );

        let stencil_ref = draws[representative]
            .batch_key
            .render_state
            .stencil_reference();
        if last_stencil_ref != Some(stencil_ref) {
            rpass.set_stencil_reference(stencil_ref);
            last_stencil_ref = Some(stencil_ref);
        }

        let inst_range = instance_range_for_draw_group(group, supports_base_instance);

        issue_material_pipeline_passes(
            rpass,
            encode,
            &draws[representative],
            ActivePipelineSelection { pipelines },
            &inst_range,
            &mut last_mesh,
            &mut last_pipeline,
        );
    }

    crate::profiling::plot_world_mesh_subpass(subpass_batch_count, subpass_input_draws);
}

/// Updates @group(2) dynamic offset and rebinds the per-draw slab when the row offset changes.
///
/// `slab_first_instance` is the slab-coordinate start of the current group's
/// `instance_range`. On base-instance-capable devices the dynamic offset is always zero
/// (rows are addressed via `first_instance`), so the rebind occurs once at most. On
/// downlevel paths each group carries `instance_count == 1` and the slab row is selected
/// via the dynamic offset.
fn bind_per_draw_slab_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    per_draw_bind_group: &wgpu::BindGroup,
    gpu_limits: &crate::gpu::GpuLimits,
    slab_first_instance: usize,
    instance_count: u32,
    supports_base_instance: bool,
    last_per_draw_dyn_offset: &mut Option<u32>,
) {
    let storage_align = gpu_limits.min_storage_buffer_offset_alignment();
    let per_draw_dyn_offset = if supports_base_instance {
        // Base-instance path: all rows accessed via `first_instance`; dynamic offset is
        // always zero for the entire pass so the bind is skipped after the first draw.
        0u32
    } else {
        // Downlevel: `first_instance` is always zero; select the draw row via dynamic offset.
        debug_assert_eq!(instance_count, 1);
        let raw = (slab_first_instance * PER_DRAW_UNIFORM_STRIDE) as u32;
        debug_assert_eq!(
            raw % storage_align,
            0,
            "per-draw offset must match min_storage_buffer_offset_alignment"
        );
        raw
    };
    if *last_per_draw_dyn_offset != Some(per_draw_dyn_offset) {
        rpass.set_bind_group(2, per_draw_bind_group, &[per_draw_dyn_offset]);
        *last_per_draw_dyn_offset = Some(per_draw_dyn_offset);
    }
}

/// Per-batch pipeline selection for [`issue_material_pipeline_passes`].
struct ActivePipelineSelection<'a> {
    /// Per-material pipeline objects in pass order.
    pipelines: &'a MaterialPipelineSet,
}

/// Walks the pipeline set for `item` and issues one [`draw_mesh_submesh_instanced`] per pipeline.
///
/// `last_pipeline` is updated and consulted across batches so that adjacent draws sharing a
/// pipeline (the typical case within a precomputed batch) skip the redundant `set_pipeline`.
fn issue_material_pipeline_passes(
    rpass: &mut wgpu::RenderPass<'_>,
    encode: &mut crate::backend::WorldMeshForwardEncodeRefs<'_>,
    item: &WorldMeshDrawItem,
    pipeline_sel: ActivePipelineSelection<'_>,
    inst_range: &std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
    last_pipeline: &mut Option<*const wgpu::RenderPipeline>,
) {
    let skin_cache = encode.skin_cache;
    for pipeline in pipeline_sel.pipelines.iter() {
        let pipeline_id: *const wgpu::RenderPipeline = pipeline;
        if *last_pipeline != Some(pipeline_id) {
            rpass.set_pipeline(pipeline);
            *last_pipeline = Some(pipeline_id);
        }
        draw_mesh_submesh_instanced(
            rpass,
            item,
            WorldMeshDrawGpuRefs {
                mesh_pool: encode.mesh_pool(),
                skin_cache,
            },
            EmbeddedVertexStreamFlags {
                embedded_uv: item.batch_key.embedded_needs_uv0,
                embedded_color: item.batch_key.embedded_needs_color,
                embedded_extended_vertex_streams: item
                    .batch_key
                    .embedded_needs_extended_vertex_streams,
            },
            inst_range.clone(),
            last_mesh,
        );
    }
}

/// Resolves the `instance_range` argument to `draw_indexed` for one [`DrawGroup`].
///
/// On base-instance-capable devices, the group's slab range is passed verbatim — the GPU
/// `instance_index` walks `instance_range.start..instance_range.end`, addressing the
/// per-draw slab directly. On downlevel devices, every group has `instance_range.len() == 1`
/// (forced by `build_instance_plan`'s `supports_base_instance = false` gate), and the slab
/// row is reached via the dynamic offset, so the draw range collapses to `0..1`.
fn instance_range_for_draw_group(
    group: &DrawGroup,
    supports_base_instance: bool,
) -> std::ops::Range<u32> {
    if supports_base_instance {
        group.instance_range.clone()
    } else {
        debug_assert_eq!(
            group.instance_range.end - group.instance_range.start,
            1,
            "downlevel groups must be singletons"
        );
        0..1
    }
}

/// Binds one vertex slot only when the buffer identity or range has changed since the last bind.
///
/// Using `global_id()` rather than pointer equality is safe because wgpu `Buffer`s are
/// refcounted and their IDs are stable for the lifetime of the object.
macro_rules! bind_vertex_if_changed {
    ($rpass:expr, $slot:expr, $buf:expr, $id:expr, $last:expr) => {{
        let slot: usize = $slot;
        if $last[slot] != Some($id) {
            $rpass.set_vertex_buffer(slot as u32, $buf);
            $last[slot] = Some($id);
        }
    }};
}

/// Binds mesh streams and issues one indexed draw for `item` over `instances`.
pub(crate) fn draw_mesh_submesh_instanced(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    streams: EmbeddedVertexStreamFlags,
    instances: std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
) {
    let Some(mesh) = resident_draw_mesh(item, gpu, streams) else {
        return;
    };
    let Some(normals_bind) = mesh.normals_buffer.as_deref() else {
        return;
    };

    if !bind_primary_vertex_streams(rpass, item, gpu, mesh, normals_bind, last_mesh) {
        return;
    }
    if !bind_optional_vertex_streams(rpass, mesh, streams, last_mesh) {
        return;
    }

    bind_index_buffer_if_changed(rpass, mesh, last_mesh);

    let first = item.first_index;
    let end = first.saturating_add(item.index_count);
    rpass.draw_indexed(first..end, 0, instances);
}

/// Returns the resident mesh for a drawable item after validating required stream readiness.
fn resident_draw_mesh<'a>(
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'a>,
    streams: EmbeddedVertexStreamFlags,
) -> Option<&'a GpuMesh> {
    if item.mesh_asset_id < 0 || item.node_id < 0 || item.index_count == 0 {
        return None;
    }
    let mesh = gpu.mesh_pool.get_mesh(item.mesh_asset_id)?;
    if streams.embedded_extended_vertex_streams && !mesh.extended_vertex_streams_ready() {
        logger::trace!(
            "WorldMeshForward: extended vertex streams missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return None;
    }
    mesh.debug_streams_ready().then_some(mesh)
}

/// Binds position and normal streams, choosing static mesh buffers or the deformation cache.
fn bind_primary_vertex_streams(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    mesh: &GpuMesh,
    normals_bind: &wgpu::Buffer,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if item.world_space_deformed || item.blendshape_deformed {
        bind_deformed_primary_streams(rpass, item, gpu, normals_bind, last_mesh)
    } else {
        bind_static_primary_streams(rpass, mesh, normals_bind, last_mesh)
    }
}

/// Binds static mesh position and normal streams.
fn bind_static_primary_streams(
    rpass: &mut wgpu::RenderPass<'_>,
    mesh: &GpuMesh,
    normals_bind: &wgpu::Buffer,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    let Some(pos) = mesh.positions_buffer.as_deref() else {
        return false;
    };
    bind_vertex_if_changed!(
        rpass,
        0,
        pos.slice(..),
        BufferBindId::full(pos),
        last_mesh.vertex
    );
    bind_vertex_if_changed!(
        rpass,
        1,
        normals_bind.slice(..),
        BufferBindId::full(normals_bind),
        last_mesh.vertex
    );
    true
}

/// Binds deformation-cache position and normal streams.
fn bind_deformed_primary_streams(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    normals_bind: &wgpu::Buffer,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    let Some(cache) = gpu.skin_cache else {
        return false;
    };
    let key = SkinCacheKey::from_draw_parts(item.space_id, item.skinned, item.instance_id);
    let Some(entry) = cache.lookup_current(&key) else {
        logger::trace!(
            "world mesh forward: current skin cache miss for space {:?} renderable {} instance {:?} node {}",
            item.space_id,
            item.renderable_index,
            item.instance_id,
            item.node_id
        );
        return false;
    };
    let pos_buf = cache.positions_arena();
    let pos_range = entry.positions.byte_range();
    bind_vertex_if_changed!(
        rpass,
        0,
        pos_buf.slice(pos_range.start..pos_range.end),
        BufferBindId::ranged(pos_buf, pos_range.start, pos_range.end),
        last_mesh.vertex
    );
    if item.world_space_deformed {
        let Some(nrm_r) = entry.normals.as_ref() else {
            return false;
        };
        let nrm_buf = cache.normals_arena();
        let nrm_range = nrm_r.byte_range();
        bind_vertex_if_changed!(
            rpass,
            1,
            nrm_buf.slice(nrm_range.start..nrm_range.end),
            BufferBindId::ranged(nrm_buf, nrm_range.start, nrm_range.end),
            last_mesh.vertex
        );
    } else {
        bind_vertex_if_changed!(
            rpass,
            1,
            normals_bind.slice(..),
            BufferBindId::full(normals_bind),
            last_mesh.vertex
        );
    }
    true
}

/// Binds UV, color, tangent, and extra UV streams required by the material reflection.
fn bind_optional_vertex_streams(
    rpass: &mut wgpu::RenderPass<'_>,
    mesh: &GpuMesh,
    streams: EmbeddedVertexStreamFlags,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if streams.embedded_uv || streams.embedded_color || streams.embedded_extended_vertex_streams {
        let Some(uv) = mesh.uv0_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            2,
            uv.slice(..),
            BufferBindId::full(uv),
            last_mesh.vertex
        );
    }
    if streams.embedded_color || streams.embedded_extended_vertex_streams {
        let Some(color) = mesh.color_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            3,
            color.slice(..),
            BufferBindId::full(color),
            last_mesh.vertex
        );
    }
    if streams.embedded_extended_vertex_streams {
        let (Some(tangent), Some(uv1), Some(uv2), Some(uv3)) = (
            mesh.tangent_buffer.as_deref(),
            mesh.uv1_buffer.as_deref(),
            mesh.uv2_buffer.as_deref(),
            mesh.uv3_buffer.as_deref(),
        ) else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            4,
            tangent.slice(..),
            BufferBindId::full(tangent),
            last_mesh.vertex
        );
        bind_vertex_if_changed!(
            rpass,
            5,
            uv1.slice(..),
            BufferBindId::full(uv1),
            last_mesh.vertex
        );
        bind_vertex_if_changed!(
            rpass,
            6,
            uv2.slice(..),
            BufferBindId::full(uv2),
            last_mesh.vertex
        );
        bind_vertex_if_changed!(
            rpass,
            7,
            uv3.slice(..),
            BufferBindId::full(uv3),
            last_mesh.vertex
        );
    }
    true
}

/// Binds the mesh index buffer when it differs from the last submitted index stream.
fn bind_index_buffer_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    mesh: &GpuMesh,
    last_mesh: &mut LastMeshBindState,
) {
    let index_key = (
        core::ptr::from_ref(mesh.index_buffer.as_ref()).addr(),
        mesh.index_format,
    );
    if last_mesh.index != Some(index_key) {
        rpass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);
        last_mesh.index = Some(index_key);
    }
}

#[cfg(test)]
mod tests {
    use super::instance_range_for_draw_group;
    use crate::render_graph::world_mesh_draw_prep::DrawGroup;

    #[test]
    fn no_base_instance_draws_from_zero() {
        let group = DrawGroup {
            representative_draw_idx: 17,
            instance_range: 17..18,
        };
        assert_eq!(instance_range_for_draw_group(&group, false), 0..1);
    }

    #[test]
    fn base_instance_uses_slab_range() {
        let group = DrawGroup {
            representative_draw_idx: 17,
            instance_range: 17..20,
        };
        assert_eq!(instance_range_for_draw_group(&group, true), 17..20);
    }
}
