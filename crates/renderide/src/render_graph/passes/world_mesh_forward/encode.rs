//! Encode indexed draws and material bind groups for graph-managed world-mesh forward passes.

use crate::backend::mesh_deform::GpuSkinCache;
use crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::gpu::GpuLimits;
use crate::materials::{MaterialPassDesc, MaterialPipelineSet};
use crate::render_graph::frame_params::PrecomputedMaterialBind;
use crate::render_graph::world_mesh_draw_prep::for_each_instance_batch;
use crate::render_graph::WorldMeshDrawItem;
use crate::resources::MeshPool;

/// Embedded material vertex stream requirements for one draw (matches pipeline reflection flags).
pub(crate) struct EmbeddedVertexStreamFlags {
    /// UV0 stream at `@location(2)`.
    pub embedded_uv: bool,
    /// Vertex color at `@location(3)`.
    pub embedded_color: bool,
    /// Extended streams (tangents, extra UVs) at `@location(4)` and above.
    pub embedded_extended_vertex_streams: bool,
}

/// GPU mesh pool and optional skin cache for [`draw_mesh_submesh_instanced`].
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
    ptr: usize,
    byte_offset: u64,
    byte_len: Option<u64>,
}

impl BufferBindId {
    /// Full-buffer bind (`buf.slice(..)`).
    fn full(buf: &wgpu::Buffer) -> Self {
        Self {
            ptr: buf as *const wgpu::Buffer as usize,
            byte_offset: 0,
            byte_len: None,
        }
    }

    /// Ranged bind (`buf.slice(byte_start..byte_end)`).
    fn ranged(buf: &wgpu::Buffer, byte_start: u64, byte_end: u64) -> Self {
        Self {
            ptr: buf as *const wgpu::Buffer as usize,
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
    fn new() -> Self {
        Self {
            vertex: [None; 8],
            index: None,
        }
    }
}

/// Draw indices, bind groups, and precomputed-batch table for one mesh-forward raster subpass.
///
/// Pipelines and `@group(1)` bind groups are pre-resolved in
/// [`crate::render_graph::passes::world_mesh_forward::execute_helpers::precompute_material_resolve_batches`]
/// during the prepare pass, so this struct carries no material-system references and makes no
/// LRU cache lookups during recording.
pub(crate) struct ForwardDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Indices into `draws` for this subpass (opaque or intersect).
    pub draw_indices: &'c [usize],
    /// Full sorted world mesh draw list for the view.
    pub draws: &'c [WorldMeshDrawItem],
    /// Pre-resolved pipelines and bind groups; one entry per unique batch-key run in `draws`.
    pub precomputed: &'c [PrecomputedMaterialBind],
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
    /// Whether `draw_indexed` may use non-zero `first_instance` / base instance.
    pub supports_base_instance: bool,
    /// Whether the packed frame light buffer contains any point/spot light.
    pub has_local_lights: bool,
}

fn should_skip_pipeline_pass(
    declared_passes: &[crate::materials::MaterialPassDesc],
    pass_idx: usize,
    has_local_lights: bool,
) -> bool {
    !has_local_lights
        && declared_passes.get(pass_idx).is_some_and(|pass| {
            pass.material_state == crate::materials::MaterialPassState::UnityForwardAdd
        })
}

/// Records one raster subpass using the pre-resolved batch table built by the prepare pass.
///
/// For each instance batch the cursor advances through `precomputed` to find the covering entry
/// — O(1) amortised since both lists are in draw-index order. Pipelines and `@group(1)` bind
/// groups are bound directly from the table; no cache lookups occur during recording.
pub(crate) fn draw_subset(batch: ForwardDrawBatch<'_, '_, '_, '_>) {
    profiling::scope!("world_mesh::draw_subset");
    let ForwardDrawBatch {
        rpass,
        draw_indices,
        draws,
        precomputed,
        encode,
        gpu_limits,
        frame_bg,
        empty_bg,
        per_draw_bind_group,
        supports_base_instance,
        has_local_lights,
    } = batch;

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_stencil_ref: Option<u32> = None;
    // Cursor into `precomputed`; advances monotonically as draw indices increase.
    let mut batch_cursor: usize = 0;
    // Track which precomputed batch is currently bound to avoid redundant set_bind_group(1).
    let mut bound_batch_cursor: Option<usize> = None;

    rpass.set_bind_group(0, frame_bg, &[]);

    for_each_instance_batch(draws, draw_indices, supports_base_instance, |inst_batch| {
        let first_idx = inst_batch.first_draw_index;

        // Advance the cursor to the precomputed batch that covers `first_idx`.
        while batch_cursor + 1 < precomputed.len()
            && precomputed[batch_cursor].last_draw_idx < first_idx
        {
            batch_cursor += 1;
        }

        let pc = &precomputed[batch_cursor];
        debug_assert!(
            first_idx >= pc.first_draw_idx && first_idx <= pc.last_draw_idx,
            "precomputed batch [{}, {}] should cover draw index {}",
            pc.first_draw_idx,
            pc.last_draw_idx,
            first_idx,
        );

        let Some(pipelines) = pc.pipelines.as_ref() else {
            return; // pipeline unavailable for this batch — skip draws
        };

        // Bind @group(1) once per unique batch; skip when the cursor hasn't advanced.
        if bound_batch_cursor != Some(batch_cursor) {
            let material_bg = pc.bind_group.as_deref().unwrap_or(empty_bg);
            rpass.set_bind_group(1, material_bg, &[]);
            bound_batch_cursor = Some(batch_cursor);
        }

        bind_per_draw_slab_if_changed(
            rpass,
            per_draw_bind_group,
            gpu_limits,
            first_idx,
            inst_batch.instance_count,
            supports_base_instance,
            &mut last_per_draw_dyn_offset,
        );

        let stencil_ref = draws[first_idx].batch_key.render_state.stencil_reference();
        if last_stencil_ref != Some(stencil_ref) {
            rpass.set_stencil_reference(stencil_ref);
            last_stencil_ref = Some(stencil_ref);
        }

        let inst_range =
            instance_range_for_batch(first_idx, inst_batch.instance_count, supports_base_instance);

        issue_material_pipeline_passes(
            rpass,
            encode,
            &draws[first_idx],
            ActivePipelineSelection {
                pipelines,
                declared_passes: pc.declared_passes,
                has_local_lights,
            },
            &inst_range,
            &mut last_mesh,
        );
    });
}

/// Updates @group(2) dynamic offset and rebinds the per-draw slab when the row offset changes.
///
/// On base-instance-capable devices the dynamic offset is always zero, so the rebind occurs once
/// at most. On downlevel paths each instance batch carries one draw, so `instance_count == 1`.
fn bind_per_draw_slab_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    per_draw_bind_group: &wgpu::BindGroup,
    gpu_limits: &crate::gpu::GpuLimits,
    first_idx: usize,
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
        let raw = (first_idx * PER_DRAW_UNIFORM_STRIDE) as u32;
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
///
/// Bundled so the dispatcher doesn't thread three separate arguments through.
struct ActivePipelineSelection<'a> {
    /// Per-material pipeline objects in pass order.
    pipelines: &'a MaterialPipelineSet,
    /// Pass descriptors declared by the material, parallel to `pipelines`.
    declared_passes: &'a [MaterialPassDesc],
    /// Whether the current view has any local lights requiring the lit variant.
    has_local_lights: bool,
}

/// Walks the pipeline set for `item`, skipping pass variants that the material doesn't declare
/// and issuing one [`draw_mesh_submesh_instanced`] per remaining pipeline.
fn issue_material_pipeline_passes(
    rpass: &mut wgpu::RenderPass<'_>,
    encode: &mut crate::backend::WorldMeshForwardEncodeRefs<'_>,
    item: &WorldMeshDrawItem,
    pipeline_sel: ActivePipelineSelection<'_>,
    inst_range: &std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
) {
    let skin_cache = encode.skin_cache;
    for (pass_idx, pipeline) in pipeline_sel.pipelines.iter().enumerate() {
        if should_skip_pipeline_pass(
            pipeline_sel.declared_passes,
            pass_idx,
            pipeline_sel.has_local_lights,
        ) {
            continue;
        }
        rpass.set_pipeline(pipeline);
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

fn instance_range_for_batch(
    first_draw_index: usize,
    instance_count: u32,
    supports_base_instance: bool,
) -> std::ops::Range<u32> {
    if supports_base_instance {
        let start = first_draw_index as u32;
        start..start + instance_count
    } else {
        0..instance_count
    }
}

/// Binds one vertex slot only when the buffer identity or range has changed since the last bind.
///
/// Using `global_id()` rather than pointer equality is safe because wgpu `Buffer`s are
/// refcounted and their IDs are stable for the lifetime of the object.
macro_rules! bind_vertex_if_changed {
    ($rpass:expr, $slot:expr, $buf:expr, $id:expr, $last:expr) => {{
        if $last[$slot as usize] != Some($id) {
            $rpass.set_vertex_buffer($slot, $buf);
            $last[$slot as usize] = Some($id);
        }
    }};
}

#[expect(
    clippy::too_many_lines,
    reason = "hot draw path keeps bind/set decisions inline for register reuse"
)]
pub(crate) fn draw_mesh_submesh_instanced(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    streams: EmbeddedVertexStreamFlags,
    instances: std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
) {
    if item.mesh_asset_id < 0 || item.node_id < 0 || item.index_count == 0 {
        return;
    }
    let EmbeddedVertexStreamFlags {
        embedded_uv,
        embedded_color,
        embedded_extended_vertex_streams,
    } = streams;
    if embedded_extended_vertex_streams
        && !gpu
            .mesh_pool
            .get_mesh(item.mesh_asset_id)
            .is_some_and(|mesh| mesh.extended_vertex_streams_ready())
    {
        logger::trace!(
            "WorldMeshForward: extended vertex streams missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return;
    }
    let Some(mesh) = gpu.mesh_pool.get_mesh(item.mesh_asset_id) else {
        return;
    };
    if !mesh.debug_streams_ready() {
        return;
    }
    let Some(normals_bind) = mesh.normals_buffer.as_deref() else {
        return;
    };

    let use_deformed = item.world_space_deformed;
    let use_blend_only = mesh.num_blendshapes > 0;
    let needs_cache_stream = use_deformed || use_blend_only;

    if needs_cache_stream {
        let Some(cache) = gpu.skin_cache else {
            return;
        };
        let key = (item.space_id, item.node_id);
        let Some(entry) = cache.lookup(&key) else {
            logger::trace!(
                "world mesh forward: skin cache miss for space {:?} node {}",
                item.space_id,
                item.node_id
            );
            return;
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
        if use_deformed {
            let Some(nrm_r) = entry.normals.as_ref() else {
                return;
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
    } else {
        let Some(pos) = mesh.positions_buffer.as_deref() else {
            return;
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
    }
    if embedded_uv || embedded_color || embedded_extended_vertex_streams {
        let Some(uv) = mesh.uv0_buffer.as_deref() else {
            return;
        };
        bind_vertex_if_changed!(
            rpass,
            2,
            uv.slice(..),
            BufferBindId::full(uv),
            last_mesh.vertex
        );
    }
    if embedded_color || embedded_extended_vertex_streams {
        let Some(color) = mesh.color_buffer.as_deref() else {
            return;
        };
        bind_vertex_if_changed!(
            rpass,
            3,
            color.slice(..),
            BufferBindId::full(color),
            last_mesh.vertex
        );
    }
    if embedded_extended_vertex_streams {
        let (Some(tangent), Some(uv1), Some(uv2), Some(uv3)) = (
            mesh.tangent_buffer.as_deref(),
            mesh.uv1_buffer.as_deref(),
            mesh.uv2_buffer.as_deref(),
            mesh.uv3_buffer.as_deref(),
        ) else {
            return;
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

    let index_key = (
        mesh.index_buffer.as_ref() as *const wgpu::Buffer as usize,
        mesh.index_format,
    );
    if last_mesh.index != Some(index_key) {
        rpass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);
        last_mesh.index = Some(index_key);
    }

    let first = item.first_index;
    let end = first.saturating_add(item.index_count);
    rpass.draw_indexed(first..end, 0, instances);
}

#[cfg(test)]
mod tests {
    use crate::materials::{pass_from_kind, PassKind};

    use super::{instance_range_for_batch, should_skip_pipeline_pass};

    #[test]
    fn no_base_instance_draws_from_zero() {
        assert_eq!(instance_range_for_batch(17, 1, false), 0..1);
    }

    #[test]
    fn base_instance_uses_sorted_draw_slot() {
        assert_eq!(instance_range_for_batch(17, 3, true), 17..20);
    }

    #[test]
    fn skips_forward_add_only_when_no_local_lights() {
        let passes = [
            pass_from_kind(PassKind::ForwardBase, "fs_main"),
            pass_from_kind(PassKind::ForwardAdd, "fs_delta"),
        ];

        assert!(!should_skip_pipeline_pass(&passes, 0, false));
        assert!(should_skip_pipeline_pass(&passes, 1, false));
        assert!(!should_skip_pipeline_pass(&passes, 1, true));
    }
}
