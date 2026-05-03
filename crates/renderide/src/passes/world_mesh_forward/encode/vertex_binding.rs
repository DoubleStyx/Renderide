//! Vertex / index buffer binding helpers for forward mesh draw recording.
//!
//! Owns the per-render-pass last-bound state ([`LastMeshBindState`]) and the
//! family of `bind_*_vertex_streams` functions that issue
//! [`wgpu::RenderPass::set_vertex_buffer`] only when the slot changes. Used by
//! [`super::draw_subset`] via [`draw_mesh_submesh_instanced`].

use crate::assets::mesh::GpuMesh;
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::mesh_deform::{GpuSkinCache, SkinCacheKey};
use crate::world_mesh::WorldMeshDrawItem;

/// Embedded material vertex stream requirements for one draw (matches pipeline reflection flags).
#[derive(Clone, Copy)]
pub(super) struct EmbeddedVertexStreamFlags {
    /// UV0 stream at `@location(2)`.
    embedded_uv: bool,
    /// Vertex color at `@location(3)`.
    embedded_color: bool,
    /// UV1 at `@location(5)` without requiring the full extended stream set.
    embedded_uv1: bool,
    /// Extended streams (tangents, extra UVs) at `@location(4)` and above.
    embedded_extended_vertex_streams: bool,
}

impl EmbeddedVertexStreamFlags {
    /// Slot used by the compact UV1 layout when the pipeline does not use full extended streams.
    fn compact_uv1_slot(self) -> Option<usize> {
        if !self.embedded_uv1 || self.embedded_extended_vertex_streams {
            return None;
        }
        Some(if self.embedded_color { 4 } else { 3 })
    }
}

/// GPU mesh pool and optional skin cache for [`draw_mesh_submesh_instanced`].
#[derive(Clone, Copy)]
pub(super) struct WorldMeshDrawGpuRefs<'a> {
    /// Resident meshes and vertex buffers.
    mesh_pool: &'a crate::gpu_pools::MeshPool,
    /// Skin/deform cache when the draw uses deformed or blendshape streams.
    skin_cache: Option<&'a GpuSkinCache>,
}

/// Compact identity for a [`wgpu::Buffer`] sub-range used to skip redundant vertex / index binds.
///
/// `byte_len == None` encodes a full-buffer `.slice(..)` bind; `Some(n)` is a ranged bind
/// of `byte_offset..byte_offset + n`. Two `BufferBindId`s are equal when they refer to the
/// same buffer object, offset, and length -- a sufficient condition for the bind to be a no-op.
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
/// buffer. Reset at every new render pass (i.e. at the start of [`super::draw_subset`]).
pub(super) struct LastMeshBindState {
    /// Last bound buffer identity per vertex slot 0-7; `None` = never bound this pass.
    vertex: [Option<BufferBindId>; 8],
    /// Last bound index buffer (pointer-as-usize identity) and format; `None` = never bound.
    index: Option<(usize, wgpu::IndexFormat)>,
}

impl LastMeshBindState {
    /// Builds empty bind-state tracking for a fresh render pass.
    pub(super) fn new() -> Self {
        Self {
            vertex: [None; 8],
            index: None,
        }
    }
}

/// Binds one vertex slot only when the buffer identity or range has changed since the last bind.
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
pub(super) fn draw_mesh_submesh_instanced(
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
    if !bind_optional_vertex_streams(rpass, item, gpu, mesh, streams, last_mesh) {
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
    let mesh = gpu.mesh_pool.get(item.mesh_asset_id)?;
    if streams.embedded_extended_vertex_streams && !mesh.extended_vertex_streams_ready() {
        logger::trace!(
            "WorldMeshForward: extended vertex streams missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return None;
    }
    if streams.embedded_uv1
        && !streams.embedded_extended_vertex_streams
        && !mesh.uv1_vertex_stream_ready()
    {
        logger::trace!(
            "WorldMeshForward: UV1 vertex stream missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
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
    if let Some(nrm_r) = entry.normals.as_ref() {
        let nrm_buf = cache.normals_arena();
        let nrm_range = nrm_r.byte_range();
        bind_vertex_if_changed!(
            rpass,
            1,
            nrm_buf.slice(nrm_range.start..nrm_range.end),
            BufferBindId::ranged(nrm_buf, nrm_range.start, nrm_range.end),
            last_mesh.vertex
        );
        return true;
    }
    if item.world_space_deformed {
        return false;
    }
    bind_vertex_if_changed!(
        rpass,
        1,
        normals_bind.slice(..),
        BufferBindId::full(normals_bind),
        last_mesh.vertex
    );
    true
}

/// Binds UV, color, tangent, and extra UV streams required by the material reflection.
fn bind_optional_vertex_streams(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    mesh: &GpuMesh,
    streams: EmbeddedVertexStreamFlags,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if streams.embedded_uv
        || streams.embedded_color
        || streams.embedded_uv1
        || streams.embedded_extended_vertex_streams
    {
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
    if let Some(slot) = streams.compact_uv1_slot() {
        let Some(uv1) = mesh.uv1_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            slot,
            uv1.slice(..),
            BufferBindId::full(uv1),
            last_mesh.vertex
        );
    }
    if streams.embedded_extended_vertex_streams {
        let (Some(uv1), Some(uv2), Some(uv3)) = (
            mesh.uv1_buffer.as_deref(),
            mesh.uv2_buffer.as_deref(),
            mesh.uv3_buffer.as_deref(),
        ) else {
            return false;
        };
        if !bind_tangent_stream(rpass, item, gpu, mesh, last_mesh) {
            return false;
        }
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

fn bind_tangent_stream(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    mesh: &GpuMesh,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if item.world_space_deformed || item.blendshape_deformed {
        let Some(cache) = gpu.skin_cache else {
            return false;
        };
        let key = SkinCacheKey::from_draw_parts(item.space_id, item.skinned, item.instance_id);
        let Some(entry) = cache.lookup_current(&key) else {
            logger::trace!(
                "WorldMeshForward: deformed tangent cache miss for mesh_asset_id {}; draw skipped",
                item.mesh_asset_id
            );
            return false;
        };
        let Some(tangent_range) = entry.tangents.as_ref() else {
            logger::trace!(
                "WorldMeshForward: deformed tangent stream missing for mesh_asset_id {}; draw skipped",
                item.mesh_asset_id
            );
            return false;
        };
        let tangent_buf = cache.tangents_arena();
        let range = tangent_range.byte_range();
        bind_vertex_if_changed!(
            rpass,
            4,
            tangent_buf.slice(range.start..range.end),
            BufferBindId::ranged(tangent_buf, range.start, range.end),
            last_mesh.vertex
        );
        return true;
    }

    let Some(tangent) = mesh.tangent_buffer.as_deref() else {
        return false;
    };
    bind_vertex_if_changed!(
        rpass,
        4,
        tangent.slice(..),
        BufferBindId::full(tangent),
        last_mesh.vertex
    );
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

/// Resolves the per-encode-call refs needed by [`draw_mesh_submesh_instanced`].
pub(super) fn gpu_refs_for_encode<'a>(
    encode: &'a WorldMeshForwardEncodeRefs<'_>,
) -> WorldMeshDrawGpuRefs<'a> {
    WorldMeshDrawGpuRefs {
        mesh_pool: encode.mesh_pool(),
        skin_cache: encode.skin_cache,
    }
}

/// Embedded vertex stream flags resolved from one draw item's batch key.
pub(super) fn streams_for_item(item: &WorldMeshDrawItem) -> EmbeddedVertexStreamFlags {
    EmbeddedVertexStreamFlags {
        embedded_uv: item.batch_key.embedded_needs_uv0,
        embedded_color: item.batch_key.embedded_needs_color,
        embedded_uv1: item.batch_key.embedded_needs_uv1,
        embedded_extended_vertex_streams: item.batch_key.embedded_needs_extended_vertex_streams,
    }
}
