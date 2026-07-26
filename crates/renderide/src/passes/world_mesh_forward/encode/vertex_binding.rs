//! Vertex / index buffer binding helpers for forward mesh draw recording.
//!
//! Owns the per-render-pass last-bound state ([`LastMeshBindState`]) and the
//! family of `bind_*_vertex_streams` functions that issue
//! [`wgpu::RenderPass::set_vertex_buffer`] only when the slot changes. Used by
//! [`super::draw_subset`] via [`draw_mesh_submesh_instanced`].

use crate::assets::mesh::GpuMesh;
use crate::gpu_pools::geometry_arena::{ArenaStream, GeometryAllocation, GeometryArena};
use crate::mesh_deform::{GpuSkinCache, SkinCacheKey};
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::world_mesh::WorldMeshDrawItem;

/// Embedded material vertex stream requirements for one draw (matches pipeline reflection flags).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::passes::world_mesh_forward) struct EmbeddedVertexStreamFlags {
    /// UV0 stream at `@location(2)`.
    embedded_uv: bool,
    /// Vertex color at `@location(3)`.
    embedded_color: bool,
    /// Tangent at `@location(4)`.
    embedded_tangent: bool,
    /// Whether `@location(4)` carries raw shader payload instead of a geometric tangent.
    embedded_raw_tangent_payload: bool,
    /// Whether `@location(1)` carries raw shader payload instead of a lighting normal.
    embedded_raw_normal_payload: bool,
    /// UV1 at `@location(5)`.
    embedded_uv1: bool,
    /// UV2 at `@location(6)`.
    embedded_uv2: bool,
    /// UV3 at `@location(7)`.
    embedded_uv3: bool,
    /// Packed UV0-UV3 stream.
    embedded_wide_low_uvs: bool,
    /// Packed UV4-UV7 stream.
    embedded_wide_high_uvs: bool,
}

impl EmbeddedVertexStreamFlags {
    fn wide_low_uv_slot(self) -> Option<usize> {
        self.embedded_wide_low_uvs.then_some(2)
    }

    fn wide_high_uv_slot(self) -> Option<usize> {
        self.slot_after([self.embedded_wide_low_uvs], self.embedded_wide_high_uvs)
    }

    fn uv_slot(self) -> Option<usize> {
        self.slot_after(
            [self.embedded_wide_low_uvs, self.embedded_wide_high_uvs],
            self.compact_uv_enabled(),
        )
    }

    fn color_slot(self) -> Option<usize> {
        self.slot_after(
            [
                self.embedded_wide_low_uvs,
                self.embedded_wide_high_uvs,
                self.compact_uv_enabled(),
            ],
            self.embedded_color,
        )
    }

    fn tangent_slot(self) -> Option<usize> {
        self.slot_after(
            [
                self.embedded_wide_low_uvs,
                self.embedded_wide_high_uvs,
                self.compact_uv_enabled(),
                self.embedded_color,
            ],
            self.embedded_tangent,
        )
    }

    fn uv1_slot(self) -> Option<usize> {
        if self.embedded_wide_low_uvs {
            return None;
        }
        self.slot_after(
            [
                self.embedded_wide_high_uvs,
                self.compact_uv_enabled(),
                self.embedded_color,
                self.embedded_tangent,
            ],
            self.embedded_uv1,
        )
    }

    fn uv2_slot(self) -> Option<usize> {
        if self.embedded_wide_low_uvs {
            return None;
        }
        self.slot_after(
            [
                self.embedded_wide_high_uvs,
                self.compact_uv_enabled(),
                self.embedded_color,
                self.embedded_tangent,
                self.embedded_uv1,
            ],
            self.embedded_uv2,
        )
    }

    fn uv3_slot(self) -> Option<usize> {
        if self.embedded_wide_low_uvs {
            return None;
        }
        self.slot_after(
            [
                self.embedded_wide_high_uvs,
                self.compact_uv_enabled(),
                self.embedded_color,
                self.embedded_tangent,
                self.embedded_uv1,
                self.embedded_uv2,
            ],
            self.embedded_uv3,
        )
    }

    fn slot_after<const N: usize>(self, preceding: [bool; N], enabled: bool) -> Option<usize> {
        if !enabled {
            return None;
        }
        Some(2 + preceding.into_iter().filter(|active| *active).count())
    }

    fn compact_uv_enabled(self) -> bool {
        self.embedded_uv && !self.embedded_wide_low_uvs
    }
}

/// Visits each optional forward vertex stream these flags require as `(slot, arena stream)`, in the
/// same order as [`bind_optional_vertex_streams`]. Position (`@location(0)`) and normal
/// (`@location(1)`) are always bound and are not visited here. Shared by [`forward_arena_alloc`] and
/// [`bind_forward_arena_streams`] so the residency check and the bind stay in lockstep.
fn for_each_forward_arena_stream(
    flags: EmbeddedVertexStreamFlags,
    mut visit: impl FnMut(usize, ArenaStream),
) {
    if let Some(slot) = flags.wide_low_uv_slot() {
        visit(slot, ArenaStream::WideLow);
    }
    if let Some(slot) = flags.wide_high_uv_slot() {
        visit(slot, ArenaStream::WideHigh);
    }
    if let Some(slot) = flags.uv_slot() {
        visit(slot, ArenaStream::Uv0);
    }
    if let Some(slot) = flags.color_slot() {
        visit(slot, ArenaStream::Color);
    }
    if let Some(slot) = flags.tangent_slot() {
        let stream = if flags.embedded_raw_tangent_payload {
            ArenaStream::RawTangent
        } else {
            ArenaStream::Tangent
        };
        visit(slot, stream);
    }
    if let Some(slot) = flags.uv1_slot() {
        visit(slot, ArenaStream::Uv1);
    }
    if let Some(slot) = flags.uv2_slot() {
        visit(slot, ArenaStream::Uv2);
    }
    if let Some(slot) = flags.uv3_slot() {
        visit(slot, ArenaStream::Uv3);
    }
}

/// Returns the arena allocation for a static forward draw whose every required stream (normal plus
/// each stream `flags` enables) is resident, or [`None`] when the mesh is not in the arena or is
/// missing a stream. The caller guarantees the draw is non-skinned.
pub(in crate::passes::world_mesh_forward) fn forward_arena_alloc(
    item: &WorldMeshDrawItem,
    arena: &GeometryArena,
    flags: EmbeddedVertexStreamFlags,
) -> Option<GeometryAllocation> {
    let alloc =
        arena.mesh_for_index_span(item.mesh_asset_id, item.first_index, item.index_count)?;
    if !alloc.has_stream(ArenaStream::Normal) {
        return None;
    }
    let mut resident = true;
    for_each_forward_arena_stream(flags, |_slot, stream| {
        if !alloc.has_stream(stream) {
            resident = false;
        }
    });
    resident.then_some(alloc)
}

/// Binds the arena's position (`@location(0)`), normal (`@location(1)`), each required optional
/// stream, and the selected full index buffer for a `multi_draw` batch sharing `flags`.
///
/// The full arena buffers are stable across material runs, so this goes through
/// [`LastMeshBindState`] instead of resubmitting every vertex/index bind for every run. Returns
/// `false` if a required stream buffer is absent (caller must not draw the batch).
pub(super) fn bind_forward_arena_streams(
    rpass: &mut wgpu::RenderPass<'_>,
    arena: &GeometryArena,
    flags: EmbeddedVertexStreamFlags,
    narrow_indices: bool,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    bind_full_vertex_buffer_if_changed(rpass, 0, arena.position_buffer(), last_mesh);
    let Some(normal) = arena.stream_buffer(ArenaStream::Normal) else {
        return false;
    };
    bind_full_vertex_buffer_if_changed(rpass, 1, normal, last_mesh);
    let mut ok = true;
    for_each_forward_arena_stream(flags, |slot, stream| match arena.stream_buffer(stream) {
        Some(buffer) => bind_full_vertex_buffer_if_changed(rpass, slot, buffer, last_mesh),
        None => ok = false,
    });
    if !ok {
        return false;
    }
    let format = arena_index_format(narrow_indices);
    let index_buffer = if narrow_indices {
        arena.index_buffer_u16()
    } else {
        arena.index_buffer_u32()
    };
    bind_full_index_buffer_if_changed(rpass, index_buffer, format, last_mesh);
    true
}

/// Forward vertex stream flags for `item`, exposed to the indirect batch collector.
pub(in crate::passes::world_mesh_forward) fn forward_stream_flags(
    item: &WorldMeshDrawItem,
) -> EmbeddedVertexStreamFlags {
    streams_for_item(item)
}

/// GPU mesh pool and optional skin cache for [`draw_mesh_submesh_instanced`].
#[derive(Clone, Copy)]
pub(super) struct WorldMeshDrawGpuRefs<'a> {
    /// Resident meshes and vertex buffers.
    mesh_pool: &'a crate::gpu_pools::MeshPool,
    /// Skin/deform cache when the draw uses deformed or blendshape streams.
    skin_cache: Option<&'a GpuSkinCache>,
    /// Canonical shared geometry storage for immutable static draws.
    geometry_arena: Option<&'a GeometryArena>,
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

const MESH_FORWARD_VERTEX_BIND_SLOTS: usize = 16;

/// Per-render-pass last-bound vertex and index buffer state for bind deduplication.
///
/// Tracks the last-submitted buffer identity for each mesh-forward vertex slot and the index
/// buffer. Reset at every new render pass (i.e. at the start of [`super::draw_subset`]).
pub(super) struct LastMeshBindState {
    /// Last bound buffer identity per vertex slot; `None` = never bound this pass.
    vertex: [Option<BufferBindId>; MESH_FORWARD_VERTEX_BIND_SLOTS],
    /// Last bound index buffer identity/range and format; `None` = never bound.
    index: Option<(BufferBindId, wgpu::IndexFormat)>,
}

impl LastMeshBindState {
    /// Builds empty bind-state tracking for a fresh render pass.
    pub(super) fn new() -> Self {
        Self {
            vertex: [None; MESH_FORWARD_VERTEX_BIND_SLOTS],
            index: None,
        }
    }

    /// Records a full/ranged vertex binding and reports whether the render pass must submit it.
    fn replace_vertex_if_changed(&mut self, slot: usize, next: BufferBindId) -> bool {
        if self.vertex[slot] == Some(next) {
            return false;
        }
        self.vertex[slot] = Some(next);
        true
    }

    /// Records an index binding and reports whether the render pass must submit it.
    fn replace_index_if_changed(&mut self, next: (BufferBindId, wgpu::IndexFormat)) -> bool {
        if self.index == Some(next) {
            return false;
        }
        self.index = Some(next);
        true
    }
}

/// Binds one full arena vertex buffer only when the slot's buffer identity changed.
fn bind_full_vertex_buffer_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    slot: usize,
    buffer: &wgpu::Buffer,
    last_mesh: &mut LastMeshBindState,
) {
    if last_mesh.replace_vertex_if_changed(slot, BufferBindId::full(buffer)) {
        rpass.set_vertex_buffer(slot as u32, buffer.slice(..));
    }
}

/// Binds one full arena index buffer only when its buffer identity or format changed.
fn bind_full_index_buffer_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    buffer: &wgpu::Buffer,
    format: wgpu::IndexFormat,
    last_mesh: &mut LastMeshBindState,
) {
    if last_mesh.replace_index_if_changed((BufferBindId::full(buffer), format)) {
        rpass.set_index_buffer(buffer.slice(..), format);
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

#[inline]
fn draw_uses_deformed_primary_streams(item: &WorldMeshDrawItem) -> bool {
    item.world_space_deformed || item.blendshape_deformed
}

#[inline]
fn draw_uses_deformed_tangent_stream(item: &WorldMeshDrawItem, mesh: &GpuMesh) -> bool {
    item.world_space_deformed || (item.blendshape_deformed && mesh.blendshape_has_tangent_deltas)
}

#[cfg(test)]
#[inline]
fn draw_uses_deformed_tangent_stream_for_flags(
    world_space_deformed: bool,
    blendshape_deformed: bool,
    blendshape_has_tangent_deltas: bool,
) -> bool {
    world_space_deformed || (blendshape_deformed && blendshape_has_tangent_deltas)
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
    if draw_static_forward_from_arena(rpass, item, gpu, streams, instances.clone(), last_mesh) {
        return;
    }
    let Some(mesh) = resident_draw_mesh(item, gpu, streams) else {
        return;
    };
    let Some(normals_bind) = mesh.normals_buffer.as_deref() else {
        return;
    };

    if !bind_primary_vertex_streams(rpass, item, gpu, mesh, normals_bind, streams, last_mesh) {
        return;
    }
    if !bind_optional_vertex_streams(rpass, item, gpu, mesh, streams, last_mesh) {
        return;
    }

    if !bind_index_buffer_if_changed(rpass, mesh, last_mesh) {
        return;
    }

    let first = item.first_index;
    let end = first.saturating_add(item.index_count);
    rpass.draw_indexed(first..end, 0, instances);
}

/// Binds position and normal streams and issues one indexed draw for the GTAO normal prepass.
pub(super) fn draw_mesh_submesh_normals_instanced(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    instances: std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
) {
    if draw_static_normals_from_arena(rpass, item, gpu, instances.clone(), last_mesh) {
        return;
    }
    let Some(mesh) = resident_depth_draw_mesh(item, gpu) else {
        return;
    };
    let Some(normals_bind) = mesh.normals_buffer.as_deref() else {
        return;
    };

    if !bind_primary_vertex_streams(
        rpass,
        item,
        gpu,
        mesh,
        normals_bind,
        EmbeddedVertexStreamFlags::default(),
        last_mesh,
    ) {
        return;
    }

    if !bind_index_buffer_if_changed(rpass, mesh, last_mesh) {
        return;
    }

    let first = item.first_index;
    let end = first.saturating_add(item.index_count);
    rpass.draw_indexed(first..end, 0, instances);
}

/// Binds the position stream and issues one indexed draw for the depth prepass.
pub(super) fn draw_mesh_submesh_depth_instanced(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    instances: std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
) {
    if draw_static_depth_from_arena(rpass, item, gpu, instances.clone(), last_mesh) {
        return;
    }
    let Some(mesh) = resident_depth_draw_mesh(item, gpu) else {
        return;
    };

    if !bind_position_vertex_stream(rpass, item, gpu, mesh, last_mesh) {
        return;
    }

    if !bind_index_buffer_if_changed(rpass, mesh, last_mesh) {
        return;
    }

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
    if item.node_id < 0 || item.index_count == 0 {
        return None;
    }
    let mesh = gpu.mesh_pool.get(item.mesh_asset_id)?;
    if streams.embedded_tangent
        && streams.embedded_raw_tangent_payload
        && !mesh.raw_tangent_vertex_stream_ready()
    {
        logger::trace!(
            "WorldMeshForward: raw tangent payload stream missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return None;
    }
    if streams.embedded_tangent
        && !streams.embedded_raw_tangent_payload
        && !mesh.tangent_vertex_stream_ready()
    {
        logger::trace!(
            "WorldMeshForward: tangent vertex stream missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return None;
    }
    if !streams.embedded_wide_low_uvs && streams.embedded_uv1 && !mesh.uv1_vertex_stream_ready() {
        logger::trace!(
            "WorldMeshForward: UV1 vertex stream missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return None;
    }
    if !streams.embedded_wide_low_uvs && streams.embedded_uv2 && !mesh.uv2_vertex_stream_ready() {
        logger::trace!(
            "WorldMeshForward: UV2 vertex stream missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return None;
    }
    if !streams.embedded_wide_low_uvs && streams.embedded_uv3 && !mesh.uv3_vertex_stream_ready() {
        logger::trace!(
            "WorldMeshForward: UV3 vertex stream missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return None;
    }
    if streams.embedded_wide_low_uvs && !mesh.wide_low_uv_vertex_stream_ready() {
        logger::trace!(
            "WorldMeshForward: wide low UV vertex stream missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return None;
    }
    if streams.embedded_wide_high_uvs && !mesh.wide_high_uv_vertex_stream_ready() {
        logger::trace!(
            "WorldMeshForward: wide high UV vertex stream missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return None;
    }
    mesh.debug_streams_ready().then_some(mesh)
}

/// Returns the resident mesh for a depth-only draw after basic draw validation.
fn resident_depth_draw_mesh<'a>(
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'a>,
) -> Option<&'a GpuMesh> {
    if item.node_id < 0 || item.index_count == 0 {
        return None;
    }
    gpu.mesh_pool.get(item.mesh_asset_id)
}

#[inline]
fn draw_can_use_static_geometry(item: &WorldMeshDrawItem, gpu: WorldMeshDrawGpuRefs<'_>) -> bool {
    item.node_id >= 0
        && item.index_count != 0
        && !item.skinned
        && !item.world_space_deformed
        && !item.blendshape_deformed
        && gpu
            .mesh_pool
            .get(item.mesh_asset_id)
            .is_some_and(GpuMesh::uses_shared_static_geometry)
}

fn draw_static_forward_from_arena(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    streams: EmbeddedVertexStreamFlags,
    instances: std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if !draw_can_use_static_geometry(item, gpu) {
        return false;
    }
    let Some(arena) = gpu.geometry_arena else {
        return false;
    };
    let Some(allocation) =
        arena.mesh_for_index_span(item.mesh_asset_id, item.first_index, item.index_count)
    else {
        return false;
    };
    let Some(mesh) = gpu.mesh_pool.get(item.mesh_asset_id) else {
        return false;
    };
    if !bind_forward_arena_stream_ranges(rpass, arena, mesh, allocation, streams, last_mesh) {
        return false;
    }
    bind_arena_index_range_if_changed(rpass, arena, allocation, last_mesh);
    issue_arena_indexed_draw(rpass, item, instances);
    true
}

fn draw_static_normals_from_arena(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    instances: std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if !draw_can_use_static_geometry(item, gpu) {
        return false;
    }
    let Some(arena) = gpu.geometry_arena else {
        return false;
    };
    let Some(allocation) =
        arena.mesh_for_index_span(item.mesh_asset_id, item.first_index, item.index_count)
    else {
        return false;
    };
    let Some(mesh) = gpu.mesh_pool.get(item.mesh_asset_id) else {
        return false;
    };
    bind_arena_position_range(rpass, arena, allocation, last_mesh);
    if !bind_forward_stream_slot(
        rpass,
        arena,
        mesh,
        allocation,
        1,
        ArenaStream::Normal,
        last_mesh,
    ) {
        return false;
    }
    bind_arena_index_range_if_changed(rpass, arena, allocation, last_mesh);
    issue_arena_indexed_draw(rpass, item, instances);
    true
}

fn draw_static_depth_from_arena(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    instances: std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if !draw_can_use_static_geometry(item, gpu) {
        return false;
    }
    let Some((arena, allocation)) = gpu.geometry_arena.and_then(|arena| {
        arena
            .mesh_for_index_span(item.mesh_asset_id, item.first_index, item.index_count)
            .map(|allocation| (arena, allocation))
    }) else {
        return false;
    };
    bind_arena_position_range(rpass, arena, allocation, last_mesh);
    bind_arena_index_range_if_changed(rpass, arena, allocation, last_mesh);
    issue_arena_indexed_draw(rpass, item, instances);
    true
}

/// Binds positions from the arena and each remaining stream from whichever source holds it.
///
/// Optional streams can be absent from the arena while the mesh still owns its dedicated copy: the
/// shared buffer for a stream is only grown for meshes that actually supply it, and that growth can
/// be refused. Both sources address vertices mesh-locally here (the arena binds this mesh's slice,
/// a dedicated buffer binds whole), so falling back per slot draws the mesh instead of dropping it.
fn bind_forward_arena_stream_ranges(
    rpass: &mut wgpu::RenderPass<'_>,
    arena: &GeometryArena,
    mesh: &GpuMesh,
    allocation: GeometryAllocation,
    streams: EmbeddedVertexStreamFlags,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    bind_arena_position_range(rpass, arena, allocation, last_mesh);
    if !bind_forward_stream_slot(
        rpass,
        arena,
        mesh,
        allocation,
        1,
        ArenaStream::Normal,
        last_mesh,
    ) {
        return false;
    }
    let mut ready = true;
    for_each_forward_arena_stream(streams, |slot, stream| {
        if !bind_forward_stream_slot(rpass, arena, mesh, allocation, slot, stream, last_mesh) {
            ready = false;
        }
    });
    ready
}

/// Binds one stream slot from the arena, or from the mesh's own buffer when the arena lacks it.
fn bind_forward_stream_slot(
    rpass: &mut wgpu::RenderPass<'_>,
    arena: &GeometryArena,
    mesh: &GpuMesh,
    allocation: GeometryAllocation,
    slot: usize,
    stream: ArenaStream,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if allocation.has_stream(stream)
        && bind_arena_stream_range(rpass, slot, arena, allocation, stream, last_mesh)
    {
        return true;
    }
    let Some(buffer) = dedicated_arena_stream_buffer(mesh, stream) else {
        return false;
    };
    bind_vertex_if_changed!(
        rpass,
        slot,
        buffer.slice(..),
        BufferBindId::full(buffer),
        last_mesh.vertex
    );
    true
}

/// The mesh's own buffer for an arena stream, retained whenever the arena does not hold it.
///
/// Lazily derived streams are gated on readiness exactly as the dedicated draw path gates them, so
/// a stream still being regenerated skips the draw instead of binding stale contents.
fn dedicated_arena_stream_buffer(mesh: &GpuMesh, stream: ArenaStream) -> Option<&wgpu::Buffer> {
    let (buffer, ready) = match stream {
        ArenaStream::Normal => (mesh.normals_buffer.as_deref(), mesh.debug_streams_ready()),
        ArenaStream::Uv0 => (mesh.uv0_buffer.as_deref(), true),
        ArenaStream::Color => (mesh.color_buffer.as_deref(), true),
        ArenaStream::Tangent => (
            mesh.tangent_buffer.as_deref(),
            mesh.tangent_vertex_stream_ready(),
        ),
        ArenaStream::RawTangent => (
            mesh.raw_tangent_buffer.as_deref(),
            mesh.raw_tangent_vertex_stream_ready(),
        ),
        ArenaStream::Uv1 => (mesh.uv1_buffer.as_deref(), mesh.uv1_vertex_stream_ready()),
        ArenaStream::Uv2 => (mesh.uv2_buffer.as_deref(), mesh.uv2_vertex_stream_ready()),
        ArenaStream::Uv3 => (mesh.uv3_buffer.as_deref(), mesh.uv3_vertex_stream_ready()),
        ArenaStream::WideLow => (
            mesh.wide_low_uv_buffer.as_deref(),
            mesh.wide_low_uv_vertex_stream_ready(),
        ),
        ArenaStream::WideHigh => (
            mesh.wide_high_uv_buffer.as_deref(),
            mesh.wide_high_uv_vertex_stream_ready(),
        ),
    };
    buffer.filter(|_| ready)
}

fn bind_arena_position_range(
    rpass: &mut wgpu::RenderPass<'_>,
    arena: &GeometryArena,
    allocation: GeometryAllocation,
    last_mesh: &mut LastMeshBindState,
) {
    let buffer = arena.position_buffer();
    let range =
        arena_position_byte_range(allocation.vertices.offset_bytes, allocation.vertex_count);
    let (start, end) = (range.start, range.end);
    bind_vertex_if_changed!(
        rpass,
        0,
        buffer.slice(range),
        BufferBindId::ranged(buffer, start, end),
        last_mesh.vertex
    );
}

fn bind_arena_stream_range(
    rpass: &mut wgpu::RenderPass<'_>,
    slot: usize,
    arena: &GeometryArena,
    allocation: GeometryAllocation,
    stream: ArenaStream,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    let Some(buffer) = arena.stream_buffer(stream) else {
        return false;
    };
    let range = arena_stream_byte_range(allocation.base_vertex(), allocation.vertex_count, stream);
    let (start, end) = (range.start, range.end);
    bind_vertex_if_changed!(
        rpass,
        slot,
        buffer.slice(range),
        BufferBindId::ranged(buffer, start, end),
        last_mesh.vertex
    );
    true
}

fn bind_arena_index_range_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    arena: &GeometryArena,
    allocation: GeometryAllocation,
    last_mesh: &mut LastMeshBindState,
) {
    let format = arena_index_format(allocation.narrow_indices);
    let buffer = if allocation.narrow_indices {
        arena.index_buffer_u16()
    } else {
        arena.index_buffer_u32()
    };
    let range = arena_index_byte_range(
        allocation.indices.offset_bytes,
        allocation.indices.len_bytes,
    );
    let id = BufferBindId::ranged(buffer, range.start, range.end);
    let key = (id, format);
    if last_mesh.index != Some(key) {
        rpass.set_index_buffer(buffer.slice(range), format);
        last_mesh.index = Some(key);
    }
}

fn arena_position_byte_range(
    position_offset_bytes: u64,
    vertex_count: u32,
) -> std::ops::Range<u64> {
    let start = position_offset_bytes;
    start
        ..start.saturating_add(
            u64::from(vertex_count)
                .saturating_mul(crate::gpu_pools::geometry_arena::ARENA_POSITION_STRIDE),
        )
}

fn arena_stream_byte_range(
    base_vertex: i32,
    vertex_count: u32,
    stream: ArenaStream,
) -> std::ops::Range<u64> {
    let start = u64::from(base_vertex.max(0) as u32).saturating_mul(stream.stride());
    start..start.saturating_add(u64::from(vertex_count).saturating_mul(stream.stride()))
}

fn arena_index_format(narrow_indices: bool) -> wgpu::IndexFormat {
    if narrow_indices {
        wgpu::IndexFormat::Uint16
    } else {
        wgpu::IndexFormat::Uint32
    }
}

fn arena_index_byte_range(index_offset_bytes: u64, index_len_bytes: u64) -> std::ops::Range<u64> {
    index_offset_bytes..index_offset_bytes.saturating_add(index_len_bytes)
}

fn issue_arena_indexed_draw(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    instances: std::ops::Range<u32>,
) {
    let first = item.first_index;
    let end = first.saturating_add(item.index_count);
    rpass.draw_indexed(first..end, 0, instances);
}

/// Binds position and normal streams, choosing static mesh buffers or the deformation cache.
fn bind_primary_vertex_streams(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    mesh: &GpuMesh,
    normals_bind: &wgpu::Buffer,
    streams: EmbeddedVertexStreamFlags,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if draw_uses_deformed_primary_streams(item) && !streams.embedded_raw_normal_payload {
        bind_deformed_primary_streams(rpass, item, gpu, normals_bind, last_mesh)
    } else if draw_uses_deformed_primary_streams(item) {
        bind_deformed_position_static_normal(rpass, item, gpu, normals_bind, last_mesh)
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

/// Binds only the position stream selected for this draw.
fn bind_position_vertex_stream(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    mesh: &GpuMesh,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if draw_uses_deformed_primary_streams(item) {
        let Some(cache) = gpu.skin_cache else {
            return false;
        };
        let key = SkinCacheKey::from_draw_parts(
            item.space_id,
            item.render_context,
            item.skinned,
            item.instance_id,
        );
        let Some(entry) = cache.lookup_current(&key) else {
            logger::trace!(
                "world mesh depth prepass: current skin cache miss for space {:?} renderable {} instance {:?} node {}",
                item.space_id,
                item.renderable_index,
                item.instance_id,
                item.node_id
            );
            return false;
        };
        let pos_buf = cache.positions_arena();
        let pos_range = entry.positions.byte_range();
        let (pos_start, pos_end) = (pos_range.start, pos_range.end);
        bind_vertex_if_changed!(
            rpass,
            0,
            pos_buf.slice(pos_range),
            BufferBindId::ranged(pos_buf, pos_start, pos_end),
            last_mesh.vertex
        );
        return true;
    }

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
    let key = SkinCacheKey::from_draw_parts(
        item.space_id,
        item.render_context,
        item.skinned,
        item.instance_id,
    );
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
    let (pos_start, pos_end) = (pos_range.start, pos_range.end);
    bind_vertex_if_changed!(
        rpass,
        0,
        pos_buf.slice(pos_range),
        BufferBindId::ranged(pos_buf, pos_start, pos_end),
        last_mesh.vertex
    );
    if let Some(nrm_r) = entry.normals.as_ref() {
        let nrm_buf = cache.normals_arena();
        let nrm_range = nrm_r.byte_range();
        let (nrm_start, nrm_end) = (nrm_range.start, nrm_range.end);
        bind_vertex_if_changed!(
            rpass,
            1,
            nrm_buf.slice(nrm_range),
            BufferBindId::ranged(nrm_buf, nrm_start, nrm_end),
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

/// Binds deformed positions while preserving static normal-slot payload data.
fn bind_deformed_position_static_normal(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    normals_bind: &wgpu::Buffer,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    let Some(cache) = gpu.skin_cache else {
        return false;
    };
    let key = SkinCacheKey::from_draw_parts(
        item.space_id,
        item.render_context,
        item.skinned,
        item.instance_id,
    );
    let Some(entry) = cache.lookup_current(&key) else {
        logger::trace!(
            "world mesh forward: current skin cache miss for raw-normal payload draw in space {:?} renderable {} instance {:?} node {}",
            item.space_id,
            item.renderable_index,
            item.instance_id,
            item.node_id
        );
        return false;
    };
    let pos_buf = cache.positions_arena();
    let pos_range = entry.positions.byte_range();
    let (pos_start, pos_end) = (pos_range.start, pos_range.end);
    bind_vertex_if_changed!(
        rpass,
        0,
        pos_buf.slice(pos_range),
        BufferBindId::ranged(pos_buf, pos_start, pos_end),
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

/// Binds UV, color, tangent, and extra UV streams required by the material reflection.
fn bind_optional_vertex_streams(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    mesh: &GpuMesh,
    streams: EmbeddedVertexStreamFlags,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if let Some(slot) = streams.wide_low_uv_slot() {
        let Some(uv) = mesh.wide_low_uv_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            slot,
            uv.slice(..),
            BufferBindId::full(uv),
            last_mesh.vertex
        );
    }
    if let Some(slot) = streams.wide_high_uv_slot() {
        let Some(uv) = mesh.wide_high_uv_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            slot,
            uv.slice(..),
            BufferBindId::full(uv),
            last_mesh.vertex
        );
    }
    if let Some(slot) = streams.uv_slot() {
        let Some(uv) = mesh.uv0_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            slot,
            uv.slice(..),
            BufferBindId::full(uv),
            last_mesh.vertex
        );
    }
    if let Some(slot) = streams.color_slot() {
        let Some(color) = mesh.color_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            slot,
            color.slice(..),
            BufferBindId::full(color),
            last_mesh.vertex
        );
    }
    if let Some(slot) = streams.tangent_slot()
        && !bind_tangent_stream(rpass, item, gpu, mesh, streams, slot, last_mesh)
    {
        return false;
    }
    if let Some(slot) = streams.uv1_slot() {
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
    if let Some(slot) = streams.uv2_slot() {
        let Some(uv2) = mesh.uv2_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            slot,
            uv2.slice(..),
            BufferBindId::full(uv2),
            last_mesh.vertex
        );
    }
    if let Some(slot) = streams.uv3_slot() {
        let Some(uv3) = mesh.uv3_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            slot,
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
    streams: EmbeddedVertexStreamFlags,
    slot: usize,
    last_mesh: &mut LastMeshBindState,
) -> bool {
    if streams.embedded_raw_tangent_payload {
        let Some(tangent) = mesh.raw_tangent_buffer.as_deref() else {
            return false;
        };
        bind_vertex_if_changed!(
            rpass,
            slot,
            tangent.slice(..),
            BufferBindId::full(tangent),
            last_mesh.vertex
        );
        return true;
    }

    if draw_uses_deformed_tangent_stream(item, mesh) {
        let Some(cache) = gpu.skin_cache else {
            return false;
        };
        let key = SkinCacheKey::from_draw_parts(
            item.space_id,
            item.render_context,
            item.skinned,
            item.instance_id,
        );
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
        let (range_start, range_end) = (range.start, range.end);
        bind_vertex_if_changed!(
            rpass,
            slot,
            tangent_buf.slice(range),
            BufferBindId::ranged(tangent_buf, range_start, range_end),
            last_mesh.vertex
        );
        return true;
    }

    let Some(tangent) = mesh.tangent_buffer.as_deref() else {
        return false;
    };
    bind_vertex_if_changed!(
        rpass,
        slot,
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
) -> bool {
    let Some(index_buffer) = mesh.index_buffer.as_deref() else {
        return false;
    };
    let index_key = (BufferBindId::full(index_buffer), mesh.index_format);
    if last_mesh.index != Some(index_key) {
        rpass.set_index_buffer(index_buffer.slice(..), mesh.index_format);
        last_mesh.index = Some(index_key);
    }
    true
}

/// Resolves the per-encode-call refs needed by [`draw_mesh_submesh_instanced`].
pub(super) fn gpu_refs_for_encode<'a>(
    encode: &'a WorldMeshForwardEncodeRefs<'_>,
    geometry_arena: Option<&'a GeometryArena>,
) -> WorldMeshDrawGpuRefs<'a> {
    WorldMeshDrawGpuRefs {
        mesh_pool: encode.mesh_pool(),
        skin_cache: encode.skin_cache,
        geometry_arena,
    }
}

/// Embedded vertex stream flags resolved from one draw item's batch key.
pub(super) fn streams_for_item(item: &WorldMeshDrawItem) -> EmbeddedVertexStreamFlags {
    EmbeddedVertexStreamFlags {
        embedded_uv: item.batch_key.embedded_needs_uv0,
        embedded_color: item.batch_key.embedded_needs_color,
        embedded_tangent: item.batch_key.embedded_needs_tangent,
        embedded_raw_tangent_payload: item.batch_key.embedded_raw_tangent_payload,
        embedded_raw_normal_payload: item.batch_key.embedded_raw_normal_payload,
        embedded_uv1: item.batch_key.embedded_needs_uv1,
        embedded_uv2: item.batch_key.embedded_needs_uv2,
        embedded_uv3: item.batch_key.embedded_needs_uv3,
        embedded_wide_low_uvs: item.batch_key.embedded_needs_wide_low_uvs,
        embedded_wide_high_uvs: item.batch_key.embedded_needs_wide_high_uvs,
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu_pools::geometry_arena::ArenaStream;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    use super::{
        BufferBindId, LastMeshBindState, arena_index_byte_range, arena_index_format,
        arena_position_byte_range, arena_stream_byte_range, draw_uses_deformed_primary_streams,
        draw_uses_deformed_tangent_stream_for_flags,
    };

    fn item(
        world_space_deformed: bool,
        blendshape_deformed: bool,
    ) -> crate::world_mesh::WorldMeshDrawItem {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.world_space_deformed = world_space_deformed;
        item.blendshape_deformed = blendshape_deformed;
        item
    }

    #[test]
    fn blendshape_only_draw_uses_deformed_primary_streams() {
        assert!(draw_uses_deformed_primary_streams(&item(false, true)));
    }

    #[test]
    fn blendshape_only_without_tangent_deltas_uses_base_tangent_stream() {
        assert!(!draw_uses_deformed_tangent_stream_for_flags(
            false, true, false
        ));
    }

    #[test]
    fn blendshape_only_with_tangent_deltas_uses_deformed_tangent_stream() {
        assert!(draw_uses_deformed_tangent_stream_for_flags(
            false, true, true
        ));
    }

    #[test]
    fn world_space_skinning_uses_deformed_tangent_stream() {
        assert!(draw_uses_deformed_tangent_stream_for_flags(
            true, false, false
        ));
    }

    #[test]
    fn direct_arena_ranges_are_mesh_local_and_stride_correct() {
        let base_vertex = 32;
        let vertex_count = 7;

        assert_eq!(arena_position_byte_range(512, vertex_count), 512..624);
        assert_eq!(
            arena_stream_byte_range(base_vertex, vertex_count, ArenaStream::Normal),
            512..624
        );
        assert_eq!(
            arena_stream_byte_range(base_vertex, vertex_count, ArenaStream::Uv0),
            256..312
        );
        assert_eq!(
            arena_stream_byte_range(base_vertex, vertex_count, ArenaStream::WideLow),
            2048..2496
        );
        assert_eq!(arena_index_byte_range(768, 256), 768..1024);
    }

    #[test]
    fn direct_arena_index_width_follows_allocation() {
        assert_eq!(arena_index_format(true), wgpu::IndexFormat::Uint16);
        assert_eq!(arena_index_format(false), wgpu::IndexFormat::Uint32);
    }

    #[test]
    fn arena_full_buffer_state_skips_only_identical_rebinds() {
        let full = BufferBindId {
            ptr: 7,
            byte_offset: 0,
            byte_len: None,
        };
        let ranged = BufferBindId {
            ptr: 7,
            byte_offset: 64,
            byte_len: Some(128),
        };
        let mut state = LastMeshBindState::new();

        assert!(state.replace_vertex_if_changed(0, full));
        assert!(!state.replace_vertex_if_changed(0, full));
        assert!(state.replace_vertex_if_changed(0, ranged));
        assert!(state.replace_vertex_if_changed(0, full));

        assert!(state.replace_index_if_changed((full, wgpu::IndexFormat::Uint16)));
        assert!(!state.replace_index_if_changed((full, wgpu::IndexFormat::Uint16)));
        assert!(state.replace_index_if_changed((full, wgpu::IndexFormat::Uint32)));
    }
}
