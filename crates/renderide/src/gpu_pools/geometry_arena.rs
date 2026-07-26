//! Shared growable geometry buffers with byte-range suballocation.
//!
//! Position offsets define `base_vertex`; derived streams use the same vertex slot. Separate
//! `u16` and `u32` index buffers preserve each mesh's index width.
use hashbrown::{HashMap, HashSet};

use crate::assets::mesh::MeshDerivedStreamMask;
use crate::mesh_deform::range_alloc::{Range, RangeAllocator};

/// Suballocation alignment (WebGPU storage offset; also a multiple of the 2/4-byte index widths).
const GEOMETRY_ARENA_ALIGN: u64 = 256;

/// Minimum initial position-arena capacity in bytes (`vec4<f32>` positions).
const MIN_VERTEX_ARENA_BYTES: u64 = 1024 * 1024;

/// Minimum initial per-width index-arena capacity in bytes.
const MIN_INDEX_ARENA_BYTES: u64 = 256 * 1024;

/// Minimum initial optional-stream capacity in bytes.
const MIN_STREAM_ARENA_BYTES: u64 = 256 * 1024;

/// Minimum saving required to compact while both buffer generations are resident.
const MIN_COMPACTION_RECLAIM_BYTES: u64 = 8 * 1024 * 1024;

/// A newly materialized optional stream this far beyond the start of its buffer is considered
/// sparse enough to request a packed rebuild at the next frame boundary.
const OPTIONAL_STREAM_SPARSE_OFFSET_BYTES: u64 = 8 * 1024 * 1024;

/// Optional-stream growth beyond this size must be justified by the payload that triggers it.
const MAX_UNJUSTIFIED_STREAM_GROWTH_BYTES: u64 = 16 * 1024 * 1024;

/// Share of that growth the triggering copy must fill, as a divisor (`1/4`).
const MIN_STREAM_GROWTH_PAYLOAD_SHARE: u64 = 4;

/// Bytes per position vertex in the arena (`vec4<f32>`, matching `GpuMesh::positions_buffer`).
pub(crate) const ARENA_POSITION_STRIDE: u64 = 16;

/// Optional per-vertex streams sharing the position allocation's vertex slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArenaStream {
    /// Bind-pose normal (`vec4<f32>`).
    Normal,
    /// UV0 (`vec2<f32>`).
    Uv0,
    /// Vertex color (`vec4<f32>`).
    Color,
    /// Tangent (`vec4<f32>`).
    Tangent,
    /// Raw tangent payload (`vec4<f32>`) used by UI shaders.
    RawTangent,
    /// UV1 (`vec2<f32>`).
    Uv1,
    /// UV2 (`vec2<f32>`).
    Uv2,
    /// UV3 (`vec2<f32>`).
    Uv3,
    /// Packed UV0-UV3 wide page (`4 * vec4<f32>`).
    WideLow,
    /// Packed UV4-UV7 wide page (`4 * vec4<f32>`).
    WideHigh,
}

impl ArenaStream {
    /// Every optional stream, in slot order.
    pub(crate) const ALL: [ArenaStream; 10] = [
        ArenaStream::Normal,
        ArenaStream::Uv0,
        ArenaStream::Color,
        ArenaStream::Tangent,
        ArenaStream::RawTangent,
        ArenaStream::Uv1,
        ArenaStream::Uv2,
        ArenaStream::Uv3,
        ArenaStream::WideLow,
        ArenaStream::WideHigh,
    ];

    /// Number of optional streams.
    const COUNT: usize = Self::ALL.len();

    /// Dense position of this stream in the arena's per-stream tables and the residency bitmask.
    #[inline]
    const fn slot(self) -> usize {
        self as usize
    }

    /// Bytes per vertex for this stream.
    #[inline]
    pub(crate) const fn stride(self) -> u64 {
        match self {
            ArenaStream::Normal
            | ArenaStream::Color
            | ArenaStream::Tangent
            | ArenaStream::RawTangent => 16,
            ArenaStream::Uv0 | ArenaStream::Uv1 | ArenaStream::Uv2 | ArenaStream::Uv3 => 8,
            ArenaStream::WideLow | ArenaStream::WideHigh => 64,
        }
    }

    /// Residency-bit for this stream.
    #[inline]
    const fn bit(self) -> u16 {
        1 << self.slot()
    }

    /// Debug/allocation label for this stream's buffer.
    #[inline]
    const fn label(self) -> &'static str {
        match self {
            ArenaStream::Normal => "geometry_normal_arena",
            ArenaStream::Uv0 => "geometry_uv0_arena",
            ArenaStream::Color => "geometry_color_arena",
            ArenaStream::Tangent => "geometry_tangent_arena",
            ArenaStream::RawTangent => "geometry_raw_tangent_arena",
            ArenaStream::Uv1 => "geometry_uv1_arena",
            ArenaStream::Uv2 => "geometry_uv2_arena",
            ArenaStream::Uv3 => "geometry_uv3_arena",
            ArenaStream::WideLow => "geometry_wide_low_uv_arena",
            ArenaStream::WideHigh => "geometry_wide_high_uv_arena",
        }
    }
}

/// Expected byte demand used to right-size a newly created geometry arena.
///
/// Fresh allocations are first-fit and therefore use the same monotonically increasing position
/// offsets modelled here. Optional streams share those base-vertex offsets, including holes for
/// meshes that do not provide a particular stream.
#[derive(Clone, Debug)]
pub(crate) struct GeometryArenaCapacityHint {
    position_bytes: u64,
    index16_bytes: u64,
    index32_bytes: u64,
    stream_bytes: [u64; ArenaStream::COUNT],
}

impl Default for GeometryArenaCapacityHint {
    fn default() -> Self {
        Self {
            position_bytes: 0,
            index16_bytes: 0,
            index32_bytes: 0,
            stream_bytes: [0; ArenaStream::COUNT],
        }
    }
}

impl GeometryArenaCapacityHint {
    /// Adds one mesh in the same order it will be allocated by [`GeometryArena::ensure_mesh`].
    pub(crate) fn include_mesh(
        &mut self,
        vertex_count: u32,
        index_count: u32,
        narrow_indices: bool,
        streams: impl IntoIterator<Item = ArenaStream>,
    ) {
        if vertex_count == 0 || index_count == 0 {
            return;
        }

        let position_offset = align_up(self.position_bytes, GEOMETRY_ARENA_ALIGN);
        let position_payload = u64::from(vertex_count).saturating_mul(ARENA_POSITION_STRIDE);
        self.position_bytes =
            position_offset.saturating_add(align_up(position_payload, GEOMETRY_ARENA_ALIGN));

        let index_payload =
            u64::from(index_count).saturating_mul(if narrow_indices { 2 } else { 4 });
        let index_total = if narrow_indices {
            &mut self.index16_bytes
        } else {
            &mut self.index32_bytes
        };
        *index_total = align_up(*index_total, GEOMETRY_ARENA_ALIGN)
            .saturating_add(align_up(index_payload, GEOMETRY_ARENA_ALIGN));

        let base_vertex = position_offset / ARENA_POSITION_STRIDE;
        for stream in streams {
            let required = base_vertex
                .saturating_add(u64::from(vertex_count))
                .saturating_mul(stream.stride());
            self.stream_bytes[stream.slot()] = self.stream_bytes[stream.slot()].max(required);
        }
    }
}

/// Which end of an arena a new allocation is taken from.
///
/// Optional streams are addressed by the position allocation's base vertex, so a mesh placed high
/// in the position arena would force every stream it supplies to span everything below it. Meshes
/// that supply no stream take the top instead, leaving the low region for the ones that do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArenaPlacement {
    /// First fit from the start.
    Low,
    /// Last fit from the top.
    High,
}

/// Byte ranges for one mesh's geometry inside the shared arenas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GeometryAllocation {
    /// Position byte range in the position arena (the master allocation).
    pub(crate) vertices: Range,
    /// Index byte range in the matching (`u16`/`u32`) index arena.
    pub(crate) indices: Range,
    /// Vertex count of this mesh (position elements).
    pub(crate) vertex_count: u32,
    /// Number of addressable indices copied from the mesh source.
    pub(crate) index_count: u32,
    /// Whether the indices are `u16` (else `u32`).
    pub(crate) narrow_indices: bool,
    /// Residency bitmask over [`ArenaStream`] slots.
    stream_bits: u16,
}

impl GeometryAllocation {
    /// Bytes per index for this allocation's format.
    #[inline]
    pub(crate) fn index_size(self) -> u64 {
        if self.narrow_indices { 2 } else { 4 }
    }

    /// `base_vertex` for an indirect draw (position-range offset / stride). Indexes every stream.
    #[inline]
    pub(crate) fn base_vertex(self) -> i32 {
        i32::try_from(self.vertices.first_element_index(ARENA_POSITION_STRIDE)).unwrap_or(0)
    }

    /// First index of the mesh's index range, in elements of its format.
    #[inline]
    pub(crate) fn first_index_base(self) -> u32 {
        self.indices.first_element_index(self.index_size())
    }

    /// Whether a mesh-local indexed draw stays inside this allocation's copied payload.
    ///
    /// Indirect draws bind the complete arena, so WebGPU cannot validate this boundary for us.
    /// Rejecting a stale submesh span here prevents it from indexing the following allocation.
    #[inline]
    pub(crate) fn contains_index_span(self, first_index: u32, index_count: u32) -> bool {
        first_index
            .checked_add(index_count)
            .is_some_and(|end| end <= self.index_count)
    }

    /// Whether `stream` is resident for this mesh.
    #[inline]
    pub(crate) fn has_stream(self, stream: ArenaStream) -> bool {
        self.stream_bits & stream.bit() != 0
    }

    /// Derived vertex streams that are authoritative in the shared arena for this allocation.
    pub(crate) fn derived_stream_mask(self) -> MeshDerivedStreamMask {
        let mut mask = MeshDerivedStreamMask::POSITION;
        for stream in ArenaStream::ALL {
            if self.has_stream(stream) {
                mask |= match stream {
                    ArenaStream::Normal => MeshDerivedStreamMask::NORMAL,
                    ArenaStream::Uv0 => MeshDerivedStreamMask::UV0,
                    ArenaStream::Color => MeshDerivedStreamMask::COLOR,
                    ArenaStream::Tangent => MeshDerivedStreamMask::TANGENT,
                    ArenaStream::RawTangent => MeshDerivedStreamMask::RAW_TANGENT,
                    ArenaStream::Uv1 => MeshDerivedStreamMask::UV1,
                    ArenaStream::Uv2 => MeshDerivedStreamMask::UV2,
                    ArenaStream::Uv3 => MeshDerivedStreamMask::UV3,
                    ArenaStream::WideLow => MeshDerivedStreamMask::WIDE_UV_LOW,
                    ArenaStream::WideHigh => MeshDerivedStreamMask::WIDE_UV_HIGH,
                };
            }
        }
        mask
    }
}

/// Source stream buffers for one mesh upload into the arena. `optional` lists whichever decomposed
/// streams the mesh has; a stream absent from it leaves [`GeometryAllocation::has_stream`] false.
pub(crate) struct MeshStreamSources<'a> {
    /// Position stream (`vec4<f32>`), required.
    pub(crate) position: &'a wgpu::Buffer,
    /// Index buffer, required.
    pub(crate) index: &'a wgpu::Buffer,
    /// Whether `index` is `u16` (else `u32`).
    pub(crate) narrow: bool,
    /// Mesh vertex count.
    pub(crate) vertex_count: u32,
    /// Mesh index count.
    pub(crate) index_count: u32,
    /// Present optional streams, each `(kind, source buffer)`.
    pub(crate) optional: &'a [(ArenaStream, &'a wgpu::Buffer)],
}

#[derive(Clone, Copy)]
struct GeometryEntry {
    allocation: GeometryAllocation,
    position_source: usize,
    index_source: usize,
    stream_sources: [Option<usize>; ArenaStream::COUNT],
    /// Whether the core position/index copies reached submit-resource retention.
    ///
    /// A newly allocated entry remains visible to passes recorded in the same graph, but it must
    /// not survive into a later graph when recording aborted before the retain/submit handoff.
    core_committed: bool,
    /// Optional residency known to have reached submit-resource retention.
    committed_stream_bits: u16,
    /// Source identities corresponding to [`Self::committed_stream_bits`].
    ///
    /// Keeping this snapshot matters when an existing optional stream is recopied from a newer
    /// source: an aborted recording must restore the previous identity so the next attempt does
    /// not mistake the unsubmitted copy for current GPU contents.
    committed_stream_sources: [Option<usize>; ArenaStream::COUNT],
    /// Streams this mesh supplies that were refused because its position offset made them sparse.
    ///
    /// Packed rebuilds order by supplied streams, so carrying the refusal moves this mesh into the
    /// low region where the copy fits densely on the next attempt.
    refused_stream_bits: u16,
}

impl GeometryEntry {
    /// Commits every copy recorded into this entry before submit-resource retention.
    fn commit_recorded_state(&mut self) {
        self.core_committed = true;
        self.committed_stream_bits = self.allocation.stream_bits;
        self.committed_stream_sources = self.stream_sources;
    }

    /// Restores optional residency to the last state that reached submit-resource retention.
    fn rollback_unsubmitted_optional_streams(&mut self) {
        self.allocation.stream_bits = self.committed_stream_bits;
        self.stream_sources = self.committed_stream_sources;
    }
}

/// One growable GPU buffer paired with its byte-range allocator.
struct GrowableGeometryBuffer {
    buffer: wgpu::Buffer,
    alloc: RangeAllocator,
    usage: wgpu::BufferUsages,
    label: &'static str,
    /// Buffer authoritative before the current, not-yet-retained recording attempt first grew it.
    ///
    /// Growth swaps the live handle immediately so later passes in the same graph bind the
    /// replacement. If recording aborts, the recorded old-to-new copy never executes; retaining
    /// this checkpoint lets the next synchronization restore the submitted buffer instead.
    rollback_buffer: Option<wgpu::Buffer>,
}

impl GrowableGeometryBuffer {
    fn try_new(
        device: &wgpu::Device,
        capacity: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Option<Self> {
        let capacity = capacity.max(GEOMETRY_ARENA_ALIGN);
        let buffer = create_arena_buffer(device, label, capacity, usage)?;
        Some(Self {
            buffer,
            alloc: RangeAllocator::new(capacity, GEOMETRY_ARENA_ALIGN),
            usage,
            label,
            rollback_buffer: None,
        })
    }

    /// Restores the submitted buffer and reconstructs its allocator after an unsubmitted growth.
    fn rollback_unsubmitted_replacement(&mut self, live_ranges: Vec<Range>) {
        let Some(original) = self.rollback_buffer.take() else {
            return;
        };
        let capacity = original.size();
        self.buffer = original;
        self.alloc = rebuild_range_allocator(capacity, live_ranges)
            .expect("committed geometry ranges must fit the restored arena buffer");
    }
}

/// Submitted arena state retained while a compacted replacement is still transactional.
///
/// The old buffers are GPU copy sources for compaction and remain the rollback authority until the
/// graph hands all referenced resources to deferred submission.
struct GeometryCompactionRollback {
    positions: GrowableGeometryBuffer,
    indices32: GrowableGeometryBuffer,
    indices16: GrowableGeometryBuffer,
    streams: [Option<wgpu::Buffer>; ArenaStream::COUNT],
    entries: HashMap<i32, GeometryEntry>,
}

#[derive(Clone, Copy, Debug)]
struct GeometryCompactionRow {
    old: GeometryAllocation,
    new: GeometryAllocation,
}

struct GeometryCompactionPlan {
    rows: Vec<GeometryCompactionRow>,
    entries: HashMap<i32, GeometryEntry>,
    position_capacity: u64,
    index32_capacity: u64,
    index16_capacity: u64,
    stream_capacities: [u64; ArenaStream::COUNT],
    copy_bytes: u64,
}

impl GeometryCompactionPlan {
    fn allocated_bytes(&self) -> u64 {
        self.position_capacity
            .saturating_add(self.index16_capacity)
            .saturating_add(self.index32_capacity)
            .saturating_add(self.stream_capacities.iter().copied().sum::<u64>())
    }
}

/// One frame-boundary arena-reclamation result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GeometryArenaReclaimStats {
    /// Whether a pending high-water/sparsity request was evaluated.
    pub(crate) evaluated: bool,
    /// Steady-state live backing bytes released by the packed replacement.
    pub(crate) reclaimed_bytes: u64,
    /// Committed payload bytes copied entirely on the GPU.
    pub(crate) copy_bytes: u64,
}

/// Shared vertex + index arenas for suballocated mesh geometry.
pub(crate) struct GeometryArena {
    positions: GrowableGeometryBuffer,
    indices32: GrowableGeometryBuffer,
    indices16: GrowableGeometryBuffer,
    /// Lazy optional-stream buffers indexed by [`ArenaStream::slot`], each grown far enough for
    /// every resident allocation that supplies that stream.
    streams: [Option<wgpu::Buffer>; ArenaStream::COUNT],
    entries: HashMap<i32, GeometryEntry>,
    max_buffer_size: u64,
    /// Mesh-pool mutation cursor used only for incremental synchronization.
    mesh_pool_generation: u64,
    /// Monotonic revision of allocation offsets and stream residency consumed by retained indirect
    /// planning. Unlike `mesh_pool_generation`, this also advances for GPU-only compaction.
    allocation_generation: u64,
    /// Static mesh ids whose population commands reached submit-resource handoff and whose
    /// dedicated upload-source buffers can be released at the next backend preparation point.
    source_release_ready: HashSet<i32>,
    /// Static mesh ids populated while recording the current command buffers. These are promoted
    /// to `source_release_ready` only when submit resources are retained.
    source_release_pending: HashSet<i32>,
    /// Buffers replaced by growth or used as one-shot copy sources during this frame. They must
    /// survive until the deferred driver thread accepts the command buffers that reference them.
    retired_buffers: Vec<wgpu::Buffer>,
    /// Original optional-stream handles before their first creation/growth in the current,
    /// not-yet-retained recording attempt. The outer `Option` marks a replacement; the inner
    /// `Option` is `None` when the stream buffer did not exist before the attempt.
    stream_rollback_buffers: [Option<Option<wgpu::Buffer>>; ArenaStream::COUNT],
    /// Smallest stream capacity the device has refused, per [`ArenaStream::slot`], or `0`.
    ///
    /// Optional streams are addressed by position base vertex, so one late stream-bearing mesh can
    /// demand a buffer spanning the whole vertex arena. Refusing a repeat of a size the driver
    /// already rejected keeps the frame off a per-frame allocation retry until compaction packs
    /// those meshes forward and lowers the requirement.
    stream_growth_denied_bytes: [u64; ArenaStream::COUNT],
    /// Whether a stream copy was refused for sparsity since the last published layout.
    stream_sparse_refused: [bool; ArenaStream::COUNT],
    /// Submitted state before an in-flight compacting rebuild.
    compaction_rollback: Option<GeometryCompactionRollback>,
    /// A removal or sparse late optional stream requested a packed rebuild evaluation.
    reclamation_requested: bool,
}

impl GeometryArena {
    /// Returns the resident allocation only when a mesh-local draw range is safe to issue.
    ///
    /// Render passes bind the complete index arena for indirect draws, so this is the last
    /// corruption boundary before stale/bad metadata could read into the next mesh.
    pub(crate) fn mesh_for_index_span(
        &self,
        asset_id: i32,
        first_index: u32,
        index_count: u32,
    ) -> Option<GeometryAllocation> {
        let allocation = self.mesh(asset_id)?;
        if allocation.contains_index_span(first_index, index_count) {
            return Some(allocation);
        }

        static INVALID_SPAN_LOG: std::sync::LazyLock<
            crate::diagnostics::log_once::KeyedLogOnce<i32>,
        > = std::sync::LazyLock::new(crate::diagnostics::log_once::KeyedLogOnce::new);
        if INVALID_SPAN_LOG.should_log(asset_id) {
            logger::warn!(
                "geometry arena rejected mesh {asset_id} index span {first_index}+{index_count}; \
                 allocation contains {} indices",
                allocation.index_count
            );
        }
        None
    }

    /// Creates arenas sized for the static meshes expected in the first population pass.
    ///
    /// Returns `None` when the device cannot back the core position/index arenas, leaving every
    /// mesh on its dedicated buffers instead of publishing arenas that fail on first use.
    pub(crate) fn new_with_capacity_hint(
        device: &wgpu::Device,
        max_buffer_size: u64,
        hint: &GeometryArenaCapacityHint,
    ) -> Option<Self> {
        let cap = max_buffer_size.max(GEOMETRY_ARENA_ALIGN);
        let vertex_usage = wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let index_usage = wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let positions = create_core_arena(
            device,
            initial_arena_capacity(hint.position_bytes, MIN_VERTEX_ARENA_BYTES, cap, true),
            MIN_VERTEX_ARENA_BYTES.min(cap),
            vertex_usage,
            "geometry_position_arena",
        )?;
        let indices32 = create_core_arena(
            device,
            initial_arena_capacity(hint.index32_bytes, MIN_INDEX_ARENA_BYTES, cap, true),
            MIN_INDEX_ARENA_BYTES.min(cap),
            index_usage,
            "geometry_index32_arena",
        )?;
        let indices16 = create_core_arena(
            device,
            initial_arena_capacity(hint.index16_bytes, MIN_INDEX_ARENA_BYTES, cap, true),
            MIN_INDEX_ARENA_BYTES.min(cap),
            index_usage,
            "geometry_index16_arena",
        )?;
        let mut streams = [const { None }; ArenaStream::COUNT];
        for stream in ArenaStream::ALL {
            let required = hint.stream_bytes[stream.slot()];
            if required == 0 {
                continue;
            }
            let capacity = initial_arena_capacity(required, MIN_STREAM_ARENA_BYTES, cap, false);
            // A refused optional stream stays absent: the lazy path retries it per mesh, and the
            // meshes that supply it keep their dedicated buffers until then.
            streams[stream.slot()] = create_stream_buffer(device, capacity, stream.label());
        }
        Some(Self {
            positions,
            indices32,
            indices16,
            streams,
            entries: HashMap::new(),
            max_buffer_size: cap,
            mesh_pool_generation: 0,
            allocation_generation: 0,
            source_release_ready: HashSet::new(),
            source_release_pending: HashSet::new(),
            retired_buffers: Vec::new(),
            stream_rollback_buffers: std::array::from_fn(|_| None),
            stream_growth_denied_bytes: [0; ArenaStream::COUNT],
            stream_sparse_refused: [false; ArenaStream::COUNT],
            compaction_rollback: None,
            reclamation_requested: false,
        })
    }

    /// Shared position buffer (slot 0) for an indirect batch.
    #[inline]
    pub(crate) fn position_buffer(&self) -> &wgpu::Buffer {
        &self.positions.buffer
    }

    /// Shared buffer for `stream`, when at least one resident mesh supplied it.
    #[inline]
    pub(crate) fn stream_buffer(&self, stream: ArenaStream) -> Option<&wgpu::Buffer> {
        self.streams[stream.slot()].as_ref()
    }

    /// Shared `u32` index buffer.
    #[inline]
    pub(crate) fn index_buffer_u32(&self) -> &wgpu::Buffer {
        &self.indices32.buffer
    }

    /// Shared `u16` index buffer.
    #[inline]
    pub(crate) fn index_buffer_u16(&self) -> &wgpu::Buffer {
        &self.indices16.buffer
    }

    /// Cached allocation for a mesh already copied into the arena.
    #[inline]
    pub(crate) fn mesh(&self, asset_id: i32) -> Option<GeometryAllocation> {
        self.entries.get(&asset_id).map(|entry| entry.allocation)
    }

    /// Mesh-pool generation that authored the arena's current allocation map.
    ///
    /// Buffer growth preserves offsets, while resident mesh replacement/removal, optional-stream
    /// residency, and packed reclamation can invalidate a retained indirect command.
    #[inline]
    pub(crate) fn allocation_generation(&self) -> u64 {
        self.allocation_generation
    }

    /// Total bytes reserved by the arena's live backing buffers.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.positions
            .buffer
            .size()
            .saturating_add(self.indices16.buffer.size())
            .saturating_add(self.indices32.buffer.size())
            .saturating_add(
                self.streams
                    .iter()
                    .filter_map(Option::as_ref)
                    .map(wgpu::Buffer::size)
                    .sum::<u64>(),
            )
    }

    /// Bytes occupied by resident mesh ranges, excluding unallocated buffer headroom.
    pub(crate) fn resident_allocation_bytes(&self) -> u64 {
        self.entries.values().fold(0u64, |total, entry| {
            let allocation = entry.allocation;
            let core = allocation
                .vertices
                .len_bytes
                .saturating_add(allocation.indices.len_bytes);
            let optional = ArenaStream::ALL
                .into_iter()
                .filter(|&stream| allocation.has_stream(stream))
                .map(|stream| u64::from(allocation.vertex_count).saturating_mul(stream.stride()))
                .sum::<u64>();
            total.saturating_add(core).saturating_add(optional)
        })
    }

    /// Whether a removal or sparse optional stream requested frame-boundary reclamation.
    #[inline]
    pub(crate) fn reclamation_requested(&self) -> bool {
        self.reclamation_requested
    }

    /// Whether the resident mesh table changed since this arena last synchronized its entries.
    ///
    /// The population pass uses this before deciding whether to record. Generation drift must wake
    /// the pass even when the current frame has zero draws, otherwise removed meshes can leave
    /// stale high-water buffers resident indefinitely.
    #[inline]
    pub(crate) fn needs_mesh_pool_synchronization(&self, current_generation: u64) -> bool {
        mesh_pool_synchronization_needed(self.mesh_pool_generation, current_generation)
    }

    /// Rebuilds committed arena contents into smaller packed buffers entirely through GPU copies.
    ///
    /// This must run before any arena consumer is recorded for the frame. The replacement stays
    /// transactional until [`Self::retain_submit_resources`]: an aborted graph restores the old
    /// handles, allocation map, and allocators without consulting released CPU upload sources.
    pub(crate) fn reclaim_high_water_at_frame_boundary(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> GeometryArenaReclaimStats {
        if !self.reclamation_requested {
            return GeometryArenaReclaimStats::default();
        }
        self.reclamation_requested = false;

        // Population synchronization rolls back any prior unsubmitted attempt before reaching this
        // method. Refuse nested transactions defensively if another caller violates that ordering.
        if self.compaction_rollback.is_some()
            || self.positions.rollback_buffer.is_some()
            || self.indices32.rollback_buffer.is_some()
            || self.indices16.rollback_buffer.is_some()
            || self.stream_rollback_buffers.iter().any(Option::is_some)
        {
            self.reclamation_requested = true;
            return GeometryArenaReclaimStats {
                evaluated: true,
                ..Default::default()
            };
        }
        if self.entries.values().any(|entry| {
            !entry.core_committed
                || entry.allocation.stream_bits != entry.committed_stream_bits
                || entry.stream_sources != entry.committed_stream_sources
        }) {
            // Do not make a second transactional layer out of copies that have not themselves
            // reached retention yet. The ordinary next-frame synchronization will commit or roll
            // them back before retrying.
            self.reclamation_requested = true;
            return GeometryArenaReclaimStats {
                evaluated: true,
                ..Default::default()
            };
        }

        let Some(plan) = build_compaction_plan(&self.entries, self.max_buffer_size) else {
            return GeometryArenaReclaimStats {
                evaluated: true,
                ..Default::default()
            };
        };
        let old_allocated_bytes = self.allocated_bytes();
        let new_allocated_bytes = plan.allocated_bytes();
        // A refused stream is a layout problem, not a size one: the rebuild pays for itself by
        // moving that mesh low enough for its copy to fit, even when it frees little. It must
        // actually move something, otherwise an already packed arena would recopy itself forever.
        let repacks_refused_stream = self.stream_sparse_refused.iter().any(|&refused| refused)
            && new_allocated_bytes <= old_allocated_bytes
            && plan
                .rows
                .iter()
                .any(|row| row.new.vertices.offset_bytes != row.old.vertices.offset_bytes);
        if !repacks_refused_stream
            && !compaction_is_worthwhile(old_allocated_bytes, new_allocated_bytes)
        {
            return GeometryArenaReclaimStats {
                evaluated: true,
                ..Default::default()
            };
        }
        for stream in ArenaStream::ALL {
            if plan.stream_capacities[stream.slot()] != 0 && self.streams[stream.slot()].is_none() {
                // Residency metadata without an authoritative stream buffer is an invariant
                // violation. Keep the current submitted arena instead of manufacturing content.
                return GeometryArenaReclaimStats {
                    evaluated: true,
                    ..Default::default()
                };
            }
        }

        let Some((position_alloc, index32_alloc, index16_alloc)) =
            rebuild_compacted_core_allocators(
                &plan.entries,
                plan.position_capacity,
                plan.index32_capacity,
                plan.index16_capacity,
            )
        else {
            // Never publish packed entries beside allocators that still describe the complete
            // replacement buffers as free. A subsequent mesh allocation would otherwise reuse a
            // live range and overwrite valid geometry without tripping any draw-span validation.
            return GeometryArenaReclaimStats {
                evaluated: true,
                ..Default::default()
            };
        };
        // Every replacement buffer must exist before the swap. A device refusal here keeps the
        // current arena authoritative rather than publishing a layout with an unusable buffer.
        let (Some(mut new_positions), Some(mut new_indices32), Some(mut new_indices16)) = (
            GrowableGeometryBuffer::try_new(
                device,
                plan.position_capacity,
                self.positions.usage,
                self.positions.label,
            ),
            GrowableGeometryBuffer::try_new(
                device,
                plan.index32_capacity,
                self.indices32.usage,
                self.indices32.label,
            ),
            GrowableGeometryBuffer::try_new(
                device,
                plan.index16_capacity,
                self.indices16.usage,
                self.indices16.label,
            ),
        ) else {
            return GeometryArenaReclaimStats {
                evaluated: true,
                ..Default::default()
            };
        };
        new_positions.alloc = position_alloc;
        new_indices32.alloc = index32_alloc;
        new_indices16.alloc = index16_alloc;
        let mut new_streams = [const { None }; ArenaStream::COUNT];
        for stream in ArenaStream::ALL {
            let capacity = plan.stream_capacities[stream.slot()];
            if capacity == 0 {
                continue;
            }
            let Some(buffer) = create_stream_buffer(device, capacity, stream.label()) else {
                // Planned residency without a buffer would leave draws binding nothing, so drop
                // the whole rebuild instead of a single stream.
                return GeometryArenaReclaimStats {
                    evaluated: true,
                    ..Default::default()
                };
            };
            new_streams[stream.slot()] = Some(buffer);
        }
        self.stream_growth_denied_bytes = [0; ArenaStream::COUNT];
        self.stream_sparse_refused = [false; ArenaStream::COUNT];

        for row in &plan.rows {
            encoder.copy_buffer_to_buffer(
                &self.positions.buffer,
                row.old.vertices.offset_bytes,
                &new_positions.buffer,
                row.new.vertices.offset_bytes,
                row.old.vertices.len_bytes,
            );
            let old_indices = if row.old.narrow_indices {
                &self.indices16.buffer
            } else {
                &self.indices32.buffer
            };
            let new_indices = if row.new.narrow_indices {
                &new_indices16.buffer
            } else {
                &new_indices32.buffer
            };
            encoder.copy_buffer_to_buffer(
                old_indices,
                row.old.indices.offset_bytes,
                new_indices,
                row.new.indices.offset_bytes,
                row.old.indices.len_bytes,
            );
            for stream in ArenaStream::ALL {
                if !row.old.has_stream(stream) {
                    continue;
                }
                let stride = stream.stride();
                let copy_bytes = u64::from(row.old.vertex_count).saturating_mul(stride);
                if copy_bytes == 0 {
                    continue;
                }
                let old_offset = u64::try_from(row.old.base_vertex().max(0))
                    .unwrap_or(0)
                    .saturating_mul(stride);
                let new_offset = u64::try_from(row.new.base_vertex().max(0))
                    .unwrap_or(0)
                    .saturating_mul(stride);
                encoder.copy_buffer_to_buffer(
                    self.streams[stream.slot()]
                        .as_ref()
                        .expect("validated resident optional stream"),
                    old_offset,
                    new_streams[stream.slot()]
                        .as_ref()
                        .expect("planned resident optional stream"),
                    new_offset,
                    copy_bytes,
                );
            }
        }

        let checkpoint = GeometryCompactionRollback {
            positions: std::mem::replace(&mut self.positions, new_positions),
            indices32: std::mem::replace(&mut self.indices32, new_indices32),
            indices16: std::mem::replace(&mut self.indices16, new_indices16),
            streams: std::mem::replace(&mut self.streams, new_streams),
            entries: std::mem::replace(&mut self.entries, plan.entries),
        };
        self.compaction_rollback = Some(checkpoint);
        self.bump_allocation_generation();

        GeometryArenaReclaimStats {
            evaluated: true,
            reclaimed_bytes: old_allocated_bytes.saturating_sub(new_allocated_bytes),
            copy_bytes: plan.copy_bytes,
        }
    }

    /// Marks a shared-static mesh as awaiting successful submit-resource handoff.
    pub(crate) fn mark_source_release_pending(&mut self, asset_id: i32) {
        if self.entries.contains_key(&asset_id) {
            self.source_release_pending.insert(asset_id);
        }
    }

    /// Drains shared-static meshes confirmed resident by a completed population-recording step.
    pub(crate) fn take_source_release_ready_asset_ids(&mut self) -> Vec<i32> {
        let mut asset_ids: Vec<_> = self.source_release_ready.drain().collect();
        asset_ids.sort_unstable();
        asset_ids
    }

    /// Retains all arena buffers used by the current frame and transfers ownership of buffers
    /// replaced by in-frame growth into the deferred-submit payload.
    pub(crate) fn retain_submit_resources(
        &mut self,
        resources: &mut crate::gpu::GpuRetainedResources,
    ) {
        resources.retain_buffers([
            self.positions.buffer.clone(),
            self.indices32.buffer.clone(),
            self.indices16.buffer.clone(),
        ]);
        resources.retain_buffers(
            self.streams
                .iter()
                .filter_map(|stream| stream.as_ref().cloned()),
        );
        resources.retain_buffers(self.retired_buffers.drain(..));
        if let Some(checkpoint) = self.compaction_rollback.take() {
            resources.retain_buffers([
                checkpoint.positions.buffer,
                checkpoint.indices32.buffer,
                checkpoint.indices16.buffer,
            ]);
            resources.retain_buffers(checkpoint.streams.into_iter().flatten());
        }
        self.positions.rollback_buffer = None;
        self.indices32.rollback_buffer = None;
        self.indices16.rollback_buffer = None;
        for rollback in &mut self.stream_rollback_buffers {
            *rollback = None;
        }
        for entry in self.entries.values_mut() {
            entry.commit_recorded_state();
        }
        self.source_release_ready
            .extend(self.source_release_pending.drain());
    }

    /// Invalidates arena allocations for meshes replaced or updated in the resident mesh pool.
    ///
    /// Mesh uploads can rewrite buffers in place, so comparing source buffer handles is not enough.
    /// The pool's mutation log supplies the missing content-generation signal while allowing
    /// unchanged allocations to remain resident.
    pub(crate) fn synchronize_mesh_pool(&mut self, mesh_pool: &crate::gpu_pools::MeshPool) {
        self.rollback_unsubmitted_population();
        let delta = mesh_pool.mutation_delta_since(self.mesh_pool_generation);
        if delta.requires_full_rebuild {
            // A mutation-log overflow cannot identify which individual assets changed. Preserve
            // allocations whose current mesh explicitly records committed shared-core residency:
            // full mesh replacement resets that bit, while released immutable meshes no longer
            // have dedicated position/index sources from which they could be rebuilt.
            let stale_asset_ids = self
                .entries
                .keys()
                .copied()
                .filter(|&asset_id| {
                    mesh_pool
                        .get(asset_id)
                        .is_none_or(|mesh| !mesh.has_shared_static_core_residency())
                })
                .collect::<Vec<_>>();
            for asset_id in stale_asset_ids {
                self.remove_entry(asset_id);
            }
        } else {
            for &asset_id in delta.changed_asset_ids {
                self.remove_entry(asset_id);
            }
        }
        self.mesh_pool_generation = delta.current_generation;
    }

    /// Rolls back arena metadata recorded by a graph that never reached submit-resource retention.
    ///
    /// New core allocations are removed so the next population attempt records their position and
    /// index copies again. Existing committed cores survive (their dedicated sources may already
    /// have been released), while optional residency/source identities return to the last
    /// committed snapshot and are recopied from any currently available dedicated streams.
    fn rollback_unsubmitted_population(&mut self) {
        let uncommitted_core_asset_ids = self
            .entries
            .iter()
            .filter_map(|(&asset_id, entry)| (!entry.core_committed).then_some(asset_id))
            .collect::<Vec<_>>();
        for asset_id in uncommitted_core_asset_ids {
            self.remove_entry(asset_id);
        }
        for entry in self.entries.values_mut() {
            entry.rollback_unsubmitted_optional_streams();
        }
        self.rollback_unsubmitted_buffer_replacements();
        self.rollback_unsubmitted_compaction();
        self.source_release_pending.clear();
        // Every retained recording drains this list. Anything left here belongs exclusively to
        // the attempt just rolled back and is no longer referenced by executable command buffers.
        self.retired_buffers.clear();
    }

    /// Restores buffers replaced by copy commands that never reached submit-resource retention.
    ///
    /// Core allocators grew alongside their buffers, so rebuilding their free maps from the
    /// surviving committed entries is part of the same transaction. Optional streams share
    /// position offsets and therefore need only their previous handles restored.
    fn rollback_unsubmitted_buffer_replacements(&mut self) {
        let position_ranges = self
            .entries
            .values()
            .map(|entry| entry.allocation.vertices)
            .collect::<Vec<_>>();
        self.positions
            .rollback_unsubmitted_replacement(position_ranges);

        let index32_ranges = self
            .entries
            .values()
            .filter(|entry| !entry.allocation.narrow_indices)
            .map(|entry| entry.allocation.indices)
            .collect::<Vec<_>>();
        self.indices32
            .rollback_unsubmitted_replacement(index32_ranges);

        let index16_ranges = self
            .entries
            .values()
            .filter(|entry| entry.allocation.narrow_indices)
            .map(|entry| entry.allocation.indices)
            .collect::<Vec<_>>();
        self.indices16
            .rollback_unsubmitted_replacement(index16_ranges);

        for slot in 0..ArenaStream::COUNT {
            if let Some(original) = self.stream_rollback_buffers[slot].take() {
                self.streams[slot] = original;
            }
        }
    }

    /// Restores the last submitted arena generation after an unretained compacting rebuild.
    fn rollback_unsubmitted_compaction(&mut self) {
        let Some(checkpoint) = self.compaction_rollback.take() else {
            return;
        };
        self.positions = checkpoint.positions;
        self.indices32 = checkpoint.indices32;
        self.indices16 = checkpoint.indices16;
        self.streams = checkpoint.streams;
        self.entries = checkpoint.entries;
        self.reclamation_requested = true;
        // Never reuse the revision exposed by the aborted layout. A retained cache that observed
        // either side must rebuild against the restored offsets.
        self.bump_allocation_generation();
    }

    /// Copies a resident mesh's streams into the arena via `encoder`, caching the allocation.
    /// Idempotent per `asset_id`; a later call with additional streams copies the missing ones into
    /// the same allocation. Returns [`None`] when the position or index arena is full.
    pub(crate) fn ensure_mesh(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        asset_id: i32,
        sources: &MeshStreamSources<'_>,
    ) -> Option<GeometryAllocation> {
        if let Some(existing) = self.entries.get(&asset_id).copied() {
            let core_sources_match = existing.position_source == buffer_identity(sources.position)
                && existing.index_source == buffer_identity(sources.index)
                && existing.allocation.vertex_count == sources.vertex_count
                && existing.allocation.index_count == sources.index_count
                && existing.allocation.narrow_indices == sources.narrow;
            if core_sources_match {
                let updated = self.copy_optional_streams(
                    device,
                    encoder,
                    existing,
                    sources.vertex_count,
                    sources.optional,
                );
                let allocation = updated.allocation;
                if allocation.stream_bits != existing.allocation.stream_bits {
                    self.bump_allocation_generation();
                }
                self.entries.insert(asset_id, updated);
                return Some(allocation);
            }
            self.remove_entry(asset_id);
        }

        let position_bytes = u64::from(sources.vertex_count) * ARENA_POSITION_STRIDE;
        let index_bytes = u64::from(sources.index_count) * if sources.narrow { 2 } else { 4 };
        if position_bytes == 0 || index_bytes == 0 {
            return None;
        }
        // Never publish an allocation whose source cannot provide the complete declared payload.
        // The allocator pads ranges to 256 bytes, so a partial core copy would otherwise look
        // resident and an indirect draw could read unwritten data or the following mesh.
        if aligned_copy_size(position_bytes, sources.position.size()) < position_bytes
            || aligned_copy_size(index_bytes, sources.index.size()) < index_bytes
        {
            return None;
        }
        let placement = if sources.optional.is_empty() {
            ArenaPlacement::High
        } else {
            ArenaPlacement::Low
        };
        let vertices = allocate_growing(
            device,
            encoder,
            &mut self.positions,
            position_bytes,
            placement,
            self.max_buffer_size,
            &mut self.retired_buffers,
        )?;
        let index_range = {
            let index_arena = if sources.narrow {
                &mut self.indices16
            } else {
                &mut self.indices32
            };
            match allocate_growing(
                device,
                encoder,
                index_arena,
                index_bytes,
                ArenaPlacement::Low,
                self.max_buffer_size,
                &mut self.retired_buffers,
            ) {
                Some(range) => range,
                None => {
                    self.positions.alloc.free(vertices);
                    return None;
                }
            }
        };

        copy_buffer_aligned(
            encoder,
            sources.position,
            &self.positions.buffer,
            vertices.offset_bytes,
            position_bytes,
        );
        let index_dst = if sources.narrow {
            &self.indices16.buffer
        } else {
            &self.indices32.buffer
        };
        copy_buffer_aligned(
            encoder,
            sources.index,
            index_dst,
            index_range.offset_bytes,
            index_bytes,
        );
        // The backend may release the per-mesh source handles at the next frame boundary. Keep
        // explicit clones in this submit payload so deferred driver-thread submission cannot race
        // that release.
        self.retired_buffers.push(sources.position.clone());
        self.retired_buffers.push(sources.index.clone());

        let entry = GeometryEntry {
            allocation: GeometryAllocation {
                vertices,
                indices: index_range,
                vertex_count: sources.vertex_count,
                index_count: sources.index_count,
                narrow_indices: sources.narrow,
                stream_bits: 0,
            },
            position_source: buffer_identity(sources.position),
            index_source: buffer_identity(sources.index),
            stream_sources: [None; ArenaStream::COUNT],
            core_committed: false,
            committed_stream_bits: 0,
            committed_stream_sources: [None; ArenaStream::COUNT],
            refused_stream_bits: 0,
        };
        let entry = self.copy_optional_streams(
            device,
            encoder,
            entry,
            sources.vertex_count,
            sources.optional,
        );
        let allocation = entry.allocation;
        self.entries.insert(asset_id, entry);
        self.bump_allocation_generation();
        Some(allocation)
    }

    /// Copies newly materialized optional streams into an already resident mesh allocation.
    ///
    /// This path deliberately does not require the dedicated position/index sources: rigid static
    /// meshes release those buffers once the core allocation is authoritative in the arena.
    pub(crate) fn ensure_optional_streams(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        asset_id: i32,
        optional: &[(ArenaStream, &wgpu::Buffer)],
    ) -> Option<GeometryAllocation> {
        let entry = self.entries.get(&asset_id).copied()?;
        let vertex_count = entry.allocation.vertex_count;
        let updated = self.copy_optional_streams(device, encoder, entry, vertex_count, optional);
        let allocation = updated.allocation;
        if allocation.stream_bits != entry.allocation.stream_bits {
            self.bump_allocation_generation();
        }
        self.entries.insert(asset_id, updated);
        Some(allocation)
    }

    /// Copies any optional streams present in `sources` but not yet resident for `alloc`, creating
    /// the target stream buffer lazily. Returns the allocation with an updated residency mask.
    fn copy_optional_streams(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mut entry: GeometryEntry,
        vertex_count: u32,
        optional: &[(ArenaStream, &wgpu::Buffer)],
    ) -> GeometryEntry {
        let alloc = &mut entry.allocation;
        let base_vertex = u64::from(alloc.base_vertex().max(0) as u32);
        let vertex_count = u64::from(vertex_count);
        for &(stream, src) in optional {
            let slot = stream.slot();
            let source_identity = buffer_identity(src);
            if alloc.stream_bits & stream.bit() != 0
                && entry.stream_sources[slot] == Some(source_identity)
            {
                continue;
            }
            let stride = stream.stride();
            let dst_offset = base_vertex * stride;
            let payload_bytes = vertex_count * stride;
            let copy_bytes = aligned_copy_size(payload_bytes, src.size());
            if copy_bytes < payload_bytes {
                // A partial stream cannot be advertised as arena-resident: consumers address the
                // complete vertex range, and a later GPU-only compaction has no CPU source from
                // which to repair missing bytes.
                continue;
            }
            let required_size = dst_offset.saturating_add(copy_bytes);
            let Some(dst) =
                self.ensure_stream_buffer(device, encoder, stream, required_size, payload_bytes)
            else {
                // The mesh keeps drawing this stream from its own buffer. Recording the demand
                // moves it into the low region on the next packed rebuild, where the copy fits.
                entry.refused_stream_bits |= stream.bit();
                continue;
            };
            copy_buffer_aligned(encoder, src, dst, base_vertex * stride, payload_bytes);
            self.retired_buffers.push(src.clone());
            alloc.stream_bits |= stream.bit();
            entry.refused_stream_bits &= !stream.bit();
            entry.stream_sources[slot] = Some(source_identity);
            if dst_offset >= OPTIONAL_STREAM_SPARSE_OFFSET_BYTES
                && dst_offset >= copy_bytes.saturating_mul(2)
            {
                // Reorder stream-bearing meshes toward the front on the next frame, after this
                // copy has reached submit-resource retention.
                self.reclamation_requested = true;
            }
        }
        entry
    }

    /// Lazily creates `stream`'s buffer and grows it through `required_size`.
    ///
    /// `payload_bytes` is what this copy actually writes. Streams are addressed by position base
    /// vertex, so a mesh sitting high in the position arena can require a buffer far larger than
    /// its own payload; growth that lopsided is refused and the mesh keeps drawing that stream from
    /// its own buffer.
    fn ensure_stream_buffer(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        stream: ArenaStream,
        required_size: u64,
        payload_bytes: u64,
    ) -> Option<&wgpu::Buffer> {
        if required_size > self.max_buffer_size {
            return None;
        }
        let slot = stream.slot();
        let denied = self.stream_growth_denied_bytes[slot];
        if denied != 0 && required_size >= denied {
            return None;
        }
        let growth = required_size
            .saturating_sub(self.streams[slot].as_ref().map_or(0, wgpu::Buffer::size));
        if growth > MAX_UNJUSTIFIED_STREAM_GROWTH_BYTES
            && payload_bytes.saturating_mul(MIN_STREAM_GROWTH_PAYLOAD_SHARE) < growth
        {
            self.refuse_sparse_stream_growth(slot);
            return None;
        }
        if self.streams[slot].is_none() {
            let capacity = initial_stream_capacity(required_size, self.max_buffer_size)?;
            let Some(buffer) = create_stream_buffer(device, capacity, stream.label()) else {
                self.deny_stream_growth(slot, capacity);
                return None;
            };
            if self.stream_rollback_buffers[slot].is_none() {
                self.stream_rollback_buffers[slot] = Some(None);
            }
            self.streams[slot] = Some(buffer);
        }
        let current_size = self.streams[slot].as_ref()?.size();
        if required_size > current_size {
            let new_size = grow_capacity(current_size, required_size, self.max_buffer_size)?;
            let Some(replacement) = create_stream_buffer(device, new_size, stream.label()) else {
                // The existing buffer stays authoritative for the meshes already resident in it.
                self.deny_stream_growth(slot, new_size);
                return None;
            };
            if self.stream_rollback_buffers[slot].is_none() {
                self.stream_rollback_buffers[slot] = Some(self.streams[slot].clone());
            }
            encoder.copy_buffer_to_buffer(
                self.streams[slot].as_ref()?,
                0,
                &replacement,
                0,
                current_size,
            );
            if let Some(previous) = self.streams[slot].replace(replacement) {
                self.retired_buffers.push(previous);
            }
        }
        self.streams[slot].as_ref()
    }

    /// Records a refused stream capacity and asks for a packed rebuild that can lower it.
    fn deny_stream_growth(&mut self, slot: usize, capacity: u64) {
        let denied = &mut self.stream_growth_denied_bytes[slot];
        *denied = if *denied == 0 {
            capacity
        } else {
            (*denied).min(capacity)
        };
        self.reclamation_requested = true;
    }

    /// Notes a stream copy refused for sparsity, requesting one packed rebuild per slot.
    ///
    /// The refusal recomputes from live offsets on every attempt, so the request is latched per
    /// slot: without that, an unmovable mesh would re-plan a rebuild every frame.
    fn refuse_sparse_stream_growth(&mut self, slot: usize) {
        if self.stream_sparse_refused[slot] {
            return;
        }
        self.stream_sparse_refused[slot] = true;
        self.reclamation_requested = true;
    }

    fn remove_entry(&mut self, asset_id: i32) {
        self.source_release_ready.remove(&asset_id);
        self.source_release_pending.remove(&asset_id);
        let Some(entry) = self.entries.remove(&asset_id) else {
            return;
        };
        self.positions.alloc.free(entry.allocation.vertices);
        if entry.allocation.narrow_indices {
            self.indices16.alloc.free(entry.allocation.indices);
        } else {
            self.indices32.alloc.free(entry.allocation.indices);
        }
        // A freed range can lower what the next optional stream needs, so retry refused sizes.
        self.stream_growth_denied_bytes = [0; ArenaStream::COUNT];
        self.stream_sparse_refused = [false; ArenaStream::COUNT];
        self.reclamation_requested = true;
        self.bump_allocation_generation();
    }

    #[inline]
    fn bump_allocation_generation(&mut self) {
        self.allocation_generation = self.allocation_generation.wrapping_add(1);
    }
}

/// Copies `bytes` from `src` offset 0 into `dst` at `dst_offset`, rounding the size up to the copy
/// alignment (source stream buffers upload padded to the same alignment) and clamping to the source
/// size so the copy stays valid for odd `u16` index counts and short streams.
fn copy_buffer_aligned(
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    dst_offset: u64,
    bytes: u64,
) {
    let size = aligned_copy_size(bytes, src.size());
    if size == 0 {
        return;
    }
    encoder.copy_buffer_to_buffer(src, 0, dst, dst_offset, size);
}

fn aligned_copy_size(bytes: u64, source_size: u64) -> u64 {
    let align = wgpu::COPY_BUFFER_ALIGNMENT - 1;
    ((bytes + align) & !align).min(source_size & !align)
}

fn buffer_identity(buffer: &wgpu::Buffer) -> usize {
    let pointer: *const wgpu::Buffer = buffer;
    pointer as usize
}

#[inline]
fn mesh_pool_synchronization_needed(synchronized_generation: u64, current_generation: u64) -> bool {
    synchronized_generation != current_generation
}

/// Plans one deterministic packed layout without consulting CPU upload sources.
///
/// Entries with wider optional layouts are placed first. Since optional streams share position
/// base vertices, this bounds rare wide-stream buffers by the meshes that actually use them instead
/// of every core-only mesh allocated earlier in the arena's lifetime.
fn build_compaction_plan(
    entries: &HashMap<i32, GeometryEntry>,
    maximum: u64,
) -> Option<GeometryCompactionPlan> {
    let mut ordered = entries
        .iter()
        .map(|(&asset_id, &entry)| (asset_id, entry))
        .collect::<Vec<_>>();
    ordered.sort_unstable_by(|(left_id, left), (right_id, right)| {
        optional_layout_weight(*right)
            .cmp(&optional_layout_weight(*left))
            .then_with(|| {
                right
                    .allocation
                    .vertex_count
                    .cmp(&left.allocation.vertex_count)
            })
            .then_with(|| left_id.cmp(right_id))
    });

    let position_required = ordered.iter().try_fold(0u64, |total, (_, entry)| {
        total.checked_add(entry.allocation.vertices.len_bytes)
    })?;
    let index16_required = ordered
        .iter()
        .filter(|(_, entry)| entry.allocation.narrow_indices)
        .try_fold(0u64, |total, (_, entry)| {
            total.checked_add(entry.allocation.indices.len_bytes)
        })?;
    let index32_required = ordered
        .iter()
        .filter(|(_, entry)| !entry.allocation.narrow_indices)
        .try_fold(0u64, |total, (_, entry)| {
            total.checked_add(entry.allocation.indices.len_bytes)
        })?;
    if position_required > maximum || index16_required > maximum || index32_required > maximum {
        return None;
    }

    let position_capacity =
        initial_arena_capacity(position_required, MIN_VERTEX_ARENA_BYTES, maximum, true);
    let index16_capacity =
        initial_arena_capacity(index16_required, MIN_INDEX_ARENA_BYTES, maximum, true);
    let index32_capacity =
        initial_arena_capacity(index32_required, MIN_INDEX_ARENA_BYTES, maximum, true);
    let mut position_alloc = RangeAllocator::new(position_capacity, GEOMETRY_ARENA_ALIGN);
    let mut index16_alloc = RangeAllocator::new(index16_capacity, GEOMETRY_ARENA_ALIGN);
    let mut index32_alloc = RangeAllocator::new(index32_capacity, GEOMETRY_ARENA_ALIGN);
    let mut stream_required = [0u64; ArenaStream::COUNT];
    let mut rows = Vec::with_capacity(ordered.len());
    let mut packed_entries = HashMap::with_capacity(ordered.len());
    let mut copy_bytes = 0u64;

    for (asset_id, mut entry) in ordered {
        let old = entry.allocation;
        // Keep the packed layout split the same way live allocation splits it: streams low, core
        // only meshes at the top. Packing everything densely from zero would put the next
        // stream-bearing mesh above every core-only one and reopen the sparse-stream problem.
        let placement = if optional_layout_weight(entry) == 0 {
            ArenaPlacement::High
        } else {
            ArenaPlacement::Low
        };
        let vertices = allocate_placed(&mut position_alloc, old.vertices.len_bytes, placement)?;
        let indices = if old.narrow_indices {
            index16_alloc.allocate(old.indices.len_bytes)?
        } else {
            index32_alloc.allocate(old.indices.len_bytes)?
        };
        let new = GeometryAllocation {
            vertices,
            indices,
            vertex_count: old.vertex_count,
            index_count: old.index_count,
            narrow_indices: old.narrow_indices,
            stream_bits: old.stream_bits,
        };
        let base_vertex = vertices.offset_bytes / ARENA_POSITION_STRIDE;
        for stream in ArenaStream::ALL {
            if !old.has_stream(stream) {
                continue;
            }
            let payload = u64::from(old.vertex_count).checked_mul(stream.stride())?;
            let required = base_vertex
                .checked_add(u64::from(old.vertex_count))?
                .checked_mul(stream.stride())?;
            stream_required[stream.slot()] = stream_required[stream.slot()].max(required);
            copy_bytes = copy_bytes.checked_add(payload)?;
        }
        copy_bytes = copy_bytes
            .checked_add(old.vertices.len_bytes)?
            .checked_add(old.indices.len_bytes)?;
        entry.allocation = new;
        rows.push(GeometryCompactionRow { old, new });
        packed_entries.insert(asset_id, entry);
    }

    let mut stream_capacities = [0u64; ArenaStream::COUNT];
    for stream in ArenaStream::ALL {
        let required = stream_required[stream.slot()];
        if required == 0 {
            continue;
        }
        if required > maximum {
            return None;
        }
        stream_capacities[stream.slot()] =
            initial_arena_capacity(required, MIN_STREAM_ARENA_BYTES, maximum, false);
    }

    Some(GeometryCompactionPlan {
        rows,
        entries: packed_entries,
        position_capacity,
        index32_capacity,
        index16_capacity,
        stream_capacities,
        copy_bytes,
    })
}

/// Total stride of the streams a mesh needs in the arena, resident or refused for sparsity.
fn optional_layout_weight(entry: GeometryEntry) -> u64 {
    let needed = entry.allocation.stream_bits | entry.refused_stream_bits;
    ArenaStream::ALL
        .into_iter()
        .filter(|&stream| needed & stream.bit() != 0)
        .map(ArenaStream::stride)
        .sum()
}

/// Requires both an absolute saving and a 25% capacity reduction before compaction.
fn compaction_is_worthwhile(old_allocated: u64, new_allocated: u64) -> bool {
    let reclaimed = old_allocated.saturating_sub(new_allocated);
    reclaimed >= MIN_COMPACTION_RECLAIM_BYTES && reclaimed >= old_allocated.saturating_add(3) / 4
}

/// Reconstructs an allocator with `live_ranges` reserved at their existing offsets.
///
/// Rebuilds the free map by reserving sorted gaps and live ranges, then freeing the gaps.
fn rebuild_range_allocator(capacity: u64, mut live_ranges: Vec<Range>) -> Option<RangeAllocator> {
    live_ranges.sort_unstable_by_key(|range| range.offset_bytes);
    let mut alloc = RangeAllocator::new(capacity, GEOMETRY_ARENA_ALIGN);
    let mut reserved_gaps = Vec::new();
    let mut cursor = 0u64;

    for range in live_ranges {
        let end = range.offset_bytes.checked_add(range.len_bytes)?;
        if range.len_bytes == 0
            || range.offset_bytes < cursor
            || end > capacity
            || range.offset_bytes % GEOMETRY_ARENA_ALIGN != 0
            || range.len_bytes % GEOMETRY_ARENA_ALIGN != 0
        {
            return None;
        }
        if range.offset_bytes > cursor {
            let gap = alloc.allocate(range.offset_bytes - cursor)?;
            if gap.offset_bytes != cursor || gap.len_bytes != range.offset_bytes - cursor {
                return None;
            }
            reserved_gaps.push(gap);
        }
        let reserved = alloc.allocate(range.len_bytes)?;
        if reserved != range {
            return None;
        }
        cursor = end;
    }

    for gap in reserved_gaps {
        alloc.free(gap);
    }
    Some(alloc)
}

/// Reconstructs the three packed core allocators from the allocation map that will be published.
///
/// Destination buffers are created with completely free allocators. Compaction must reserve every
/// copied live range before those buffers become authoritative; otherwise the first later mesh
/// allocation can overwrite a resident packed mesh at offset zero.
fn rebuild_compacted_core_allocators(
    entries: &HashMap<i32, GeometryEntry>,
    position_capacity: u64,
    index32_capacity: u64,
    index16_capacity: u64,
) -> Option<(RangeAllocator, RangeAllocator, RangeAllocator)> {
    let positions = rebuild_range_allocator(
        position_capacity,
        entries
            .values()
            .map(|entry| entry.allocation.vertices)
            .collect(),
    )?;
    let indices32 = rebuild_range_allocator(
        index32_capacity,
        entries
            .values()
            .filter(|entry| !entry.allocation.narrow_indices)
            .map(|entry| entry.allocation.indices)
            .collect(),
    )?;
    let indices16 = rebuild_range_allocator(
        index16_capacity,
        entries
            .values()
            .filter(|entry| entry.allocation.narrow_indices)
            .map(|entry| entry.allocation.indices)
            .collect(),
    )?;
    Some((positions, indices32, indices16))
}

fn allocate_growing(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    arena: &mut GrowableGeometryBuffer,
    bytes: u64,
    placement: ArenaPlacement,
    max_buffer_size: u64,
    retired_buffers: &mut Vec<wgpu::Buffer>,
) -> Option<Range> {
    if let Some(range) = allocate_placed(&mut arena.alloc, bytes, placement) {
        return Some(range);
    }

    let required_capacity = arena
        .alloc
        .capacity()
        .saturating_add(align_up(bytes, GEOMETRY_ARENA_ALIGN));
    let new_capacity = grow_capacity(arena.buffer.size(), required_capacity, max_buffer_size)?;
    // Growth that the device refuses leaves the current buffer authoritative and fails this one
    // allocation, so the mesh keeps its dedicated buffers for the frame.
    let replacement = create_arena_buffer(device, arena.label, new_capacity, arena.usage)?;
    encoder.copy_buffer_to_buffer(&arena.buffer, 0, &replacement, 0, arena.buffer.size());
    if arena.rollback_buffer.is_none() {
        arena.rollback_buffer = Some(arena.buffer.clone());
    }
    let previous = std::mem::replace(&mut arena.buffer, replacement);
    retired_buffers.push(previous);
    arena.alloc.grow_to(new_capacity);
    allocate_placed(&mut arena.alloc, bytes, placement)
}

fn allocate_placed(
    alloc: &mut RangeAllocator,
    bytes: u64,
    placement: ArenaPlacement,
) -> Option<Range> {
    match placement {
        ArenaPlacement::Low => alloc.allocate(bytes),
        ArenaPlacement::High => alloc.allocate_high(bytes),
    }
}

fn grow_capacity(current: u64, required: u64, maximum: u64) -> Option<u64> {
    if required > maximum {
        return None;
    }
    let mut capacity = current.max(GEOMETRY_ARENA_ALIGN);
    while capacity < required {
        let doubled = capacity.saturating_mul(2).min(maximum);
        if doubled == capacity {
            return None;
        }
        capacity = doubled;
    }
    Some(capacity)
}

fn initial_stream_capacity(required: u64, maximum: u64) -> Option<u64> {
    let floor = MIN_STREAM_ARENA_BYTES
        .min(maximum)
        .max(GEOMETRY_ARENA_ALIGN);
    grow_capacity(floor, required.max(GEOMETRY_ARENA_ALIGN), maximum)
}

/// Initial arena sizing with optional 12.5% growth headroom for the core streams.
///
/// Optional streams use their exact modelled high-water mark because a wide-UV stream can be four
/// times larger than positions and speculative headroom there is expensive.
fn initial_arena_capacity(required: u64, floor: u64, maximum: u64, headroom: bool) -> u64 {
    let floor = floor.min(maximum).max(GEOMETRY_ARENA_ALIGN);
    let required = if headroom {
        required.saturating_add(required / 8)
    } else {
        required
    };
    align_up(required.max(floor), GEOMETRY_ARENA_ALIGN).min(maximum)
}

fn align_up(bytes: u64, alignment: u64) -> u64 {
    bytes
        .saturating_add(alignment.saturating_sub(1))
        .checked_div(alignment)
        .unwrap_or(0)
        .saturating_mul(alignment)
}

fn create_stream_buffer(
    device: &wgpu::Device,
    capacity: u64,
    label: &'static str,
) -> Option<wgpu::Buffer> {
    create_arena_buffer(
        device,
        label,
        capacity.max(GEOMETRY_ARENA_ALIGN),
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )
}

/// Creates one core arena, falling back to `floor` bytes before reporting failure.
fn create_core_arena(
    device: &wgpu::Device,
    capacity: u64,
    floor: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
) -> Option<GrowableGeometryBuffer> {
    if let Some(arena) = GrowableGeometryBuffer::try_new(device, capacity, usage, label) {
        return Some(arena);
    }
    if floor >= capacity {
        return None;
    }
    GrowableGeometryBuffer::try_new(device, floor, usage, label)
}

/// Creates one arena buffer, returning `None` when the device refuses the allocation.
///
/// An out-of-memory `create_buffer` still hands back a handle, and every later copy, bind, and
/// submit against it is a validation error that takes the frame down with it. Catching the refusal
/// at the allocation site keeps the invalid handle out of the arena entirely.
fn create_arena_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> Option<wgpu::Buffer> {
    let out_of_memory_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    });
    let internal_error = pollster::block_on(internal_scope.pop());
    let out_of_memory_error = pollster::block_on(out_of_memory_scope.pop());
    if let Some(error) = internal_error.or(out_of_memory_error) {
        static ALLOCATION_FAILURE_LOG: std::sync::LazyLock<
            crate::diagnostics::log_once::KeyedLogOnce<&'static str>,
        > = std::sync::LazyLock::new(crate::diagnostics::log_once::KeyedLogOnce::new);
        if ALLOCATION_FAILURE_LOG.should_log(label) {
            logger::warn!(
                "geometry arena could not allocate {label} at {size} bytes: {error}; \
                 affected meshes stay on dedicated buffers"
            );
        }
        return None;
    }
    crate::profiling::note_resource_churn!(Buffer, "gpu_pools::geometry_arena");
    Some(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_vertex_and_first_index_convert_by_width() {
        let wide = GeometryAllocation {
            vertices: Range {
                offset_bytes: 1024,
                len_bytes: 512,
            },
            indices: Range {
                offset_bytes: 2048,
                len_bytes: 256,
            },
            vertex_count: 32,
            index_count: 60,
            narrow_indices: false,
            stream_bits: 0,
        };
        assert_eq!(wide.base_vertex(), 64); // 1024 / 16
        assert_eq!(wide.first_index_base(), 512); // 2048 / 4

        let narrow = GeometryAllocation {
            vertices: Range {
                offset_bytes: 0,
                len_bytes: 512,
            },
            indices: Range {
                offset_bytes: 2048,
                len_bytes: 256,
            },
            vertex_count: 32,
            index_count: 60,
            narrow_indices: true,
            stream_bits: 0,
        };
        assert_eq!(narrow.first_index_base(), 1024); // 2048 / 2
        assert_eq!(narrow.index_size(), 2);
    }

    #[test]
    fn indexed_draw_span_is_limited_to_copied_payload_not_padded_allocation() {
        let alloc = GeometryAllocation {
            vertices: Range {
                offset_bytes: 0,
                len_bytes: 256,
            },
            // The allocator rounds this up to 256 bytes, but only 12 u32 indices are valid.
            indices: Range {
                offset_bytes: 0,
                len_bytes: 256,
            },
            vertex_count: 8,
            index_count: 12,
            narrow_indices: false,
            stream_bits: 0,
        };

        assert!(alloc.contains_index_span(0, 12));
        assert!(alloc.contains_index_span(3, 9));
        assert!(!alloc.contains_index_span(3, 10));
        assert!(!alloc.contains_index_span(u32::MAX, 2));
    }

    #[test]
    fn aligned_copy_never_returns_an_invalid_partial_alignment() {
        assert_eq!(aligned_copy_size(6, 8), 8);
        assert_eq!(aligned_copy_size(6, 6), 4);
        assert_eq!(aligned_copy_size(16, 15), 12);
        assert_eq!(aligned_copy_size(16, 16), 16);
    }

    #[test]
    fn stream_residency_bits_are_distinct() {
        let mut alloc = GeometryAllocation {
            vertices: Range {
                offset_bytes: 0,
                len_bytes: 512,
            },
            indices: Range {
                offset_bytes: 0,
                len_bytes: 256,
            },
            vertex_count: 32,
            index_count: 60,
            narrow_indices: false,
            stream_bits: 0,
        };
        for stream in ArenaStream::ALL {
            assert!(!alloc.has_stream(stream));
        }
        alloc.stream_bits |= ArenaStream::Tangent.bit();
        assert!(alloc.has_stream(ArenaStream::Tangent));
        assert!(!alloc.has_stream(ArenaStream::Normal));
        let mask = alloc.derived_stream_mask();
        assert!(mask.contains(MeshDerivedStreamMask::POSITION));
        assert!(mask.contains(MeshDerivedStreamMask::TANGENT));
        assert!(!mask.contains(MeshDerivedStreamMask::NORMAL));
    }

    #[test]
    fn entry_commit_captures_recorded_core_and_optional_state() {
        let mut stream_sources = [None; ArenaStream::COUNT];
        stream_sources[ArenaStream::Tangent.slot()] = Some(41);
        let mut entry = GeometryEntry {
            allocation: GeometryAllocation {
                vertices: Range {
                    offset_bytes: 0,
                    len_bytes: 256,
                },
                indices: Range {
                    offset_bytes: 0,
                    len_bytes: 256,
                },
                vertex_count: 16,
                index_count: 24,
                narrow_indices: false,
                stream_bits: ArenaStream::Tangent.bit(),
            },
            position_source: 1,
            index_source: 2,
            stream_sources,
            core_committed: false,
            committed_stream_bits: 0,
            committed_stream_sources: [None; ArenaStream::COUNT],
            refused_stream_bits: 0,
        };

        entry.commit_recorded_state();

        assert!(entry.core_committed);
        assert_eq!(entry.committed_stream_bits, ArenaStream::Tangent.bit());
        assert_eq!(
            entry.committed_stream_sources[ArenaStream::Tangent.slot()],
            Some(41)
        );
    }

    #[test]
    fn entry_rollback_restores_last_committed_optional_state() {
        let committed_bits = ArenaStream::Normal.bit();
        let mut committed_sources = [None; ArenaStream::COUNT];
        committed_sources[ArenaStream::Normal.slot()] = Some(7);
        let mut staged_sources = committed_sources;
        staged_sources[ArenaStream::Tangent.slot()] = Some(99);
        let mut entry = GeometryEntry {
            allocation: GeometryAllocation {
                vertices: Range {
                    offset_bytes: 0,
                    len_bytes: 256,
                },
                indices: Range {
                    offset_bytes: 0,
                    len_bytes: 256,
                },
                vertex_count: 16,
                index_count: 24,
                narrow_indices: false,
                stream_bits: committed_bits | ArenaStream::Tangent.bit(),
            },
            position_source: 1,
            index_source: 2,
            stream_sources: staged_sources,
            core_committed: true,
            committed_stream_bits: committed_bits,
            committed_stream_sources: committed_sources,
            refused_stream_bits: 0,
        };

        entry.rollback_unsubmitted_optional_streams();

        assert_eq!(entry.allocation.stream_bits, committed_bits);
        assert_eq!(entry.stream_sources, committed_sources);
    }

    #[test]
    fn growth_doubles_and_respects_device_limit() {
        assert_eq!(grow_capacity(256, 257, 1024), Some(512));
        assert_eq!(grow_capacity(256, 900, 1024), Some(1024));
        assert_eq!(grow_capacity(256, 1025, 1024), None);
    }

    #[test]
    fn rollback_allocator_rebuild_preserves_committed_offsets_and_holes() {
        let mut alloc = rebuild_range_allocator(
            2048,
            vec![
                Range {
                    offset_bytes: 256,
                    len_bytes: 256,
                },
                Range {
                    offset_bytes: 1024,
                    len_bytes: 512,
                },
            ],
        )
        .expect("valid committed ranges");

        assert_eq!(alloc.capacity(), 2048);
        assert_eq!(alloc.used_bytes(), 768);
        assert_eq!(
            alloc.allocate(512),
            Some(Range {
                offset_bytes: 512,
                len_bytes: 512,
            })
        );
        assert_eq!(
            alloc.allocate(256),
            Some(Range {
                offset_bytes: 0,
                len_bytes: 256,
            })
        );
        assert_eq!(
            alloc.allocate(512),
            Some(Range {
                offset_bytes: 1536,
                len_bytes: 512,
            })
        );
        assert_eq!(alloc.allocate(256), None);
    }

    #[test]
    fn rollback_allocator_rebuild_rejects_ranges_outside_original_buffer() {
        assert!(
            rebuild_range_allocator(
                1024,
                vec![Range {
                    offset_bytes: 1024,
                    len_bytes: 256,
                }]
            )
            .is_none()
        );
        assert!(
            rebuild_range_allocator(
                1024,
                vec![
                    Range {
                        offset_bytes: 256,
                        len_bytes: 512,
                    },
                    Range {
                        offset_bytes: 512,
                        len_bytes: 256,
                    },
                ]
            )
            .is_none()
        );
    }

    #[test]
    fn optional_streams_start_small_and_grow_to_required_offset() {
        assert_eq!(
            initial_stream_capacity(256, 8 * 1024 * 1024),
            Some(MIN_STREAM_ARENA_BYTES)
        );
        assert_eq!(
            initial_stream_capacity(MIN_STREAM_ARENA_BYTES + 1, 8 * 1024 * 1024),
            Some(2 * MIN_STREAM_ARENA_BYTES)
        );
        assert_eq!(
            initial_stream_capacity(256, 512 * 1024),
            Some(MIN_STREAM_ARENA_BYTES)
        );
        assert_eq!(initial_stream_capacity(1025, 1024), None);
    }

    #[test]
    fn capacity_hint_models_aligned_mesh_offsets_and_sparse_streams() {
        let mut hint = GeometryArenaCapacityHint::default();
        hint.include_mesh(3, 3, true, [ArenaStream::Normal, ArenaStream::WideLow]);
        hint.include_mesh(4, 6, false, [ArenaStream::Uv0]);

        assert_eq!(hint.position_bytes, 512);
        assert_eq!(hint.index16_bytes, 256);
        assert_eq!(hint.index32_bytes, 256);
        assert_eq!(hint.stream_bytes[ArenaStream::Normal.slot()], 48);
        assert_eq!(hint.stream_bytes[ArenaStream::WideLow.slot()], 192);
        // The second mesh starts at position byte 256, or base vertex 16.
        assert_eq!(hint.stream_bytes[ArenaStream::Uv0.slot()], 20 * 8);
    }

    #[test]
    fn initial_capacity_uses_actual_demand_instead_of_a_fixed_large_reservation() {
        assert_eq!(
            initial_arena_capacity(0, MIN_VERTEX_ARENA_BYTES, 64 * 1024 * 1024, true),
            MIN_VERTEX_ARENA_BYTES
        );
        assert_eq!(
            initial_arena_capacity(
                8 * 1024 * 1024,
                MIN_VERTEX_ARENA_BYTES,
                64 * 1024 * 1024,
                true,
            ),
            9 * 1024 * 1024
        );
        assert_eq!(
            initial_arena_capacity(
                80 * 1024 * 1024,
                MIN_VERTEX_ARENA_BYTES,
                64 * 1024 * 1024,
                true,
            ),
            64 * 1024 * 1024
        );
    }

    fn committed_entry(
        vertex_offset: u64,
        vertex_count: u32,
        narrow_indices: bool,
        stream_bits: u16,
    ) -> GeometryEntry {
        GeometryEntry {
            allocation: GeometryAllocation {
                vertices: Range {
                    offset_bytes: vertex_offset,
                    len_bytes: GEOMETRY_ARENA_ALIGN,
                },
                indices: Range {
                    offset_bytes: vertex_offset,
                    len_bytes: GEOMETRY_ARENA_ALIGN,
                },
                vertex_count,
                index_count: u32::try_from(GEOMETRY_ARENA_ALIGN / 4).unwrap(),
                narrow_indices,
                stream_bits,
            },
            position_source: 1,
            index_source: 2,
            stream_sources: [None; ArenaStream::COUNT],
            core_committed: true,
            committed_stream_bits: stream_bits,
            committed_stream_sources: [None; ArenaStream::COUNT],
            refused_stream_bits: 0,
        }
    }

    #[test]
    fn compaction_places_rare_wide_streams_before_core_only_meshes() {
        let mut entries = HashMap::new();
        for asset_id in 0..4 {
            entries.insert(
                asset_id,
                committed_entry(
                    u64::try_from(asset_id).unwrap_or(0) * 16 * 1024 * 1024,
                    16,
                    false,
                    0,
                ),
            );
        }
        entries.insert(
            99,
            committed_entry(64 * 1024 * 1024, 16, false, ArenaStream::WideHigh.bit()),
        );

        let plan =
            build_compaction_plan(&entries, 256 * 1024 * 1024).expect("valid compacted layout");
        let wide = plan.entries.get(&99).expect("wide-stream entry");
        assert_eq!(wide.allocation.vertices.offset_bytes, 0);
        assert_eq!(
            plan.stream_capacities[ArenaStream::WideHigh.slot()],
            MIN_STREAM_ARENA_BYTES
        );
        assert_eq!(plan.rows.len(), entries.len());
    }

    #[test]
    fn compaction_moves_a_refused_stream_mesh_into_the_low_region() {
        let mut refused = committed_entry(64 * 1024 * 1024, 16, false, 0);
        refused.refused_stream_bits = ArenaStream::Color.bit();
        let mut entries = HashMap::new();
        for asset_id in 0..4 {
            entries.insert(
                asset_id,
                committed_entry(u64::try_from(asset_id).unwrap_or(0) * 16 * 1024 * 1024, 16, false, 0),
            );
        }
        entries.insert(99, refused);

        let plan =
            build_compaction_plan(&entries, 256 * 1024 * 1024).expect("valid compacted layout");

        assert_eq!(
            plan.entries
                .get(&99)
                .expect("refused entry")
                .allocation
                .vertices
                .offset_bytes,
            0
        );
    }

    #[test]
    fn compaction_plan_preserves_formats_residency_and_payload_size() {
        let normal = ArenaStream::Normal.bit();
        let uv = ArenaStream::Uv0.bit();
        let entries = HashMap::from([
            (10, committed_entry(4096, 8, true, normal | uv)),
            (20, committed_entry(8192, 4, false, 0)),
        ]);

        let plan =
            build_compaction_plan(&entries, 64 * 1024 * 1024).expect("valid compacted layout");
        let narrow = plan.entries.get(&10).expect("narrow entry").allocation;
        let wide = plan.entries.get(&20).expect("wide entry").allocation;
        assert!(narrow.narrow_indices);
        assert!(!wide.narrow_indices);
        assert!(narrow.has_stream(ArenaStream::Normal));
        assert!(narrow.has_stream(ArenaStream::Uv0));
        assert_eq!(narrow.vertices.offset_bytes, 0);
        // The core-only mesh takes the top so the stream-bearing mesh keeps the low region.
        assert_eq!(
            wide.vertices.offset_bytes,
            plan.position_capacity - GEOMETRY_ARENA_ALIGN
        );
        assert_eq!(
            plan.copy_bytes,
            // Two aligned core ranges per mesh plus the narrow mesh's normal and UV payloads.
            4 * GEOMETRY_ARENA_ALIGN + 8 * (16 + 8)
        );
    }

    #[test]
    fn compacted_allocators_reserve_every_published_live_range() {
        let entries = HashMap::from([
            (
                10,
                committed_entry(4096, 8, true, ArenaStream::Normal.bit()),
            ),
            (20, committed_entry(8192, 4, false, 0)),
            (30, committed_entry(12288, 6, true, 0)),
        ]);
        let plan =
            build_compaction_plan(&entries, 64 * 1024 * 1024).expect("valid compacted layout");
        let (mut positions, mut indices32, mut indices16) = rebuild_compacted_core_allocators(
            &plan.entries,
            plan.position_capacity,
            plan.index32_capacity,
            plan.index16_capacity,
        )
        .expect("packed live ranges fit their replacement buffers");

        let live_position_bytes = plan
            .entries
            .values()
            .map(|entry| entry.allocation.vertices.len_bytes)
            .sum::<u64>();
        let live_index32_bytes = plan
            .entries
            .values()
            .filter(|entry| !entry.allocation.narrow_indices)
            .map(|entry| entry.allocation.indices.len_bytes)
            .sum::<u64>();
        let live_index16_bytes = plan
            .entries
            .values()
            .filter(|entry| entry.allocation.narrow_indices)
            .map(|entry| entry.allocation.indices.len_bytes)
            .sum::<u64>();

        assert_eq!(positions.used_bytes(), live_position_bytes);
        assert_eq!(indices32.used_bytes(), live_index32_bytes);
        assert_eq!(indices16.used_bytes(), live_index16_bytes);
        // Positions split by stream residency, so fresh headroom is not simply the packed tail.
        // What must hold is that it never lands on a live range.
        let next_position = positions
            .allocate(GEOMETRY_ARENA_ALIGN)
            .expect("position headroom");
        assert!(
            plan.entries.values().all(|entry| {
                let live = entry.allocation.vertices;
                next_position.offset_bytes >= live.offset_bytes + live.len_bytes
                    || next_position.offset_bytes + next_position.len_bytes <= live.offset_bytes
            }),
            "fresh position allocation overlapped a packed live range"
        );
        assert_eq!(
            indices32
                .allocate(GEOMETRY_ARENA_ALIGN)
                .expect("u32 index headroom")
                .offset_bytes,
            live_index32_bytes
        );
        assert_eq!(
            indices16
                .allocate(GEOMETRY_ARENA_ALIGN)
                .expect("u16 index headroom")
                .offset_bytes,
            live_index16_bytes
        );
    }

    #[test]
    fn compaction_requires_material_absolute_and_fractional_savings() {
        assert!(!compaction_is_worthwhile(
            64 * 1024 * 1024,
            60 * 1024 * 1024
        ));
        assert!(!compaction_is_worthwhile(
            64 * 1024 * 1024,
            49 * 1024 * 1024
        ));
        assert!(compaction_is_worthwhile(64 * 1024 * 1024, 48 * 1024 * 1024));
        assert!(compaction_is_worthwhile(
            512 * 1024 * 1024,
            128 * 1024 * 1024
        ));
    }

    #[test]
    fn mesh_pool_generation_drift_wakes_zero_draw_population() {
        assert!(!mesh_pool_synchronization_needed(41, 41));
        assert!(mesh_pool_synchronization_needed(41, 42));
        // A pool/device reset also requires the conservative synchronization path.
        assert!(mesh_pool_synchronization_needed(41, 0));
    }
}
