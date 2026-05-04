//! Mesh packed-buffer layout matching `Renderite.Shared.MeshBuffer.ComputeBufferLayout`.
//!
//! Regions: vertices -> indices -> bone_counts -> bone_weights -> bind_poses -> blendshape_data.

use crate::shared::{
    BlendshapeBufferDescriptor, IndexBufferFormat, SubmeshBufferDescriptor,
    VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType,
};
use hashbrown::{HashMap, HashSet};

/// Bytes per sparse position entry on the GPU: `vertex_index: u32` + `delta.xyz: f32`.
pub const BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE: usize = 16;

/// Bytes per sparse packed normal or tangent entry: `vertex_index: u32` + three snorm16 channels.
pub const BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_SIZE: usize = 12;

/// Number of `u32` words per sparse position entry in the GPU buffer.
pub const BLENDSHAPE_POSITION_SPARSE_ENTRY_WORDS: u32 = 4;

/// Number of `u32` words per sparse packed normal or tangent entry in the GPU buffer.
pub const BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS: u32 = 3;

/// Packed normal and tangent deltas are clamped to this absolute component range.
pub const BLENDSHAPE_PACKED_VECTOR_DELTA_RANGE: f32 = 2.0;

/// Deltas smaller than this magnitude (length squared) are dropped as non-influencing.
pub const BLENDSHAPE_DELTA_EPSILON_SQ: f32 = 1e-14;

fn vertex_format_size(format: VertexAttributeFormat) -> i32 {
    match format {
        VertexAttributeFormat::Float32 => 4,
        VertexAttributeFormat::Half16 => 2,
        VertexAttributeFormat::UNorm8 => 1,
        VertexAttributeFormat::UNorm16 => 2,
        VertexAttributeFormat::SInt8 => 1,
        VertexAttributeFormat::SInt16 => 2,
        VertexAttributeFormat::SInt32 => 4,
        VertexAttributeFormat::UInt8 => 1,
        VertexAttributeFormat::UInt16 => 2,
        VertexAttributeFormat::UInt32 => 4,
    }
}

/// Interleaved vertex stride from [`VertexAttributeDescriptor`] list (host order).
pub fn compute_vertex_stride(attrs: &[VertexAttributeDescriptor]) -> i32 {
    attrs
        .iter()
        .map(|a| vertex_format_size(a.format) * a.dimensions)
        .sum()
}

/// Total index count from submeshes (`max(index_start + index_count)`).
pub fn compute_index_count(submeshes: &[SubmeshBufferDescriptor]) -> i32 {
    submeshes
        .iter()
        .map(|s| s.index_start + s.index_count)
        .max()
        .unwrap_or(0)
}

/// Bytes per index for [`IndexBufferFormat`].
pub fn index_bytes_per_element(format: IndexBufferFormat) -> i32 {
    match format {
        IndexBufferFormat::UInt16 => 2,
        IndexBufferFormat::UInt32 => 4,
    }
}

/// Maximum allowed mesh buffer size in bytes (`MeshBuffer.MAX_BUFFER_SIZE`).
pub const MAX_BUFFER_SIZE: usize = 2_147_483_648;

/// Byte offsets for each region of the host mesh payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshBufferLayout {
    /// Byte length of the interleaved vertex region at the start of the host buffer.
    pub vertex_size: usize,
    /// Byte offset where the index buffer begins.
    pub index_buffer_start: usize,
    /// Byte length of the index buffer region.
    pub index_buffer_length: usize,
    /// Byte offset where optional per-vertex bone count bytes begin.
    pub bone_counts_start: usize,
    /// Byte length of the bone counts region (or zero).
    pub bone_counts_length: usize,
    /// Byte offset where packed bone weight tail data begins.
    pub bone_weights_start: usize,
    /// Byte length of the bone weights region (or zero).
    pub bone_weights_length: usize,
    /// Byte offset where inverse bind-pose matrices begin.
    pub bind_poses_start: usize,
    /// Byte length of bind pose data (or zero).
    pub bind_poses_length: usize,
    /// Byte offset where packed blendshape delta payload begins.
    pub blendshape_data_start: usize,
    /// Byte length of blendshape payload (or zero).
    pub blendshape_data_length: usize,
    /// Total bytes required to cover all regions (validation vs mapped SHM).
    pub total_buffer_length: usize,
}

fn compute_blendshape_data_length(
    blendshape_buffers: &[BlendshapeBufferDescriptor],
    vertex_count: i32,
) -> usize {
    let vertex_count = vertex_count.max(0) as usize;
    let bytes_per_channel = 12 * vertex_count;
    blendshape_buffers
        .iter()
        .map(|d| {
            let mut len = 0;
            if d.data_flags.positions() {
                len += bytes_per_channel;
            }
            if d.data_flags.normals() {
                len += bytes_per_channel;
            }
            if d.data_flags.tangets() {
                len += bytes_per_channel;
            }
            len
        })
        .sum()
}

/// Computes layout per `MeshBuffer.ComputeBufferLayout`.
pub fn compute_mesh_buffer_layout(
    vertex_stride: i32,
    vertex_count: i32,
    index_count: i32,
    index_bytes: i32,
    bone_count: i32,
    bone_weight_count: i32,
    blendshape_buffers: Option<&[BlendshapeBufferDescriptor]>,
) -> Result<MeshBufferLayout, &'static str> {
    let vertex_stride = vertex_stride.max(0) as usize;
    let vertex_count = vertex_count.max(0) as usize;
    let index_count = index_count.max(0) as usize;
    let index_bytes = index_bytes.max(0) as usize;
    let bone_count = bone_count.max(0) as usize;
    let bone_weight_count = bone_weight_count.max(0) as usize;

    let vertex_size = vertex_stride
        .checked_mul(vertex_count)
        .ok_or("Mesh buffer size overflow")?;
    let index_buffer_length = index_count
        .checked_mul(index_bytes)
        .ok_or("Mesh buffer size overflow")?;
    let index_buffer_start = vertex_size;
    let bone_counts_start = index_buffer_start + index_buffer_length;
    let bone_counts_length = vertex_count;
    let bone_weights_start = bone_counts_start + bone_counts_length;
    let bone_weights_length = bone_weight_count
        .checked_mul(8)
        .ok_or("Mesh buffer size overflow")?;
    let bind_poses_start = bone_weights_start + bone_weights_length;
    let bind_poses_length = bone_count
        .checked_mul(64)
        .ok_or("Mesh buffer size overflow")?;
    let blendshape_data_start = bind_poses_start + bind_poses_length;
    let blendshape_data_length = blendshape_buffers.map_or(0, |b| {
        compute_blendshape_data_length(b, vertex_count as i32)
    });
    let total_buffer_length = blendshape_data_start + blendshape_data_length;

    if total_buffer_length > MAX_BUFFER_SIZE {
        return Err("Mesh buffer size exceeds maximum allowed size of 2 GB.");
    }

    Ok(MeshBufferLayout {
        vertex_size,
        index_buffer_start,
        index_buffer_length,
        bone_counts_start,
        bone_counts_length,
        bone_weights_start,
        bone_weights_length,
        bind_poses_start,
        bind_poses_length,
        blendshape_data_start,
        blendshape_data_length,
        total_buffer_length,
    })
}

/// Extracts bind pose matrices from raw bytes (64 bytes per matrix).
pub fn extract_bind_poses(raw: &[u8], bone_count: usize) -> Option<Vec<[[f32; 4]; 4]>> {
    const MATRIX_BYTES: usize = 64;
    let need = bone_count.checked_mul(MATRIX_BYTES)?;
    if raw.len() < need {
        return None;
    }
    let mut poses = Vec::with_capacity(bone_count);
    for i in 0..bone_count {
        let start = i * MATRIX_BYTES;
        let slice = &raw[start..start + MATRIX_BYTES];
        poses.push(bytemuck::pod_read_unaligned(slice));
    }
    Some(poses)
}

/// GPU-ready channel-sparse blendshape deltas and CPU scatter ranges.
pub struct BlendshapeGpuPack {
    /// Tightly packed `u32` words containing position, normal, and tangent sparse sections.
    pub sparse_deltas: Vec<u8>,
    /// Per-frame sparse ranges sorted by shape and frame weight.
    pub frame_ranges: Vec<BlendshapeFrameRange>,
    /// Per-shape spans into [`Self::frame_ranges`].
    pub shape_frame_spans: Vec<BlendshapeFrameSpan>,
    /// Logical blendshape slot count (`max(blendshape_index) + 1`).
    pub num_blendshapes: i32,
    /// Whether any sparse row carries a nonzero position delta.
    pub has_position_deltas: bool,
    /// Whether any sparse row carries a nonzero normal delta.
    pub has_normal_deltas: bool,
    /// Whether any sparse row carries a nonzero tangent delta.
    pub has_tangent_deltas: bool,
    /// Whether any packed normal or tangent component was clamped to the supported delta range.
    pub clamped_packed_deltas: bool,
}

/// Sparse range and metadata for one Unity blendshape frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendshapeFrameRange {
    /// Logical blendshape index from [`BlendshapeBufferDescriptor::blendshape_index`].
    pub shape_index: u32,
    /// Host frame index from [`BlendshapeBufferDescriptor::frame_index`].
    pub frame_index: i32,
    /// Unity frame weight from [`BlendshapeBufferDescriptor::frame_weight`].
    pub frame_weight: f32,
    /// First `u32` word of this frame's position entries in [`BlendshapeGpuPack::sparse_deltas`].
    pub position_first_word: u32,
    /// Number of sparse position entries in this frame.
    pub position_count: u32,
    /// First `u32` word of this frame's packed normal entries in [`BlendshapeGpuPack::sparse_deltas`].
    pub normal_first_word: u32,
    /// Number of sparse packed normal entries in this frame.
    pub normal_count: u32,
    /// First `u32` word of this frame's packed tangent entries in [`BlendshapeGpuPack::sparse_deltas`].
    pub tangent_first_word: u32,
    /// Number of sparse packed tangent entries in this frame.
    pub tangent_count: u32,
}

/// Span of frame rows belonging to one logical blendshape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlendshapeFrameSpan {
    /// First row in [`BlendshapeGpuPack::frame_ranges`].
    pub first_frame: u32,
    /// Number of rows for this logical shape.
    pub frame_count: u32,
}

/// Weighted contribution for one frame range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendshapeFrameCoefficient {
    /// Index into the frame range slice passed to [`select_blendshape_frame_coefficients`].
    pub frame_range_index: usize,
    /// Interpolated multiplier applied to the frame delta.
    pub effective_weight: f32,
}

/// Mutable extraction accumulator for one blendshape frame.
#[derive(Clone, Debug)]
struct PendingBlendshapeFrame {
    /// Logical blendshape index.
    shape_index: u32,
    /// Host frame index.
    frame_index: i32,
    /// Unity frame weight.
    frame_weight: f32,
    /// Nonzero per-vertex deltas in this frame, keyed by vertex index.
    entries: HashMap<u32, PendingBlendshapeDelta>,
}

/// One sparse vertex delta row before deterministic sorting and byte packing.
#[derive(Clone, Copy, Debug, Default)]
struct PendingBlendshapeDelta {
    position: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 3],
}

impl PendingBlendshapeDelta {
    fn has_any_channel(self) -> bool {
        vector_has_nonzero_delta(self.position)
            || vector_has_nonzero_delta(self.normal)
            || vector_has_nonzero_delta(self.tangent)
    }
}

/// Blendshape delta stream channel carried by a host descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BlendshapeDeltaChannel {
    Position,
    Normal,
    Tangent,
}

impl BlendshapeDeltaChannel {
    fn label(self) -> &'static str {
        match self {
            Self::Position => "position",
            Self::Normal => "normal",
            Self::Tangent => "tangent",
        }
    }

    fn set_delta(self, row: &mut PendingBlendshapeDelta, delta: [f32; 3]) {
        match self {
            Self::Position => row.position = delta,
            Self::Normal => row.normal = delta,
            Self::Tangent => row.tangent = delta,
        }
    }
}

/// Returns whether a coefficient is finite and nonzero enough to dispatch.
fn coefficient_is_active(weight: f32) -> bool {
    weight.is_finite() && weight != 0.0
}

/// Adds one frame coefficient when the frame has sparse entries.
fn maybe_frame_coefficient(
    frame_range_index: usize,
    effective_weight: f32,
    range: &BlendshapeFrameRange,
) -> Option<BlendshapeFrameCoefficient> {
    if !frame_range_has_entries(range) || !coefficient_is_active(effective_weight) {
        return None;
    }
    Some(BlendshapeFrameCoefficient {
        frame_range_index,
        effective_weight,
    })
}

fn frame_range_has_entries(range: &BlendshapeFrameRange) -> bool {
    range.position_count != 0 || range.normal_count != 0 || range.tangent_count != 0
}

/// Selects up to two sparse frame ranges for a Unity blendshape runtime weight.
pub fn select_blendshape_frame_coefficients(
    shape_index: u32,
    weight: f32,
    shape_frame_spans: &[BlendshapeFrameSpan],
    frame_ranges: &[BlendshapeFrameRange],
) -> [Option<BlendshapeFrameCoefficient>; 2] {
    if !coefficient_is_active(weight) {
        return [None, None];
    }
    let Some(span) = shape_frame_spans.get(shape_index as usize).copied() else {
        return [None, None];
    };
    let first = span.first_frame as usize;
    let count = span.frame_count as usize;
    let Some(end) = first.checked_add(count) else {
        return [None, None];
    };
    let Some(frames) = frame_ranges.get(first..end) else {
        return [None, None];
    };
    let valid_frame_count = frames
        .iter()
        .filter(|range| range.frame_weight.is_finite())
        .count();
    if valid_frame_count == 0 {
        return [None, None];
    }
    if valid_frame_count == 1 {
        let Some((local_index, range)) = frames
            .iter()
            .enumerate()
            .find(|(_, range)| range.frame_weight.is_finite() && range.frame_weight != 0.0)
        else {
            return [None, None];
        };
        return [
            maybe_frame_coefficient(first + local_index, weight / range.frame_weight, range),
            None,
        ];
    }

    let Some((lo_local, hi_local)) = select_frame_segment(frames, weight) else {
        return [None, None];
    };
    let lo = &frames[lo_local];
    let hi = &frames[hi_local];
    let denom = hi.frame_weight - lo.frame_weight;
    if !denom.is_finite() || denom == 0.0 {
        return [None, None];
    }
    let t = (weight - lo.frame_weight) / denom;
    if !t.is_finite() {
        return [None, None];
    }
    [
        maybe_frame_coefficient(first + lo_local, 1.0 - t, lo),
        maybe_frame_coefficient(first + hi_local, t, hi),
    ]
}

/// Chooses the sorted frame segment that surrounds or nearest-extrapolates `weight`.
fn select_frame_segment(frames: &[BlendshapeFrameRange], weight: f32) -> Option<(usize, usize)> {
    let mut previous_valid = None;
    let mut penultimate_valid = None;
    for (index, range) in frames.iter().enumerate() {
        if !range.frame_weight.is_finite() {
            continue;
        }
        let Some(previous) = previous_valid else {
            previous_valid = Some(index);
            continue;
        };
        if weight <= frames[index].frame_weight {
            return Some((previous, index));
        }
        penultimate_valid = Some(previous);
        previous_valid = Some(index);
    }
    Some((penultimate_valid?, previous_valid?))
}

/// Returns whether any runtime blendshape weight selects a nonempty sparse frame range.
pub fn blendshape_deform_is_active(
    num_blendshapes: u32,
    shape_frame_spans: &[BlendshapeFrameSpan],
    frame_ranges: &[BlendshapeFrameRange],
    blend_weights: &[f32],
) -> bool {
    if num_blendshapes == 0
        || shape_frame_spans.len() != num_blendshapes as usize
        || frame_ranges.is_empty()
    {
        return false;
    }
    (0..num_blendshapes).any(|shape_index| {
        let weight = blend_weights
            .get(shape_index as usize)
            .copied()
            .unwrap_or(0.0);
        select_blendshape_frame_coefficients(shape_index, weight, shape_frame_spans, frame_ranges)
            .into_iter()
            .flatten()
            .any(|coefficient| coefficient_is_active(coefficient.effective_weight))
    })
}

/// Computes the logical blendshape slot count from descriptor indices.
fn blendshape_slot_count(blendshape_buffers: &[BlendshapeBufferDescriptor]) -> Option<usize> {
    const MAX_BLENDSHAPES: usize = 4096;
    let num_blendshapes = blendshape_buffers
        .iter()
        .map(|d| d.blendshape_index.max(0) + 1)
        .max()
        .unwrap_or(0) as usize;
    if num_blendshapes == 0 {
        return None;
    }
    if num_blendshapes > MAX_BLENDSHAPES {
        logger::warn!(
            "extract_blendshape_offsets: num_blendshapes={num_blendshapes} exceeds cap {MAX_BLENDSHAPES}"
        );
        return None;
    }
    Some(num_blendshapes)
}

/// Returns whether a vector delta is large enough to influence a sparse row.
fn vector_has_nonzero_delta(delta: [f32; 3]) -> bool {
    let [x, y, z] = delta;
    let mag_sq = z.mul_add(z, x.mul_add(x, y * y));
    mag_sq > BLENDSHAPE_DELTA_EPSILON_SQ
}

/// Reads one descriptor channel into sparse pending entries.
fn read_pending_channel_entries(
    raw: &[u8],
    byte_offset: usize,
    vertex_count: usize,
    duplicate_frame: bool,
) -> Option<Vec<(u32, [f32; 3])>> {
    const VECTOR3_BYTES: usize = 12;
    let chunk_len = VECTOR3_BYTES * vertex_count;
    if byte_offset + chunk_len > raw.len() {
        return None;
    }
    let mut entries = Vec::new();
    for v in 0..vertex_count {
        let src_offset = byte_offset + v * VECTOR3_BYTES;
        let x = f32::from_le_bytes(raw[src_offset..src_offset + 4].try_into().ok()?);
        let y = f32::from_le_bytes(raw[src_offset + 4..src_offset + 8].try_into().ok()?);
        let z = f32::from_le_bytes(raw[src_offset + 8..src_offset + 12].try_into().ok()?);
        let delta = [x, y, z];
        if !duplicate_frame && vector_has_nonzero_delta(delta) {
            entries.push((v as u32, delta));
        }
    }
    Some(entries)
}

/// Returns the mutable frame accumulator for `descriptor`, creating it if this is the first channel
/// observed for that shape/frame pair.
fn pending_frame_for_descriptor<'a>(
    per_shape: &'a mut [Vec<PendingBlendshapeFrame>],
    shape_index: usize,
    descriptor: &BlendshapeBufferDescriptor,
) -> Option<&'a mut PendingBlendshapeFrame> {
    let frames = per_shape.get_mut(shape_index)?;
    let frame_index = descriptor.frame_index;
    let index = match frames
        .iter()
        .position(|frame| frame.frame_index == frame_index)
    {
        Some(index) => index,
        None => {
            frames.push(PendingBlendshapeFrame {
                shape_index: shape_index as u32,
                frame_index,
                frame_weight: descriptor.frame_weight,
                entries: HashMap::new(),
            });
            frames.len() - 1
        }
    };
    frames.get_mut(index)
}

/// Merges one descriptor channel into the shape/frame sparse accumulator.
fn merge_pending_channel_entries(
    frame: &mut PendingBlendshapeFrame,
    channel: BlendshapeDeltaChannel,
    entries: Vec<(u32, [f32; 3])>,
) {
    for (vertex_index, delta) in entries {
        channel.set_delta(frame.entries.entry(vertex_index).or_default(), delta);
    }
}

/// Extracts descriptor streams into per-shape pending blendshape frames.
fn collect_pending_blendshape_frames(
    raw: &[u8],
    layout: &MeshBufferLayout,
    blendshape_buffers: &[BlendshapeBufferDescriptor],
    vertex_count: usize,
    num_blendshapes: usize,
) -> Option<Vec<Vec<PendingBlendshapeFrame>>> {
    const VECTOR3_BYTES: usize = 12;
    let mut per_shape: Vec<Vec<PendingBlendshapeFrame>> = Vec::with_capacity(num_blendshapes);
    per_shape.resize_with(num_blendshapes, Vec::new);
    let mut seen_channels: Vec<HashSet<(i32, BlendshapeDeltaChannel)>> =
        Vec::with_capacity(num_blendshapes);
    seen_channels.resize_with(num_blendshapes, HashSet::new);
    let mut byte_offset = layout.blendshape_data_start;

    for descriptor in blendshape_buffers {
        let bi = descriptor.blendshape_index.max(0) as usize;
        if bi >= num_blendshapes {
            continue;
        }
        for (has_channel, channel) in [
            (
                descriptor.data_flags.positions(),
                BlendshapeDeltaChannel::Position,
            ),
            (
                descriptor.data_flags.normals(),
                BlendshapeDeltaChannel::Normal,
            ),
            (
                descriptor.data_flags.tangets(),
                BlendshapeDeltaChannel::Tangent,
            ),
        ] {
            if !has_channel {
                continue;
            }
            let chunk_len = VECTOR3_BYTES * vertex_count;
            let duplicate_frame = !seen_channels[bi].insert((descriptor.frame_index, channel));
            if duplicate_frame {
                logger::warn!(
                    "extract_blendshape_offsets: duplicate {} frame shape={} frame={} skipped",
                    channel.label(),
                    descriptor.blendshape_index,
                    descriptor.frame_index
                );
            }
            let entries =
                read_pending_channel_entries(raw, byte_offset, vertex_count, duplicate_frame)?;
            if !duplicate_frame {
                let frame = pending_frame_for_descriptor(&mut per_shape, bi, descriptor)?;
                merge_pending_channel_entries(frame, channel, entries);
            }
            byte_offset += chunk_len;
        }
    }
    Some(per_shape)
}

/// Converts pending frames into the packed sparse byte blob and frame spans.
fn build_blendshape_gpu_pack(
    mut per_shape: Vec<Vec<PendingBlendshapeFrame>>,
    num_blendshapes: usize,
) -> BlendshapeGpuPack {
    let mut sparse_deltas = Vec::new();
    let frame_count: usize = per_shape.iter().map(Vec::len).sum();
    let mut frame_ranges = Vec::with_capacity(frame_count);
    let mut shape_frame_spans = vec![BlendshapeFrameSpan::default(); num_blendshapes];
    let mut has_position_deltas = false;
    let mut has_normal_deltas = false;
    let mut has_tangent_deltas = false;
    let mut clamped_packed_deltas = false;

    for (s, frames) in per_shape.iter_mut().enumerate() {
        frames.sort_by(|a, b| {
            a.frame_weight
                .total_cmp(&b.frame_weight)
                .then(a.frame_index.cmp(&b.frame_index))
        });
        let first_frame = frame_ranges.len() as u32;
        append_sorted_pending_frames(
            frames,
            &mut sparse_deltas,
            &mut frame_ranges,
            &mut has_position_deltas,
            &mut has_normal_deltas,
            &mut has_tangent_deltas,
            &mut clamped_packed_deltas,
        );
        shape_frame_spans[s] = BlendshapeFrameSpan {
            first_frame,
            frame_count: frame_ranges.len() as u32 - first_frame,
        };
    }

    BlendshapeGpuPack {
        sparse_deltas,
        frame_ranges,
        shape_frame_spans,
        num_blendshapes: num_blendshapes as i32,
        has_position_deltas,
        has_normal_deltas,
        has_tangent_deltas,
        clamped_packed_deltas,
    }
}

/// Appends sorted pending frames to the sparse byte blob and frame metadata.
fn append_sorted_pending_frames(
    frames: &[PendingBlendshapeFrame],
    sparse_deltas: &mut Vec<u8>,
    frame_ranges: &mut Vec<BlendshapeFrameRange>,
    has_position_deltas: &mut bool,
    has_normal_deltas: &mut bool,
    has_tangent_deltas: &mut bool,
    clamped_packed_deltas: &mut bool,
) {
    for frame in frames {
        let mut entries: Vec<(u32, PendingBlendshapeDelta)> = frame
            .entries
            .iter()
            .filter_map(|(&vertex_index, &delta)| {
                delta.has_any_channel().then_some((vertex_index, delta))
            })
            .collect();
        entries.sort_by_key(|(vertex_index, _)| *vertex_index);
        let position_first_word = sparse_word_len(sparse_deltas);
        let mut position_count = 0;
        for (vi, delta) in entries.iter().filter_map(|(vi, delta)| {
            vector_has_nonzero_delta(delta.position).then_some((*vi, *delta))
        }) {
            *has_position_deltas = true;
            append_position_sparse_entry(sparse_deltas, vi, delta.position);
            position_count += 1;
        }

        let normal_first_word = sparse_word_len(sparse_deltas);
        let mut normal_count = 0;
        for (vi, delta) in entries.iter().filter_map(|(vi, delta)| {
            vector_has_nonzero_delta(delta.normal).then_some((*vi, *delta))
        }) {
            *has_normal_deltas = true;
            *clamped_packed_deltas |=
                append_packed_vector_sparse_entry(sparse_deltas, vi, delta.normal);
            normal_count += 1;
        }

        let tangent_first_word = sparse_word_len(sparse_deltas);
        let mut tangent_count = 0;
        for (vi, delta) in entries.iter().filter_map(|(vi, delta)| {
            vector_has_nonzero_delta(delta.tangent).then_some((*vi, *delta))
        }) {
            *has_tangent_deltas = true;
            *clamped_packed_deltas |=
                append_packed_vector_sparse_entry(sparse_deltas, vi, delta.tangent);
            tangent_count += 1;
        }

        frame_ranges.push(BlendshapeFrameRange {
            shape_index: frame.shape_index,
            frame_index: frame.frame_index,
            frame_weight: frame.frame_weight,
            position_first_word,
            position_count,
            normal_first_word,
            normal_count,
            tangent_first_word,
            tangent_count,
        });
    }
}

fn sparse_word_len(sparse_deltas: &[u8]) -> u32 {
    (sparse_deltas.len() / size_of::<u32>()) as u32
}

fn append_position_sparse_entry(sparse_deltas: &mut Vec<u8>, vertex_index: u32, delta: [f32; 3]) {
    sparse_deltas.extend_from_slice(&vertex_index.to_le_bytes());
    for component in delta {
        sparse_deltas.extend_from_slice(&component.to_le_bytes());
    }
}

fn append_packed_vector_sparse_entry(
    sparse_deltas: &mut Vec<u8>,
    vertex_index: u32,
    delta: [f32; 3],
) -> bool {
    let (x, x_clamped) = pack_snorm16_delta_component(delta[0]);
    let (y, y_clamped) = pack_snorm16_delta_component(delta[1]);
    let (z, z_clamped) = pack_snorm16_delta_component(delta[2]);
    let xy = u32::from(x) | (u32::from(y) << 16);
    let z_word = u32::from(z);
    sparse_deltas.extend_from_slice(&vertex_index.to_le_bytes());
    sparse_deltas.extend_from_slice(&xy.to_le_bytes());
    sparse_deltas.extend_from_slice(&z_word.to_le_bytes());
    x_clamped || y_clamped || z_clamped
}

fn pack_snorm16_delta_component(component: f32) -> (u16, bool) {
    let finite = component.is_finite();
    let input = if finite { component } else { 0.0 };
    let clamped = input.clamp(
        -BLENDSHAPE_PACKED_VECTOR_DELTA_RANGE,
        BLENDSHAPE_PACKED_VECTOR_DELTA_RANGE,
    );
    let scaled = (clamped / BLENDSHAPE_PACKED_VECTOR_DELTA_RANGE * 32767.0).round();
    let signed = scaled.clamp(-32767.0, 32767.0) as i16;
    (signed as u16, !finite || clamped != input)
}

/// Repacks host blendshape position, normal, and tangent deltas into frame-aware sparse GPU storage.
///
/// Position, normal, and tangent deltas are encoded as separate sparse channel ranges so empty
/// channels and vertices do not allocate GPU rows.
pub fn extract_blendshape_offsets(
    raw: &[u8],
    layout: &MeshBufferLayout,
    blendshape_buffers: &[BlendshapeBufferDescriptor],
    vertex_count: i32,
) -> Option<BlendshapeGpuPack> {
    if blendshape_buffers.is_empty() || vertex_count <= 0 {
        return None;
    }
    let vertex_count = vertex_count as usize;
    let num_blendshapes = blendshape_slot_count(blendshape_buffers)?;
    let required_len = layout.blendshape_data_start + layout.blendshape_data_length;
    if raw.len() < required_len {
        return None;
    }
    let per_shape = collect_pending_blendshape_frames(
        raw,
        layout,
        blendshape_buffers,
        vertex_count,
        num_blendshapes,
    )?;
    Some(build_blendshape_gpu_pack(per_shape, num_blendshapes))
}

/// Returns byte offset and size of the first attribute of `target` type in the interleaved vertex.
pub fn attribute_offset_and_size(
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
) -> Option<(usize, usize)> {
    let mut offset: i32 = 0;
    for a in attrs {
        let size = (vertex_format_size(a.format) * a.dimensions) as usize;
        if (a.attribute as i16) == (target as i16) {
            return Some((offset as usize, size));
        }
        offset += size as i32;
    }
    None
}

/// Extracts a float3 position stream and a normal stream from interleaved vertices into dense
/// `vec4<f32>` storage (16 bytes each per vertex).
///
/// Position must be at least three-component `float32`. Normal is allowed to be absent or
/// unsupported; in that case a stable +Z normal is synthesized so UI meshes that do not upload
/// normals still satisfy the shared raster vertex layout.
pub fn extract_float3_position_normal_as_vec4_streams(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<(Vec<u8>, Vec<u8>)> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }
    let pos = attribute_offset_and_size(attrs, VertexAttributeType::Position)?;
    let pos_attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (VertexAttributeType::Position as i16))?;
    if pos_attr.format != VertexAttributeFormat::Float32 || pos_attr.dimensions < 3 {
        return None;
    }
    if pos.1 < 12 {
        return None;
    }

    let mut pos_out = vec![0u8; vertex_count * 16];
    let mut nrm_out = vec![0u8; vertex_count * 16];
    let one = 1.0f32.to_le_bytes();
    fill_normal_stream_with_forward_z(&mut nrm_out);

    let nrm = attribute_offset_and_size(attrs, VertexAttributeType::Normal);
    let nrm_attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (VertexAttributeType::Normal as i16));
    let nrm_offset = if matches!(
        (nrm, nrm_attr),
        (Some((_, sz)), Some(attr))
            if attr.format == VertexAttributeFormat::Float32 && attr.dimensions >= 3 && sz >= 12
    ) {
        nrm.map(|(off, _)| off)
    } else {
        None
    };

    for i in 0..vertex_count {
        let base = i * stride;
        let p0 = base + pos.0;
        if p0 + 12 > vertex_data.len() {
            return None;
        }
        let po = i * 16;
        pos_out[po..po + 12].copy_from_slice(&vertex_data[p0..p0 + 12]);
        pos_out[po + 12..po + 16].copy_from_slice(&one);

        if let Some(nrm_offset) = nrm_offset {
            let n0 = base + nrm_offset;
            if n0 + 12 > vertex_data.len() {
                return None;
            }
            let no = i * 16;
            nrm_out[no..no + 12].copy_from_slice(&vertex_data[n0..n0 + 12]);
        }
    }
    Some((pos_out, nrm_out))
}

fn fill_normal_stream_with_forward_z(out: &mut [u8]) {
    let zero = 0.0f32.to_le_bytes();
    let one = 1.0f32.to_le_bytes();
    for chunk in out.chunks_exact_mut(16) {
        chunk[0..4].copy_from_slice(&zero);
        chunk[4..8].copy_from_slice(&zero);
        chunk[8..12].copy_from_slice(&one);
        chunk[12..16].copy_from_slice(&zero);
    }
}

/// Dense `vec2<f32>` UV stream (`8` bytes per vertex) for embedded materials (e.g. world Unlit).
///
/// When [`VertexAttributeType::UV0`] is missing or not `float32`x2, returns **zeros** so a vertex buffer
/// slot can always be bound.
pub fn uv0_float2_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<Vec<u8>> {
    vertex_float2_stream_bytes(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::UV0,
    )
}

/// Dense `vec2<f32>` vertex stream for an arbitrary float2 attribute.
///
/// Missing or unsupported attributes return zeros so optional embedded shader streams can still
/// bind a stable vertex buffer slot.
pub fn vertex_float2_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
) -> Option<Vec<u8>> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }
    let mut out = vec![0u8; vertex_count * 8];
    let Some((off, sz)) = attribute_offset_and_size(attrs, target) else {
        return Some(out);
    };
    let attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (target as i16))?;
    if attr.format != VertexAttributeFormat::Float32 || attr.dimensions < 2 {
        return Some(out);
    }
    if sz < 8 {
        return Some(out);
    }
    for i in 0..vertex_count {
        let base = i * stride + off;
        if base + 8 > vertex_data.len() {
            return None;
        }
        let o = i * 8;
        out[o..o + 8].copy_from_slice(&vertex_data[base..base + 8]);
    }
    Some(out)
}

/// Dense `vec4<f32>` vertex stream for an arbitrary float attribute.
///
/// Missing or unsupported attributes return `default` per vertex.
#[cfg(test)]
pub fn vertex_float4_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
    default: [f32; 4],
) -> Option<Vec<u8>> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }
    let mut out = vec![0u8; vertex_count * 16];
    for chunk in out.chunks_exact_mut(16) {
        for (component, value) in default.iter().enumerate() {
            let o = component * 4;
            chunk[o..o + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    let Some((off, sz)) = attribute_offset_and_size(attrs, target) else {
        return Some(out);
    };
    let attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (target as i16))?;
    if attr.format != VertexAttributeFormat::Float32 || attr.dimensions < 1 {
        return Some(out);
    }
    let dims = attr.dimensions.clamp(1, 4) as usize;
    if sz < dims * 4 {
        return Some(out);
    }
    for i in 0..vertex_count {
        let base = i * stride + off;
        if base + dims * 4 > vertex_data.len() {
            return None;
        }
        let o = i * 16;
        for c in 0..dims {
            let src = base + c * 4;
            out[o + c * 4..o + c * 4 + 4].copy_from_slice(&vertex_data[src..src + 4]);
        }
    }

    Some(out)
}

/// Dense `vec4<f32>` color stream (`16` bytes per vertex) for UI / text embedded materials.
///
/// Missing or unsupported color attributes default to opaque white so non-colored meshes keep
/// rendering correctly while UI meshes can consume the host color stream when present.
pub fn color_float4_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<Vec<u8>> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }
    let mut out = vec![0u8; vertex_count * 16];
    fill_color_stream_with_white(&mut out);

    let Some((off, _sz)) = attribute_offset_and_size(attrs, VertexAttributeType::Color) else {
        return Some(out);
    };
    let color_attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (VertexAttributeType::Color as i16))?;

    for i in 0..vertex_count {
        let base = i * stride + off;
        if base >= vertex_data.len() {
            return None;
        }
        let Some(rgba) = decode_vertex_color(vertex_data, base, *color_attr) else {
            return Some(out);
        };
        let o = i * 16;
        for (component, value) in rgba.into_iter().enumerate() {
            out[o + component * 4..o + component * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    Some(out)
}

fn fill_color_stream_with_white(out: &mut [u8]) {
    let one = 1.0f32.to_le_bytes();
    for chunk in out.chunks_exact_mut(16) {
        chunk[0..4].copy_from_slice(&one);
        chunk[4..8].copy_from_slice(&one);
        chunk[8..12].copy_from_slice(&one);
        chunk[12..16].copy_from_slice(&one);
    }
}

fn decode_vertex_color(
    vertex_data: &[u8],
    base: usize,
    attr: VertexAttributeDescriptor,
) -> Option<[f32; 4]> {
    let dims = attr.dimensions.clamp(1, 4) as usize;
    let mut rgba = [1.0f32; 4];
    match attr.format {
        VertexAttributeFormat::UNorm8 | VertexAttributeFormat::UInt8 => {
            let end = base.checked_add(dims)?;
            let src = vertex_data.get(base..end)?;
            for (i, byte) in src.iter().take(dims).enumerate() {
                rgba[i] = f32::from(*byte) / 255.0;
            }
        }
        VertexAttributeFormat::UNorm16 | VertexAttributeFormat::UInt16 => {
            let end = base.checked_add(dims.checked_mul(2)?)?;
            let src = vertex_data.get(base..end)?;
            for (i, chunk) in src.chunks(2).take(dims).enumerate() {
                rgba[i] = f32::from(u16::from_le_bytes(chunk.try_into().ok()?)) / 65535.0;
            }
        }
        VertexAttributeFormat::Float32 => {
            let end = base.checked_add(dims.checked_mul(4)?)?;
            let src = vertex_data.get(base..end)?;
            for (i, chunk) in src.chunks(4).take(dims).enumerate() {
                rgba[i] = f32::from_le_bytes(chunk.try_into().ok()?);
            }
        }
        _ => return None,
    }
    Some(rgba)
}

/// Splits the mesh tail `bone_weights` region into GPU storage buffers for the skinning shader:
/// `array<vec4<u32>>` joint indices and `array<vec4<f32>>` weights per vertex.
///
/// Supports either **4 influences** (`32 * vertex_count` bytes as `(f32 weight, i32 index)` tuples)
/// or **1 influence** (`8 * vertex_count` bytes).
pub fn split_bone_weights_tail_for_gpu(
    bone_weights_tail: &[u8],
    vertex_count: usize,
) -> Option<(Vec<u8>, Vec<u8>)> {
    if vertex_count == 0 {
        return None;
    }
    let four_inf = vertex_count * 32;
    let one_inf = vertex_count * 8;
    let span = if bone_weights_tail.len() >= four_inf {
        4usize
    } else if bone_weights_tail.len() >= one_inf {
        1usize
    } else {
        return None;
    };

    let mut idx_bytes = vec![0u8; vertex_count * 16];
    let mut wt_bytes = vec![0u8; vertex_count * 16];

    for v in 0..vertex_count {
        for k in 0..4 {
            let (w, j) = if k < span {
                let off = v * (span * 8) + k * 8;
                if off + 8 > bone_weights_tail.len() {
                    return None;
                }
                let w_raw = f32::from_le_bytes(bone_weights_tail[off..off + 4].try_into().ok()?);
                let j = i32::from_le_bytes(bone_weights_tail[off + 4..off + 8].try_into().ok()?);
                // Match legacy skinned VB build: unmapped bones must not contribute (index 0 only if weight > 0).
                if j < 0 {
                    (0.0f32, 0u32)
                } else {
                    (w_raw, j as u32)
                }
            } else {
                (0.0f32, 0u32)
            };
            let wb = v * 16 + k * 4;
            wt_bytes[wb..wb + 4].copy_from_slice(&w.to_le_bytes());
            idx_bytes[wb..wb + 4].copy_from_slice(&j.to_le_bytes());
        }
    }
    Some((idx_bytes, wt_bytes))
}
