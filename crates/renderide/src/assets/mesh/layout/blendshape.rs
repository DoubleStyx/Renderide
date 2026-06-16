//! Blendshape frame selection and sparse GPU delta packing.

use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use crate::cpu_parallelism::{
    admit_blendshape_channel_tasks, admit_blendshape_pack_shapes, current_reference_worker_count,
    record_parallel_admission,
};
#[cfg(test)]
pub use crate::render_contract::{
    BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_SIZE, BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE,
};
pub use crate::render_contract::{
    BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS, BLENDSHAPE_POSITION_SPARSE_ENTRY_WORDS,
    BlendshapeFrameRange, BlendshapeFrameSpan, BlendshapeGpuPack,
};
use crate::shared::BlendshapeBufferDescriptor;

use super::buffer_layout::MeshBufferLayout;

/// Packed normal and tangent deltas are clamped to this absolute component range.
pub const BLENDSHAPE_PACKED_VECTOR_DELTA_RANGE: f32 = 2.0;

/// Deltas smaller than this magnitude (length squared) are dropped as non-influencing.
pub const BLENDSHAPE_DELTA_EPSILON_SQ: f32 = 1e-14;

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

#[derive(Clone, Copy, Debug)]
struct PendingBlendshapeChannelTask {
    descriptor: BlendshapeBufferDescriptor,
    shape_index: usize,
    channel: BlendshapeDeltaChannel,
    byte_offset: usize,
    duplicate_frame: bool,
}

struct PendingBlendshapeChannelResult {
    task: PendingBlendshapeChannelTask,
    entries: Vec<(u32, Vec3)>,
}

struct PackedBlendshapeShape {
    sparse_deltas: Vec<u8>,
    frame_ranges: Vec<BlendshapeFrameRange>,
    has_position_deltas: bool,
    has_normal_deltas: bool,
    has_tangent_deltas: bool,
    clamped_packed_deltas: bool,
}

/// One sparse vertex delta row before deterministic sorting and byte packing.
#[derive(Clone, Copy, Debug, Default)]
struct PendingBlendshapeDelta {
    position: Vec3,
    normal: Vec3,
    tangent: Vec3,
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

    fn set_delta(self, row: &mut PendingBlendshapeDelta, delta: Vec3) {
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
fn vector_has_nonzero_delta(delta: Vec3) -> bool {
    delta.length_squared() > BLENDSHAPE_DELTA_EPSILON_SQ
}

/// Reads one descriptor channel into sparse pending entries.
fn read_pending_channel_entries(
    raw: &[u8],
    byte_offset: usize,
    vertex_count: usize,
    duplicate_frame: bool,
) -> Option<Vec<(u32, Vec3)>> {
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
        let delta = Vec3::new(x, y, z);
        if !duplicate_frame && vector_has_nonzero_delta(delta) {
            entries.push((v as u32, delta));
        }
    }
    Some(entries)
}

fn read_pending_channel_task(
    raw: &[u8],
    task: PendingBlendshapeChannelTask,
    vertex_count: usize,
) -> Option<PendingBlendshapeChannelResult> {
    let entries =
        read_pending_channel_entries(raw, task.byte_offset, vertex_count, task.duplicate_frame)?;
    Some(PendingBlendshapeChannelResult { task, entries })
}

fn pending_frame_for_shape_descriptor<'a>(
    frames: &'a mut Vec<PendingBlendshapeFrame>,
    shape_index: usize,
    descriptor: &BlendshapeBufferDescriptor,
) -> Option<&'a mut PendingBlendshapeFrame> {
    let frame_index = descriptor.frame_index;
    let index = if let Some(index) = frames
        .iter()
        .position(|frame| frame.frame_index == frame_index)
    {
        index
    } else {
        frames.push(PendingBlendshapeFrame {
            shape_index: shape_index as u32,
            frame_index,
            frame_weight: descriptor.frame_weight,
            entries: HashMap::new(),
        });
        frames.len() - 1
    };
    frames.get_mut(index)
}

/// Merges one descriptor channel into the shape/frame sparse accumulator.
fn merge_pending_channel_entries(
    frame: &mut PendingBlendshapeFrame,
    channel: BlendshapeDeltaChannel,
    entries: Vec<(u32, Vec3)>,
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
    let mut tasks = Vec::new();

    for &descriptor in blendshape_buffers {
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
            tasks.push(PendingBlendshapeChannelTask {
                descriptor,
                shape_index: bi,
                channel,
                byte_offset,
                duplicate_frame,
            });
            byte_offset += chunk_len;
        }
    }
    let sample_count = tasks.len().saturating_mul(vertex_count);
    let admission =
        admit_blendshape_channel_tasks(tasks.len(), sample_count, current_reference_worker_count());
    record_parallel_admission(
        "blendshape_channel_extract",
        sample_count,
        tasks.len(),
        admission,
    );
    let results = if let Some(chunk_size) = admission.chunk_size() {
        tasks
            .par_iter()
            .copied()
            .with_min_len(chunk_size)
            .map(|task| read_pending_channel_task(raw, task, vertex_count))
            .collect::<Vec<_>>()
    } else {
        tasks
            .iter()
            .copied()
            .map(|task| read_pending_channel_task(raw, task, vertex_count))
            .collect::<Vec<_>>()
    };
    for result in results {
        let result = result?;
        if result.task.duplicate_frame {
            continue;
        }
        let frames = per_shape.get_mut(result.task.shape_index)?;
        let frame = pending_frame_for_shape_descriptor(
            frames,
            result.task.shape_index,
            &result.task.descriptor,
        )?;
        merge_pending_channel_entries(frame, result.task.channel, result.entries);
    }
    Some(per_shape)
}

/// Converts pending frames into the packed sparse byte blob and frame spans.
fn build_blendshape_gpu_pack(
    per_shape: Vec<Vec<PendingBlendshapeFrame>>,
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
    let sparse_entry_count = per_shape
        .iter()
        .flatten()
        .map(|frame| frame.entries.len())
        .sum::<usize>();

    let shape_jobs = per_shape.into_iter().collect::<Vec<_>>();
    let admission = admit_blendshape_pack_shapes(
        shape_jobs.len(),
        sparse_entry_count,
        current_reference_worker_count(),
    );
    record_parallel_admission(
        "blendshape_shape_pack",
        sparse_entry_count,
        shape_jobs.len(),
        admission,
    );
    let packed_shapes = if let Some(chunk_size) = admission.chunk_size() {
        shape_jobs
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(pack_pending_shape_frames)
            .collect::<Vec<_>>()
    } else {
        shape_jobs
            .into_iter()
            .map(pack_pending_shape_frames)
            .collect::<Vec<_>>()
    };

    for (s, packed_shape) in packed_shapes.into_iter().enumerate() {
        let first_frame = frame_ranges.len() as u32;
        let sparse_word_offset = sparse_word_len(&sparse_deltas);
        for mut range in packed_shape.frame_ranges {
            range.position_first_word =
                range.position_first_word.saturating_add(sparse_word_offset);
            range.normal_first_word = range.normal_first_word.saturating_add(sparse_word_offset);
            range.tangent_first_word = range.tangent_first_word.saturating_add(sparse_word_offset);
            frame_ranges.push(range);
        }
        sparse_deltas.extend_from_slice(&packed_shape.sparse_deltas);
        has_position_deltas |= packed_shape.has_position_deltas;
        has_normal_deltas |= packed_shape.has_normal_deltas;
        has_tangent_deltas |= packed_shape.has_tangent_deltas;
        clamped_packed_deltas |= packed_shape.clamped_packed_deltas;
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

fn pack_pending_shape_frames(mut frames: Vec<PendingBlendshapeFrame>) -> PackedBlendshapeShape {
    frames.sort_by(|a, b| {
        a.frame_weight
            .total_cmp(&b.frame_weight)
            .then(a.frame_index.cmp(&b.frame_index))
    });
    let mut sparse_deltas = Vec::new();
    let mut frame_ranges = Vec::with_capacity(frames.len());
    let mut has_position_deltas = false;
    let mut has_normal_deltas = false;
    let mut has_tangent_deltas = false;
    let mut clamped_packed_deltas = false;
    append_sorted_pending_frames(
        &frames,
        &mut sparse_deltas,
        &mut frame_ranges,
        &mut has_position_deltas,
        &mut has_normal_deltas,
        &mut has_tangent_deltas,
        &mut clamped_packed_deltas,
    );
    PackedBlendshapeShape {
        sparse_deltas,
        frame_ranges,
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

fn append_position_sparse_entry(sparse_deltas: &mut Vec<u8>, vertex_index: u32, delta: Vec3) {
    sparse_deltas.extend_from_slice(&vertex_index.to_le_bytes());
    for component in delta.to_array() {
        sparse_deltas.extend_from_slice(&component.to_le_bytes());
    }
}

fn append_packed_vector_sparse_entry(
    sparse_deltas: &mut Vec<u8>,
    vertex_index: u32,
    delta: Vec3,
) -> bool {
    let (x, x_clamped) = pack_snorm16_delta_component(delta.x);
    let (y, y_clamped) = pack_snorm16_delta_component(delta.y);
    let (z, z_clamped) = pack_snorm16_delta_component(delta.z);
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
