use std::ops::Range;
use std::sync::Arc;

use glam::{Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::shared::{TrailRenderBufferUpload, TrailTextureMode};

use super::bounds::bounds_for_trails;
use super::ids::trail_render_buffer_mesh_asset_id;
use super::types::{
    ParticleRenderBufferError, TrailRenderBufferAsset, checked_range, nonnegative_count,
    photondust_particle_color_to_linear, read_pod_at,
};
use super::upload::{
    GeneratedExtraStreams, GeneratedMeshUploadInput, generated_vertex_stride,
    prepared_generated_derived_streams, push_generated_vertex,
};

/// Minimum trail points before building texture-mode meshes in parallel is worthwhile.
const TRAIL_PARALLEL_POINT_MIN: usize = 512;
/// Trail points targeted per decode worker.
const TRAIL_DECODE_PARALLEL_CHUNK_POINTS: usize = 256;
/// Trail points required before trail decode fans out across Rayon.
pub(super) const TRAIL_DECODE_PARALLEL_MIN_POINTS: usize = TRAIL_DECODE_PARALLEL_CHUNK_POINTS * 2;
/// Trail points targeted per generated mesh worker.
const TRAIL_MESH_PARALLEL_CHUNK_POINTS: usize = 256;
/// Trail points required before one texture-mode mesh build fans out across Rayon.
const TRAIL_MESH_PARALLEL_MIN_POINTS: usize = TRAIL_MESH_PARALLEL_CHUNK_POINTS * 2;

/// Number of bytes in one PhotonDust trail-offset row.
pub(super) const TRAIL_OFFSET_BYTES: usize = 16;

/// CPU output from building a trail render buffer.
#[derive(Debug)]
pub(crate) struct TrailRenderBufferBuild {
    /// Resident trail render-buffer metadata.
    pub(crate) asset: TrailRenderBufferAsset,
    /// Generated trail mesh inputs ready for renderer-thread GPU upload.
    pub(crate) meshes: Vec<GeneratedMeshUploadInput>,
}

/// Builds trail render-buffer metadata and generated mesh bytes without touching the GPU.
pub(crate) fn build_trail_render_buffer_cpu(
    raw: Arc<[u8]>,
    upload: &TrailRenderBufferUpload,
) -> Result<TrailRenderBufferBuild, ParticleRenderBufferError> {
    profiling::scope!("particle::build_trail_render_buffer");
    let asset_id = upload.asset_id;
    let trails_count = nonnegative_count("trail", asset_id, "trails_count", upload.trails_count)?;
    let trail_point_count = nonnegative_count(
        "trail",
        asset_id,
        "trail_point_count",
        upload.trail_point_count,
    )?;
    let trails = decode_trails(raw.as_ref(), upload, trails_count, trail_point_count)?;
    let meshes = build_trail_mesh_inputs(asset_id, &trails)?;
    Ok(TrailRenderBufferBuild {
        asset: TrailRenderBufferAsset {
            asset_id,
            trails_count,
            trail_point_count,
        },
        meshes,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TrailOffset {
    pub(super) offset: usize,
    pub(super) capacity: usize,
    pub(super) start: usize,
    pub(super) count: usize,
}

impl TrailOffset {
    pub(super) fn point_index(self, logical_index: usize) -> Option<usize> {
        if logical_index >= self.count || self.capacity == 0 {
            return None;
        }
        Some(self.offset + ((self.start + logical_index) % self.capacity))
    }
}

#[derive(Clone, Copy)]
pub(super) struct TrailPoint {
    /// Trail point center in the render-buffer renderer's local space.
    pub(super) position: Vec3,
    /// Trail point color converted from PhotonDust sRGB to linear vertex color.
    pub(super) color: Vec4,
    /// Trail width from PhotonDust.
    pub(super) width: f32,
}

pub(super) struct TrailPolyline {
    /// Ordered trail points after applying the host ring-buffer offset row.
    pub(super) points: Vec<TrailPoint>,
    /// Cumulative local-space distance at each point.
    pub(super) distances: Vec<f32>,
    /// Total polyline distance clamped away from zero for stretch-mode UVs.
    pub(super) total_distance: f32,
}

pub(super) fn decode_trails(
    raw: &[u8],
    upload: &TrailRenderBufferUpload,
    trails_count: usize,
    trail_point_count: usize,
) -> Result<Vec<TrailPolyline>, ParticleRenderBufferError> {
    let asset_id = upload.asset_id;
    let trail_offsets = checked_range(
        "trail",
        asset_id,
        raw.len(),
        "trails",
        upload.trails_offset,
        trails_count,
        TRAIL_OFFSET_BYTES,
    )?;
    let positions = checked_range(
        "trail",
        asset_id,
        raw.len(),
        "positions",
        upload.positions_offset,
        trail_point_count,
        12,
    )?;
    let colors = checked_range(
        "trail",
        asset_id,
        raw.len(),
        "colors",
        upload.colors_offset,
        trail_point_count,
        16,
    )?;
    let sizes = checked_range(
        "trail",
        asset_id,
        raw.len(),
        "sizes",
        upload.sizes_offset,
        trail_point_count,
        4,
    )?;
    let offsets = (0..trails_count)
        .map(|index| {
            let row: [i32; 4] = read_pod_at(raw, &trail_offsets, index);
            decode_trail_offset(row, trail_point_count)
        })
        .collect::<Vec<_>>();
    let total_points = offsets
        .iter()
        .filter_map(|offset| offset.as_ref().map(|offset| offset.count))
        .sum::<usize>();
    let chunks = trail_chunks_by_point_budget(&offsets, TRAIL_DECODE_PARALLEL_CHUNK_POINTS);
    if trail_decode_parallel_is_worthwhile(total_points, chunks.len()) {
        profiling::scope!("particles::decode_trails_parallel");
        let per_chunk = chunks
            .par_iter()
            .with_min_len(1)
            .map(|range| {
                decode_trail_range(
                    raw,
                    &positions,
                    &colors,
                    &sizes,
                    trail_point_count,
                    &offsets,
                    range.clone(),
                )
            })
            .collect::<Vec<_>>();
        let mut trails = Vec::with_capacity(trails_count);
        for mut chunk in per_chunk {
            trails.append(&mut chunk);
        }
        return Ok(trails);
    }
    profiling::scope!("particles::decode_trails_serial");
    let mut trails = Vec::with_capacity(trails_count);
    for offset in offsets.iter().take(trails_count).copied() {
        trails.push(decode_trail_polyline(
            raw,
            &positions,
            &colors,
            &sizes,
            trail_point_count,
            offset,
        ));
    }
    Ok(trails)
}

/// Returns whether trail decode has enough work to produce at least two useful Rayon chunks.
pub(super) fn trail_decode_parallel_is_worthwhile(total_points: usize, chunk_count: usize) -> bool {
    total_points >= TRAIL_DECODE_PARALLEL_MIN_POINTS
        && chunk_count >= 2
        && rayon::current_num_threads() > 1
}

/// Decodes one contiguous trail-index range into ordered polylines.
fn decode_trail_range(
    raw: &[u8],
    positions: &Range<usize>,
    colors: &Range<usize>,
    sizes: &Range<usize>,
    trail_point_count: usize,
    offsets: &[Option<TrailOffset>],
    range: Range<usize>,
) -> Vec<TrailPolyline> {
    let mut trails = Vec::with_capacity(range.end.saturating_sub(range.start));
    for index in range {
        trails.push(decode_trail_polyline(
            raw,
            positions,
            colors,
            sizes,
            trail_point_count,
            offsets[index],
        ));
    }
    trails
}

/// Decodes one logical trail after offset-ring normalization.
fn decode_trail_polyline(
    raw: &[u8],
    positions: &Range<usize>,
    colors: &Range<usize>,
    sizes: &Range<usize>,
    trail_point_count: usize,
    offset: Option<TrailOffset>,
) -> TrailPolyline {
    let mut points = Vec::with_capacity(offset.map_or(0, |o| o.count));
    if let Some(offset) = offset {
        for logical_index in 0..offset.count {
            let Some(point_index) = offset.point_index(logical_index) else {
                continue;
            };
            if point_index >= trail_point_count {
                continue;
            }
            let p: [f32; 3] = read_pod_at(raw, positions, point_index);
            let c: [f32; 4] = read_pod_at(raw, colors, point_index);
            let width = read_pod_at::<f32>(raw, sizes, point_index).max(0.0);
            points.push(TrailPoint {
                position: Vec3::from_array(p),
                color: photondust_particle_color_to_linear(Vec4::from_array(c)),
                width,
            });
        }
    }
    let distances = trail_distances(&points);
    let total_distance = distances.last().copied().unwrap_or(0.0).max(1e-6);
    TrailPolyline {
        points,
        distances,
        total_distance,
    }
}

fn decode_trail_offset(row: [i32; 4], trail_point_count: usize) -> Option<TrailOffset> {
    let [offset, capacity, start, count] = row;
    if offset < 0 || capacity <= 0 || start < 0 || count <= 0 {
        return None;
    }
    let offset = offset as usize;
    let capacity = capacity as usize;
    if offset >= trail_point_count || offset.saturating_add(capacity) > trail_point_count {
        return None;
    }
    Some(TrailOffset {
        offset,
        capacity,
        start: start as usize,
        count: (count as usize).min(capacity),
    })
}

/// Builds generated trail mesh upload inputs for every supported texture coordinate mode.
fn build_trail_mesh_inputs(
    asset_id: i32,
    trails: &[TrailPolyline],
) -> Result<Vec<GeneratedMeshUploadInput>, ParticleRenderBufferError> {
    let modes = [
        TrailTextureMode::Stretch,
        TrailTextureMode::Tile,
        TrailTextureMode::DistributePerSegment,
        TrailTextureMode::RepeatPerSegment,
    ];
    let inputs: Result<Vec<_>, ParticleRenderBufferError> = modes
        .into_iter()
        .map(|mode| {
            let mesh_asset_id = trail_render_buffer_mesh_asset_id(asset_id, mode).ok_or(
                ParticleRenderBufferError::GeneratedIdOverflow {
                    kind: "trail",
                    asset_id,
                },
            )?;
            Ok((mode, mesh_asset_id))
        })
        .collect();
    let inputs = inputs?;
    if trail_mesh_parallel_is_worthwhile(trails) {
        return inputs
            .into_par_iter()
            .map(|(mode, mesh_asset_id)| {
                build_trail_mesh_input(mesh_asset_id, asset_id, trails, mode)
            })
            .collect();
    }
    inputs
        .into_iter()
        .map(|(mode, mesh_asset_id)| build_trail_mesh_input(mesh_asset_id, asset_id, trails, mode))
        .collect()
}

/// Returns whether trail mesh generation has enough point work to build modes in parallel.
fn trail_mesh_parallel_is_worthwhile(trails: &[TrailPolyline]) -> bool {
    trails.iter().map(|trail| trail.points.len()).sum::<usize>() >= TRAIL_PARALLEL_POINT_MIN
        && rayon::current_num_threads() > 1
}

fn build_trail_mesh_input(
    mesh_asset_id: i32,
    source_asset_id: i32,
    trails: &[TrailPolyline],
    texture_mode: TrailTextureMode,
) -> Result<GeneratedMeshUploadInput, ParticleRenderBufferError> {
    let vertex_count = trails
        .iter()
        .map(|trail| trail.points.len().saturating_mul(2))
        .try_fold(0usize, |acc, count| acc.checked_add(count))
        .ok_or(ParticleRenderBufferError::MeshTooLarge {
            kind: "trail",
            asset_id: source_asset_id,
        })?;
    let segment_count = trails
        .iter()
        .map(|trail| trail.points.len().saturating_sub(1))
        .try_fold(0usize, |acc, count| acc.checked_add(count))
        .ok_or(ParticleRenderBufferError::MeshTooLarge {
            kind: "trail",
            asset_id: source_asset_id,
        })?;
    let index_count =
        segment_count
            .checked_mul(6)
            .ok_or(ParticleRenderBufferError::MeshTooLarge {
                kind: "trail",
                asset_id: source_asset_id,
            })?;
    if vertex_count > u32::MAX as usize || index_count > i32::MAX as usize {
        return Err(ParticleRenderBufferError::MeshTooLarge {
            kind: "trail",
            asset_id: source_asset_id,
        });
    }

    let mut vertices = Vec::with_capacity(vertex_count * generated_vertex_stride());
    let mut indices = Vec::with_capacity(index_count * 4);
    let trail_vertex_offsets = trail_vertex_offsets(trails);
    let trail_point_count = trail_vertex_offsets.last().copied().unwrap_or(0) / 2;
    let chunks = trail_chunks_by_point_budget_from_trails(trails, TRAIL_MESH_PARALLEL_CHUNK_POINTS);
    if trail_mesh_inner_parallel_is_worthwhile(trail_point_count, chunks.len()) {
        profiling::scope!("particles::build_trail_mesh_inner_parallel");
        let per_chunk = chunks
            .par_iter()
            .with_min_len(1)
            .map(|range| {
                build_trail_mesh_chunk(trails, &trail_vertex_offsets, texture_mode, range.clone())
            })
            .collect::<Vec<_>>();
        for mut chunk in per_chunk {
            vertices.append(&mut chunk.vertices);
            indices.append(&mut chunk.indices);
        }
    } else {
        profiling::scope!("particles::build_trail_mesh_inner_serial");
        let mut chunk =
            build_trail_mesh_chunk(trails, &trail_vertex_offsets, texture_mode, 0..trails.len());
        vertices.append(&mut chunk.vertices);
        indices.append(&mut chunk.indices);
    }

    Ok(GeneratedMeshUploadInput {
        kind: "trail",
        source_asset_id,
        mesh_asset_id,
        prepared_derived_streams: prepared_generated_derived_streams(
            &vertices,
            vertex_count,
            GeneratedExtraStreams::default(),
        ),
        vertices,
        indices,
        vertex_count,
        index_count,
        bounds: bounds_for_trails(trails),
    })
}

/// Packed generated trail data for one trail-index chunk.
pub(super) struct TrailMeshChunk {
    /// Generated vertex bytes for the chunk.
    pub(super) vertices: Vec<u8>,
    /// Generated index bytes for the chunk.
    pub(super) indices: Vec<u8>,
}

/// Returns whether one generated trail mesh has enough point work for at least two worker chunks.
fn trail_mesh_inner_parallel_is_worthwhile(total_points: usize, chunk_count: usize) -> bool {
    total_points >= TRAIL_MESH_PARALLEL_MIN_POINTS
        && chunk_count >= 2
        && rayon::current_num_threads() > 1
}

/// Builds one contiguous trail-index range into local vertex and index buffers.
pub(super) fn build_trail_mesh_chunk(
    trails: &[TrailPolyline],
    trail_vertex_offsets: &[usize],
    texture_mode: TrailTextureMode,
    range: Range<usize>,
) -> TrailMeshChunk {
    let vertex_count =
        trail_vertex_offsets[range.end].saturating_sub(trail_vertex_offsets[range.start]);
    let segment_count = trails[range.clone()]
        .iter()
        .map(|trail| trail.points.len().saturating_sub(1))
        .sum::<usize>();
    let mut vertices = Vec::with_capacity(vertex_count * generated_vertex_stride());
    let mut indices = Vec::with_capacity(segment_count * 6 * size_of::<u32>());
    let normal = Vec3::Z;
    for trail_index in range {
        let trail = &trails[trail_index];
        if trail.points.len() < 2 {
            continue;
        }
        for point_index in 0..trail.points.len() {
            let point = trail.points[point_index];
            let tangent = trail_tangent(&trail.points, point_index);
            let side = trail_side(tangent);
            let half_width = point.width * 0.5;
            let v = trail_v_coordinate(
                texture_mode,
                &trail.distances,
                trail.total_distance,
                point_index,
            );
            push_generated_vertex(
                &mut vertices,
                point.position - side * half_width,
                normal,
                Vec2::new(0.0, v),
                point.color,
            );
            push_generated_vertex(
                &mut vertices,
                point.position + side * half_width,
                normal,
                Vec2::new(1.0, v),
                point.color,
            );
        }
        let base_vertex = trail_vertex_offsets[trail_index] as u32;
        for segment_index in 0..trail.points.len() - 1 {
            let a = base_vertex + (segment_index as u32) * 2;
            for index in [a, a + 1, a + 2, a + 2, a + 1, a + 3] {
                indices.extend_from_slice(bytemuck::bytes_of(&index));
            }
        }
    }
    TrailMeshChunk { vertices, indices }
}

/// Prefix-sums generated trail vertex counts per source trail.
pub(super) fn trail_vertex_offsets(trails: &[TrailPolyline]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(trails.len() + 1);
    let mut total = 0usize;
    offsets.push(0);
    for trail in trails {
        total = total.saturating_add(trail.points.len().saturating_mul(2));
        offsets.push(total);
    }
    offsets
}

/// Builds trail-index chunks by accumulating decoded point count.
pub(super) fn trail_chunks_by_point_budget_from_trails(
    trails: &[TrailPolyline],
    target_points: usize,
) -> Vec<Range<usize>> {
    trail_chunks_by_point_budget_impl(trails.len(), target_points, |index| {
        trails[index].points.len()
    })
}

/// Builds trail-index chunks by accumulating offset point count.
pub(super) fn trail_chunks_by_point_budget(
    offsets: &[Option<TrailOffset>],
    target_points: usize,
) -> Vec<Range<usize>> {
    trail_chunks_by_point_budget_impl(offsets.len(), target_points, |index| {
        offsets[index].map_or(0, |offset| offset.count)
    })
}

/// Chunks an ordered trail domain without reordering any source trail.
fn trail_chunks_by_point_budget_impl(
    len: usize,
    target_points: usize,
    point_count: impl Fn(usize) -> usize,
) -> Vec<Range<usize>> {
    if len == 0 {
        return Vec::new();
    }
    let target_points = target_points.max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut points = 0usize;
    for index in 0..len {
        points = points.saturating_add(point_count(index));
        if points >= target_points && index + 1 < len {
            chunks.push(start..index + 1);
            start = index + 1;
            points = 0;
        }
    }
    chunks.push(start..len);
    chunks
}

pub(super) fn trail_distances(points: &[TrailPoint]) -> Vec<f32> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut distances = Vec::with_capacity(points.len());
    let mut total = 0.0;
    distances.push(0.0);
    for pair in points.windows(2) {
        total += (pair[1].position - pair[0].position).length();
        distances.push(total);
    }
    distances
}

pub(super) fn trail_v_coordinate(
    texture_mode: TrailTextureMode,
    distances: &[f32],
    total: f32,
    point_index: usize,
) -> f32 {
    match texture_mode {
        TrailTextureMode::Stretch => distances.get(point_index).copied().unwrap_or(0.0) / total,
        TrailTextureMode::Tile => distances.get(point_index).copied().unwrap_or(0.0),
        TrailTextureMode::DistributePerSegment => {
            let denom = distances.len().saturating_sub(1).max(1) as f32;
            point_index as f32 / denom
        }
        TrailTextureMode::RepeatPerSegment => point_index as f32,
    }
}

fn trail_tangent(points: &[TrailPoint], point_index: usize) -> Vec3 {
    let prev = point_index
        .checked_sub(1)
        .and_then(|index| points.get(index))
        .map(|point| point.position);
    let next = points.get(point_index + 1).map(|point| point.position);
    match (prev, next) {
        (Some(prev), Some(next)) => safe_normalize(next - prev, Vec3::X),
        (None, Some(next)) => safe_normalize(next - points[point_index].position, Vec3::X),
        (Some(prev), None) => safe_normalize(points[point_index].position - prev, Vec3::X),
        (None, None) => Vec3::X,
    }
}

fn trail_side(tangent: Vec3) -> Vec3 {
    let side = tangent.cross(Vec3::Z);
    if side.length_squared() > 1e-8 {
        return side.normalize();
    }
    safe_normalize(tangent.cross(Vec3::Y), Vec3::Y)
}

fn safe_normalize(value: Vec3, fallback: Vec3) -> Vec3 {
    if value.length_squared() > 1e-8 && value.is_finite() {
        value.normalize()
    } else {
        fallback
    }
}
