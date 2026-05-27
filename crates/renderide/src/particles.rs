//! PhotonDust render-buffer decoding and generated mesh helpers.

use std::{ops::Range, sync::Arc};

use glam::{Quat, Vec2, Vec3, Vec4};
use rayon::prelude::*;
use thiserror::Error;

use crate::assets::mesh::{
    GpuMesh, MeshGpuUploadContext, compute_and_validate_mesh_layout, try_upload_mesh_from_raw,
};
use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::{
    IndexBufferFormat, MeshUploadData, PointRenderBufferUpload, RenderBoundingBox,
    SubmeshBufferDescriptor, SubmeshTopology, TrailRenderBufferUpload, TrailTextureMode,
    VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType,
};

/// Number of billboard vertices generated for one point particle.
const BILLBOARD_VERTICES_PER_POINT: usize = 4;
/// Number of billboard indices generated for one point particle.
const BILLBOARD_INDICES_PER_POINT: usize = 12;
/// Minimum point particles before Rayon decode/fill scheduling is worthwhile.
const POINT_PARTICLE_PARALLEL_MIN: usize = 2_048;
/// Point particle chunk size used by parallel vertex/index fill.
const POINT_PARTICLE_PARALLEL_CHUNK: usize = 1_024;
/// Minimum trail points before building texture-mode meshes in parallel is worthwhile.
const TRAIL_PARALLEL_POINT_MIN: usize = 1_024;
/// Trail points targeted per decode worker.
const TRAIL_DECODE_PARALLEL_CHUNK_POINTS: usize = 1_024;
/// Trail points required before trail decode fans out across Rayon.
const TRAIL_DECODE_PARALLEL_MIN_POINTS: usize = TRAIL_DECODE_PARALLEL_CHUNK_POINTS * 2;
/// Trail points targeted per generated mesh worker.
const TRAIL_MESH_PARALLEL_CHUNK_POINTS: usize = 1_024;
/// Trail points required before one texture-mode mesh build fans out across Rayon.
const TRAIL_MESH_PARALLEL_MIN_POINTS: usize = TRAIL_MESH_PARALLEL_CHUNK_POINTS * 2;
/// Cheap bound scans use larger chunks so scheduling overhead stays amortized.
const PARTICLE_BOUNDS_PARALLEL_CHUNK_POINTS: usize = 16_384;
/// Point or trail point count required before bounds reduction uses Rayon.
const PARTICLE_BOUNDS_PARALLEL_MIN_POINTS: usize = PARTICLE_BOUNDS_PARALLEL_CHUNK_POINTS * 2;
/// Number of bytes in one PhotonDust trail-offset row.
const TRAIL_OFFSET_BYTES: usize = 16;
/// Generated particle mesh id tag for billboard quads.
const BILLBOARD_MESH_KIND: i32 = 1;
/// Generated particle mesh id tag for stretch trail ribbons.
const TRAIL_STRETCH_MESH_KIND: i32 = 2;
/// Generated particle mesh id tag for tiled trail ribbons.
const TRAIL_TILE_MESH_KIND: i32 = 3;
/// Generated particle mesh id tag for distributed trail ribbons.
const TRAIL_DISTRIBUTE_MESH_KIND: i32 = 4;
/// Generated particle mesh id tag for per-segment repeated trail ribbons.
const TRAIL_REPEAT_MESH_KIND: i32 = 5;
/// Number of generated mesh ids reserved per source render-buffer asset.
const GENERATED_MESH_KIND_STRIDE: i64 = 8;

/// CPU metadata retained for a resident PhotonDust point render buffer.
#[derive(Clone, Debug)]
pub(crate) struct PointRenderBufferAsset {
    /// Host point render-buffer asset id.
    pub(crate) asset_id: i32,
    /// Number of particles decoded from the latest upload.
    pub(crate) count: usize,
    /// Texture-sheet frame grid copied from the upload.
    pub(crate) frame_grid_size: glam::IVec2,
    /// CPU point data retained for mesh-particle renderers.
    pub(crate) points: Arc<[PointParticle]>,
}

/// CPU metadata retained for a resident PhotonDust trail render buffer.
#[derive(Clone, Debug)]
pub(crate) struct TrailRenderBufferAsset {
    /// Host trail render-buffer asset id.
    pub(crate) asset_id: i32,
    /// Number of logical trails decoded from the latest upload.
    pub(crate) trails_count: usize,
    /// Number of trail point slots decoded from the latest upload.
    pub(crate) trail_point_count: usize,
}

/// Meshes and metadata produced by a point render-buffer upload.
#[derive(Debug)]
pub(crate) struct PointRenderBufferMeshUpload {
    /// Resident point render-buffer metadata.
    pub(crate) asset: PointRenderBufferAsset,
    /// Generated billboard quad mesh for the point buffer.
    pub(crate) billboard_mesh: GpuMesh,
}

/// Meshes and metadata produced by a trail render-buffer upload.
#[derive(Debug)]
pub(crate) struct TrailRenderBufferMeshUpload {
    /// Resident trail render-buffer metadata.
    pub(crate) asset: TrailRenderBufferAsset,
    /// Generated trail meshes for the supported texture modes.
    pub(crate) meshes: Vec<GpuMesh>,
}

/// Existing generated trail meshes, keyed by texture mode, for in-place GPU buffer reuse.
#[derive(Default)]
pub(crate) struct TrailRenderBufferExistingMeshes {
    /// Existing stretch-mode trail mesh.
    pub(crate) stretch: Option<GpuMesh>,
    /// Existing tiled trail mesh.
    pub(crate) tile: Option<GpuMesh>,
    /// Existing per-segment distributed trail mesh.
    pub(crate) distribute_per_segment: Option<GpuMesh>,
    /// Existing per-segment repeated trail mesh.
    pub(crate) repeat_per_segment: Option<GpuMesh>,
}

impl TrailRenderBufferExistingMeshes {
    /// Takes the existing mesh for `mode`, leaving that slot empty.
    fn take_mode(&mut self, mode: TrailTextureMode) -> Option<GpuMesh> {
        match mode {
            TrailTextureMode::Stretch => self.stretch.take(),
            TrailTextureMode::Tile => self.tile.take(),
            TrailTextureMode::DistributePerSegment => self.distribute_per_segment.take(),
            TrailTextureMode::RepeatPerSegment => self.repeat_per_segment.take(),
        }
    }
}

/// Error raised while validating or generating a PhotonDust render-buffer mesh.
#[derive(Debug, Error)]
pub(crate) enum ParticleRenderBufferError {
    /// The host sent a negative count for a required row array.
    #[error("{kind} render buffer {asset_id}: negative {field} {value}")]
    NegativeCount {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
        /// Field that carried the invalid count.
        field: &'static str,
        /// Invalid value.
        value: i32,
    },
    /// A required payload offset was negative.
    #[error("{kind} render buffer {asset_id}: missing required {field} offset")]
    MissingOffset {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
        /// Missing field name.
        field: &'static str,
    },
    /// A payload byte range overflowed or fell outside the shared-memory copy.
    #[error(
        "{kind} render buffer {asset_id}: {field} byte range offset={offset} len={len} exceeds raw len {raw_len}"
    )]
    RangeOutOfBounds {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
        /// Field being read.
        field: &'static str,
        /// Requested byte offset.
        offset: i32,
        /// Requested byte length.
        len: usize,
        /// Available raw bytes.
        raw_len: usize,
    },
    /// The generated mesh id cannot fit into the renderer's signed asset id space.
    #[error("{kind} render buffer {asset_id}: generated mesh id overflow")]
    GeneratedIdOverflow {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
    },
    /// The generated vertex or index count exceeded supported limits.
    #[error("{kind} render buffer {asset_id}: generated mesh is too large")]
    MeshTooLarge {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
    },
    /// Mesh layout validation failed for generated geometry.
    #[error("{kind} render buffer {asset_id}: generated mesh layout is invalid")]
    InvalidMeshLayout {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
    },
    /// GPU upload failed for generated geometry.
    #[error("{kind} render buffer {asset_id}: generated mesh GPU upload failed")]
    GpuUploadFailed {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
    },
}

/// Returns the generated mesh asset id for a point-buffer billboard mesh.
pub(crate) fn billboard_render_buffer_mesh_asset_id(asset_id: i32) -> Option<i32> {
    generated_mesh_asset_id(asset_id, BILLBOARD_MESH_KIND)
}

/// Returns the generated mesh asset id for a trail-buffer texture mode.
pub(crate) fn trail_render_buffer_mesh_asset_id(
    asset_id: i32,
    mode: TrailTextureMode,
) -> Option<i32> {
    let kind = match mode {
        TrailTextureMode::Stretch => TRAIL_STRETCH_MESH_KIND,
        TrailTextureMode::Tile => TRAIL_TILE_MESH_KIND,
        TrailTextureMode::DistributePerSegment => TRAIL_DISTRIBUTE_MESH_KIND,
        TrailTextureMode::RepeatPerSegment => TRAIL_REPEAT_MESH_KIND,
    };
    generated_mesh_asset_id(asset_id, kind)
}

/// Returns all generated mesh ids owned by a point render-buffer asset.
pub(crate) fn point_render_buffer_generated_mesh_ids(asset_id: i32) -> impl Iterator<Item = i32> {
    std::iter::once(billboard_render_buffer_mesh_asset_id(asset_id)).flatten()
}

/// Returns all generated mesh ids owned by a trail render-buffer asset.
pub(crate) fn trail_render_buffer_generated_mesh_ids(asset_id: i32) -> impl Iterator<Item = i32> {
    [
        trail_render_buffer_mesh_asset_id(asset_id, TrailTextureMode::Stretch),
        trail_render_buffer_mesh_asset_id(asset_id, TrailTextureMode::Tile),
        trail_render_buffer_mesh_asset_id(asset_id, TrailTextureMode::DistributePerSegment),
        trail_render_buffer_mesh_asset_id(asset_id, TrailTextureMode::RepeatPerSegment),
    ]
    .into_iter()
    .flatten()
}

/// Returns whether `asset_id` belongs to the generated PhotonDust mesh id range.
pub(crate) fn is_generated_particle_mesh_asset_id(asset_id: i32) -> bool {
    generated_mesh_kind(asset_id).is_some()
}

/// Returns whether `asset_id` is a generated PhotonDust billboard mesh id.
pub(crate) fn is_generated_billboard_mesh_asset_id(asset_id: i32) -> bool {
    generated_mesh_kind(asset_id) == Some(BILLBOARD_MESH_KIND)
}

/// Returns whether `asset_id` is a generated PhotonDust trail mesh id.
#[cfg(test)]
pub(crate) fn is_generated_trail_mesh_asset_id(asset_id: i32) -> bool {
    matches!(
        generated_mesh_kind(asset_id),
        Some(
            TRAIL_STRETCH_MESH_KIND
                | TRAIL_TILE_MESH_KIND
                | TRAIL_DISTRIBUTE_MESH_KIND
                | TRAIL_REPEAT_MESH_KIND
        )
    )
}

/// Builds the generated billboard mesh for a point render-buffer upload.
pub(crate) fn build_point_render_buffer_upload(
    gpu: MeshGpuUploadContext<'_>,
    raw: Arc<[u8]>,
    upload: &PointRenderBufferUpload,
    existing: Option<GpuMesh>,
) -> Result<PointRenderBufferMeshUpload, ParticleRenderBufferError> {
    profiling::scope!("particle::build_point_render_buffer");
    let asset_id = upload.asset_id;
    let count = nonnegative_count("point", asset_id, "count", upload.count)?;
    let points = decode_point_particles(raw.as_ref(), upload, count)?;
    let mesh_asset_id = billboard_render_buffer_mesh_asset_id(asset_id).ok_or(
        ParticleRenderBufferError::GeneratedIdOverflow {
            kind: "point",
            asset_id,
        },
    )?;
    let mesh = build_billboard_mesh(
        gpu,
        mesh_asset_id,
        asset_id,
        &points,
        upload.frame_grid_size,
        existing,
    )?;
    let points: Arc<[PointParticle]> = Arc::from(points.into_boxed_slice());
    Ok(PointRenderBufferMeshUpload {
        asset: PointRenderBufferAsset {
            asset_id,
            count: points.len(),
            frame_grid_size: upload.frame_grid_size,
            points,
        },
        billboard_mesh: mesh,
    })
}

/// Builds the generated ribbon meshes for a trail render-buffer upload.
pub(crate) fn build_trail_render_buffer_upload(
    gpu: MeshGpuUploadContext<'_>,
    raw: Arc<[u8]>,
    upload: &TrailRenderBufferUpload,
    existing: TrailRenderBufferExistingMeshes,
) -> Result<TrailRenderBufferMeshUpload, ParticleRenderBufferError> {
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
    let meshes = build_trail_meshes(gpu, asset_id, &trails, existing)?;
    Ok(TrailRenderBufferMeshUpload {
        asset: TrailRenderBufferAsset {
            asset_id,
            trails_count,
            trail_point_count,
        },
        meshes,
    })
}

fn generated_mesh_asset_id(source_asset_id: i32, kind: i32) -> Option<i32> {
    if source_asset_id < 0 || !(0..GENERATED_MESH_KIND_STRIDE as i32).contains(&kind) {
        return None;
    }
    let encoded = i64::from(source_asset_id)
        .checked_mul(GENERATED_MESH_KIND_STRIDE)?
        .checked_add(i64::from(kind))?
        .checked_add(2)?;
    let id = -encoded;
    (id >= i64::from(i32::MIN) && id <= -2).then_some(id as i32)
}

fn generated_mesh_kind(asset_id: i32) -> Option<i32> {
    if asset_id >= -1 {
        return None;
    }
    let encoded = i64::from(asset_id).checked_neg()?;
    let payload = encoded.checked_sub(2)?;
    let kind = (payload % GENERATED_MESH_KIND_STRIDE) as i32;
    matches!(
        kind,
        BILLBOARD_MESH_KIND
            | TRAIL_STRETCH_MESH_KIND
            | TRAIL_TILE_MESH_KIND
            | TRAIL_DISTRIBUTE_MESH_KIND
            | TRAIL_REPEAT_MESH_KIND
    )
    .then_some(kind)
}

fn nonnegative_count(
    kind: &'static str,
    asset_id: i32,
    field: &'static str,
    value: i32,
) -> Result<usize, ParticleRenderBufferError> {
    if value < 0 {
        return Err(ParticleRenderBufferError::NegativeCount {
            kind,
            asset_id,
            field,
            value,
        });
    }
    Ok(value as usize)
}

fn checked_range(
    kind: &'static str,
    asset_id: i32,
    raw_len: usize,
    field: &'static str,
    offset: i32,
    count: usize,
    stride: usize,
) -> Result<Range<usize>, ParticleRenderBufferError> {
    if offset < 0 {
        return Err(ParticleRenderBufferError::MissingOffset {
            kind,
            asset_id,
            field,
        });
    }
    let len = count
        .checked_mul(stride)
        .ok_or(ParticleRenderBufferError::RangeOutOfBounds {
            kind,
            asset_id,
            field,
            offset,
            len: usize::MAX,
            raw_len,
        })?;
    let start = offset as usize;
    let end = start
        .checked_add(len)
        .ok_or(ParticleRenderBufferError::RangeOutOfBounds {
            kind,
            asset_id,
            field,
            offset,
            len,
            raw_len,
        })?;
    if end > raw_len {
        return Err(ParticleRenderBufferError::RangeOutOfBounds {
            kind,
            asset_id,
            field,
            offset,
            len,
            raw_len,
        });
    }
    Ok(start..end)
}

fn checked_optional_range(
    kind: &'static str,
    asset_id: i32,
    raw_len: usize,
    field: &'static str,
    offset: i32,
    count: usize,
    stride: usize,
) -> Result<Option<Range<usize>>, ParticleRenderBufferError> {
    if offset < 0 {
        return Ok(None);
    }
    checked_range(kind, asset_id, raw_len, field, offset, count, stride).map(Some)
}

fn read_pod_at<T: bytemuck::Pod>(raw: &[u8], range: &Range<usize>, index: usize) -> T {
    let stride = size_of::<T>();
    let start = range.start + index * stride;
    bytemuck::pod_read_unaligned(&raw[start..start + stride])
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PointParticle {
    /// Particle center in the render-buffer renderer's local space.
    pub(crate) position: Vec3,
    /// Particle rotation in the render-buffer renderer's local space.
    pub(crate) rotation: Quat,
    /// Particle size from PhotonDust.
    pub(crate) size: Vec3,
    /// Particle color from PhotonDust.
    pub(crate) color: Vec4,
    /// Optional texture-sheet frame index.
    pub(crate) frame_index: Option<u16>,
}

fn decode_point_particles(
    raw: &[u8],
    upload: &PointRenderBufferUpload,
    count: usize,
) -> Result<Vec<PointParticle>, ParticleRenderBufferError> {
    let asset_id = upload.asset_id;
    let positions = checked_range(
        "point",
        asset_id,
        raw.len(),
        "positions",
        upload.positions_offset,
        count,
        12,
    )?;
    let rotations = checked_range(
        "point",
        asset_id,
        raw.len(),
        "rotations",
        upload.rotations_offset,
        count,
        16,
    )?;
    let sizes = checked_range(
        "point",
        asset_id,
        raw.len(),
        "sizes",
        upload.sizes_offset,
        count,
        12,
    )?;
    let colors = checked_range(
        "point",
        asset_id,
        raw.len(),
        "colors",
        upload.colors_offset,
        count,
        16,
    )?;
    let frames = checked_optional_range(
        "point",
        asset_id,
        raw.len(),
        "frame_indexes",
        upload.frame_indexes_offset,
        count,
        2,
    )?;
    let decode = |index| {
        let p: [f32; 3] = read_pod_at(raw, &positions, index);
        let r: [f32; 4] = read_pod_at(raw, &rotations, index);
        let s: [f32; 3] = read_pod_at(raw, &sizes, index);
        let c: [f32; 4] = read_pod_at(raw, &colors, index);
        let frame_index = frames
            .as_ref()
            .map(|frame_range| read_pod_at::<u16>(raw, frame_range, index));
        PointParticle {
            position: Vec3::from_array(p),
            rotation: Quat::from_xyzw(r[0], r[1], r[2], r[3]),
            size: Vec3::from_array(s),
            color: Vec4::from_array(c),
            frame_index,
        }
    };
    let points = if point_parallel_is_worthwhile(count) {
        (0..count).into_par_iter().map(decode).collect()
    } else {
        (0..count).map(decode).collect()
    };
    Ok(points)
}

fn build_billboard_mesh(
    gpu: MeshGpuUploadContext<'_>,
    mesh_asset_id: i32,
    source_asset_id: i32,
    points: &[PointParticle],
    frame_grid_size: glam::IVec2,
    existing: Option<GpuMesh>,
) -> Result<GpuMesh, ParticleRenderBufferError> {
    let vertex_count = points
        .len()
        .checked_mul(BILLBOARD_VERTICES_PER_POINT)
        .ok_or(ParticleRenderBufferError::MeshTooLarge {
            kind: "point",
            asset_id: source_asset_id,
        })?;
    let index_count = points
        .len()
        .checked_mul(BILLBOARD_INDICES_PER_POINT)
        .ok_or(ParticleRenderBufferError::MeshTooLarge {
            kind: "point",
            asset_id: source_asset_id,
        })?;
    if vertex_count > u32::MAX as usize || index_count > i32::MAX as usize {
        return Err(ParticleRenderBufferError::MeshTooLarge {
            kind: "point",
            asset_id: source_asset_id,
        });
    }

    let mut vertices = vec![0u8; vertex_count * generated_vertex_stride()];
    let mut indices = vec![0u8; index_count * size_of::<u32>()];
    fill_billboard_buffers(points, frame_grid_size, &mut vertices, &mut indices);

    upload_generated_mesh(
        gpu,
        GeneratedMeshUploadInput {
            kind: "point",
            source_asset_id,
            mesh_asset_id,
            vertices,
            indices,
            vertex_count,
            index_count,
            bounds: bounds_for_points(points),
        },
        existing,
    )
}

/// Returns whether point decode/fill work is large enough to amortize Rayon scheduling.
fn point_parallel_is_worthwhile(count: usize) -> bool {
    count >= POINT_PARTICLE_PARALLEL_MIN && rayon::current_num_threads() > 1
}

/// Fills packed billboard vertex and index buffers for `points`.
fn fill_billboard_buffers(
    points: &[PointParticle],
    frame_grid_size: glam::IVec2,
    vertices: &mut [u8],
    indices: &mut [u8],
) {
    let vertex_chunk_len = BILLBOARD_VERTICES_PER_POINT * generated_vertex_stride();
    let index_chunk_len = BILLBOARD_INDICES_PER_POINT * size_of::<u32>();
    if point_parallel_is_worthwhile(points.len()) {
        points
            .par_chunks(POINT_PARTICLE_PARALLEL_CHUNK)
            .zip(vertices.par_chunks_mut(vertex_chunk_len * POINT_PARTICLE_PARALLEL_CHUNK))
            .zip(indices.par_chunks_mut(index_chunk_len * POINT_PARTICLE_PARALLEL_CHUNK))
            .enumerate()
            .for_each(
                |(chunk_index, ((point_chunk, vertex_chunk), index_chunk))| {
                    let base_particle = chunk_index * POINT_PARTICLE_PARALLEL_CHUNK;
                    fill_billboard_chunk(
                        point_chunk,
                        base_particle,
                        frame_grid_size,
                        vertex_chunk,
                        index_chunk,
                    );
                },
            );
    } else {
        fill_billboard_chunk(points, 0, frame_grid_size, vertices, indices);
    }
}

/// Fills one contiguous point chunk into matching vertex and index chunks.
fn fill_billboard_chunk(
    points: &[PointParticle],
    base_particle: usize,
    frame_grid_size: glam::IVec2,
    vertices: &mut [u8],
    indices: &mut [u8],
) {
    let vertex_stride = generated_vertex_stride();
    let vertex_chunk_len = BILLBOARD_VERTICES_PER_POINT * vertex_stride;
    let index_chunk_len = BILLBOARD_INDICES_PER_POINT * size_of::<u32>();
    for (local_index, point) in points.iter().enumerate() {
        let particle_index = base_particle + local_index;
        let vertex_start = local_index * vertex_chunk_len;
        let index_start = local_index * index_chunk_len;
        fill_billboard_particle(
            point,
            particle_index,
            frame_grid_size,
            &mut vertices[vertex_start..vertex_start + vertex_chunk_len],
            &mut indices[index_start..index_start + index_chunk_len],
        );
    }
}

/// Fills the four billboard vertices and twelve duplicated indices for one point.
fn fill_billboard_particle(
    point: &PointParticle,
    particle_index: usize,
    frame_grid_size: glam::IVec2,
    vertices: &mut [u8],
    indices: &mut [u8],
) {
    let (_, _, roll) = point.rotation.to_euler(glam::EulerRot::XYZ);
    let point_data = Vec3::new(point.size.x * 0.5, point.size.y * 0.5, roll);
    for (corner_index, corner) in [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
    ]
    .into_iter()
    .enumerate()
    {
        let vertex_stride = generated_vertex_stride();
        let vertex_start = corner_index * vertex_stride;
        let uv = particle_frame_uv(corner, point.frame_index, frame_grid_size);
        write_generated_vertex(
            &mut vertices[vertex_start..vertex_start + vertex_stride],
            point.position,
            point_data,
            uv,
            point.color,
        );
    }

    let base_vertex = (particle_index * BILLBOARD_VERTICES_PER_POINT) as u32;
    write_u32s(
        indices,
        &[
            base_vertex,
            base_vertex + 1,
            base_vertex + 2,
            base_vertex + 2,
            base_vertex + 1,
            base_vertex + 3,
            base_vertex,
            base_vertex + 2,
            base_vertex + 1,
            base_vertex + 2,
            base_vertex + 3,
            base_vertex + 1,
        ],
    );
}

fn particle_frame_uv(corner: Vec2, frame_index: Option<u16>, frame_grid_size: glam::IVec2) -> Vec2 {
    let columns = frame_grid_size.x.max(0) as u32;
    let rows = frame_grid_size.y.max(0) as u32;
    let Some(frame_index) = frame_index else {
        return corner;
    };
    if columns == 0 || rows == 0 {
        return corner;
    }
    let frame_count = columns.saturating_mul(rows).max(1);
    let frame = u32::from(frame_index).min(frame_count - 1);
    let column = frame % columns;
    let row = frame / columns;
    Vec2::new(
        (column as f32 + corner.x) / columns as f32,
        (row as f32 + corner.y) / rows as f32,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TrailOffset {
    offset: usize,
    capacity: usize,
    start: usize,
    count: usize,
}

impl TrailOffset {
    fn point_index(self, logical_index: usize) -> Option<usize> {
        if logical_index >= self.count || self.capacity == 0 {
            return None;
        }
        Some(self.offset + ((self.start + logical_index) % self.capacity))
    }
}

#[derive(Clone, Copy)]
struct TrailPoint {
    position: Vec3,
    color: Vec4,
    width: f32,
}

struct TrailPolyline {
    /// Ordered trail points after applying the host ring-buffer offset row.
    points: Vec<TrailPoint>,
    /// Cumulative local-space distance at each point.
    distances: Vec<f32>,
    /// Total polyline distance clamped away from zero for stretch-mode UVs.
    total_distance: f32,
}

fn decode_trails(
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
fn trail_decode_parallel_is_worthwhile(total_points: usize, chunk_count: usize) -> bool {
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
                color: Vec4::from_array(c),
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

/// Builds the generated trail meshes for every supported texture coordinate mode.
fn build_trail_meshes(
    gpu: MeshGpuUploadContext<'_>,
    asset_id: i32,
    trails: &[TrailPolyline],
    mut existing: TrailRenderBufferExistingMeshes,
) -> Result<Vec<GpuMesh>, ParticleRenderBufferError> {
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
            Ok((mode, mesh_asset_id, existing.take_mode(mode)))
        })
        .collect();
    let inputs = inputs?;
    if trail_mesh_parallel_is_worthwhile(trails) {
        return inputs
            .into_par_iter()
            .map(|(mode, mesh_asset_id, existing)| {
                build_trail_mesh(gpu, mesh_asset_id, asset_id, trails, mode, existing)
            })
            .collect();
    }
    inputs
        .into_iter()
        .map(|(mode, mesh_asset_id, existing)| {
            build_trail_mesh(gpu, mesh_asset_id, asset_id, trails, mode, existing)
        })
        .collect()
}

/// Returns whether trail mesh generation has enough point work to build modes in parallel.
fn trail_mesh_parallel_is_worthwhile(trails: &[TrailPolyline]) -> bool {
    trails.iter().map(|trail| trail.points.len()).sum::<usize>() >= TRAIL_PARALLEL_POINT_MIN
        && rayon::current_num_threads() > 1
}

fn build_trail_mesh(
    gpu: MeshGpuUploadContext<'_>,
    mesh_asset_id: i32,
    source_asset_id: i32,
    trails: &[TrailPolyline],
    texture_mode: TrailTextureMode,
    existing: Option<GpuMesh>,
) -> Result<GpuMesh, ParticleRenderBufferError> {
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

    upload_generated_mesh(
        gpu,
        GeneratedMeshUploadInput {
            kind: "trail",
            source_asset_id,
            mesh_asset_id,
            vertices,
            indices,
            vertex_count,
            index_count,
            bounds: bounds_for_trails(trails),
        },
        existing,
    )
}

/// Packed generated trail data for one trail-index chunk.
struct TrailMeshChunk {
    /// Generated vertex bytes for the chunk.
    vertices: Vec<u8>,
    /// Generated index bytes for the chunk.
    indices: Vec<u8>,
}

/// Returns whether one generated trail mesh has enough point work for at least two worker chunks.
fn trail_mesh_inner_parallel_is_worthwhile(total_points: usize, chunk_count: usize) -> bool {
    total_points >= TRAIL_MESH_PARALLEL_MIN_POINTS
        && chunk_count >= 2
        && rayon::current_num_threads() > 1
}

/// Builds one contiguous trail-index range into local vertex and index buffers.
fn build_trail_mesh_chunk(
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
fn trail_vertex_offsets(trails: &[TrailPolyline]) -> Vec<usize> {
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
fn trail_chunks_by_point_budget_from_trails(
    trails: &[TrailPolyline],
    target_points: usize,
) -> Vec<Range<usize>> {
    trail_chunks_by_point_budget_impl(trails.len(), target_points, |index| {
        trails[index].points.len()
    })
}

/// Builds trail-index chunks by accumulating offset point count.
fn trail_chunks_by_point_budget(
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

fn trail_distances(points: &[TrailPoint]) -> Vec<f32> {
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

fn trail_v_coordinate(
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

fn generated_vertex_stride() -> usize {
    12 + 12 + 8 + 16
}

fn push_generated_vertex(out: &mut Vec<u8>, position: Vec3, normal: Vec3, uv: Vec2, color: Vec4) {
    push_f32s(out, &position.to_array());
    push_f32s(out, &normal.to_array());
    push_f32s(out, &uv.to_array());
    push_f32s(out, &color.to_array());
}

fn push_f32s<const N: usize>(out: &mut Vec<u8>, values: &[f32; N]) {
    out.extend_from_slice(bytemuck::cast_slice(values));
}

/// Writes one generated vertex into an already sized byte slice.
fn write_generated_vertex(out: &mut [u8], position: Vec3, normal: Vec3, uv: Vec2, color: Vec4) {
    let mut cursor = 0usize;
    cursor += write_f32s(&mut out[cursor..], &position.to_array());
    cursor += write_f32s(&mut out[cursor..], &normal.to_array());
    cursor += write_f32s(&mut out[cursor..], &uv.to_array());
    let _ = write_f32s(&mut out[cursor..], &color.to_array());
}

/// Writes tightly packed `f32` values into `out` and returns the byte count written.
fn write_f32s<const N: usize>(out: &mut [u8], values: &[f32; N]) -> usize {
    let bytes = bytemuck::cast_slice(values);
    out[..bytes.len()].copy_from_slice(bytes);
    bytes.len()
}

/// Writes tightly packed `u32` values into `out`.
fn write_u32s(out: &mut [u8], values: &[u32]) {
    out.copy_from_slice(bytemuck::cast_slice(values));
}

/// Inputs needed to publish one renderer-generated mesh into the mesh pool.
struct GeneratedMeshUploadInput {
    /// Human-readable source kind used in diagnostics.
    kind: &'static str,
    /// Host asset id that produced the generated mesh.
    source_asset_id: i32,
    /// Renderer-generated mesh asset id.
    mesh_asset_id: i32,
    /// Packed interleaved vertex bytes.
    vertices: Vec<u8>,
    /// Packed `u32` index bytes.
    indices: Vec<u8>,
    /// Number of vertices in `vertices`.
    vertex_count: usize,
    /// Number of indices in `indices`.
    index_count: usize,
    /// Local-space bounds for the generated mesh.
    bounds: RenderBoundingBox,
}

fn upload_generated_mesh(
    gpu: MeshGpuUploadContext<'_>,
    input: GeneratedMeshUploadInput,
    existing: Option<GpuMesh>,
) -> Result<GpuMesh, ParticleRenderBufferError> {
    let data = generated_mesh_upload_data(
        input.mesh_asset_id,
        input.vertex_count,
        input.index_count,
        input.bounds,
    )?;
    let layout = compute_and_validate_mesh_layout(&data).ok_or(
        ParticleRenderBufferError::InvalidMeshLayout {
            kind: input.kind,
            asset_id: input.source_asset_id,
        },
    )?;
    let mut raw = vec![0u8; layout.total_buffer_length];
    if input.vertices.len() == layout.vertex_size
        && input.indices.len() == layout.index_buffer_length
    {
        raw[..layout.vertex_size].copy_from_slice(&input.vertices);
        raw[layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length]
            .copy_from_slice(&input.indices);
    } else {
        return Err(ParticleRenderBufferError::InvalidMeshLayout {
            kind: input.kind,
            asset_id: input.source_asset_id,
        });
    }
    try_upload_mesh_from_raw(gpu, &raw, &data, existing, &layout).ok_or(
        ParticleRenderBufferError::GpuUploadFailed {
            kind: input.kind,
            asset_id: input.source_asset_id,
        },
    )
}

fn generated_mesh_upload_data(
    mesh_asset_id: i32,
    vertex_count: usize,
    index_count: usize,
    bounds: RenderBoundingBox,
) -> Result<MeshUploadData, ParticleRenderBufferError> {
    let vertex_count_i32 = i32::try_from(vertex_count).map_err(|_conversion_error| {
        ParticleRenderBufferError::MeshTooLarge {
            kind: "generated",
            asset_id: mesh_asset_id,
        }
    })?;
    let index_count_i32 = i32::try_from(index_count).map_err(|_conversion_error| {
        ParticleRenderBufferError::MeshTooLarge {
            kind: "generated",
            asset_id: mesh_asset_id,
        }
    })?;
    Ok(MeshUploadData {
        buffer: SharedMemoryBufferDescriptor {
            length: 1,
            ..Default::default()
        },
        vertex_count: vertex_count_i32,
        index_buffer_format: IndexBufferFormat::UInt32,
        vertex_attributes: vec![
            VertexAttributeDescriptor {
                attribute: VertexAttributeType::Position,
                format: VertexAttributeFormat::Float32,
                dimensions: 3,
            },
            VertexAttributeDescriptor {
                attribute: VertexAttributeType::Normal,
                format: VertexAttributeFormat::Float32,
                dimensions: 3,
            },
            VertexAttributeDescriptor {
                attribute: VertexAttributeType::UV0,
                format: VertexAttributeFormat::Float32,
                dimensions: 2,
            },
            VertexAttributeDescriptor {
                attribute: VertexAttributeType::Color,
                format: VertexAttributeFormat::Float32,
                dimensions: 4,
            },
        ],
        submeshes: if index_count_i32 > 0 {
            vec![SubmeshBufferDescriptor {
                topology: SubmeshTopology::Triangles,
                index_start: 0,
                index_count: index_count_i32,
                bounds,
            }]
        } else {
            Vec::new()
        },
        bounds,
        asset_id: mesh_asset_id,
        ..Default::default()
    })
}

fn bounds_for_points(points: &[PointParticle]) -> RenderBoundingBox {
    if particle_bounds_parallel_is_worthwhile(points.len()) {
        return points
            .par_chunks(PARTICLE_BOUNDS_PARALLEL_CHUNK_POINTS)
            .with_min_len(1)
            .map(|chunk| {
                let mut bounds = BoundsAccumulator::default();
                for point in chunk {
                    let radius = point.size.abs().max_element() * 0.5;
                    bounds.include(point.position - Vec3::splat(radius));
                    bounds.include(point.position + Vec3::splat(radius));
                }
                bounds
            })
            .reduce(BoundsAccumulator::default, |mut a, b| {
                a.merge(b);
                a
            })
            .finish();
    }
    let mut bounds = BoundsAccumulator::default();
    for point in points {
        let radius = point.size.abs().max_element() * 0.5;
        bounds.include(point.position - Vec3::splat(radius));
        bounds.include(point.position + Vec3::splat(radius));
    }
    bounds.finish()
}

fn bounds_for_trails(trails: &[TrailPolyline]) -> RenderBoundingBox {
    let point_count = trails.iter().map(|trail| trail.points.len()).sum::<usize>();
    let chunks =
        trail_chunks_by_point_budget_from_trails(trails, PARTICLE_BOUNDS_PARALLEL_CHUNK_POINTS);
    if particle_bounds_parallel_is_worthwhile(point_count) && chunks.len() >= 2 {
        return chunks
            .par_iter()
            .with_min_len(1)
            .map(|range| {
                let mut bounds = BoundsAccumulator::default();
                for trail in &trails[range.clone()] {
                    for point in &trail.points {
                        let radius = point.width.abs() * 0.5;
                        bounds.include(point.position - Vec3::splat(radius));
                        bounds.include(point.position + Vec3::splat(radius));
                    }
                }
                bounds
            })
            .reduce(BoundsAccumulator::default, |mut a, b| {
                a.merge(b);
                a
            })
            .finish();
    }
    let mut bounds = BoundsAccumulator::default();
    for trail in trails {
        for point in &trail.points {
            let radius = point.width.abs() * 0.5;
            bounds.include(point.position - Vec3::splat(radius));
            bounds.include(point.position + Vec3::splat(radius));
        }
    }
    bounds.finish()
}

/// Returns whether a cheap bounds reduction has at least two useful chunks.
fn particle_bounds_parallel_is_worthwhile(point_count: usize) -> bool {
    point_count >= PARTICLE_BOUNDS_PARALLEL_MIN_POINTS && rayon::current_num_threads() > 1
}

#[derive(Default)]
struct BoundsAccumulator {
    min: Option<Vec3>,
    max: Option<Vec3>,
}

impl BoundsAccumulator {
    fn include(&mut self, point: Vec3) {
        if !point.is_finite() {
            return;
        }
        self.min = Some(self.min.map_or(point, |min| min.min(point)));
        self.max = Some(self.max.map_or(point, |max| max.max(point)));
    }

    fn merge(&mut self, other: Self) {
        if let Some(min) = other.min {
            self.include(min);
        }
        if let Some(max) = other.max {
            self.include(max);
        }
    }

    fn finish(self) -> RenderBoundingBox {
        match (self.min, self.max) {
            (Some(min), Some(max)) => RenderBoundingBox {
                center: (min + max) * 0.5,
                extents: (max - min).abs() * 0.5,
            },
            _ => RenderBoundingBox {
                center: Vec3::ZERO,
                extents: Vec3::ZERO,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_mesh_ids_are_negative_and_distinct_by_kind() {
        assert_eq!(billboard_render_buffer_mesh_asset_id(1), Some(-11));
        assert_eq!(
            trail_render_buffer_mesh_asset_id(1, TrailTextureMode::Stretch),
            Some(-12)
        );
        assert_ne!(
            trail_render_buffer_mesh_asset_id(1, TrailTextureMode::Stretch),
            trail_render_buffer_mesh_asset_id(1, TrailTextureMode::Tile)
        );
    }

    #[test]
    fn generated_mesh_ids_reject_negative_source_assets() {
        assert_eq!(billboard_render_buffer_mesh_asset_id(-1), None);
    }

    #[test]
    fn generated_mesh_classification_is_kind_specific() {
        let billboard = billboard_render_buffer_mesh_asset_id(3).unwrap();
        let trail = trail_render_buffer_mesh_asset_id(3, TrailTextureMode::Stretch).unwrap();

        assert!(is_generated_particle_mesh_asset_id(billboard));
        assert!(is_generated_billboard_mesh_asset_id(billboard));
        assert!(!is_generated_trail_mesh_asset_id(billboard));
        assert!(is_generated_particle_mesh_asset_id(trail));
        assert!(!is_generated_billboard_mesh_asset_id(trail));
        assert!(is_generated_trail_mesh_asset_id(trail));
        assert!(!is_generated_particle_mesh_asset_id(-2));
        assert!(!is_generated_billboard_mesh_asset_id(-2));
    }

    #[test]
    fn point_decode_reads_float3_sizes() {
        let positions_offset = 0;
        let rotations_offset = positions_offset + 12;
        let sizes_offset = rotations_offset + 16;
        let colors_offset = sizes_offset + 12;
        let frames_offset = colors_offset + 16;
        let mut raw = Vec::new();
        raw.extend_from_slice(bytemuck::cast_slice(&[[1.0f32, 2.0, 3.0]]));
        raw.extend_from_slice(bytemuck::cast_slice(&[[0.0f32, 0.0, 0.0, 1.0]]));
        raw.extend_from_slice(bytemuck::cast_slice(&[[4.0f32, 5.0, 6.0]]));
        raw.extend_from_slice(bytemuck::cast_slice(&[[0.25f32, 0.5, 0.75, 1.0]]));
        raw.extend_from_slice(bytemuck::cast_slice(&[7u16]));
        let upload = PointRenderBufferUpload {
            asset_id: 12,
            count: 1,
            positions_offset,
            rotations_offset,
            sizes_offset,
            colors_offset,
            frame_indexes_offset: frames_offset,
            frame_grid_size: glam::IVec2::new(4, 4),
            ..Default::default()
        };

        let points = decode_point_particles(&raw, &upload, 1).unwrap();

        assert_eq!(points[0].position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(points[0].size, Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(points[0].frame_index, Some(7));
    }

    #[test]
    fn billboard_fill_writes_stable_point_indices() {
        let points = vec![
            PointParticle {
                position: Vec3::new(1.0, 2.0, 3.0),
                rotation: Quat::IDENTITY,
                size: Vec3::splat(1.0),
                color: Vec4::ONE,
                frame_index: None,
            },
            PointParticle {
                position: Vec3::new(4.0, 5.0, 6.0),
                rotation: Quat::IDENTITY,
                size: Vec3::splat(2.0),
                color: Vec4::ONE,
                frame_index: Some(1),
            },
        ];
        let mut vertices =
            vec![0u8; points.len() * BILLBOARD_VERTICES_PER_POINT * generated_vertex_stride()];
        let mut indices = vec![0u8; points.len() * BILLBOARD_INDICES_PER_POINT * size_of::<u32>()];

        fill_billboard_buffers(&points, glam::IVec2::new(2, 1), &mut vertices, &mut indices);

        let index_words: &[u32] = bytemuck::cast_slice(&indices);
        assert_eq!(
            index_words,
            &[
                0, 1, 2, 2, 1, 3, 0, 2, 1, 2, 3, 1, 4, 5, 6, 6, 5, 7, 4, 6, 5, 6, 7, 5
            ]
        );
        let first_vertex: &[f32] = bytemuck::cast_slice(&vertices[..generated_vertex_stride()]);
        assert_eq!(&first_vertex[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&first_vertex[6..8], &[0.0, 0.0]);
    }

    #[test]
    fn trail_decode_precomputes_distances() {
        let trails_offset = 0;
        let positions_offset = trails_offset + TRAIL_OFFSET_BYTES as i32;
        let colors_offset = positions_offset + 3 * 12;
        let sizes_offset = colors_offset + 3 * 16;
        let mut raw = Vec::new();
        raw.extend_from_slice(bytemuck::cast_slice(&[[0i32, 3, 0, 3]]));
        raw.extend_from_slice(bytemuck::cast_slice(&[
            [0.0f32, 0.0, 0.0],
            [3.0f32, 0.0, 0.0],
            [3.0f32, 4.0, 0.0],
        ]));
        raw.extend_from_slice(bytemuck::cast_slice(&[
            [1.0f32, 1.0, 1.0, 1.0],
            [1.0f32, 1.0, 1.0, 1.0],
            [1.0f32, 1.0, 1.0, 1.0],
        ]));
        raw.extend_from_slice(bytemuck::cast_slice(&[1.0f32, 1.0, 1.0]));
        let upload = TrailRenderBufferUpload {
            asset_id: 77,
            trails_count: 1,
            trail_point_count: 3,
            trails_offset,
            positions_offset,
            colors_offset,
            sizes_offset,
            ..Default::default()
        };

        let trails = decode_trails(&raw, &upload, 1, 3).unwrap();

        assert_eq!(trails[0].distances, vec![0.0, 3.0, 7.0]);
        assert_eq!(trails[0].total_distance, 7.0);
    }

    #[test]
    fn trail_offset_wraps_inside_capacity() {
        let offset = TrailOffset {
            offset: 10,
            capacity: 4,
            start: 3,
            count: 4,
        };

        let indexes: Vec<_> = (0..4)
            .filter_map(|logical_index| offset.point_index(logical_index))
            .collect();

        assert_eq!(indexes, vec![13, 10, 11, 12]);
    }

    #[test]
    fn trail_texture_modes_generate_expected_v_coordinates() {
        let distances = vec![0.0, 2.0, 5.0];

        assert_eq!(
            trail_v_coordinate(TrailTextureMode::Stretch, &distances, 5.0, 1),
            0.4
        );
        assert_eq!(
            trail_v_coordinate(TrailTextureMode::Tile, &distances, 5.0, 1),
            2.0
        );
        assert_eq!(
            trail_v_coordinate(TrailTextureMode::DistributePerSegment, &distances, 5.0, 1),
            0.5
        );
        assert_eq!(
            trail_v_coordinate(TrailTextureMode::RepeatPerSegment, &distances, 5.0, 2),
            2.0
        );
    }

    #[test]
    fn trail_mesh_chunks_merge_to_full_mesh_bytes() {
        let trails = vec![
            test_trail(&[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ]),
            test_trail(&[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::new(0.0, 3.0, 0.0),
            ]),
            test_trail(&[Vec3::new(1.0, 1.0, 0.0), Vec3::new(2.0, 2.0, 0.0)]),
        ];
        let offsets = trail_vertex_offsets(&trails);
        let full = build_trail_mesh_chunk(&trails, &offsets, TrailTextureMode::Stretch, 0..3);
        let mut chunked = TrailMeshChunk {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
        for range in [0..1, 1..3] {
            let mut chunk =
                build_trail_mesh_chunk(&trails, &offsets, TrailTextureMode::Stretch, range);
            chunked.vertices.append(&mut chunk.vertices);
            chunked.indices.append(&mut chunk.indices);
        }

        assert_eq!(chunked.vertices, full.vertices);
        assert_eq!(chunked.indices, full.indices);
    }

    #[test]
    fn trail_chunk_budget_requires_two_chunks_for_parallel_decode() {
        let offsets = vec![
            Some(TrailOffset {
                offset: 0,
                capacity: 4,
                start: 0,
                count: 4,
            }),
            Some(TrailOffset {
                offset: 4,
                capacity: 4,
                start: 0,
                count: 4,
            }),
        ];
        let chunks = trail_chunks_by_point_budget(&offsets, 4);

        assert_eq!(chunks, vec![0..1, 1..2]);
        assert!(
            trail_decode_parallel_is_worthwhile(TRAIL_DECODE_PARALLEL_MIN_POINTS, 2)
                || rayon::current_num_threads() == 1
        );
        assert!(!trail_decode_parallel_is_worthwhile(
            TRAIL_DECODE_PARALLEL_MIN_POINTS,
            1
        ));
    }

    fn test_trail(points: &[Vec3]) -> TrailPolyline {
        let points = points
            .iter()
            .copied()
            .map(|position| TrailPoint {
                position,
                color: Vec4::ONE,
                width: 1.0,
            })
            .collect::<Vec<_>>();
        let distances = trail_distances(&points);
        let total_distance = distances.last().copied().unwrap_or(0.0).max(1e-6);
        TrailPolyline {
            points,
            distances,
            total_distance,
        }
    }
}
