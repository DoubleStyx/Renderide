use std::sync::Arc;

use glam::{Quat, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::assets::mesh::{GpuMesh, MeshGpuUploadContext};
use crate::shared::PointRenderBufferUpload;

use super::bounds::bounds_for_points;
use super::ids::billboard_render_buffer_mesh_asset_id;
use super::types::{
    ParticleRenderBufferError, PointParticle, PointRenderBufferAsset, PointRenderBufferMeshUpload,
    checked_optional_range, checked_range, nonnegative_count, photondust_particle_color_to_linear,
    read_pod_at,
};
use super::upload::{
    GeneratedMeshUploadInput, generated_vertex_stride, upload_generated_mesh,
    write_generated_vertex, write_u32s,
};

/// Number of billboard vertices generated for one point particle.
pub(super) const BILLBOARD_VERTICES_PER_POINT: usize = 4;
/// Number of billboard indices generated for one point particle.
pub(super) const BILLBOARD_INDICES_PER_POINT: usize = 12;
/// Minimum point particles before Rayon decode/fill scheduling is worthwhile.
const POINT_PARTICLE_PARALLEL_MIN: usize = 2_048;
/// Point particle chunk size used by parallel vertex/index fill.
const POINT_PARTICLE_PARALLEL_CHUNK: usize = 1_024;

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

pub(super) fn decode_point_particles(
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
            color: photondust_particle_color_to_linear(Vec4::from_array(c)),
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
pub(super) fn fill_billboard_buffers(
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
    let row = rows - 1 - frame / columns;
    Vec2::new(
        (column as f32 + corner.x) / columns as f32,
        (row as f32 + corner.y) / rows as f32,
    )
}
