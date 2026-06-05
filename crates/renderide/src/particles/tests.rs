use std::mem::size_of;

use glam::{Quat, Vec3, Vec4};

use super::ids::{
    billboard_render_buffer_mesh_asset_id, is_generated_billboard_mesh_asset_id,
    is_generated_particle_mesh_asset_id, is_generated_trail_mesh_asset_id,
    trail_render_buffer_mesh_asset_id,
};
use super::point::{
    BILLBOARD_INDICES_PER_POINT, BILLBOARD_VERTICES_PER_POINT, billboard_extra_streams,
    decode_point_particles, fill_billboard_buffers,
};
use super::trail::{
    TRAIL_DECODE_PARALLEL_MIN_POINTS, TRAIL_OFFSET_BYTES, TrailMeshChunk, TrailOffset, TrailPoint,
    TrailPolyline, build_trail_mesh_chunk, decode_trails, trail_chunks_by_point_budget,
    trail_decode_parallel_is_worthwhile, trail_distances, trail_v_coordinate, trail_vertex_offsets,
};
use super::types::{ParticleDrawKind, ParticleDrawParams, PointParticle};
use super::upload::{
    GeneratedExtraStreams, generated_vertex_stride, prepared_generated_derived_streams,
};
use crate::shared::{
    BillboardAlignment, MeshAlignment, MotionVectorMode, PointRenderBufferUpload,
    TrailRenderBufferUpload, TrailTextureMode,
};

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
    raw.extend_from_slice(bytemuck::cast_slice(&[[0.25f32, 0.5, 1.25, 0.75]]));
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
    assert!(
        Vec4::new(0.050_876_09, 0.214_041_14, 1.25, 0.75).distance_squared(points[0].color) < 1e-5
    );
    assert_eq!(points[0].frame_index, Some(7));
}

#[test]
fn billboard_fill_writes_front_facing_point_indices() {
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
    assert_eq!(index_words, &[0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7]);
    let first_vertex: &[f32] = bytemuck::cast_slice(&vertices[..generated_vertex_stride()]);
    assert_eq!(&first_vertex[..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&first_vertex[3..5], &[0.5, 0.5]);
    assert_eq!(&first_vertex[6..8], &[0.0, 0.0]);
    let second_first_vertex_start = BILLBOARD_VERTICES_PER_POINT * generated_vertex_stride();
    let second_first_vertex: &[f32] = bytemuck::cast_slice(
        &vertices[second_first_vertex_start..second_first_vertex_start + generated_vertex_stride()],
    );
    assert_eq!(&second_first_vertex[6..8], &[0.5, 0.0]);
    let second_last_vertex_start = (BILLBOARD_VERTICES_PER_POINT + 3) * generated_vertex_stride();
    let second_last_vertex: &[f32] = bytemuck::cast_slice(
        &vertices[second_last_vertex_start..second_last_vertex_start + generated_vertex_stride()],
    );
    assert_eq!(&second_last_vertex[6..8], &[1.0, 1.0]);
}

#[test]
fn billboard_frame_uvs_advance_top_to_bottom() {
    let points: Vec<_> = (0u16..4)
        .map(|frame_index| PointParticle {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            size: Vec3::ONE,
            color: Vec4::ONE,
            frame_index: Some(frame_index),
        })
        .collect();
    let mut vertices =
        vec![0u8; points.len() * BILLBOARD_VERTICES_PER_POINT * generated_vertex_stride()];
    let mut indices = vec![0u8; points.len() * BILLBOARD_INDICES_PER_POINT * size_of::<u32>()];

    fill_billboard_buffers(&points, glam::IVec2::new(2, 2), &mut vertices, &mut indices);

    let uv = |point_index: usize, corner_index: usize| {
        let vertex_stride = generated_vertex_stride();
        let vertex_start =
            (point_index * BILLBOARD_VERTICES_PER_POINT + corner_index) * vertex_stride;
        let vertex: &[f32] =
            bytemuck::cast_slice(&vertices[vertex_start..vertex_start + vertex_stride]);
        [vertex[6], vertex[7]]
    };

    assert_eq!(uv(0, 0), [0.0, 0.5]);
    assert_eq!(uv(0, 3), [0.5, 1.0]);
    assert_eq!(uv(1, 0), [0.5, 0.5]);
    assert_eq!(uv(1, 3), [1.0, 1.0]);
    assert_eq!(uv(2, 0), [0.0, 0.0]);
    assert_eq!(uv(2, 3), [0.5, 0.5]);
    assert_eq!(uv(3, 0), [0.5, 0.0]);
    assert_eq!(uv(3, 3), [1.0, 0.5]);
}

#[test]
fn generated_particle_derived_streams_match_vertex_payloads() {
    let points = vec![PointParticle {
        position: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::IDENTITY,
        size: Vec3::splat(2.0),
        color: Vec4::new(0.25, 0.5, 0.75, 1.0),
        frame_index: None,
    }];
    let mut vertices =
        vec![0u8; points.len() * BILLBOARD_VERTICES_PER_POINT * generated_vertex_stride()];
    let mut indices = vec![0u8; points.len() * BILLBOARD_INDICES_PER_POINT * size_of::<u32>()];

    fill_billboard_buffers(&points, glam::IVec2::ONE, &mut vertices, &mut indices);
    let prepared = prepared_generated_derived_streams(
        &vertices,
        points.len() * BILLBOARD_VERTICES_PER_POINT,
        GeneratedExtraStreams::default(),
    );

    let positions = prepared.positions.as_deref().expect("positions");
    let normals = prepared.normals.as_deref().expect("normals");
    let uv0 = prepared.uv0.as_deref().expect("uv0");
    let color = prepared.color.as_deref().expect("color");
    let first_position: &[f32] = bytemuck::cast_slice(&positions[..16]);
    let first_normal: &[f32] = bytemuck::cast_slice(&normals[..16]);
    let first_uv: &[f32] = bytemuck::cast_slice(&uv0[..8]);
    let first_color: &[f32] = bytemuck::cast_slice(&color[..16]);

    assert_eq!(first_position, &[1.0, 2.0, 3.0, 1.0]);
    assert_eq!(first_normal, &[1.0, 1.0, 0.0, 0.0]);
    assert_eq!(first_uv, &[0.0, 0.0]);
    assert_eq!(first_color, &[0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn billboard_extra_streams_pack_particle_orientation() {
    let points = vec![PointParticle {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        size: Vec3::ONE,
        color: Vec4::ONE,
        frame_index: None,
    }];

    let streams = billboard_extra_streams(&points);
    let raw_tangent = streams.point_forward_upz.as_deref().expect("raw tangent");
    let uv1 = streams.point_up_xy.as_deref().expect("uv1");
    let tangent: &[f32] = bytemuck::cast_slice(&raw_tangent[..16]);
    let up_xy: &[f32] = bytemuck::cast_slice(&uv1[..8]);

    assert_eq!(tangent, &[0.0, 0.0, 1.0, 0.0]);
    assert_eq!(up_xy, &[0.0, 1.0]);
    assert_eq!(raw_tangent.len(), BILLBOARD_VERTICES_PER_POINT * 16);
    assert_eq!(uv1.len(), BILLBOARD_VERTICES_PER_POINT * 8);
}

#[test]
fn particle_draw_params_pack_uniform_rows() {
    let billboard = ParticleDrawParams::billboard(
        BillboardAlignment::Direction,
        0.125,
        0.75,
        MotionVectorMode::Object,
    );
    assert_eq!(billboard.kind, ParticleDrawKind::Billboard);
    assert_eq!(
        billboard.to_uniform_rows()[0],
        [ParticleDrawKind::Billboard as u32 as f32, 4.0, 0.125, 0.75]
    );

    let mesh = ParticleDrawParams::mesh(
        MeshAlignment::Facing,
        Vec4::new(0.25, 0.5, 0.75, 0.5),
        Some(13),
    );
    let mesh_rows = mesh.to_uniform_rows();
    assert_eq!(mesh_rows[0][0], ParticleDrawKind::Mesh as u32 as f32);
    assert_eq!(mesh_rows[0][1], MeshAlignment::Facing as u32 as f32);
    assert_eq!(mesh_rows[1], [0.25, 0.5, 0.75, 0.5]);
    assert_eq!(mesh_rows[2][1], 13.0);

    let trail = ParticleDrawParams::trail(TrailTextureMode::Tile, MotionVectorMode::Camera, true);
    let trail_rows = trail.to_uniform_rows();
    assert_eq!(trail_rows[0][0], ParticleDrawKind::Trail as u32 as f32);
    assert_eq!(trail_rows[2][0], MotionVectorMode::Camera as u32 as f32);
    assert_eq!(trail_rows[2][2], TrailTextureMode::Tile as u32 as f32);
    assert_eq!(trail_rows[2][3], 1.0);
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
        [0.25f32, 0.5, 1.25, 0.75],
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
    assert!(
        Vec4::new(0.050_876_09, 0.214_041_14, 1.25, 0.75)
            .distance_squared(trails[0].points[0].color)
            < 1e-5
    );
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
        let mut chunk = build_trail_mesh_chunk(&trails, &offsets, TrailTextureMode::Stretch, range);
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
