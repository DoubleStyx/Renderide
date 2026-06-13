use std::mem::size_of;

use glam::{Quat, Vec3, Vec4};

use super::bounds::{bounds_for_points, bounds_for_trails};
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
    TrailPolyline, build_trail_mesh_chunk, build_trail_render_buffer_cpu, decode_trails,
    trail_chunks_by_point_budget, trail_chunks_by_point_budget_from_trails,
    trail_decode_parallel_is_worthwhile, trail_distances, trail_v_coordinate, trail_vertex_offsets,
};
use super::types::{
    ParticleDrawKind, ParticleDrawParams, ParticleRenderBufferError, PointParticle,
    checked_optional_range, checked_range, nonnegative_count, photondust_particle_color_to_linear,
    read_pod_at,
};
use super::upload::{
    GeneratedExtraStreams, generated_mesh_upload_data, generated_vertex_stride,
    prepared_generated_derived_streams, push_generated_vertex, write_generated_vertex,
};
use crate::shared::{
    BillboardAlignment, MeshAlignment, MotionVectorMode, PointRenderBufferUpload,
    TrailRenderBufferUpload, TrailTextureMode,
};

fn point_at(position: Vec3, size: Vec3) -> PointParticle {
    PointParticle {
        position,
        rotation: Quat::IDENTITY,
        size,
        color: Vec4::ONE,
        frame_index: None,
    }
}

#[test]
fn point_bounds_use_max_size_radius_and_skip_non_finite_positions() {
    let empty = bounds_for_points(&[]);
    assert_eq!(empty.center, Vec3::ZERO);
    assert_eq!(empty.extents, Vec3::ZERO);

    // Radius comes from the absolute dominant size component, so negative sizes still expand.
    let single = bounds_for_points(&[point_at(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(-4.0, 1.0, 2.0),
    )]);
    assert_eq!(single.center, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(single.extents, Vec3::splat(2.0));

    let with_nan = bounds_for_points(&[
        point_at(Vec3::new(1.0, 2.0, 3.0), Vec3::new(-4.0, 1.0, 2.0)),
        point_at(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::ONE),
        point_at(Vec3::splat(f32::INFINITY), Vec3::ONE),
    ]);
    assert_eq!(with_nan.center, single.center);
    assert_eq!(with_nan.extents, single.extents);
}

#[test]
fn trail_bounds_merge_trails_with_width_radius() {
    let mut near = test_trail(&[Vec3::ZERO]);
    near.points[0].width = 2.0;
    let mut far = test_trail(&[Vec3::new(10.0, 0.0, 0.0)]);
    far.points[0].width = 4.0;

    let bounds = bounds_for_trails(&[near, far]);

    assert_eq!(bounds.center, Vec3::new(5.5, 0.0, 0.0));
    assert_eq!(bounds.extents, Vec3::new(6.5, 2.0, 2.0));
}

#[test]
fn point_bounds_parallel_reduction_matches_serial_min_max() {
    // Above the parallel admission threshold the chunked rayon reduction must produce exactly
    // the serial min/max result; f32 min/max is order-independent so equality is exact.
    let points: Vec<_> = (0..9_000)
        .map(|index| {
            let f = index as f32;
            point_at(Vec3::new(f, -f * 0.5, (f % 17.0) - 8.0), Vec3::splat(1.0))
        })
        .collect();

    let bounds = bounds_for_points(&points);

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for point in &points {
        let radius = point.size.abs().max_element() * 0.5;
        min = min.min(point.position - Vec3::splat(radius));
        max = max.max(point.position + Vec3::splat(radius));
    }
    assert_eq!(bounds.center, (min + max) * 0.5);
    assert_eq!(bounds.extents, (max - min).abs() * 0.5);
}

#[test]
fn render_buffer_validation_rejects_bad_host_rows() {
    assert!(matches!(
        nonnegative_count("point", 7, "count", -3),
        Err(ParticleRenderBufferError::NegativeCount {
            field: "count",
            value: -3,
            ..
        })
    ));
    assert_eq!(nonnegative_count("point", 7, "count", 5).unwrap(), 5);

    assert!(matches!(
        checked_range("point", 7, 64, "positions", -1, 1, 12),
        Err(ParticleRenderBufferError::MissingOffset {
            field: "positions",
            ..
        })
    ));
    assert!(matches!(
        checked_range("point", 7, 64, "positions", 0, usize::MAX, 16),
        Err(ParticleRenderBufferError::RangeOutOfBounds { .. })
    ));
    assert!(matches!(
        checked_range("point", 7, 64, "positions", 60, 1, 12),
        Err(ParticleRenderBufferError::RangeOutOfBounds {
            offset: 60,
            len: 12,
            raw_len: 64,
            ..
        })
    ));
    assert_eq!(
        checked_range("point", 7, 64, "positions", 4, 5, 12).unwrap(),
        4..64
    );

    // A negative offset on an optional stream means "absent", not an error.
    assert_eq!(
        checked_optional_range("point", 7, 64, "frame_indexes", -1, 5, 2).unwrap(),
        None
    );
    assert_eq!(
        checked_optional_range("point", 7, 64, "frame_indexes", 0, 5, 2).unwrap(),
        Some(0..10)
    );
}

#[test]
fn photondust_color_conversion_applies_srgb_only_to_ldr_channels() {
    use crate::color_space::srgb_channel_to_linear;

    let converted = photondust_particle_color_to_linear(Vec4::new(0.5, -0.5, 2.0, 0.5));
    assert_eq!(converted.x, srgb_channel_to_linear(0.5));
    assert_eq!(converted.y, srgb_channel_to_linear(-0.5));
    assert_eq!(converted.z, 2.0);
    assert_eq!(converted.w, 0.5);

    // Channels at or beyond +/-1 are HDR passthrough; alpha is never converted.
    let hdr = photondust_particle_color_to_linear(Vec4::new(1.0, -1.0, -1.5, 0.25));
    assert_eq!(hdr, Vec4::new(1.0, -1.0, -1.5, 0.25));
}

#[test]
fn read_pod_at_handles_unaligned_ranges() {
    let mut raw = vec![0xAAu8];
    raw.extend_from_slice(&1.5f32.to_le_bytes());
    raw.extend_from_slice(&(-2.5f32).to_le_bytes());

    let range = 1..raw.len();
    assert_eq!(read_pod_at::<f32>(&raw, &range, 0), 1.5);
    assert_eq!(read_pod_at::<f32>(&raw, &range, 1), -2.5);
}

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
    let raw_tangent = streams.raw_tangent.as_deref().expect("raw tangent");
    let uv1 = streams.uv1.as_deref().expect("uv1");
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

#[test]
fn point_decode_rejects_truncated_payload() {
    // Sized for one particle but the host claims two: the first stream read must fail cleanly.
    let mut raw = Vec::new();
    raw.extend_from_slice(bytemuck::cast_slice(&[[1.0f32, 2.0, 3.0]]));
    let upload = PointRenderBufferUpload {
        asset_id: 9,
        count: 2,
        positions_offset: 0,
        rotations_offset: 0,
        sizes_offset: 0,
        colors_offset: 0,
        frame_indexes_offset: -1,
        ..Default::default()
    };

    let err = decode_point_particles(&raw, &upload, 2).unwrap_err();

    assert!(matches!(
        err,
        ParticleRenderBufferError::RangeOutOfBounds {
            field: "positions",
            ..
        }
    ));
}

#[test]
fn billboard_frame_uv_clamps_overflow_and_ignores_empty_grids() {
    let corner_uv = |frame_index: Option<u16>, grid: glam::IVec2| {
        let points = vec![PointParticle {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            size: Vec3::ONE,
            color: Vec4::ONE,
            frame_index,
        }];
        let mut vertices = vec![0u8; BILLBOARD_VERTICES_PER_POINT * generated_vertex_stride()];
        let mut indices = vec![0u8; BILLBOARD_INDICES_PER_POINT * size_of::<u32>()];
        fill_billboard_buffers(&points, grid, &mut vertices, &mut indices);
        let vertex: &[f32] = bytemuck::cast_slice(&vertices[..generated_vertex_stride()]);
        [vertex[6], vertex[7]]
    };

    // Frame 9 in a 2x2 sheet clamps to the last frame (3): column 1, bottom row.
    assert_eq!(corner_uv(Some(9), glam::IVec2::new(2, 2)), [0.5, 0.0]);
    // Zero or negative grids fall back to the raw quad corner.
    assert_eq!(corner_uv(Some(2), glam::IVec2::new(0, 0)), [0.0, 0.0]);
    assert_eq!(corner_uv(Some(2), glam::IVec2::new(-2, 3)), [0.0, 0.0]);
}

#[test]
fn billboard_roll_encodes_z_rotation_in_point_data() {
    let points = vec![PointParticle {
        position: Vec3::ZERO,
        rotation: Quat::from_rotation_z(0.5),
        size: Vec3::new(2.0, 4.0, 0.0),
        color: Vec4::ONE,
        frame_index: None,
    }];
    let mut vertices = vec![0u8; BILLBOARD_VERTICES_PER_POINT * generated_vertex_stride()];
    let mut indices = vec![0u8; BILLBOARD_INDICES_PER_POINT * size_of::<u32>()];

    fill_billboard_buffers(&points, glam::IVec2::ONE, &mut vertices, &mut indices);

    let vertex: &[f32] = bytemuck::cast_slice(&vertices[..generated_vertex_stride()]);
    assert_eq!(&vertex[3..5], &[1.0, 2.0]);
    assert!((vertex[5] - 0.5).abs() < 1e-6);
}

#[test]
fn billboard_parallel_fill_keeps_per_particle_layout() {
    // Above the 512-point admission threshold the chunked parallel fill must produce the same
    // per-particle vertex placement and absolute index bases as the serial layout contract.
    let count = 600usize;
    let points: Vec<_> = (0..count)
        .map(|index| point_at(Vec3::new(index as f32, 0.0, 0.0), Vec3::ONE))
        .collect();
    let mut vertices = vec![0u8; count * BILLBOARD_VERTICES_PER_POINT * generated_vertex_stride()];
    let mut indices = vec![0u8; count * BILLBOARD_INDICES_PER_POINT * size_of::<u32>()];

    fill_billboard_buffers(&points, glam::IVec2::ONE, &mut vertices, &mut indices);

    let index_words: &[u32] = bytemuck::cast_slice(&indices);
    for particle in 0..count {
        let base = (particle * BILLBOARD_VERTICES_PER_POINT) as u32;
        assert_eq!(
            &index_words[particle * BILLBOARD_INDICES_PER_POINT
                ..(particle + 1) * BILLBOARD_INDICES_PER_POINT],
            &[base, base + 1, base + 2, base + 2, base + 1, base + 3]
        );
        let vertex_start = particle * BILLBOARD_VERTICES_PER_POINT * generated_vertex_stride();
        let vertex: &[f32] =
            bytemuck::cast_slice(&vertices[vertex_start..vertex_start + generated_vertex_stride()]);
        assert_eq!(&vertex[..3], &[particle as f32, 0.0, 0.0]);
    }
}

#[test]
fn invalid_trail_offset_rows_decode_to_empty_polylines() {
    let point_count = 4usize;
    let rows = [
        [-1i32, 2, 0, 2], // negative offset
        [0, 0, 0, 2],     // zero capacity
        [0, 2, 0, 0],     // zero count
        [1, 4, 0, 4],     // offset + capacity exceeds the point table
    ];
    let trails_offset = 0;
    let positions_offset = trails_offset + (rows.len() * TRAIL_OFFSET_BYTES) as i32;
    let colors_offset = positions_offset + (point_count * 12) as i32;
    let sizes_offset = colors_offset + (point_count * 16) as i32;
    let mut raw = Vec::new();
    raw.extend_from_slice(bytemuck::cast_slice(&rows));
    raw.extend_from_slice(bytemuck::cast_slice(&[[0.0f32; 3]; 4]));
    raw.extend_from_slice(bytemuck::cast_slice(&[[1.0f32; 4]; 4]));
    raw.extend_from_slice(bytemuck::cast_slice(&[1.0f32; 4]));
    let upload = TrailRenderBufferUpload {
        asset_id: 3,
        trails_count: rows.len() as i32,
        trail_point_count: point_count as i32,
        trails_offset,
        positions_offset,
        colors_offset,
        sizes_offset,
        ..Default::default()
    };

    let trails = decode_trails(&raw, &upload, rows.len(), point_count).unwrap();

    assert_eq!(trails.len(), rows.len());
    for trail in &trails {
        assert!(trail.points.is_empty());
        assert!(trail.distances.is_empty());
    }
}

#[test]
fn trail_count_clamps_to_capacity_and_wraps_ring_start() {
    // count 5 over capacity 2 clamps to 2; start 1 reads the ring as [p1, p0].
    let trails_offset = 0;
    let positions_offset = trails_offset + TRAIL_OFFSET_BYTES as i32;
    let colors_offset = positions_offset + 2 * 12;
    let sizes_offset = colors_offset + 2 * 16;
    let mut raw = Vec::new();
    raw.extend_from_slice(bytemuck::cast_slice(&[[0i32, 2, 1, 5]]));
    raw.extend_from_slice(bytemuck::cast_slice(&[
        [0.0f32, 0.0, 0.0],
        [5.0f32, 0.0, 0.0],
    ]));
    raw.extend_from_slice(bytemuck::cast_slice(&[[1.0f32; 4]; 2]));
    raw.extend_from_slice(bytemuck::cast_slice(&[1.0f32, -2.0]));
    let upload = TrailRenderBufferUpload {
        asset_id: 4,
        trails_count: 1,
        trail_point_count: 2,
        trails_offset,
        positions_offset,
        colors_offset,
        sizes_offset,
        ..Default::default()
    };

    let trails = decode_trails(&raw, &upload, 1, 2).unwrap();

    assert_eq!(trails[0].points.len(), 2);
    assert_eq!(trails[0].points[0].position, Vec3::new(5.0, 0.0, 0.0));
    assert_eq!(trails[0].points[1].position, Vec3::ZERO);
    // Negative host widths clamp to zero during decode.
    assert_eq!(trails[0].points[0].width, 0.0);
    assert_eq!(trails[0].points[1].width, 1.0);
}

#[test]
fn trail_sides_stay_finite_for_z_aligned_and_degenerate_tangents() {
    // A trail running along +Z has a tangent parallel to the ribbon normal; the side vector must
    // come from the Y fallback instead of a zero cross product.
    let z_trail = vec![test_trail(&[
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 2.0),
    ])];
    let offsets = trail_vertex_offsets(&z_trail);
    let chunk = build_trail_mesh_chunk(&z_trail, &offsets, TrailTextureMode::Stretch, 0..1);
    let floats: &[f32] = bytemuck::cast_slice(&chunk.vertices);
    assert!(floats.iter().all(|value| value.is_finite()));
    let first: &[f32] = bytemuck::cast_slice(&chunk.vertices[..generated_vertex_stride()]);
    assert!((first[0].abs() - 0.5).abs() < 1e-6);

    // Coincident points give a zero-length tangent; vertices must still be finite.
    let degenerate = vec![test_trail(&[Vec3::ONE, Vec3::ONE])];
    let offsets = trail_vertex_offsets(&degenerate);
    let chunk = build_trail_mesh_chunk(&degenerate, &offsets, TrailTextureMode::Stretch, 0..1);
    let floats: &[f32] = bytemuck::cast_slice(&chunk.vertices);
    assert!(floats.iter().all(|value| value.is_finite()));
}

#[test]
fn trail_chunk_budget_handles_empty_and_zero_targets() {
    assert!(trail_chunks_by_point_budget(&[], 4).is_empty());

    let trails = vec![test_trail(&[Vec3::ZERO, Vec3::X, Vec3::Y])];
    // A zero target clamps to one point per chunk instead of looping or panicking.
    assert_eq!(
        trail_chunks_by_point_budget_from_trails(&trails, 0),
        vec![0..1]
    );
}

#[test]
fn single_point_trails_emit_no_vertices_and_keep_base_vertices_aligned() {
    // A trail with one point produces no ribbon segments. The declared vertex count, the
    // per-trail base-vertex offsets, and the emitted vertex bytes must all agree so the next
    // trail's indices reference its own vertices.
    let trails = vec![
        test_trail(&[Vec3::ZERO]),
        test_trail(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 2.0, 0.0)]),
    ];
    let offsets = trail_vertex_offsets(&trails);
    assert_eq!(offsets, vec![0, 0, 4]);

    let chunk = build_trail_mesh_chunk(&trails, &offsets, TrailTextureMode::Stretch, 0..2);
    assert_eq!(
        chunk.vertices.len(),
        offsets.last().copied().unwrap_or(0) * generated_vertex_stride()
    );
    let index_words: &[u32] = bytemuck::cast_slice(&chunk.indices);
    assert_eq!(index_words, &[0, 1, 2, 2, 1, 3]);
}

#[test]
fn single_point_trail_build_declares_matching_vertex_count() {
    // End-to-end: one freshly spawned single-point trail followed by a two-point trail. The
    // generated mesh's declared vertex count must match the interleaved bytes actually built.
    let trails_offset = 0;
    let positions_offset = trails_offset + 2 * TRAIL_OFFSET_BYTES as i32;
    let colors_offset = positions_offset + 3 * 12;
    let sizes_offset = colors_offset + 3 * 16;
    let mut raw = Vec::new();
    raw.extend_from_slice(bytemuck::cast_slice(&[[0i32, 1, 0, 1], [1i32, 2, 0, 2]]));
    raw.extend_from_slice(bytemuck::cast_slice(&[
        [0.0f32, 0.0, 0.0],
        [1.0f32, 0.0, 0.0],
        [2.0f32, 0.0, 0.0],
    ]));
    raw.extend_from_slice(bytemuck::cast_slice(&[[1.0f32; 4]; 3]));
    raw.extend_from_slice(bytemuck::cast_slice(&[1.0f32, 1.0, 1.0]));
    let upload = TrailRenderBufferUpload {
        asset_id: 5,
        trails_count: 2,
        trail_point_count: 3,
        trails_offset,
        positions_offset,
        colors_offset,
        sizes_offset,
        ..Default::default()
    };

    let build = build_trail_render_buffer_cpu(raw.into(), &upload).unwrap();

    for mesh in &build.meshes {
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(
            mesh.vertices.len(),
            mesh.vertex_count * generated_vertex_stride()
        );
        assert_eq!(mesh.index_count, 6);
        let index_words: &[u32] = bytemuck::cast_slice(&mesh.indices);
        assert_eq!(index_words, &[0, 1, 2, 2, 1, 3]);
    }
}

#[test]
fn generated_mesh_ids_saturate_at_signed_encoding_boundary() {
    // The encoding packs source id * 8 + kind + 2 into the negative i32 space; the largest
    // billboard-encodable source asset is 268_435_455.
    const LAST_ENCODABLE: i32 = 268_435_455;

    let boundary = billboard_render_buffer_mesh_asset_id(LAST_ENCODABLE);
    assert!(boundary.is_some());
    assert!(is_generated_billboard_mesh_asset_id(boundary.unwrap()));
    assert_eq!(
        billboard_render_buffer_mesh_asset_id(LAST_ENCODABLE + 1),
        None
    );
    assert_eq!(billboard_render_buffer_mesh_asset_id(i32::MAX), None);
}

#[test]
fn trail_mesh_ids_round_trip_all_texture_modes() {
    let modes = [
        TrailTextureMode::Stretch,
        TrailTextureMode::Tile,
        TrailTextureMode::DistributePerSegment,
        TrailTextureMode::RepeatPerSegment,
    ];

    let ids: Vec<_> = modes
        .into_iter()
        .map(|mode| trail_render_buffer_mesh_asset_id(9, mode).unwrap())
        .collect();

    for &id in &ids {
        assert!(is_generated_trail_mesh_asset_id(id));
        assert!(!is_generated_billboard_mesh_asset_id(id));
    }
    let mut deduped = ids.clone();
    deduped.sort_unstable();
    deduped.dedup();
    assert_eq!(deduped.len(), ids.len());
}

#[test]
fn generated_vertex_push_and_write_produce_identical_bytes() {
    let position = Vec3::new(1.0, -2.0, 3.5);
    let normal = Vec3::new(0.0, 0.5, -1.0);
    let uv = glam::Vec2::new(0.25, 0.75);
    let color = Vec4::new(0.1, 0.2, 0.3, 0.4);

    let mut pushed = Vec::new();
    push_generated_vertex(&mut pushed, position, normal, uv, color);
    let mut written = vec![0u8; generated_vertex_stride()];
    write_generated_vertex(&mut written, position, normal, uv, color);

    assert_eq!(pushed, written);
}

#[test]
fn generated_mesh_upload_data_validates_counts() {
    let bounds = crate::shared::RenderBoundingBox {
        center: Vec3::ZERO,
        extents: Vec3::ONE,
    };

    // No indices means no submesh rows rather than a degenerate zero-length submesh.
    let empty = generated_mesh_upload_data(-11, 8, 0, bounds).unwrap();
    assert!(empty.submeshes.is_empty());
    assert_eq!(empty.vertex_count, 8);

    let populated = generated_mesh_upload_data(-11, 8, 12, bounds).unwrap();
    assert_eq!(populated.submeshes.len(), 1);
    assert_eq!(populated.submeshes[0].index_count, 12);

    assert!(matches!(
        generated_mesh_upload_data(-11, i32::MAX as usize + 1, 0, bounds),
        Err(ParticleRenderBufferError::MeshTooLarge { .. })
    ));
    assert!(matches!(
        generated_mesh_upload_data(-11, 8, i32::MAX as usize + 1, bounds),
        Err(ParticleRenderBufferError::MeshTooLarge { .. })
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
