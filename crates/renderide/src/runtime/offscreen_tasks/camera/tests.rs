//! Camera render-task readback and packing tests.

use glam::{IVec2, Vec3};

use super::super::cube_capture::{CubeCaptureBasisMode, CubeCaptureFace};
use super::result_write::{output_byte_count, pack_rgba8_to_host_buffer};
use super::*;

#[test]
fn camera_render_tasks_use_master_msaa_policy() {
    assert_eq!(
        RenderPathProfile::camera_readback(ViewPostProcessing::disabled()).sample_count_policy(),
        crate::render_graph::compiled::RenderPathSampleCountPolicy::MasterMsaa
    );
}

#[test]
fn camera360_requests_use_fov_threshold() {
    let regular = CameraRenderParameters {
        fov: 179.999,
        ..Default::default()
    };
    let camera360_min = CameraRenderParameters {
        fov: 180.0,
        ..Default::default()
    };
    let camera360_full = CameraRenderParameters {
        fov: 360.0,
        ..Default::default()
    };

    assert!(!camera360::camera_render_parameters_request_camera360(
        &regular
    ));
    assert!(camera360::camera_render_parameters_request_camera360(
        &camera360_min
    ));
    assert!(camera360::camera_render_parameters_request_camera360(
        &camera360_full
    ));
}

#[test]
fn camera360_face_size_matches_output_texel_budget() {
    assert_eq!(
        camera360::camera360_face_size_for_extent(CameraTaskExtent {
            width: 3840,
            height: 2160
        })
        .expect("face size"),
        2048
    );
    assert_eq!(
        camera360::camera360_face_size_for_extent(CameraTaskExtent {
            width: 6,
            height: 1
        })
        .expect("face size"),
        1
    );
    assert_eq!(
        camera360::camera360_face_size_for_extent(CameraTaskExtent {
            width: 48,
            height: 32
        })
        .expect("face size"),
        16
    );
}

#[test]
fn camera360_uses_copied_cubemap_basis_for_face_captures() {
    assert_eq!(
        camera360::CAMERA360_CUBE_BASIS_MODE,
        CubeCaptureBasisMode::Camera360Copied
    );
    assert_eq!(
        CubeCaptureFace::PosY
            .basis_for(camera360::CAMERA360_CUBE_BASIS_MODE)
            .right,
        Vec3::NEG_X
    );
    assert_ne!(
        CubeCaptureFace::PosY
            .basis_for(camera360::CAMERA360_CUBE_BASIS_MODE)
            .right,
        CubeCaptureFace::PosY.basis().right
    );
}

#[test]
fn readback_layout_removes_row_padding_contract() {
    let layout = compute_readback_layout(
        wgpu::Extent3d {
            width: 17,
            height: 3,
            depth_or_array_layers: 1,
        },
        4096,
    )
    .expect("layout");

    assert_eq!(layout.bytes_per_row_tight, 68);
    assert_eq!(
        layout.bytes_per_row_padded,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
    );
    assert_eq!(
        layout.buffer_size,
        u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * 3
    );
}

#[test]
fn copy_padded_rows_to_tight_strips_padding() {
    let layout = ReadbackLayout {
        width: 2,
        height: 2,
        bytes_per_row_tight: 8,
        bytes_per_row_padded: 12,
        buffer_size: 24,
    };
    let padded = [
        1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99, 10, 11, 12, 13, 14, 15, 16, 17, 88, 88, 88, 88,
    ];

    let tight = copy_padded_rows_to_tight(&padded, &layout).expect("copy rows");

    assert_eq!(
        tight,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
    );
}

#[test]
fn copy_padded_rows_to_tight_rejects_short_mapping() {
    let layout = ReadbackLayout {
        width: 2,
        height: 2,
        bytes_per_row_tight: 8,
        bytes_per_row_padded: 12,
        buffer_size: 24,
    };
    let err = copy_padded_rows_to_tight(&[0u8; 23], &layout).expect_err("short mapping");

    assert!(matches!(
        err,
        CameraReadbackError::MappedReadbackTooSmall {
            required: 24,
            actual: 23
        }
    ));
}

#[test]
fn pack_rgba8_preserves_rows_and_converts_formats() {
    let extent = CameraTaskExtent {
        width: 2,
        height: 2,
    };
    let rgba = [
        10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43,
    ];

    let mut argb = vec![0; 16];
    pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Argb32, &mut argb)
        .expect("argb pack");
    assert_eq!(
        argb,
        vec![
            13, 10, 11, 12, 23, 20, 21, 22, 33, 30, 31, 32, 43, 40, 41, 42
        ]
    );

    let mut rgba_out = vec![0; 16];
    pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Rgba32, &mut rgba_out)
        .expect("rgba pack");
    assert_eq!(
        rgba_out,
        vec![
            10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43
        ]
    );

    let mut bgra = vec![0; 16];
    pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Bgra32, &mut bgra)
        .expect("bgra pack");
    assert_eq!(
        bgra,
        vec![
            12, 11, 10, 13, 22, 21, 20, 23, 32, 31, 30, 33, 42, 41, 40, 43
        ]
    );

    let mut rgb = vec![0; 12];
    pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Rgb24, &mut rgb)
        .expect("rgb pack");
    assert_eq!(rgb, vec![10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]);
}

#[test]
fn pack_rgba8_rejects_small_source_without_touching_destination() {
    let extent = CameraTaskExtent {
        width: 2,
        height: 1,
    };
    let mut dst = [9u8; 8];

    let err = pack_rgba8_to_host_buffer(
        &[1, 2, 3, 4, 5, 6, 7],
        extent,
        CameraTaskOutputFormat::Rgba32,
        &mut dst,
    )
    .expect_err("small rgba source");

    assert!(matches!(
        err,
        CameraReadbackError::ResultDescriptorTooSmall {
            required: 8,
            actual: 7
        }
    ));
    assert_eq!(dst, [9u8; 8]);
}

#[test]
fn pack_rgba8_rejects_small_destination_without_writing_past_end() {
    let extent = CameraTaskExtent {
        width: 2,
        height: 1,
    };
    let rgba = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut dst = [9u8; 7];

    let error = pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Rgba32, &mut dst)
        .expect_err("small dst must fail");

    assert!(matches!(
        error,
        CameraReadbackError::ResultDescriptorTooSmall {
            required: 8,
            actual: 7
        }
    ));
    assert_eq!(dst, [9u8; 7]);
}

#[test]
fn task_extent_accepts_positive_dimensions() {
    let parameters = CameraRenderParameters {
        resolution: IVec2::new(4096, 2048),
        ..Default::default()
    };

    let extent = CameraTaskExtent::from_parameters(&parameters).expect("valid extent");

    assert_eq!(
        extent,
        CameraTaskExtent {
            width: 4096,
            height: 2048
        }
    );
    assert_eq!(extent.tuple(), (4096, 2048));
}

#[test]
fn task_extent_rejects_invalid_dimensions() {
    let parameters = CameraRenderParameters {
        resolution: IVec2::new(-1, 4),
        ..Default::default()
    };

    assert!(matches!(
        CameraTaskExtent::from_parameters(&parameters),
        Err(CameraReadbackError::InvalidExtent {
            width: -1,
            height: 4
        })
    ));
}

#[test]
fn task_extent_rejects_zero_dimensions() {
    for resolution in [IVec2::new(0, 4), IVec2::new(4, 0)] {
        let parameters = CameraRenderParameters {
            resolution,
            ..Default::default()
        };

        assert!(matches!(
            CameraTaskExtent::from_parameters(&parameters),
            Err(CameraReadbackError::InvalidExtent { .. })
        ));
    }
}

#[test]
fn compute_readback_layout_rejects_zero_extent() {
    let err = compute_readback_layout(
        wgpu::Extent3d {
            width: 0,
            height: 1,
            depth_or_array_layers: 1,
        },
        4096,
    )
    .expect_err("zero width");

    assert!(matches!(
        err,
        CameraReadbackError::InvalidExtent {
            width: 0,
            height: 1
        }
    ));
}

#[test]
fn compute_readback_layout_rejects_buffers_above_limit() {
    let err = compute_readback_layout(
        wgpu::Extent3d {
            width: 64,
            height: 2,
            depth_or_array_layers: 1,
        },
        255,
    )
    .expect_err("buffer over limit");

    assert!(matches!(
        err,
        CameraReadbackError::ReadbackBufferTooLarge { size, max: 255 }
            if size > 255
    ));
}

#[test]
fn output_byte_count_detects_overflow() {
    let err = output_byte_count(
        CameraTaskExtent {
            width: u32::MAX,
            height: u32::MAX,
        },
        CameraTaskOutputFormat::Rgba32,
    )
    .expect_err("overflow");

    assert!(matches!(err, CameraReadbackError::OutputByteCountOverflow));
}

#[test]
fn draw_filter_keeps_excludes_with_only_render_list() {
    let task = CameraRenderTask {
        only_render_list: vec![1, 2],
        exclude_render_list: vec![3],
        ..Default::default()
    };

    let filter = draw_filter_from_camera_render_task(&task);

    assert!(filter.only.as_ref().is_some_and(|only| only.contains(&1)));
    assert_eq!(filter.exclude.len(), 1);
    assert!(filter.exclude.contains(&3));
}

#[test]
fn draw_filter_uses_excludes_when_only_list_is_empty() {
    let task = CameraRenderTask {
        exclude_render_list: vec![7, 9, 7],
        ..Default::default()
    };

    let filter = draw_filter_from_camera_render_task(&task);

    assert!(filter.only.is_none());
    assert_eq!(filter.exclude.len(), 2);
    assert!(filter.exclude.contains(&7));
    assert!(filter.exclude.contains(&9));
}

#[test]
fn camera_task_layer_policy_uses_render_private_ui_parameter() {
    let public = CameraRenderParameters {
        render_private_ui: false,
        ..Default::default()
    };
    let private = CameraRenderParameters {
        render_private_ui: true,
        ..Default::default()
    };

    assert_eq!(
        camera_render_task_layer_policy(&public),
        ViewLayerPolicy::camera(false)
    );
    assert_eq!(
        camera_render_task_layer_policy(&private),
        ViewLayerPolicy::camera(true)
    );
}

#[test]
fn output_format_accepts_initial_cpu_formats() {
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::ARGB32),
        Some(CameraTaskOutputFormat::Argb32)
    );
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::RGBA32),
        Some(CameraTaskOutputFormat::Rgba32)
    );
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::BGRA32),
        Some(CameraTaskOutputFormat::Bgra32)
    );
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::RGB24),
        Some(CameraTaskOutputFormat::Rgb24)
    );
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::RGBAHalf),
        None
    );
}

#[test]
fn output_format_reports_expected_byte_widths() {
    assert_eq!(CameraTaskOutputFormat::Argb32.bytes_per_pixel(), 4);
    assert_eq!(CameraTaskOutputFormat::Rgba32.bytes_per_pixel(), 4);
    assert_eq!(CameraTaskOutputFormat::Bgra32.bytes_per_pixel(), 4);
    assert_eq!(CameraTaskOutputFormat::Rgb24.bytes_per_pixel(), 3);
}

#[test]
fn alpha_coverage_repair_only_runs_for_alpha_outputs() {
    assert!(CameraTaskOutputFormat::Argb32.needs_alpha_coverage_repair());
    assert!(CameraTaskOutputFormat::Rgba32.needs_alpha_coverage_repair());
    assert!(CameraTaskOutputFormat::Bgra32.needs_alpha_coverage_repair());
    assert!(!CameraTaskOutputFormat::Rgb24.needs_alpha_coverage_repair());
}

#[test]
fn alpha_coverage_uses_reverse_z_clear_contract() {
    assert!(!alpha_coverage::depth_marks_coverage(
        crate::gpu::MAIN_FORWARD_DEPTH_CLEAR
    ));
    assert!(!alpha_coverage::depth_marks_coverage(f32::NAN));
    assert!(alpha_coverage::depth_marks_coverage(
        crate::gpu::MAIN_FORWARD_DEPTH_CLEAR + f32::EPSILON
    ));
    assert!(alpha_coverage::depth_marks_coverage(1.0));
}

#[test]
fn camera_render_task_post_processing_policy_matches_host_parameters() {
    let disabled = CameraRenderParameters {
        post_processing: false,
        screen_space_reflections: true,
        ..Default::default()
    };
    let disabled_policy = camera_render_task_post_processing(&disabled);

    assert_eq!(disabled_policy, ViewPostProcessing::disabled());

    let enabled_without_ssr = CameraRenderParameters {
        post_processing: true,
        screen_space_reflections: false,
        ..Default::default()
    };
    let enabled_without_ssr_policy = camera_render_task_post_processing(&enabled_without_ssr);

    assert_eq!(
        enabled_without_ssr_policy,
        ViewPostProcessing::new(true, false, false)
    );

    let enabled_with_ssr = CameraRenderParameters {
        post_processing: true,
        screen_space_reflections: true,
        ..Default::default()
    };
    let enabled_with_ssr_policy = camera_render_task_post_processing(&enabled_with_ssr);

    assert_eq!(
        enabled_with_ssr_policy,
        ViewPostProcessing::new(true, true, false)
    );
}
