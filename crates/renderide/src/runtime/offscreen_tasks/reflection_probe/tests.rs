//! Reflection-probe render-task planning tests.

use glam::Vec3;
use hashbrown::HashSet;

use crate::camera::HostCameraFrame;
use crate::render_graph::FrameViewClear;
use crate::shared::{
    ReflectionProbeClear, ReflectionProbeRenderTask, ReflectionProbeState,
    ReflectionProbeTimeSlicingMode, ReflectionProbeType,
};

use super::face::{finite_positive_or, host_camera_frame_for_probe_face, reflection_probe_clip};
use super::*;

#[test]
fn reflection_probe_captures_use_single_sample_policy() {
    assert_eq!(
        RenderPathProfile::reflection_probe().sample_count_policy(),
        crate::render_graph::compiled::RenderPathSampleCountPolicy::SingleSample
    );
}

#[test]
fn probe_output_format_follows_hdr_flag() {
    assert_eq!(ProbeOutputFormat::from_hdr(false), ProbeOutputFormat::Rgba8);
    assert_eq!(
        ProbeOutputFormat::from_hdr(true),
        ProbeOutputFormat::Rgba16Float
    );
    assert_eq!(
        ProbeOutputFormat::Rgba8.bytes_per_pixel(),
        RGBA8_BYTES_PER_PIXEL
    );
    assert_eq!(
        ProbeOutputFormat::Rgba16Float.bytes_per_pixel(),
        RGBA16F_BYTES_PER_PIXEL
    );
}

#[test]
fn probe_task_extent_accepts_positive_size_and_reports_square_tuple() {
    let extent = ProbeTaskExtent::from_size(8).expect("valid size");

    assert_eq!(extent.size, 8);
    assert_eq!(extent.mip_levels, mip_levels_for_edge(8));
    assert_eq!(extent.tuple(), (8, 8));
}

#[test]
fn probe_task_extent_rejects_non_positive_sizes() {
    for size in [-2, 0] {
        let err = ProbeTaskExtent::from_size(size).expect_err("invalid size");

        assert!(matches!(err, ReflectionProbeBakeError::InvalidSize { size: s } if s == size));
    }
}

#[test]
fn onchanges_individual_faces_steps_one_remaining_face() {
    let faces = onchanges::onchanges_faces_for_step(
        ReflectionProbeTimeSlicingMode::IndividualFaces,
        0b0000_0011,
    );

    assert_eq!(faces, vec![ProbeCubeFace::PosY]);
}

#[test]
fn onchanges_all_faces_at_once_steps_all_remaining_faces() {
    let faces = onchanges::onchanges_faces_for_step(
        ReflectionProbeTimeSlicingMode::AllFacesAtOnce,
        0b0011_0001,
    );

    assert_eq!(
        faces,
        vec![
            ProbeCubeFace::NegX,
            ProbeCubeFace::PosY,
            ProbeCubeFace::NegY
        ]
    );
}

#[test]
fn onchanges_faces_returns_empty_when_all_faces_are_complete() {
    let faces = onchanges::onchanges_faces_for_step(
        ReflectionProbeTimeSlicingMode::AllFacesAtOnce,
        ProbeCubeFace::ALL_MASK,
    );

    assert!(faces.is_empty());
}

#[test]
fn runtime_capture_no_time_slicing_completes_in_one_tick() {
    assert_eq!(
        onchanges::runtime_capture_ticks_to_complete(ReflectionProbeTimeSlicingMode::NoTimeSlicing),
        1
    );
}

#[test]
fn runtime_capture_all_faces_time_slicing_completes_in_nine_ticks() {
    assert_eq!(
        onchanges::runtime_capture_ticks_to_complete(
            ReflectionProbeTimeSlicingMode::AllFacesAtOnce
        ),
        9
    );
}

#[test]
fn runtime_capture_individual_faces_time_slicing_completes_in_fourteen_ticks() {
    assert_eq!(
        onchanges::runtime_capture_ticks_to_complete(
            ReflectionProbeTimeSlicingMode::IndividualFaces
        ),
        14
    );
}

#[test]
fn realtime_capture_scheduler_ignores_non_realtime_and_solid_color_states() {
    assert!(!onchanges::realtime_probe_state_needs_capture(
        ReflectionProbeState {
            r#type: ReflectionProbeType::Baked,
            ..Default::default()
        }
    ));
    assert!(!onchanges::realtime_probe_state_needs_capture(
        ReflectionProbeState {
            r#type: ReflectionProbeType::OnChanges,
            ..Default::default()
        }
    ));
    assert!(!onchanges::realtime_probe_state_needs_capture(
        ReflectionProbeState {
            r#type: ReflectionProbeType::Realtime,
            clear_flags: ReflectionProbeClear::Color,
            ..Default::default()
        }
    ));
    assert!(onchanges::realtime_probe_state_needs_capture(
        ReflectionProbeState {
            r#type: ReflectionProbeType::Realtime,
            clear_flags: ReflectionProbeClear::Skybox,
            ..Default::default()
        }
    ));
}

#[test]
fn probe_face_projection_is_square_ninety_degrees() {
    let frame = host_camera_frame_for_probe_face(
        &HostCameraFrame::default(),
        ReflectionProbeState {
            near_clip: 0.1,
            far_clip: 100.0,
            ..Default::default()
        },
        (256, 256),
        Vec3::ZERO,
        ProbeCubeFace::PosZ,
    );
    let view = frame
        .explicit_view
        .expect("probe face should use explicit camera view");

    assert!((view.proj.x_axis.x - 1.0).abs() < 1e-6);
    assert!((view.proj.y_axis.y - 1.0).abs() < 1e-6);
}

#[test]
fn reflection_probe_clip_sanitizes_invalid_near_and_far() {
    let clip = reflection_probe_clip(ReflectionProbeState {
        near_clip: f32::NAN,
        far_clip: 0.001,
        ..Default::default()
    });

    assert!(clip.near >= 0.01);
    assert!(clip.far >= clip.near + 0.01);
}

#[test]
fn finite_positive_or_replaces_non_finite_and_non_positive_values() {
    assert_eq!(finite_positive_or(5.0, 1.0), 5.0);
    assert_eq!(finite_positive_or(0.0, 1.0), 1.0);
    assert_eq!(finite_positive_or(-3.0, 1.0), 1.0);
    assert_eq!(finite_positive_or(f32::INFINITY, 1.0), 1.0);
    assert_eq!(finite_positive_or(f32::NAN, 1.0), 1.0);
}

#[test]
fn clear_from_reflection_probe_state_uses_color_when_requested() {
    let color = glam::Vec4::new(0.1, 0.2, 0.3, 0.4);
    let clear = clear_from_reflection_probe_state(ReflectionProbeState {
        clear_flags: ReflectionProbeClear::Color,
        background_color: color,
        ..Default::default()
    });

    assert_eq!(clear, FrameViewClear::color(color));
}

#[test]
fn clear_from_reflection_probe_state_defaults_to_skybox() {
    let clear = clear_from_reflection_probe_state(ReflectionProbeState {
        clear_flags: ReflectionProbeClear::Skybox,
        background_color: glam::Vec4::ONE,
        ..Default::default()
    });

    assert_eq!(clear, FrameViewClear::skybox());
}

#[test]
fn reflection_probe_profile_disables_post_processing() {
    let policy = RenderPathProfile::reflection_probe().post_processing();

    assert!(!policy.is_enabled());
    assert!(!policy.screen_space_reflections);
    assert!(!policy.motion_blur);
}

#[test]
fn non_skybox_probe_excludes_requested_transforms() {
    let task = ReflectionProbeRenderTask {
        exclude_transform_ids: vec![4, 4, 8],
        ..Default::default()
    };
    let state = ReflectionProbeState::default();

    let filter = draw_filter_from_reflection_probe_task(&task, &state);

    assert!(filter.only.is_none());
    assert_eq!(filter.exclude.len(), 2);
    assert!(filter.exclude.contains(&4));
    assert!(filter.exclude.contains(&8));
}

#[test]
fn skybox_only_probe_uses_empty_selective_filter() {
    let task = ReflectionProbeRenderTask {
        exclude_transform_ids: vec![1, 2],
        ..Default::default()
    };
    let state = ReflectionProbeState {
        flags: 0b001,
        ..Default::default()
    };

    let filter = draw_filter_from_reflection_probe_task(&task, &state);

    assert!(filter.only.as_ref().is_some_and(HashSet::is_empty));
    assert!(filter.exclude.is_empty());
}

#[test]
fn state_only_draw_filter_passes_or_suppresses_everything() {
    let regular = draw_filter_from_reflection_probe_state(&ReflectionProbeState::default());
    assert!(regular.only.is_none());
    assert!(regular.exclude.is_empty());

    let skybox_only = draw_filter_from_reflection_probe_state(&ReflectionProbeState {
        flags: 0b001,
        ..Default::default()
    });
    assert!(skybox_only.only.as_ref().is_some_and(HashSet::is_empty));
    assert!(skybox_only.exclude.is_empty());
}
