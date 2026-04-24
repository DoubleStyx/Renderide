//! Unit tests for [`super::view_math`].

use glam::{Mat4, Quat, Vec3, Vec4};
use openxr as xr;

use super::view_math::{
    headset_center_pose_from_stereo_views, openxr_pose_to_engine, openxr_pose_to_host_tracking,
    ref_from_view_matrix, view_projection_from_xr_view,
};

fn pose_identity() -> xr::Posef {
    xr::Posef {
        orientation: xr::Quaternionf {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
        position: xr::Vector3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    }
}

#[test]
fn identity_pose_maps_to_identity_ref_from_view() {
    let m = ref_from_view_matrix(&pose_identity());
    assert!(
        m.abs_diff_eq(Mat4::IDENTITY, 1e-4),
        "expected identity ref_from_view, got {m:?}"
    );
}

#[test]
fn identity_openxr_pose_maps_to_identity_engine_quat() {
    let (_p, q) = openxr_pose_to_engine(&pose_identity());
    assert!(
        q.abs_diff_eq(Quat::IDENTITY, 1e-4),
        "expected identity engine orientation, got {q:?}"
    );
}

#[test]
fn host_tracking_pose_converts_to_unity_lh() {
    // OpenXR RH (-Z forward) -> FrooxEngine/Unity LH (+Z forward):
    //   position: (x, y, -z)
    //   rotation: (-qx, -qy, qz, qw)
    let pose = xr::Posef {
        orientation: xr::Quaternionf {
            x: 0.1,
            y: 0.2,
            z: 0.3,
            w: 0.9,
        },
        position: xr::Vector3f {
            x: 1.0,
            y: 2.0,
            z: -3.0,
        },
    };
    let (p, q) = openxr_pose_to_host_tracking(&pose);
    assert!(p.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-5));
    let o = pose.orientation;
    let q_expected = Quat::from_xyzw(-o.x, -o.y, o.z, o.w).normalize();
    assert!(q.abs_diff_eq(q_expected, 1e-4));
}

#[test]
fn headset_center_pose_averages_positions_and_slerps_rotation() {
    let pose_l = xr::Posef {
        orientation: xr::Quaternionf {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
        position: xr::Vector3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    };
    let pose_r = xr::Posef {
        orientation: xr::Quaternionf {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
        position: xr::Vector3f {
            x: 0.2,
            y: 0.0,
            z: 0.0,
        },
    };
    let views = [
        xr::View {
            pose: pose_l,
            fov: xr::Fovf {
                angle_left: 0.0,
                angle_right: 0.0,
                angle_up: 0.0,
                angle_down: 0.0,
            },
        },
        xr::View {
            pose: pose_r,
            fov: xr::Fovf {
                angle_left: 0.0,
                angle_right: 0.0,
                angle_up: 0.0,
                angle_down: 0.0,
            },
        },
    ];
    let (p, q) = headset_center_pose_from_stereo_views(&views).expect("center pose");
    let (pl, _) = openxr_pose_to_host_tracking(&pose_l);
    let (pr, _) = openxr_pose_to_host_tracking(&pose_r);
    let expected_p = (pl + pr) * 0.5;
    assert!(
        p.abs_diff_eq(expected_p, 1e-4),
        "p={p:?} expected {expected_p:?}"
    );
    assert!(q.abs_diff_eq(Quat::IDENTITY, 1e-4));
}

#[test]
fn pitch_up_moves_forward_point_up_in_clip_space() {
    // OpenXR uses right-handed pose rotations with -Z forward, so physical "look up"
    // corresponds to a negative X rotation.
    let angle = -0.3_f32;
    let q_xr = Quat::from_rotation_x(angle);
    let view = xr::View {
        pose: xr::Posef {
            orientation: xr::Quaternionf {
                x: q_xr.x,
                y: q_xr.y,
                z: q_xr.z,
                w: q_xr.w,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        },
        fov: xr::Fovf {
            angle_left: -0.45,
            angle_right: 0.45,
            angle_up: 0.45,
            angle_down: -0.45,
        },
    };
    let vp = view_projection_from_xr_view(&view, 0.01, 100.0);
    // Host/scene forward is +Z (Unity LH basis). Looking up should move a forward point upward
    // in clip space, not downward.
    let clip = vp * Vec4::new(0.0, 0.0, 1.0, 1.0);
    let ndc_y = clip.y / clip.w;
    assert!(
        ndc_y > 0.0,
        "pitch up should move a forward point upward in clip space, clip={clip:?}"
    );
}

/// An OpenXR pose whose quaternion has zero length cannot be normalized; the conversion must fall
/// back to identity rather than produce NaN components.
#[test]
fn non_finite_quat_falls_back_to_identity() {
    let pose = xr::Posef {
        orientation: xr::Quaternionf {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
        position: xr::Vector3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    };
    let (_p, q) = super::view_math::openxr_pose_to_host_tracking(&pose);
    assert!(q.abs_diff_eq(Quat::IDENTITY, 1e-4));
}

/// [`headset_center_pose_from_stereo_views`] returns [`None`] for an empty view slice and the raw
/// mono pose (not the averaged one) for a single-view slice.
#[test]
fn headset_center_pose_handles_empty_and_single_view() {
    use super::view_math::{headset_pose_from_xr_view, openxr_pose_to_host_tracking};

    let views: [xr::View; 0] = [];
    assert!(headset_center_pose_from_stereo_views(&views).is_none());

    let pose = xr::Posef {
        orientation: xr::Quaternionf {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
        position: xr::Vector3f {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
    };
    let fov = xr::Fovf {
        angle_left: 0.0,
        angle_right: 0.0,
        angle_up: 0.0,
        angle_down: 0.0,
    };
    let single = [xr::View { pose, fov }];
    let (p, q) = headset_center_pose_from_stereo_views(&single).expect("single-view pose");
    let (expected_p, expected_q) = headset_pose_from_xr_view(&single[0]);
    assert!(p.abs_diff_eq(expected_p, 1e-5));
    assert!(q.abs_diff_eq(expected_q, 1e-5));

    let (raw_p, _) = openxr_pose_to_host_tracking(&pose);
    assert!(p.abs_diff_eq(raw_p, 1e-5));
}

#[test]
fn yaw_right_moves_forward_point_left_in_clip_space() {
    // OpenXR uses right-handed pose rotations with -Z forward, so physical "look right"
    // corresponds to a negative Y rotation.
    let angle = -0.3_f32;
    let q_xr = Quat::from_rotation_y(angle);
    let view = xr::View {
        pose: xr::Posef {
            orientation: xr::Quaternionf {
                x: q_xr.x,
                y: q_xr.y,
                z: q_xr.z,
                w: q_xr.w,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        },
        fov: xr::Fovf {
            angle_left: -0.45,
            angle_right: 0.45,
            angle_up: 0.45,
            angle_down: -0.45,
        },
    };
    let vp = view_projection_from_xr_view(&view, 0.01, 100.0);
    // Host/scene forward is +Z (Unity LH basis). Looking right should move a forward point to
    // the left in clip space, not to the right.
    let clip = vp * Vec4::new(0.0, 0.0, 1.0, 1.0);
    let ndc_x = clip.x / clip.w;
    assert!(
        ndc_x < 0.0,
        "yaw right should move a forward point left in clip space, clip={clip:?}"
    );
}
