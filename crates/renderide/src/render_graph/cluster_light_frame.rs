//! Single source of truth for clustered light [`ClusterParams`](crate::render_graph::passes::clustered_light)
//! view/projection and for [`crate::gpu::frame_globals::FrameGpuUniforms::view_space_z_coeffs`].
//!
//! Both the clustered light compute pass and fragment `cluster_id_from_frag` (`pbs_cluster.wgsl`)
//! must use the same world-to-view matrix (per eye in stereo) so cluster AABBs, view-space Z, and spotlight
//! axes agree in one view space.

use glam::Mat4;

use crate::scene::SceneCoordinator;

use super::frame_params::HostCameraFrame;
use super::world_mesh_cull::build_world_mesh_cull_proj_params;
use super::{effective_head_output_clip_planes, view_matrix_from_render_transform};

/// One stereo eye: world-to-view and reverse-Z projection used by clustered light compute.
#[derive(Clone, Copy, Debug)]
pub struct ClusterStereoEye {
    /// World-to-view (same basis as mesh `view_matrix_from_render_transform`).
    pub scene_view: Mat4,
    /// Reverse-Z perspective projection for this eye (matches `inv_proj` in cluster AABB build).
    pub proj: Mat4,
}

/// Camera bundle shared by [`super::passes::clustered_light::ClusteredLightPass`] and world forward frame uniforms.
#[derive(Clone, Copy, Debug)]
pub struct ClusterLightFrameParams {
    /// Desktop / mono world-to-view from the active main render space.
    pub mono_scene_view: Mat4,
    /// Main reverse-Z projection (same as [`super::world_mesh_cull::WorldMeshCullProjParams::world_proj`]).
    pub mono_proj: Mat4,
    pub near_clip: f32,
    pub far_clip: f32,
    /// Per-eye view + proj when multiview stereo uses OpenXR (or best-effort decomposed fallback).
    pub stereo_eyes: Option<(ClusterStereoEye, ClusterStereoEye)>,
}

/// Builds [`ClusterLightFrameParams`] for the current frame.
///
/// `mono_scene_view` and `mono_proj` match [`build_world_mesh_cull_proj_params`] and the forward pass
/// world projection path so [`FrameGpuUniforms::view_space_z_coeffs_from_world_to_view`] and
/// `ClusterParams.view` use the same world-to-view for mono.
pub fn cluster_light_frame_params(
    hc: &HostCameraFrame,
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
    use_multiview: bool,
) -> ClusterLightFrameParams {
    let (near_clip, far_clip) = effective_head_output_clip_planes(
        hc.near_clip,
        hc.far_clip,
        hc.output_device,
        scene
            .active_main_space()
            .map(|space| space.root_transform.scale),
    );

    let cull = build_world_mesh_cull_proj_params(scene, viewport_px, hc);
    let mono_proj = cull.world_proj;

    let mono_scene_view = scene
        .active_main_space()
        .map(|s| view_matrix_from_render_transform(&s.view_transform))
        .unwrap_or(Mat4::IDENTITY);

    let stereo_eyes = if use_multiview && hc.vr_active && hc.stereo_view_proj.is_some() {
        if let Some(((vl, pl), (vr, pr))) = hc.stereo_cluster {
            Some((
                ClusterStereoEye {
                    scene_view: vl,
                    proj: pl,
                },
                ClusterStereoEye {
                    scene_view: vr,
                    proj: pr,
                },
            ))
        } else if let Some((vp_l, vp_r)) = hc.stereo_view_proj {
            try_decompose_stereo_from_vp(vp_l, vp_r, mono_proj)
        } else {
            None
        }
    } else {
        None
    };

    ClusterLightFrameParams {
        mono_scene_view,
        mono_proj,
        near_clip,
        far_clip,
        stereo_eyes,
    }
}

/// Best-effort `V` from `P * V` when only combined view-projection is available (e.g. mirror path).
/// Assumes `mono_proj` matches the projection used inside `vp_l` / `vp_r`; otherwise culling may be wrong.
fn try_decompose_stereo_from_vp(
    vp_l: Mat4,
    vp_r: Mat4,
    mono_proj: Mat4,
) -> Option<(ClusterStereoEye, ClusterStereoEye)> {
    let inv_p = mono_proj.inverse();
    let v_l = inv_p * vp_l;
    let v_r = inv_p * vp_r;
    if !(matrix_is_finite(v_l) && matrix_is_finite(v_r)) {
        return None;
    }
    Some((
        ClusterStereoEye {
            scene_view: v_l,
            proj: mono_proj,
        },
        ClusterStereoEye {
            scene_view: v_r,
            proj: mono_proj,
        },
    ))
}

fn matrix_is_finite(m: Mat4) -> bool {
    m.to_cols_array().iter().all(|x| x.is_finite())
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::*;

    /// Mirrors `clustered_light.wgsl` spotlight sphere proxy (`SPOT_PENUMBRA_RAD` penumbra margin).
    mod spotlight_cpu {
        use glam::Vec3;

        const SPOT_PENUMBRA_RAD: f32 = 0.1;

        pub fn sphere_aabb_intersect(
            center: Vec3,
            radius: f32,
            aabb_min: Vec3,
            aabb_max: Vec3,
        ) -> bool {
            let closest = center.clamp(aabb_min, aabb_max);
            (center - closest).length_squared() <= radius * radius
        }

        pub fn spotlight_bounds_intersect_aabb(
            apex: Vec3,
            axis: Vec3,
            cos_half: f32,
            range: f32,
            aabb_min: Vec3,
            aabb_max: Vec3,
        ) -> bool {
            let cull_cos_half = (cos_half - SPOT_PENUMBRA_RAD).max(-1.0);
            if cull_cos_half >= 0.9999 {
                return sphere_aabb_intersect(apex, range, aabb_min, aabb_max);
            }
            let axis_n = axis.normalize();
            let sin_sq = (1.0 - cull_cos_half * cull_cos_half).max(0.0);
            let tan_sq = sin_sq / (cull_cos_half * cull_cos_half).max(1e-8);
            let radius = range * (0.25 + tan_sq).sqrt();
            let center = apex + axis_n * (range * 0.5);
            sphere_aabb_intersect(center, radius, aabb_min, aabb_max)
        }
    }

    #[test]
    fn mono_scene_view_matches_cull_proj_space() {
        let hc = HostCameraFrame::default();
        let scene = SceneCoordinator::default();
        let clf = cluster_light_frame_params(&hc, &scene, (1280, 720), false);
        let cull = build_world_mesh_cull_proj_params(&scene, (1280, 720), &hc);
        assert_eq!(clf.mono_proj, cull.world_proj);
    }

    #[test]
    fn view_space_z_coeffs_matches_mono_scene_view_row() {
        let hc = HostCameraFrame::default();
        let scene = SceneCoordinator::default();
        let clf = cluster_light_frame_params(&hc, &scene, (800, 600), false);
        let z = crate::gpu::frame_globals::FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(
            clf.mono_scene_view,
        );
        let m = clf.mono_scene_view;
        assert!((z[0] - m.x_axis.z).abs() < 1e-5);
        assert!((z[1] - m.y_axis.z).abs() < 1e-5);
        assert!((z[2] - m.z_axis.z).abs() < 1e-5);
        assert!((z[3] - m.w_axis.z).abs() < 1e-5);
    }

    #[test]
    fn stereo_right_eye_z_coeffs_match_decomposed_view() {
        let mut hc = HostCameraFrame::default();
        let right_view = Mat4::from_rotation_y(0.35);
        let proj = Mat4::IDENTITY;
        hc.stereo_cluster = Some(((Mat4::IDENTITY, proj), (right_view, proj)));
        hc.stereo_view_proj = Some((Mat4::IDENTITY, Mat4::IDENTITY));
        hc.vr_active = true;
        let scene = SceneCoordinator::default();
        let clf = cluster_light_frame_params(&hc, &scene, (800, 600), true);
        let right = clf.stereo_eyes.expect("stereo eyes").1;
        let z = crate::gpu::frame_globals::FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(
            right.scene_view,
        );
        let m = right.scene_view;
        assert!((z[0] - m.x_axis.z).abs() < 1e-5);
        assert!((z[1] - m.y_axis.z).abs() < 1e-5);
        assert!((z[2] - m.z_axis.z).abs() < 1e-5);
        assert!((z[3] - m.w_axis.z).abs() < 1e-5);
    }

    #[test]
    fn spotlight_sphere_proxy_changes_when_world_to_view_translation_wrong() {
        use spotlight_cpu::spotlight_bounds_intersect_aabb;

        let cos_half = 0.75;
        let range = 10.0;
        let aabb_min = Vec3::new(-0.5, -0.5, -8.0);
        let aabb_max = Vec3::new(0.5, 0.5, -0.5);

        let world_pos = Vec3::ZERO;
        let world_dir = Vec3::new(0.0, 0.0, -1.0);

        let view_correct = Mat4::IDENTITY;
        let pos_ok = view_correct.transform_point3(world_pos);
        let dir_ok = view_correct.transform_vector3(world_dir).normalize();
        let hit_ok =
            spotlight_bounds_intersect_aabb(pos_ok, dir_ok, cos_half, range, aabb_min, aabb_max);

        // Wrong world-to-view shifts the apex in view space while leaving direction unchanged (bug pattern).
        let view_wrong = Mat4::from_translation(Vec3::new(40.0, 0.0, 0.0));
        let pos_bad = view_wrong.transform_point3(world_pos);
        let dir_bad = view_wrong.transform_vector3(world_dir).normalize();
        let hit_bad =
            spotlight_bounds_intersect_aabb(pos_bad, dir_bad, cos_half, range, aabb_min, aabb_max);

        assert_ne!(
            hit_ok, hit_bad,
            "cluster culling must use the same world-to-view as the cluster grid"
        );
    }
}
