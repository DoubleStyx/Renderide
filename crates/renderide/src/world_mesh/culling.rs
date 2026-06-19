//! View-projection parameters for CPU frustum culling of world mesh draws.
//!
//! Values match [`super::passes::world_mesh_forward::WorldMeshForwardOpaquePass`] per-space `view` and
//! global projection state (`HostCameraFrame`, viewport aspect, clip planes). When
//! [`HostCameraFrame::explicit_world_to_view`] returns a secondary-camera view, frustum and Hi-Z
//! temporal paths use that world-to-view (same as the forward pass) instead of
//! [`view_matrix_for_world_mesh_render_space`].

mod eval;
pub(crate) mod frustum;
mod geometry;

use crate::camera::{HostCameraFrame, WorldProjectionSet};
pub use crate::cull_contract::{HiZTemporalState, WorldMeshCullProjParams};
use crate::hi_z_cpu::HiZCullData;
use crate::scene::SceneSpaceRead;

pub(crate) use eval::{
    CpuCullFailure, mesh_cpu_cull_with_geometry, mesh_draw_passes_cpu_cull,
    overlay_rect_clip_visible, world_aabb_visible_for_cull,
};
pub(crate) use geometry::{
    MeshCullGeometry, MeshCullTarget, mesh_world_geometry_for_cull_with_head,
};

/// Host camera + projection bundle for [`super::draw_prep::queue_draws_with_parallelism`].
pub struct WorldMeshCullInput<'a> {
    /// Shared reverse-Z projection state for the frame.
    pub proj: WorldMeshCullProjParams,
    /// Per-frame head and clip data (bone palette and overlay projection parity).
    pub host_camera: &'a HostCameraFrame,
    /// Previous-frame hierarchical depth for optional occlusion after frustum tests.
    pub hi_z: Option<HiZCullData>,
    /// View/projection from the frame that authored [`Self::hi_z`]; required for stable temporal tests.
    pub hi_z_temporal: Option<HiZTemporalState>,
}

/// Builds [`WorldMeshCullProjParams`] from viewport size and [`HostCameraFrame`].
pub fn build_world_mesh_cull_proj_params(
    scene: &(impl SceneSpaceRead + ?Sized),
    viewport_px: (u32, u32),
    hc: &HostCameraFrame,
) -> WorldMeshCullProjParams {
    let projections = WorldProjectionSet::from_scene_host(scene, viewport_px, hc);

    WorldMeshCullProjParams {
        world_proj: projections.world_proj,
        overlay_proj: projections.overlay_proj,
        vr_stereo: projections.stereo_view_proj,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`build_world_mesh_cull_proj_params`].

    use glam::Mat4;

    use crate::scene::SceneCoordinator;

    use super::{WorldMeshCullProjParams, build_world_mesh_cull_proj_params};
    use crate::camera::HostCameraFrame;

    #[test]
    fn build_world_mesh_cull_proj_params_sets_vr_stereo_only_when_active_and_pair_present() {
        use crate::camera::{EyeView, StereoViewMatrices};
        let scene = SceneCoordinator::new();
        let left_view_proj = Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0));
        let right_view_proj = Mat4::from_translation(glam::Vec3::new(-1.0, 0.0, 0.0));
        let left_eye = EyeView::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            left_view_proj,
            glam::Vec3::new(-0.03, 0.0, 0.0),
        );
        let right_eye = EyeView::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            right_view_proj,
            glam::Vec3::new(0.03, 0.0, 0.0),
        );
        let stereo = Some(StereoViewMatrices::new(left_eye, right_eye));
        let hc = HostCameraFrame {
            vr_active: true,
            stereo,
            ..Default::default()
        };
        let p = build_world_mesh_cull_proj_params(&scene, (1280, 720), &hc);
        assert_eq!(p.vr_stereo, Some((left_view_proj, right_view_proj)));

        let hc2 = HostCameraFrame {
            vr_active: false,
            stereo,
            ..Default::default()
        };
        let p2 = build_world_mesh_cull_proj_params(&scene, (1280, 720), &hc2);
        assert!(p2.vr_stereo.is_none());
    }

    #[test]
    fn build_world_mesh_cull_proj_params_overlay_independent_of_primary_ortho_task() {
        // Regression: previously the screen-overlay path borrowed `primary_ortho_task`, which the
        // host populates from the first orthographic camera task -- typically the dash camera's
        // `OrthographicSize = 0.5f` task meant for dash-RT rendering, not for the screen overlay.
        // Sharing it produced a tiny half-meter overlay frustum and pushed the dash off-screen.
        // The screen overlay now uses a dedicated unit-height ortho independent of the host's
        // task list, so injecting an unrelated host orthographic task must not change it.
        use crate::camera::{CameraClipPlanes, OrthographicProjectionSpec};

        let scene = SceneCoordinator::new();
        let hc = HostCameraFrame::default();
        let p_no = build_world_mesh_cull_proj_params(&scene, (800, 600), &hc);

        let hc_ortho = HostCameraFrame {
            primary_ortho_task: Some(OrthographicProjectionSpec::new(
                10.0,
                CameraClipPlanes::new(0.01, 5000.0),
            )),
            ..Default::default()
        };
        let p_ortho = build_world_mesh_cull_proj_params(&scene, (800, 600), &hc_ortho);

        assert_eq!(p_no.world_proj, p_ortho.world_proj);
        assert_eq!(p_no.overlay_proj, p_ortho.overlay_proj);
    }

    #[test]
    fn cull_projection_mapping_applies_to_world_overlay_and_stereo() {
        let world = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ]);
        let overlay = Mat4::from_cols_array(&[
            2.0, 3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, 17.0,
        ]);
        let stereo_left = Mat4::from_cols_array(&[
            3.0, 4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, 10.0, //
            11.0, 12.0, 13.0, 14.0, //
            15.0, 16.0, 17.0, 18.0,
        ]);
        let stereo_right = Mat4::from_cols_array(&[
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0, //
            12.0, 13.0, 14.0, 15.0, //
            16.0, 17.0, 18.0, 19.0,
        ]);
        let projection_flip = Mat4::from_scale(glam::Vec3::new(1.0, -1.0, 1.0));
        let mapped = WorldMeshCullProjParams {
            world_proj: world,
            overlay_proj: overlay,
            vr_stereo: Some((stereo_left, stereo_right)),
        }
        .map_projection_matrices(|projection| projection_flip * projection);

        assert_eq!(mapped.world_proj, projection_flip * world);
        assert_eq!(mapped.overlay_proj, projection_flip * overlay);
        assert_eq!(
            mapped.vr_stereo,
            Some((
                projection_flip * stereo_left,
                projection_flip * stereo_right
            ))
        );
    }
}
