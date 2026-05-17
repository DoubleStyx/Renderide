//! Shared projection plans for world, overlay, stereo, and secondary camera paths.

use glam::Mat4;

use crate::scene::SceneCoordinator;

use super::{
    CameraClipPlanes, HostCameraFrame, Viewport, clamp_desktop_fov_degrees,
    effective_head_output_clip_planes, reverse_z_perspective,
};

/// Projection matrices shared by world-mesh culling and forward rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldProjectionSet {
    /// Effective clip planes after host output-device and head-output adjustment.
    pub clip: CameraClipPlanes,
    /// Viewport this projection was built for.
    pub viewport: Viewport,
    /// Projection for world draws.
    pub world_proj: Mat4,
    /// Projection for overlay draws.
    pub overlay_proj: Mat4,
    /// Stereo view-projection pair when the host camera is actively stereo.
    pub stereo_view_proj: Option<(Mat4, Mat4)>,
}

impl WorldProjectionSet {
    /// Builds world/overlay projection data from scene head-output scale, viewport, and host camera.
    pub fn from_scene_host(
        scene: &SceneCoordinator,
        viewport_px: (u32, u32),
        host_camera: &HostCameraFrame,
    ) -> Self {
        let viewport = Viewport::from_tuple(viewport_px);
        let explicit_view_projection = host_camera.explicit_view_projection();
        let explicit_proj = explicit_view_projection.map(|(_, proj)| proj);
        let clip = if explicit_proj.is_some() {
            host_camera.clip
        } else {
            let root_scale = scene
                .active_main_space()
                .map(|space| space.root_transform().scale);
            let (near, far) = effective_head_output_clip_planes(
                host_camera.clip.near,
                host_camera.clip.far,
                host_camera.output_device,
                root_scale,
            );
            CameraClipPlanes::new(near, far)
        };
        let fov_rad = clamp_desktop_fov_degrees(host_camera.desktop_fov_degrees).to_radians();
        let world_proj = explicit_proj.unwrap_or_else(|| {
            reverse_z_perspective(viewport.aspect(), fov_rad, clip.near, clip.far)
        });
        let overlay_proj =
            explicit_proj.unwrap_or_else(|| HostCameraFrame::overlay_projection(viewport, clip));
        let stereo_view_proj = host_camera
            .active_stereo()
            .map(|stereo| stereo.view_proj_pair());
        Self {
            clip,
            viewport,
            world_proj,
            overlay_proj,
            stereo_view_proj,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::WorldProjectionSet;
    use crate::camera::{CameraClipPlanes, EyeView, HostCameraFrame};
    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::{HeadOutputDevice, RenderTransform};

    #[test]
    fn explicit_secondary_projection_replaces_world_and_overlay_projection() {
        let scene = SceneCoordinator::new();
        let explicit_proj = Mat4::from_scale(Vec3::new(2.0, 3.0, 1.0));
        let host_camera = HostCameraFrame {
            explicit_view: Some(EyeView::new(
                Mat4::IDENTITY,
                explicit_proj,
                Mat4::IDENTITY,
                Vec3::ZERO,
            )),
            ..Default::default()
        };

        let set = WorldProjectionSet::from_scene_host(&scene, (1280, 720), &host_camera);

        assert_eq!(set.world_proj, explicit_proj);
        assert_eq!(set.overlay_proj, explicit_proj);
    }

    #[test]
    fn explicit_secondary_projection_keeps_resolved_clip_planes() {
        let mut scene = SceneCoordinator::new();
        let root = RenderTransform {
            scale: Vec3::splat(9.0),
            ..Default::default()
        };
        scene.test_seed_space_identity_worlds(RenderSpaceId(1), vec![root], vec![-1]);
        let host_camera = HostCameraFrame {
            clip: CameraClipPlanes::new(0.0002, 0.25),
            output_device: HeadOutputDevice::Screen360,
            explicit_view: Some(EyeView::new(
                Mat4::IDENTITY,
                Mat4::IDENTITY,
                Mat4::IDENTITY,
                Vec3::ZERO,
            )),
            ..Default::default()
        };

        let set = WorldProjectionSet::from_scene_host(&scene, (1280, 720), &host_camera);

        assert_eq!(set.clip, host_camera.clip);
    }

    #[test]
    fn main_projection_scales_near_but_not_far_by_root_scale() {
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(
            RenderSpaceId(1),
            vec![RenderTransform::default()],
            vec![-1],
        );
        scene.test_set_space_root_transform(
            RenderSpaceId(1),
            RenderTransform {
                scale: Vec3::splat(0.25),
                ..Default::default()
            },
        );
        let host_camera = HostCameraFrame {
            clip: CameraClipPlanes::new(0.01, 4096.0),
            output_device: HeadOutputDevice::Screen,
            ..Default::default()
        };

        let set = WorldProjectionSet::from_scene_host(&scene, (1280, 720), &host_camera);

        assert!((set.clip.near - 0.0025).abs() < 1e-6);
        assert!((set.clip.far - 4096.0).abs() < 1e-3);
    }
}
