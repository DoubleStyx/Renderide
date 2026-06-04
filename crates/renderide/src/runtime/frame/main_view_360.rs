//! Experimental cubemap-backed 360 projection for the desktop/headless main view.

use crate::camera::{HostCameraFrame, ViewId, WorldProjectionSet};
use crate::gpu::GpuContext;
use crate::render_graph::{FrameViewClear, GraphExecuteError, RenderPathProfile};
use crate::runtime::RendererRuntime;
use crate::runtime::frame::schedule::RenderScheduleKind;
use crate::runtime::frame::view_plan::{FrameViewPlan, FrameViewPlanParams, FrameViewPlanTarget};
use crate::runtime::offscreen_tasks::camera::CAMERA_TASK_COLOR_FORMAT;
use crate::runtime::offscreen_tasks::camera::camera360::{
    CAMERA360_CUBE_BASIS_MODE, camera360_face_size_for_dimensions, project_camera360_to_equirect,
};
use crate::runtime::offscreen_tasks::cube_capture::{
    CubeCaptureExtent, CubeCaptureFace, CubeCaptureTargetError, CubeCaptureTargets,
    host_camera_frame_for_cube_face_with_basis, render_cube_capture_faces_offscreen,
};
use crate::scene::{RenderSpaceId, SceneCoordinator};

/// Cached resources for the live main-view 360 capture.
pub(in crate::runtime) struct MainView360Resources {
    cube_targets: Option<MainView360CubeTargets>,
}

impl MainView360Resources {
    /// Builds an empty live main-view 360 resource cache.
    pub(in crate::runtime) const fn new() -> Self {
        Self { cube_targets: None }
    }

    fn cube_targets(
        &mut self,
        gpu: &GpuContext,
        output_extent_px: (u32, u32),
    ) -> Result<&CubeCaptureTargets, CubeCaptureTargetError> {
        let face_size = camera360_face_size_for_dimensions(output_extent_px.0, output_extent_px.1);
        let current = self.cube_targets.take();
        let next = match current {
            Some(targets)
                if targets.face_size == face_size
                    && targets.color_format == CAMERA_TASK_COLOR_FORMAT =>
            {
                targets
            }
            _ => create_main_view_360_cube_targets(gpu, face_size)?,
        };
        Ok(&self.cube_targets.insert(next).targets)
    }
}

struct MainView360CubeTargets {
    face_size: u32,
    color_format: wgpu::TextureFormat,
    targets: CubeCaptureTargets,
}

fn create_main_view_360_cube_targets(
    gpu: &GpuContext,
    face_size: u32,
) -> Result<MainView360CubeTargets, CubeCaptureTargetError> {
    let targets = CubeCaptureTargets::create(
        gpu,
        CubeCaptureExtent::new(face_size, 1),
        CAMERA_TASK_COLOR_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        "renderide-main-view-360-cube",
    )?;
    Ok(MainView360CubeTargets {
        face_size,
        color_format: CAMERA_TASK_COLOR_FORMAT,
        targets,
    })
}

impl RendererRuntime {
    /// Desktop/headless entry point for the experimental cubemap-backed main view.
    pub(in crate::runtime) fn render_desktop_360_frame(
        &mut self,
        gpu: &mut GpuContext,
    ) -> Result<(), GraphExecuteError> {
        self.render_desktop_secondaries_frame(gpu)?;
        self.render_main_view_360(gpu)
    }

    /// Returns whether the experimental main-view 360 path is enabled in renderer settings.
    pub(in crate::runtime) fn main_view_360_enabled(&self) -> bool {
        self.config
            .settings
            .read()
            .map(|settings| settings.experimental.main_view_360_enabled)
            .unwrap_or(false)
    }

    fn render_main_view_360(&mut self, gpu: &mut GpuContext) -> Result<(), GraphExecuteError> {
        profiling::scope!("main_view_360::render");
        self.sync_master_msaa(gpu);
        let output_targets = gpu.primary_offscreen_targets();
        let main_profile = if gpu.is_headless() {
            RenderPathProfile::headless_main()
        } else {
            RenderPathProfile::desktop_main()
        };

        let RendererRuntime {
            backend,
            scene,
            host_camera,
            tick_state,
            main_view_360,
            ..
        } = self;
        let cube_targets = match main_view_360.cube_targets(gpu, output_targets.extent_px) {
            Ok(targets) => targets,
            Err(error) => {
                logger::warn!("main view 360 target allocation failed: {error}");
                return Ok(());
            }
        };
        let pose = main_view_360_pose(scene, host_camera);
        let plans = plan_main_view_360_faces(
            scene,
            host_camera,
            pose,
            cube_targets,
            output_targets.extent_px,
            tick_state.frame_time_seconds(),
            main_profile,
        );
        render_cube_capture_faces_offscreen(
            RenderScheduleKind::Camera360Capture,
            gpu,
            backend,
            scene,
            plans,
        )?;
        project_camera360_to_equirect(
            gpu,
            cube_targets,
            &output_targets.color_view,
            output_targets.color_format,
            pose.projection_rotation,
            "main_view_360_projection",
        );
        Ok(())
    }
}

fn plan_main_view_360_faces(
    scene: &SceneCoordinator,
    base_camera: &HostCameraFrame,
    pose: MainView360Pose,
    cube_targets: &CubeCaptureTargets,
    output_extent_px: (u32, u32),
    frame_time_seconds: f32,
    main_profile: RenderPathProfile,
) -> Vec<FrameViewPlan<'static>> {
    let clip = WorldProjectionSet::from_scene_host(scene, output_extent_px, base_camera).clip;
    let face_viewport = cube_targets.extent.viewport();
    let render_context = scene.active_main_render_context();
    CubeCaptureFace::ALL
        .iter()
        .copied()
        .map(|face| {
            let host_camera = host_camera_frame_for_cube_face_with_basis(
                base_camera,
                clip,
                face_viewport,
                pose.position,
                face,
                CAMERA360_CUBE_BASIS_MODE,
            );
            FrameViewPlan::new(
                &host_camera,
                FrameViewPlanParams {
                    render_context,
                    frame_time_seconds,
                    view_id: main_view_360_face_view_id(face),
                    viewport_px: face_viewport,
                    clear: FrameViewClear::skybox(),
                    profile: RenderPathProfile::cube_capture(main_profile.post_processing()),
                    target: FrameViewPlanTarget::offscreen(cube_targets.to_offscreen_handles(face)),
                },
            )
        })
        .collect()
}

fn main_view_360_pose(scene: &SceneCoordinator, base_camera: &HostCameraFrame) -> MainView360Pose {
    let position = base_camera.view_origin_world();
    let projection_rotation = main_view_360_projection_rotation(main_view_360_view_rotation(scene));
    MainView360Pose {
        position,
        projection_rotation,
    }
}

fn main_view_360_view_rotation(scene: &SceneCoordinator) -> glam::Quat {
    scene
        .active_main_space()
        .map(|space| space.view_transform().rotation)
        .unwrap_or(glam::Quat::IDENTITY)
}

fn main_view_360_projection_rotation(view_rotation: glam::Quat) -> glam::Quat {
    view_rotation * glam::Quat::from_rotation_y(std::f32::consts::PI)
}

fn main_view_360_face_view_id(face: CubeCaptureFace) -> ViewId {
    ViewId::camera360_render_task_face(RenderSpaceId(0), -1, face.view_id_face_index())
}

#[derive(Clone, Copy)]
struct MainView360Pose {
    position: glam::Vec3,
    projection_rotation: glam::Quat,
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};
    use hashbrown::HashSet;

    use super::*;
    use crate::config::{RendererSettings, RendererSettingsHandle};
    use crate::connection::ConnectionParams;
    use crate::scene::RenderSpaceId;
    use crate::shared::RenderTransform;

    fn build_runtime() -> RendererRuntime {
        let settings: RendererSettingsHandle =
            std::sync::Arc::new(std::sync::RwLock::new(RendererSettings::default()));
        RendererRuntime::new(
            Option::<ConnectionParams>::None,
            settings,
            std::path::PathBuf::from("test_config.toml"),
        )
    }

    fn copied_cubemap_center_world_direction(projection_rotation: Quat) -> Vec3 {
        -(projection_rotation * Vec3::Z)
    }

    fn assert_vec3_nearly_eq(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).length() < 1e-6,
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn main_view_360_setting_defaults_disabled() {
        let runtime = build_runtime();

        assert!(!runtime.main_view_360_enabled());
    }

    #[test]
    fn main_view_360_setting_reads_live_config() {
        let runtime = build_runtime();
        runtime
            .settings()
            .write()
            .expect("settings write")
            .experimental
            .main_view_360_enabled = true;

        assert!(runtime.main_view_360_enabled());
    }

    #[test]
    fn main_view_360_pose_uses_corrected_projection_rotation() {
        let mut scene = SceneCoordinator::new();
        let rotation = Quat::from_rotation_y(1.25);
        scene.test_seed_space_identity_worlds(
            RenderSpaceId(1),
            vec![RenderTransform::default()],
            vec![-1],
        );
        scene.test_set_space_view_transform(
            RenderSpaceId(1),
            RenderTransform {
                position: Vec3::new(2.0, 3.0, 4.0),
                rotation,
                scale: Vec3::ONE,
            },
        );
        let base_camera = HostCameraFrame {
            eye_world_position: Some(Vec3::new(2.0, 3.0, 4.0)),
            ..Default::default()
        };

        let pose = main_view_360_pose(&scene, &base_camera);

        assert_eq!(pose.position, Vec3::new(2.0, 3.0, 4.0));
        assert!(
            (pose
                .projection_rotation
                .dot(main_view_360_projection_rotation(rotation))
                .abs()
                - 1.0)
                .abs()
                < 1e-6
        );
        assert_vec3_nearly_eq(
            copied_cubemap_center_world_direction(pose.projection_rotation),
            rotation * Vec3::Z,
        );
    }

    #[test]
    fn main_view_360_projection_rotation_defaults_to_forward_center() {
        let scene = SceneCoordinator::new();
        let projection_rotation =
            main_view_360_projection_rotation(main_view_360_view_rotation(&scene));

        assert_vec3_nearly_eq(
            copied_cubemap_center_world_direction(projection_rotation),
            Vec3::Z,
        );
    }

    #[test]
    fn main_view_360_face_ids_cover_every_cube_face() {
        let ids = CubeCaptureFace::ALL
            .iter()
            .copied()
            .map(main_view_360_face_view_id)
            .collect::<HashSet<_>>();

        assert_eq!(ids.len(), CubeCaptureFace::ALL.len());
    }
}
