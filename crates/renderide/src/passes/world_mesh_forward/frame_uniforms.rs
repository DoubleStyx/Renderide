//! Frame uniform construction and upload helpers for world-mesh forward views.

use bytemuck::Zeroable;

use crate::camera::HostCameraFrame;
use crate::frame_upload_batch::GraphUploadSink;
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::graph_inputs::{OffscreenWriteTarget, PerViewFramePlan};
use crate::scene::SceneCoordinator;
use crate::world_mesh::cluster::{
    ClusterFrameParams, FrameGpuUniformBuildParams, cluster_frame_params,
    cluster_frame_params_stereo,
};

use super::camera::resolve_camera_world_pair;
use super::prepare::WorldMeshForwardPrepareFrame;

/// Per-view inputs layered on top of scene/camera state when packing frame uniforms.
struct FrameUniformInputs {
    /// Viewport extent in physical pixels.
    viewport_px: (u32, u32),
    /// Number of resident lights written for this view.
    light_count: u32,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    frame_time_seconds: f32,
    /// Effective raster sample count for this view.
    sample_count: u32,
    /// Whether the view uses stereo multiview rendering.
    use_multiview: bool,
    /// Current render target for projection-convention adjustment.
    offscreen_write_target: OffscreenWriteTarget,
    /// Reserved direct skybox specular state; specular IBL comes from reflection probes.
    skybox_specular: crate::gpu::frame_globals::SkyboxSpecularUniformParams,
}

/// Writes per-view `FrameGpuUniforms` via [`GraphUploadSink`].
pub(super) fn write_per_view_frame_uniforms(
    uploads: GraphUploadSink<'_>,
    frame: &WorldMeshForwardPrepareFrame<'_, '_>,
    frame_plan: &PerViewFramePlan,
    use_multiview: bool,
    hc: &HostCameraFrame,
) {
    let uniforms = build_frame_gpu_uniforms(
        hc,
        frame.systems.scene,
        FrameUniformInputs {
            viewport_px: frame.view.viewport_px,
            light_count: frame
                .systems
                .frame_resources
                .frame_light_count_u32(frame.view.view_id),
            frame_time_seconds: frame.view.frame_time_seconds,
            sample_count: frame.view.sample_count,
            use_multiview,
            offscreen_write_target: frame.view.offscreen_write_target,
            skybox_specular: frame
                .systems
                .frame_resources
                .skybox_specular_uniform_params(),
        },
    );
    uploads.write_buffer(
        &frame_plan.frame_uniform_buffer,
        0,
        bytemuck::bytes_of(&uniforms),
    );
}

/// Resolves cluster + camera-world scratch into [`FrameGpuUniforms`] for one view.
fn build_frame_gpu_uniforms(
    hc: &HostCameraFrame,
    scene: &SceneCoordinator,
    inputs: FrameUniformInputs,
) -> FrameGpuUniforms {
    let (vw, vh) = inputs.viewport_px;
    let (camera_world, camera_world_right) = resolve_camera_world_pair(hc);
    let ambient_light = scene.active_main_ambient_light();
    let ambient_sh = FrameGpuUniforms::ambient_sh_from_render_sh2(&ambient_light);
    let ambient_sh_valid = FrameGpuUniforms::ambient_sh_is_valid(&ambient_light);
    let stereo_cluster = inputs.use_multiview && hc.active_stereo().is_some();
    let frame_idx = hc.frame_index as u32;
    if stereo_cluster && let Some((left, right)) = cluster_frame_params_stereo(hc, scene, (vw, vh))
    {
        let left = apply_render_target_projection(left, inputs.offscreen_write_target);
        let right = apply_render_target_projection(right, inputs.offscreen_write_target);
        return left.frame_gpu_uniforms(&FrameGpuUniformBuildParams {
            camera_world_pos: camera_world,
            camera_world_pos_right: camera_world_right,
            right_z_coeffs: right.view_space_z_coeffs(),
            right_view_to_world_y_coeffs:
                FrameGpuUniforms::view_to_world_y_coeffs_from_world_to_view(right.world_to_view),
            right_x_coeffs: right.view_space_x_coeffs(),
            right_y_coeffs: right.view_space_y_coeffs(),
            right_proj_params: right.proj_params(),
            right_proj: right.proj.to_cols_array(),
            right_projection_flags: right.projection_flags,
            light_count: inputs.light_count,
            sample_count: inputs.sample_count,
            frame_index: frame_idx,
            ambient_sh_valid,
            skybox_specular: inputs.skybox_specular,
            frame_time_seconds: inputs.frame_time_seconds,
            ambient_sh,
        });
    }
    if let Some(mono) = cluster_frame_params(hc, scene, (vw, vh)) {
        let mono = apply_render_target_projection(mono, inputs.offscreen_write_target);
        let z = mono.view_space_z_coeffs();
        let x = mono.view_space_x_coeffs();
        let y = mono.view_space_y_coeffs();
        let p = mono.proj_params();
        return mono.frame_gpu_uniforms(&FrameGpuUniformBuildParams {
            camera_world_pos: camera_world,
            camera_world_pos_right: camera_world_right,
            light_count: inputs.light_count,
            right_z_coeffs: z,
            right_view_to_world_y_coeffs:
                FrameGpuUniforms::view_to_world_y_coeffs_from_world_to_view(mono.world_to_view),
            right_x_coeffs: x,
            right_y_coeffs: y,
            right_proj_params: p,
            right_proj: mono.proj.to_cols_array(),
            right_projection_flags: mono.projection_flags,
            sample_count: inputs.sample_count,
            frame_index: frame_idx,
            ambient_sh_valid,
            skybox_specular: inputs.skybox_specular,
            frame_time_seconds: inputs.frame_time_seconds,
            ambient_sh,
        });
    }
    FrameGpuUniforms::zeroed()
}

fn apply_render_target_projection(
    mut params: ClusterFrameParams,
    offscreen_write_target: OffscreenWriteTarget,
) -> ClusterFrameParams {
    params.proj = offscreen_write_target.render_projection(params.proj);
    params
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::*;
    use crate::camera::EyeView;
    use crate::gpu::frame_globals::SkyboxSpecularUniformParams;

    fn projection_with_non_unit_y() -> Mat4 {
        Mat4::from_cols_array(&[
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.25, 0.5, 1.0, 1.0, 0.0, 0.0, 0.1, 0.0,
        ])
    }

    fn explicit_camera(proj: Mat4) -> HostCameraFrame {
        HostCameraFrame {
            explicit_view: Some(EyeView::new(Mat4::IDENTITY, proj, proj, Vec3::ZERO)),
            ..HostCameraFrame::default()
        }
    }

    fn frame_inputs(offscreen_write_target: OffscreenWriteTarget) -> FrameUniformInputs {
        FrameUniformInputs {
            viewport_px: (128, 96),
            light_count: 0,
            frame_time_seconds: 0.0,
            sample_count: 1,
            use_multiview: false,
            offscreen_write_target,
            skybox_specular: SkyboxSpecularUniformParams::disabled(),
        }
    }

    #[test]
    fn frame_uniforms_keep_primary_projection_params_unchanged() {
        let scene = SceneCoordinator::new();
        let proj = projection_with_non_unit_y();
        let uniforms = build_frame_gpu_uniforms(
            &explicit_camera(proj),
            &scene,
            frame_inputs(OffscreenWriteTarget::None),
        );

        assert_eq!(
            uniforms.proj_params_left,
            FrameGpuUniforms::proj_params_from_proj(proj)
        );
        assert_eq!(uniforms.proj_left, proj.to_cols_array());
    }

    #[test]
    fn frame_uniforms_use_offscreen_render_projection_params() {
        let scene = SceneCoordinator::new();
        let proj = projection_with_non_unit_y();
        let write_target = OffscreenWriteTarget::host_render_texture(77);
        let uniforms =
            build_frame_gpu_uniforms(&explicit_camera(proj), &scene, frame_inputs(write_target));
        let expected_proj = write_target.render_projection(proj);

        assert_eq!(
            uniforms.proj_params_left,
            FrameGpuUniforms::proj_params_from_proj(expected_proj)
        );
        assert_eq!(uniforms.proj_left, expected_proj.to_cols_array());
    }
}
