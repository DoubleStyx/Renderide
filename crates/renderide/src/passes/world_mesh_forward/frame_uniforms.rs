//! Frame uniform construction and upload helpers for world-mesh forward views.

use bytemuck::Zeroable;

use crate::camera::HostCameraFrame;
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::graph_inputs::{GraphPassFrame, PerViewFramePlan};
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::scene::SceneCoordinator;
use crate::world_mesh::cluster::{
    FrameGpuUniformBuildParams, cluster_frame_params, cluster_frame_params_stereo,
};

use super::camera::resolve_camera_world_pair;

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
    /// Reserved direct skybox specular state; specular IBL comes from reflection probes.
    skybox_specular: crate::gpu::frame_globals::SkyboxSpecularUniformParams,
}

/// Writes per-view `FrameGpuUniforms` via [`GraphUploadSink`].
pub(super) fn write_per_view_frame_uniforms(
    uploads: GraphUploadSink<'_>,
    frame: &GraphPassFrame<'_>,
    frame_plan: &PerViewFramePlan,
    use_multiview: bool,
    hc: &HostCameraFrame,
) {
    let uniforms = build_frame_gpu_uniforms(
        hc,
        frame.shared.scene,
        FrameUniformInputs {
            viewport_px: frame.view.viewport_px,
            light_count: frame
                .shared
                .frame_resources
                .frame_light_count_u32(frame.view.view_id),
            frame_time_seconds: frame.view.frame_time_seconds,
            sample_count: frame.view.sample_count,
            use_multiview,
            skybox_specular: frame
                .shared
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
        return left.frame_gpu_uniforms(FrameGpuUniformBuildParams {
            camera_world_pos: camera_world,
            camera_world_pos_right: camera_world_right,
            right_z_coeffs: right.view_space_z_coeffs(),
            right_view_to_world_y_coeffs:
                FrameGpuUniforms::view_to_world_y_coeffs_from_world_to_view(right.world_to_view),
            right_proj_params: right.proj_params(),
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
        let z = mono.view_space_z_coeffs();
        let p = mono.proj_params();
        return mono.frame_gpu_uniforms(FrameGpuUniformBuildParams {
            camera_world_pos: camera_world,
            camera_world_pos_right: camera_world_right,
            light_count: inputs.light_count,
            right_z_coeffs: z,
            right_view_to_world_y_coeffs:
                FrameGpuUniforms::view_to_world_y_coeffs_from_world_to_view(mono.world_to_view),
            right_proj_params: p,
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
