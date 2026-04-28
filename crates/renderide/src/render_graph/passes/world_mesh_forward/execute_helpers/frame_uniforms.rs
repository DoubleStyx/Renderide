//! Frame uniform construction and upload helpers for world-mesh forward views.

use bytemuck::Zeroable;

use crate::backend::FrameResourceManager;
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::cluster_frame::{
    cluster_frame_params, cluster_frame_params_stereo, FrameGpuUniformBuildParams,
};
use crate::render_graph::frame_params::{FrameRenderParams, HostCameraFrame, PerViewFramePlanSlot};
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::scene::SceneCoordinator;

use super::camera::resolve_camera_world;

/// Builds [`FrameGpuUniforms`], syncs cluster viewport, and writes frame + lights.
pub(super) fn write_frame_uniforms_and_cluster(
    queue: &wgpu::Queue,
    frame_resources: &FrameResourceManager,
    hc: HostCameraFrame,
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
    use_multiview: bool,
) {
    let light_count_u = frame_resources.frame_light_count_u32();
    let uniforms = build_frame_gpu_uniforms(
        hc,
        scene,
        viewport_px,
        light_count_u,
        use_multiview,
        frame_resources.skybox_specular_uniform_params(),
    );

    frame_resources.write_frame_uniform_and_lights_from_scratch(queue, &uniforms);
}

/// Writes per-view `FrameGpuUniforms` via [`FrameUploadBatch`] or falls back to the shared frame buffer.
///
/// Multi-view paths plant a [`PerViewFramePlanSlot`] on the blackboard naming the per-view bind
/// group and uniform buffer; single-view fallbacks keep writing the shared `frame_uniform`
/// buffer directly on the GPU queue.
pub(super) fn write_per_view_frame_uniforms(
    queue: &wgpu::Queue,
    upload_batch: &FrameUploadBatch,
    frame: &mut FrameRenderParams<'_>,
    blackboard: &mut Blackboard,
    use_multiview: bool,
    hc: crate::render_graph::frame_params::HostCameraFrame,
) {
    if let Some(frame_plan) = blackboard.get::<PerViewFramePlanSlot>() {
        let uniforms = build_frame_gpu_uniforms(
            hc,
            frame.shared.scene,
            frame.view.viewport_px,
            frame.shared.frame_resources.frame_light_count_u32(),
            use_multiview,
            frame
                .shared
                .frame_resources
                .skybox_specular_uniform_params(),
        );
        upload_batch.write_buffer(
            &frame_plan.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    } else {
        write_frame_uniforms_and_cluster(
            queue,
            frame.shared.frame_resources,
            hc,
            frame.shared.scene,
            frame.view.viewport_px,
            use_multiview,
        );
    }
}

/// Resolves cluster + camera-world scratch into [`FrameGpuUniforms`] for one view.
fn build_frame_gpu_uniforms(
    hc: HostCameraFrame,
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
    light_count: u32,
    use_multiview: bool,
    skybox_specular: crate::gpu::frame_globals::SkyboxSpecularUniformParams,
) -> FrameGpuUniforms {
    let (vw, vh) = viewport_px;
    let camera_world = resolve_camera_world(&hc);
    let ambient_sh =
        FrameGpuUniforms::ambient_sh_from_render_sh2(&scene.active_main_ambient_light());
    let stereo_cluster = use_multiview && hc.vr_active && hc.stereo.is_some();
    let frame_idx = hc.frame_index as u32;
    if stereo_cluster {
        if let Some((left, right)) = cluster_frame_params_stereo(&hc, scene, (vw, vh)) {
            return left.frame_gpu_uniforms(FrameGpuUniformBuildParams {
                camera_world_pos: camera_world,
                light_count,
                right_z_coeffs: right.view_space_z_coeffs(),
                right_proj_params: right.proj_params(),
                frame_index: frame_idx,
                skybox_specular,
                ambient_sh,
            });
        }
    }
    if let Some(mono) = cluster_frame_params(&hc, scene, (vw, vh)) {
        let z = mono.view_space_z_coeffs();
        let p = mono.proj_params();
        return mono.frame_gpu_uniforms(FrameGpuUniformBuildParams {
            camera_world_pos: camera_world,
            light_count,
            right_z_coeffs: z,
            right_proj_params: p,
            frame_index: frame_idx,
            skybox_specular,
            ambient_sh,
        });
    }
    FrameGpuUniforms::zeroed()
}
