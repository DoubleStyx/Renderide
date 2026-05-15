//! Frame GPU uniform layering for [`super::ClusterFrameParams`].
//!
//! Separates the per-frame data (camera position, light count, projection right-eye parameters,
//! ambient SH) from the cluster-grid identity (clip planes, view, projection) defined on
//! [`super::ClusterFrameParams`]. The latter is shared across compute and fragment passes;
//! the former is layered on top once per submitted frame.

use crate::gpu::frame_globals::{
    ClusteredFrameGlobalsParams, FrameGpuUniforms, SkyboxSpecularUniformParams,
};

use super::ClusterFrameParams;
use super::clip::CLUSTER_COUNT_Z;

/// Per-frame values layered on top of [`ClusterFrameParams`] when packing [`FrameGpuUniforms`].
#[derive(Clone, Copy, Debug)]
pub struct FrameGpuUniformBuildParams {
    /// Left-eye or mono world-space camera position.
    pub camera_world_pos: glam::Vec3,
    /// Right-eye world-space camera position, or the same value as [`Self::camera_world_pos`] in mono mode.
    pub camera_world_pos_right: glam::Vec3,
    /// Number of resident lights written to the frame lights buffer.
    pub light_count: u32,
    /// Right-eye view-space-Z coefficients, or the mono coefficients for non-stereo frames.
    pub right_z_coeffs: [f32; 4],
    /// Right-eye projection parameters, or the mono parameters for non-stereo frames.
    pub right_proj_params: [f32; 4],
    /// Right-eye projection flags, or the mono flags for non-stereo frames.
    pub right_projection_flags: u32,
    /// Raster sample count for the current frame target.
    pub sample_count: u32,
    /// Monotonic host frame index used by temporal effects.
    pub frame_index: u32,
    /// Whether `ambient_sh` contains host-authored lighting data.
    pub ambient_sh_valid: bool,
    /// Reserved direct skybox specular state; specular IBL comes from reflection probes.
    pub skybox_specular: SkyboxSpecularUniformParams,
    /// Host ambient SH2 coefficients for indirect diffuse.
    pub ambient_sh: [[f32; 4]; 9],
}

/// Builds [`FrameGpuUniforms`] for clustered PBS materials from a cluster grid identity
/// ([`ClusterFrameParams`]) plus per-frame data.
///
/// Must stay in sync with the cluster compute pass; the grid sizing and projection coefficients
/// match what the compute uses for cluster tile / Z-slice math.
pub(super) fn build_frame_gpu_uniforms(
    cfp: &ClusterFrameParams,
    params: FrameGpuUniformBuildParams,
) -> FrameGpuUniforms {
    let params = ClusteredFrameGlobalsParams {
        camera_world_pos: params.camera_world_pos,
        camera_world_pos_right: params.camera_world_pos_right,
        view_space_z_coeffs: cfp.view_space_z_coeffs(),
        view_space_z_coeffs_right: params.right_z_coeffs,
        cluster_count_x: cfp.cluster_count_x,
        cluster_count_y: cfp.cluster_count_y,
        cluster_count_z: CLUSTER_COUNT_Z,
        near_clip: cfp.near_clip,
        far_clip: cfp.far_clip,
        light_count: params.light_count,
        viewport_width: cfp.viewport_width.max(1),
        viewport_height: cfp.viewport_height.max(1),
        proj_params_left: cfp.proj_params(),
        proj_params_right: params.right_proj_params,
        frame_index: params.frame_index,
        projection_flags_left: cfp.projection_flags,
        projection_flags_right: params.right_projection_flags,
        sample_count: params.sample_count,
        ambient_sh_valid: params.ambient_sh_valid,
        skybox_specular: params.skybox_specular,
        ambient_sh: params.ambient_sh,
    };
    FrameGpuUniforms::new_clustered(&params)
}
