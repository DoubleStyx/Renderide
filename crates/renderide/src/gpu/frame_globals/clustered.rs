//! Clustered forward + lighting frame uniform constructor.
//!
//! [`ClusteredFrameGlobalsParams`] is the input bundle and
//! [`super::FrameGpuUniforms::new_clustered`] is its sole constructor. Pure: takes
//! resolved camera / cluster / projection / SH inputs and returns the packed
//! WGSL-matched struct.

use super::skybox_specular::SkyboxSpecularUniformParams;
use super::uniforms::{FrameGpuUniforms, pack_frame_tail_flags};

/// Inputs for [`FrameGpuUniforms::new_clustered`] (clustered forward + lighting).
#[derive(Clone, Copy, Debug)]
pub struct ClusteredFrameGlobalsParams {
    /// World-space camera position for the active view.
    pub camera_world_pos: glam::Vec3,
    /// Right-eye world-space camera position; equals [`Self::camera_world_pos`] in mono mode.
    pub camera_world_pos_right: glam::Vec3,
    /// Left-eye (or mono) view-space Z coefficients from world position.
    pub view_space_z_coeffs: [f32; 4],
    /// Right-eye view-space Z coefficients; equals `view_space_z_coeffs` in mono.
    pub view_space_z_coeffs_right: [f32; 4],
    /// Cluster grid width in tiles.
    pub cluster_count_x: u32,
    /// Cluster grid height in tiles.
    pub cluster_count_y: u32,
    /// Cluster grid depth (Z slices).
    pub cluster_count_z: u32,
    /// Near clip in view space (positive forward).
    pub near_clip: f32,
    /// Far clip (reverse-Z aware).
    pub far_clip: f32,
    /// Packed light count for the frame buffer.
    pub light_count: u32,
    /// Viewport width in physical pixels.
    pub viewport_width: u32,
    /// Viewport height in physical pixels.
    pub viewport_height: u32,
    /// Left-eye (or mono) projection coefficients `(P[0][0], P[1][1], P[0][2], P[1][2])`.
    pub proj_params_left: [f32; 4],
    /// Right-eye projection coefficients; equals `proj_params_left` in mono.
    pub proj_params_right: [f32; 4],
    /// Monotonic frame index (wraps `HostCameraFrame::frame_index`).
    pub frame_index: u32,
    /// Left-eye or mono projection flags.
    pub projection_flags_left: u32,
    /// Right-eye projection flags, or the same value as [`Self::projection_flags_left`] in mono mode.
    pub projection_flags_right: u32,
    /// Raster sample count for the current frame target.
    pub sample_count: u32,
    /// Whether `ambient_sh` contains host-authored lighting data.
    pub ambient_sh_valid: bool,
    /// Skybox indirect specular sampling parameters.
    pub skybox_specular: SkyboxSpecularUniformParams,
    /// Ambient SH2 coefficients for the active main render space.
    pub ambient_sh: [[f32; 4]; 9],
}

impl FrameGpuUniforms {
    /// Builds per-frame uniforms for clustered forward and lighting.
    ///
    /// `params.view_space_z_coeffs_right` should equal `params.view_space_z_coeffs` in mono mode;
    /// `params.proj_params_right` should equal `params.proj_params_left` in mono mode.
    pub fn new_clustered(params: &ClusteredFrameGlobalsParams) -> Self {
        Self {
            camera_world_pos: [
                params.camera_world_pos.x,
                params.camera_world_pos.y,
                params.camera_world_pos.z,
                0.0,
            ],
            camera_world_pos_right: [
                params.camera_world_pos_right.x,
                params.camera_world_pos_right.y,
                params.camera_world_pos_right.z,
                0.0,
            ],
            view_space_z_coeffs: params.view_space_z_coeffs,
            view_space_z_coeffs_right: params.view_space_z_coeffs_right,
            cluster_count_x: params.cluster_count_x,
            cluster_count_y: params.cluster_count_y,
            cluster_count_z: params.cluster_count_z,
            near_clip: params.near_clip,
            far_clip: params.far_clip,
            light_count: params.light_count,
            viewport_width: params.viewport_width,
            viewport_height: params.viewport_height,
            proj_params_left: params.proj_params_left,
            proj_params_right: params.proj_params_right,
            frame_tail: [
                params.frame_index,
                params.projection_flags_left,
                params.projection_flags_right,
                pack_frame_tail_flags(params.ambient_sh_valid, params.sample_count),
            ],
            skybox_specular: params.skybox_specular.to_vec4(),
            ambient_sh: params.ambient_sh,
        }
    }
}
