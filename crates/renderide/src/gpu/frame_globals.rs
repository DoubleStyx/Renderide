//! CPU layout for `shaders/source/modules/globals.wgsl` (`FrameGlobals` at `@group(0) @binding(0)`).

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

/// Uniform block matching WGSL `FrameGlobals` (96-byte size, 16-byte aligned).
///
/// Encodes camera position, coefficients for view-space Z from world position (left and optional
/// right eye), clustered grid dimensions, clip planes, light count, viewport size, and directional
/// prefix count for clustered forward sampling.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameGpuUniforms {
    /// World-space camera position (`.w` unused).
    pub camera_world_pos: [f32; 4],
    /// World `vec4(x,y,z,1)` → view-space Z is `dot(xyz, world.xyz) + w` (third column of world-to-view).
    pub view_space_z_coeffs: [f32; 4],
    /// Same as [`Self::view_space_z_coeffs`] for the right eye when [`Self::stereo_cluster_layers`] is `2`.
    pub view_space_z_coeffs_right: [f32; 4],
    pub cluster_count_x: u32,
    pub cluster_count_y: u32,
    pub cluster_count_z: u32,
    /// `1` mono, `2` packed stereo cluster buffers (see `cluster_id_from_frag` in `pbs_cluster.wgsl`).
    pub stereo_cluster_layers: u32,
    pub near_clip: f32,
    pub far_clip: f32,
    pub light_count: u32,
    pub viewport_width: u32,
    pub viewport_height: u32,
    /// Leading directional lights in `lights` (`0..directional_light_count`); not in cluster lists.
    pub directional_light_count: u32,
    /// Matches WGSL `_pad_frame` (`vec2<u32>`).
    pub pad_frame: [u32; 2],
}

impl FrameGpuUniforms {
    /// Coefficients so `dot(coeffs.xyz, world) + coeffs.w` yields view-space Z for a world point.
    ///
    /// Uses the third row of the column-major world-to-view matrix (`glam` column vectors).
    pub fn view_space_z_coeffs_from_world_to_view(world_to_view: Mat4) -> [f32; 4] {
        let m = world_to_view;
        [m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z]
    }

    /// Builds per-frame uniforms for clustered forward and lighting.
    #[allow(clippy::too_many_arguments)]
    pub fn new_clustered(
        camera_world_pos: glam::Vec3,
        view_space_z_coeffs: [f32; 4],
        view_space_z_coeffs_right: [f32; 4],
        stereo_cluster_layers: u32,
        cluster_count_x: u32,
        cluster_count_y: u32,
        cluster_count_z: u32,
        near_clip: f32,
        far_clip: f32,
        light_count: u32,
        viewport_width: u32,
        viewport_height: u32,
        directional_light_count: u32,
    ) -> Self {
        Self {
            camera_world_pos: [
                camera_world_pos.x,
                camera_world_pos.y,
                camera_world_pos.z,
                0.0,
            ],
            view_space_z_coeffs,
            view_space_z_coeffs_right,
            cluster_count_x,
            cluster_count_y,
            cluster_count_z,
            stereo_cluster_layers: stereo_cluster_layers.clamp(1, 2),
            near_clip,
            far_clip,
            light_count,
            viewport_width,
            viewport_height,
            directional_light_count,
            pad_frame: [0, 0],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use super::*;

    #[test]
    fn frame_globals_size_96() {
        assert_eq!(std::mem::size_of::<FrameGpuUniforms>(), 96);
        assert_eq!(std::mem::size_of::<FrameGpuUniforms>() % 16, 0);
    }

    #[test]
    fn directional_light_count_field_offset_matches_wgsl_frame_globals() {
        assert_eq!(offset_of!(FrameGpuUniforms, directional_light_count), 84);
        assert_eq!(offset_of!(FrameGpuUniforms, pad_frame), 88);
    }
}
