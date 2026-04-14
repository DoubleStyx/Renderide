//! CPU layout for `shaders/source/modules/globals.wgsl` (`FrameGlobals` at `@group(0) @binding(0)`).

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

/// Uniform block matching WGSL `FrameGlobals` (80-byte size, 16-byte aligned).
///
/// Encodes camera position, per-eye coefficients for view-space Z from world position, clustered
/// grid dimensions, clip planes, light count, and viewport size for clustered forward sampling.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameGpuUniforms {
    /// World-space camera position (`.w` unused).
    pub camera_world_pos: [f32; 4],
    /// Left-eye (or mono) world → view-space Z: `dot(xyz, world) + w` (third row of world-to-view).
    pub view_space_z_coeffs: [f32; 4],
    /// Right-eye world → view-space Z. Set equal to `view_space_z_coeffs` in mono mode.
    pub view_space_z_coeffs_right: [f32; 4],
    /// Cluster grid width in tiles (X).
    pub cluster_count_x: u32,
    /// Cluster grid height in tiles (Y).
    pub cluster_count_y: u32,
    /// Depth slice count for clustered lighting (Z).
    pub cluster_count_z: u32,
    /// Camera near clip plane (view space, positive forward).
    pub near_clip: f32,
    /// Camera far clip plane (reverse-Z aware; matches shader expectations).
    pub far_clip: f32,
    /// Number of lights packed into the frame storage buffer for this pass.
    pub light_count: u32,
    /// Viewport width in pixels (physical).
    pub viewport_width: u32,
    /// Viewport height in pixels (physical).
    pub viewport_height: u32,
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
    ///
    /// `view_space_z_coeffs_right` should equal `view_space_z_coeffs` in mono mode.
    #[allow(clippy::too_many_arguments)]
    pub fn new_clustered(
        camera_world_pos: glam::Vec3,
        view_space_z_coeffs: [f32; 4],
        view_space_z_coeffs_right: [f32; 4],
        cluster_count_x: u32,
        cluster_count_y: u32,
        cluster_count_z: u32,
        near_clip: f32,
        far_clip: f32,
        light_count: u32,
        viewport_width: u32,
        viewport_height: u32,
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
            near_clip,
            far_clip,
            light_count,
            viewport_width,
            viewport_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_globals_size_80() {
        assert_eq!(std::mem::size_of::<FrameGpuUniforms>(), 80);
        assert_eq!(std::mem::size_of::<FrameGpuUniforms>() % 16, 0);
    }
}
