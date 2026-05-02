//! [`FrameGpuUniforms`] WGSL-matched uniform block + pure helpers for projection /
//! view-space-Z / ambient-SH packing.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::shared::RenderSH2;

/// Diffuse fallback used before host ambient SH arrives.
pub(super) const FALLBACK_AMBIENT_COLOR: f32 = 0.03;
/// Zeroth-order SH basis constant used for fallback packing.
pub(super) const SH_C0: f32 = 0.282_094_8;

/// Uniform block matching WGSL `FrameGlobals` (336-byte size, 16-byte aligned).
///
/// Encodes per-eye camera positions, per-eye coefficients for view-space Z from world position,
/// clustered grid dimensions, clip planes, light count, viewport size, per-eye projection
/// coefficients for screen-space-to-view unprojection, a monotonic frame index for temporal /
/// jittered effects, skybox specular environment sampling parameters, and ambient SH2.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameGpuUniforms {
    /// World-space camera position (`.w` unused).
    pub camera_world_pos: [f32; 4],
    /// Right-eye world-space camera position (`.w` unused); equals [`Self::camera_world_pos`] in mono mode.
    pub camera_world_pos_right: [f32; 4],
    /// Left-eye (or mono) world -> view-space Z: `dot(xyz, world) + w`.
    pub view_space_z_coeffs: [f32; 4],
    /// Right-eye world -> view-space Z. Set equal to `view_space_z_coeffs` in mono mode.
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
    /// Left-eye (or mono) projection coefficients: `(P[0][0], P[1][1], P[0][2], P[1][2])`.
    ///
    /// Column-major `glam::Mat4` indexing. Screen-space -> view-space unprojection (view Z known)
    /// uses `view_x = (ndc_x - c.z) * view_z / c.x` and `view_y = (ndc_y - c.w) * view_z / c.y`,
    /// where `c` is this vec4. Encodes both symmetric (desktop) and asymmetric (per-eye VR)
    /// perspective projections exactly.
    pub proj_params_left: [f32; 4],
    /// Right-eye projection coefficients (same packing as [`Self::proj_params_left`]).
    ///
    /// Equals [`Self::proj_params_left`] in mono mode.
    pub proj_params_right: [f32; 4],
    /// Packed trailing `vec4<u32>` slot: `.x` is the monotonic frame index (wraps
    /// `host_camera.frame_index`; used for temporal / jittered screen-space effects), `.yzw` are
    /// reserved padding so the struct aligns to a 16-byte boundary without tripping naga-oil's
    /// composable-identifier substitution rules (numeric-suffix names are rejected).
    pub frame_tail: [u32; 4],
    /// Skybox specular parameters: `.x` max resident LOD, `.y` enabled flag,
    /// `.z` [`super::skybox_specular::SkyboxSpecularSourceKind`] tag, `.w` reserved.
    pub skybox_specular: [f32; 4],
    /// Ambient SH2 coefficients (`RenderSH2` order), padded to WGSL `vec4<f32>` slots.
    pub ambient_sh: [[f32; 4]; 9],
}

impl FrameGpuUniforms {
    /// Coefficients so `dot(coeffs.xyz, world) + coeffs.w` yields view-space Z for a world point.
    ///
    /// Uses the third row of the column-major world-to-view matrix (`glam` column vectors).
    pub fn view_space_z_coeffs_from_world_to_view(world_to_view: Mat4) -> [f32; 4] {
        let m = world_to_view;
        [m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z]
    }

    /// Extracts `(P[0][0], P[1][1], P[0][2], P[1][2])` from a column-major perspective matrix.
    ///
    /// For symmetric projections `P[0][2]` and `P[1][2]` are zero; asymmetric (per-eye VR)
    /// projections encode the principal-point offset there. Used by screen-space passes that
    /// unproject from depth to view space without needing the full `inv_proj` matrix.
    pub fn proj_params_from_proj(proj: Mat4) -> [f32; 4] {
        [proj.x_axis.x, proj.y_axis.y, proj.z_axis.x, proj.z_axis.y]
    }

    /// Pads host SH2 coefficients into WGSL-friendly vec4 slots.
    pub fn ambient_sh_from_render_sh2(sh: &RenderSH2) -> [[f32; 4]; 9] {
        if render_sh2_is_zero(sh) {
            let sh0 = FALLBACK_AMBIENT_COLOR * (4.0 * std::f32::consts::PI * SH_C0);
            return [
                [sh0, sh0, sh0, 0.0],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
            ];
        }
        [
            [sh.sh0.x, sh.sh0.y, sh.sh0.z, 0.0],
            [sh.sh1.x, sh.sh1.y, sh.sh1.z, 0.0],
            [sh.sh2.x, sh.sh2.y, sh.sh2.z, 0.0],
            [sh.sh3.x, sh.sh3.y, sh.sh3.z, 0.0],
            [sh.sh4.x, sh.sh4.y, sh.sh4.z, 0.0],
            [sh.sh5.x, sh.sh5.y, sh.sh5.z, 0.0],
            [sh.sh6.x, sh.sh6.y, sh.sh6.z, 0.0],
            [sh.sh7.x, sh.sh7.y, sh.sh7.z, 0.0],
            [sh.sh8.x, sh.sh8.y, sh.sh8.z, 0.0],
        ]
    }
}

/// Returns true when the host SH payload is still the all-zero default.
fn render_sh2_is_zero(sh: &RenderSH2) -> bool {
    let energy = sh.sh0.abs().element_sum()
        + sh.sh1.abs().element_sum()
        + sh.sh2.abs().element_sum()
        + sh.sh3.abs().element_sum()
        + sh.sh4.abs().element_sum()
        + sh.sh5.abs().element_sum()
        + sh.sh6.abs().element_sum()
        + sh.sh7.abs().element_sum()
        + sh.sh8.abs().element_sum();
    energy < 1e-8
}
