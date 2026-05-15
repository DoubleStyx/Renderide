//! [`FrameGpuUniforms`] WGSL-matched uniform block + pure helpers for projection /
//! view-space-Z / ambient-SH packing.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::shared::RenderSH2;

/// Frame tail bit that indicates frame ambient SH2 contains host-authored data.
pub const FRAME_TAIL_AMBIENT_SH_VALID: u32 = 1 << 0;

/// Bit offset for the encoded raster sample count in `FrameGpuUniforms::frame_tail.w`.
pub const FRAME_TAIL_SAMPLE_COUNT_SHIFT: u32 = 1;

/// Bit mask for the encoded raster sample count in `FrameGpuUniforms::frame_tail.w`.
pub const FRAME_TAIL_SAMPLE_COUNT_MASK: u32 = 0xF << FRAME_TAIL_SAMPLE_COUNT_SHIFT;

/// Frame projection flag indicating that the corresponding view uses orthographic projection.
pub const FRAME_PROJECTION_FLAG_ORTHOGRAPHIC: u32 = 1;

/// Uniform block matching WGSL `FrameGlobals` (304-byte size, 16-byte aligned).
///
/// Encodes per-eye camera positions, per-eye coefficients for view-space Z from world position,
/// clustered grid dimensions, clip planes, light count, viewport size, per-eye projection
/// coefficients for screen-space-to-view unprojection, a monotonic frame index for temporal /
/// jittered effects, a reserved direct skybox specular slot, and ambient SH2.
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
    /// uses `view_x = (ndc_x + c.z) * view_z / c.x` and `view_y = (ndc_y + c.w) * view_z / c.y`,
    /// where `c` is this vec4. Encodes both symmetric (desktop) and asymmetric (per-eye VR)
    /// perspective projections exactly.
    pub proj_params_left: [f32; 4],
    /// Right-eye projection coefficients (same packing as [`Self::proj_params_left`]).
    ///
    /// Equals [`Self::proj_params_left`] in mono mode.
    pub proj_params_right: [f32; 4],
    /// Packed trailing `vec4<u32>` slot: `.x` is the monotonic frame index (wraps
    /// `host_camera.frame_index`; used for temporal / jittered screen-space effects), `.y` holds
    /// left/mono projection flags, `.z` holds right-eye projection flags, and `.w` packs
    /// frame-wide flags through [`pack_frame_tail_flags`].
    pub frame_tail: [u32; 4],
    /// Reserved direct skybox specular parameters: `.x` max resident LOD, `.y` enabled flag,
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

    /// Returns true when the host SH2 payload contains nonzero lighting data.
    pub fn ambient_sh_is_valid(sh: &RenderSH2) -> bool {
        render_sh2_is_nonzero(sh)
    }
}

/// Packs frame-wide flags into `FrameGpuUniforms::frame_tail.w`.
pub fn pack_frame_tail_flags(ambient_sh_valid: bool, sample_count: u32) -> u32 {
    let ambient_flag = if ambient_sh_valid {
        FRAME_TAIL_AMBIENT_SH_VALID
    } else {
        0
    };
    ambient_flag
        | ((encoded_frame_sample_count(sample_count) << FRAME_TAIL_SAMPLE_COUNT_SHIFT)
            & FRAME_TAIL_SAMPLE_COUNT_MASK)
}

/// Decodes the raster sample count from `FrameGpuUniforms::frame_tail.w`.
#[cfg(test)]
pub(super) fn frame_tail_sample_count(flags: u32) -> u32 {
    match (flags & FRAME_TAIL_SAMPLE_COUNT_MASK) >> FRAME_TAIL_SAMPLE_COUNT_SHIFT {
        2 => 2,
        4 => 4,
        8 => 8,
        _ => 1,
    }
}

/// Returns the encoded frame sample count payload stored in `frame_tail.w`.
fn encoded_frame_sample_count(sample_count: u32) -> u32 {
    match sample_count {
        2 => 2,
        4 => 4,
        8 => 8,
        _ => 0,
    }
}

/// Returns true when the host SH payload contains nonzero lighting data.
fn render_sh2_is_nonzero(sh: &RenderSH2) -> bool {
    let energy = sh.sh0.abs().element_sum()
        + sh.sh1.abs().element_sum()
        + sh.sh2.abs().element_sum()
        + sh.sh3.abs().element_sum()
        + sh.sh4.abs().element_sum()
        + sh.sh5.abs().element_sum()
        + sh.sh6.abs().element_sum()
        + sh.sh7.abs().element_sum()
        + sh.sh8.abs().element_sum();
    energy >= 1e-8
}
