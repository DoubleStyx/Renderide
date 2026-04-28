//! CPU layout for `shaders/source/modules/globals.wgsl` (`FrameGlobals` at `@group(0) @binding(0)`).

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::shared::RenderSH2;

/// Diffuse fallback used before host ambient SH arrives.
const FALLBACK_AMBIENT_COLOR: f32 = 0.03;
/// Zeroth-order SH basis constant used for fallback packing.
const SH_C0: f32 = 0.282_094_8;

/// Default `Projection360` field of view used by Unity material defaults.
const PROJECTION360_DEFAULT_FOV: [f32; 4] = [std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0];
/// Default texture scale/offset used by Unity `_MainTex_ST` properties.
const DEFAULT_MAIN_TEX_ST: [f32; 4] = [1.0, 1.0, 0.0, 0.0];

/// Frame-global skybox specular source encoded in [`FrameGpuUniforms::skybox_specular`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkyboxSpecularSourceKind {
    /// No resident indirect-specular source is bound.
    Disabled,
    /// `@group(0) @binding(9)` is a cubemap source.
    Cubemap,
    /// `@group(0) @binding(11)` is a Projection360 equirectangular `Texture2D` source.
    Projection360Equirect,
}

impl SkyboxSpecularSourceKind {
    /// Numeric tag consumed by WGSL.
    pub const fn to_f32(self) -> f32 {
        match self {
            Self::Disabled => 0.0,
            Self::Cubemap => 1.0,
            Self::Projection360Equirect => 2.0,
        }
    }
}

/// Uniform block matching WGSL `FrameGlobals` (320-byte size, 16-byte aligned).
///
/// Encodes camera position, per-eye coefficients for view-space Z from world position, clustered
/// grid dimensions, clip planes, light count, viewport size, per-eye projection coefficients for
/// screen-space-to-view unprojection, a monotonic frame index for temporal / jittered effects,
/// skybox specular environment sampling parameters, and ambient SH2.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameGpuUniforms {
    /// World-space camera position (`.w` unused).
    pub camera_world_pos: [f32; 4],
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
    /// Column-major `glam::Mat4` indexing. Screen-space → view-space unprojection (view Z known)
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
    /// Skybox specular parameters: `.x` max resident LOD, `.y` enabled flag, `.z` storage-V
    /// inversion flag, `.w` [`SkyboxSpecularSourceKind`] tag.
    pub skybox_specular: [f32; 4],
    /// Projection360 equirectangular `_FOV` parameters for `Texture2D` skybox specular sampling.
    pub skybox_specular_equirect_fov: [f32; 4],
    /// Projection360 equirectangular `_MainTex_ST` parameters for skybox specular sampling.
    pub skybox_specular_equirect_st: [f32; 4],
    /// Ambient SH2 coefficients (`RenderSH2` order), padded to WGSL `vec4<f32>` slots.
    pub ambient_sh: [[f32; 4]; 9],
}

/// CPU-side parameters packed into [`FrameGpuUniforms::skybox_specular`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyboxSpecularUniformParams {
    /// Highest resident source mip available for roughness-driven sampling.
    pub max_lod: f32,
    /// Whether the frame has a resident skybox source bound for indirect specular.
    pub enabled: bool,
    /// Whether the source storage orientation needs V-axis compensation in shader sampling.
    pub storage_v_inverted: bool,
    /// Source texture kind selected by the active skybox material.
    pub source_kind: SkyboxSpecularSourceKind,
    /// Projection360 equirectangular `_FOV` parameters.
    pub equirect_fov: [f32; 4],
    /// Projection360 equirectangular `_MainTex_ST` parameters.
    pub equirect_st: [f32; 4],
}

impl SkyboxSpecularUniformParams {
    /// Disabled skybox specular environment.
    pub const fn disabled() -> Self {
        Self {
            max_lod: 0.0,
            enabled: false,
            storage_v_inverted: false,
            source_kind: SkyboxSpecularSourceKind::Disabled,
            equirect_fov: PROJECTION360_DEFAULT_FOV,
            equirect_st: DEFAULT_MAIN_TEX_ST,
        }
    }

    /// Builds enabled parameters from a resident cubemap mip count and storage orientation flag.
    pub fn from_resident_mips(mip_levels_resident: u32, storage_v_inverted: bool) -> Self {
        Self::from_cubemap_resident_mips(mip_levels_resident, storage_v_inverted)
    }

    /// Builds enabled parameters from a resident cubemap mip count and storage orientation flag.
    pub fn from_cubemap_resident_mips(mip_levels_resident: u32, storage_v_inverted: bool) -> Self {
        Self {
            max_lod: mip_levels_resident.saturating_sub(1) as f32,
            enabled: mip_levels_resident > 0,
            storage_v_inverted,
            source_kind: if mip_levels_resident > 0 {
                SkyboxSpecularSourceKind::Cubemap
            } else {
                SkyboxSpecularSourceKind::Disabled
            },
            equirect_fov: PROJECTION360_DEFAULT_FOV,
            equirect_st: DEFAULT_MAIN_TEX_ST,
        }
    }

    /// Builds enabled parameters from a resident Projection360 equirect texture.
    pub fn from_equirect_resident_mips(
        mip_levels_resident: u32,
        storage_v_inverted: bool,
        equirect_fov: [f32; 4],
        equirect_st: [f32; 4],
    ) -> Self {
        Self {
            max_lod: mip_levels_resident.saturating_sub(1) as f32,
            enabled: mip_levels_resident > 0,
            storage_v_inverted,
            source_kind: if mip_levels_resident > 0 {
                SkyboxSpecularSourceKind::Projection360Equirect
            } else {
                SkyboxSpecularSourceKind::Disabled
            },
            equirect_fov,
            equirect_st,
        }
    }

    /// Packs parameters into the `vec4<f32>` layout consumed by WGSL.
    pub fn to_vec4(self) -> [f32; 4] {
        [
            self.max_lod,
            if self.enabled { 1.0 } else { 0.0 },
            if self.storage_v_inverted { 1.0 } else { 0.0 },
            self.source_kind.to_f32(),
        ]
    }
}

/// Inputs for [`FrameGpuUniforms::new_clustered`] (clustered forward + lighting).
#[derive(Clone, Copy, Debug)]
pub struct ClusteredFrameGlobalsParams {
    /// World-space camera position for the active view.
    pub camera_world_pos: glam::Vec3,
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
    /// Skybox indirect specular sampling parameters.
    pub skybox_specular: SkyboxSpecularUniformParams,
    /// Ambient SH2 coefficients for the active main render space.
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

    /// Builds per-frame uniforms for clustered forward and lighting.
    ///
    /// `params.view_space_z_coeffs_right` should equal `params.view_space_z_coeffs` in mono mode;
    /// `params.proj_params_right` should equal `params.proj_params_left` in mono mode.
    pub fn new_clustered(params: ClusteredFrameGlobalsParams) -> Self {
        Self {
            camera_world_pos: [
                params.camera_world_pos.x,
                params.camera_world_pos.y,
                params.camera_world_pos.z,
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
            frame_tail: [params.frame_index, 0, 0, 0],
            skybox_specular: params.skybox_specular.to_vec4(),
            skybox_specular_equirect_fov: params.skybox_specular.equirect_fov,
            skybox_specular_equirect_st: params.skybox_specular.equirect_st,
            ambient_sh: params.ambient_sh,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_globals_size_320() {
        assert_eq!(std::mem::size_of::<FrameGpuUniforms>(), 320);
        assert_eq!(std::mem::size_of::<FrameGpuUniforms>() % 16, 0);
    }

    #[test]
    fn z_coeffs_extracts_third_row_for_translation_only_view() {
        // Translation-only view: world-to-view z = world.z + tz (tz from row 3, w component).
        let tz = 7.0;
        let m = Mat4::from_translation(glam::Vec3::new(0.0, 0.0, tz));
        let coeffs = FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(m);
        assert_eq!(coeffs, [0.0, 0.0, 1.0, tz]);

        // Sanity: dot(coeffs.xyz, p) + coeffs.w matches (m * p).z for a sample point.
        let p = glam::Vec3::new(2.0, -3.0, 4.0);
        let view_z = (m * p.extend(1.0)).z;
        let dotted = coeffs[2].mul_add(p.z, coeffs[0].mul_add(p.x, coeffs[1] * p.y)) + coeffs[3];
        assert!((view_z - dotted).abs() < 1e-6);
    }

    #[test]
    fn z_coeffs_matches_third_component_under_yaw_rotation() {
        // Yaw should leave Z row invariant (rotation about Y keeps Z-basis).
        let m = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_3);
        let coeffs = FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(m);
        let p = glam::Vec3::new(1.5, -0.25, 2.0);
        let view_z = (m * p.extend(1.0)).z;
        let dotted = coeffs[2].mul_add(p.z, coeffs[0].mul_add(p.x, coeffs[1] * p.y)) + coeffs[3];
        assert!((view_z - dotted).abs() < 1e-5);
    }

    #[test]
    fn proj_params_extract_diagonal_and_offcenter_are_zero_for_symmetric() {
        // Symmetric perspective: [0][2] and [1][2] are zero.
        let p = Mat4::perspective_rh(60.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        let coeffs = FrameGpuUniforms::proj_params_from_proj(p);
        assert!(coeffs[0].abs() > 0.0);
        assert!(coeffs[1].abs() > 0.0);
        assert!(coeffs[2].abs() < 1e-5);
        assert!(coeffs[3].abs() < 1e-5);
    }

    #[test]
    fn new_clustered_populates_fields_including_zero_w_for_camera_pos() {
        let u = FrameGpuUniforms::new_clustered(ClusteredFrameGlobalsParams {
            camera_world_pos: glam::Vec3::new(1.0, 2.0, 3.0),
            view_space_z_coeffs: [0.1, 0.2, 0.3, 0.4],
            view_space_z_coeffs_right: [0.5, 0.6, 0.7, 0.8],
            cluster_count_x: 16,
            cluster_count_y: 9,
            cluster_count_z: 24,
            near_clip: 0.05,
            far_clip: 1000.0,
            light_count: 42,
            viewport_width: 1920,
            viewport_height: 1080,
            proj_params_left: [1.5, 2.5, 0.0, 0.0],
            proj_params_right: [1.5, 2.5, 0.1, -0.2],
            frame_index: 7,
            skybox_specular: SkyboxSpecularUniformParams::from_resident_mips(6, true),
            ambient_sh: [[0.0; 4]; 9],
        });
        assert_eq!(u.camera_world_pos, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(u.view_space_z_coeffs, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(u.view_space_z_coeffs_right, [0.5, 0.6, 0.7, 0.8]);
        assert_eq!(u.cluster_count_x, 16);
        assert_eq!(u.cluster_count_y, 9);
        assert_eq!(u.cluster_count_z, 24);
        assert_eq!(u.near_clip, 0.05);
        assert_eq!(u.far_clip, 1000.0);
        assert_eq!(u.light_count, 42);
        assert_eq!(u.viewport_width, 1920);
        assert_eq!(u.viewport_height, 1080);
        assert_eq!(u.proj_params_left, [1.5, 2.5, 0.0, 0.0]);
        assert_eq!(u.proj_params_right, [1.5, 2.5, 0.1, -0.2]);
        assert_eq!(u.frame_tail, [7, 0, 0, 0]);
        assert_eq!(u.skybox_specular, [5.0, 1.0, 1.0, 1.0]);
        assert_eq!(u.skybox_specular_equirect_fov, PROJECTION360_DEFAULT_FOV);
        assert_eq!(u.skybox_specular_equirect_st, DEFAULT_MAIN_TEX_ST);
        assert_eq!(u.ambient_sh, [[0.0; 4]; 9]);
    }

    #[test]
    fn skybox_specular_params_pack_disabled_cubemap_and_equirect() {
        assert_eq!(
            SkyboxSpecularUniformParams::disabled().to_vec4(),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            SkyboxSpecularUniformParams::from_resident_mips(6, true).to_vec4(),
            [5.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            SkyboxSpecularUniformParams::from_resident_mips(0, true).to_vec4(),
            [0.0, 0.0, 1.0, 0.0]
        );
        let equirect =
            SkyboxSpecularUniformParams::from_equirect_resident_mips(3, true, [1.0; 4], [2.0; 4]);
        assert_eq!(equirect.to_vec4(), [2.0, 1.0, 1.0, 2.0]);
        assert_eq!(equirect.equirect_fov, [1.0; 4]);
        assert_eq!(equirect.equirect_st, [2.0; 4]);
    }

    #[test]
    fn render_sh2_packs_into_vec4_slots() {
        let sh = RenderSH2 {
            sh0: glam::Vec3::new(1.0, 2.0, 3.0),
            sh8: glam::Vec3::new(4.0, 5.0, 6.0),
            ..RenderSH2::default()
        };

        let packed = FrameGpuUniforms::ambient_sh_from_render_sh2(&sh);

        assert_eq!(packed[0], [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(packed[8], [4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn zero_render_sh2_packs_startup_fallback() {
        let packed = FrameGpuUniforms::ambient_sh_from_render_sh2(&RenderSH2::default());

        assert!(packed[0][0] > 0.0);
        assert_eq!(packed[1], [0.0; 4]);
    }
}
