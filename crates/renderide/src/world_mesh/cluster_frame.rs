//! Shared **clustered forward** camera parameters for the light assignment compute pass and
//! fragment [`FrameGpuUniforms`](crate::gpu::frame_globals::FrameGpuUniforms).
//!
//! [`cluster_frame_params`] must produce **identical** `near_clip` / `far_clip`, projection, and
//! view matrix as the forward pass uses for [`FrameGpuUniforms::view_space_z_coeffs`] and cluster
//! grid dimensions; otherwise Z-slice and XY tile indices diverge and lighting pops at cluster
//! boundaries.

use glam::Mat4;

use crate::backend::CLUSTER_COUNT_Z;
use crate::backend::TILE_SIZE;
use crate::camera::{
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_perspective,
    view_matrix_from_render_transform,
};
use crate::gpu::frame_globals::{
    ClusteredFrameGlobalsParams, FrameGpuUniforms, SkyboxSpecularUniformParams,
};
use crate::render_graph::frame_params::HostCameraFrame;
use crate::scene::SceneCoordinator;

/// Minimum positive near distance used only by clustered Z slicing.
///
/// Keep in sync with `CLUSTER_NEAR_CLIP_MIN` in `shaders/source/modules/cluster_math.wgsl`.
pub const CLUSTER_NEAR_CLIP_MIN: f32 = 0.0001;
/// Minimum positive far-minus-near span used only by clustered Z slicing.
///
/// Keep in sync with `CLUSTER_FAR_CLIP_MIN_SPAN` in `shaders/source/modules/cluster_math.wgsl`.
pub const CLUSTER_FAR_CLIP_MIN_SPAN: f32 = 0.0001;

/// Returns clip planes sanitized exactly like the WGSL clustered-light helpers.
pub fn sanitize_cluster_clip_planes(near_clip: f32, far_clip: f32) -> (f32, f32) {
    let near_safe = if near_clip.is_finite() {
        near_clip.max(CLUSTER_NEAR_CLIP_MIN)
    } else {
        CLUSTER_NEAR_CLIP_MIN
    };
    let far_safe = if far_clip.is_finite() {
        far_clip.max(near_safe + CLUSTER_FAR_CLIP_MIN_SPAN)
    } else {
        near_safe + CLUSTER_FAR_CLIP_MIN_SPAN
    };
    (near_safe, far_safe)
}

/// Maps view-space Z to a clustered depth slice using the shared sanitized clip planes.
#[cfg(test)]
pub fn cluster_z_slice_from_view_z(
    view_z: f32,
    near_clip: f32,
    far_clip: f32,
    cluster_count_z: u32,
) -> u32 {
    let z_count = cluster_count_z.max(1);
    let (near_safe, far_safe) = sanitize_cluster_clip_planes(near_clip, far_clip);
    let d = (-view_z).clamp(near_safe, far_safe);
    let z = (d / near_safe).log(far_safe / near_safe) * z_count as f32;
    z.clamp(0.0, z_count.saturating_sub(1) as f32) as u32
}

/// Single source of truth for clustered lighting: clip planes, projection, main-space view, and grid size.
///
/// Use the same value for [`FrameGpuUniforms::new_clustered`] and for building clustered light
/// compute uniforms (see [`cluster_params_for_compute`] in [`super::passes::clustered_light`]).
#[derive(Clone, Copy, Debug)]
pub struct ClusterFrameParams {
    /// Effective near clip (positive distance), **same** as [`FrameGpuUniforms::near_clip`].
    pub near_clip: f32,
    /// Effective far clip (positive distance), **same** as [`FrameGpuUniforms::far_clip`].
    pub far_clip: f32,
    /// World-to-view for the active main space (handedness fix applied).
    pub world_to_view: Mat4,
    /// Reverse-Z perspective matching the desktop forward path (`world_mesh_forward`).
    pub proj: Mat4,
    /// Cluster grid width in tiles (matches [`FrameGpuUniforms::cluster_count_x`]).
    pub cluster_count_x: u32,
    /// Cluster grid height in tiles (matches [`FrameGpuUniforms::cluster_count_y`]).
    pub cluster_count_y: u32,
    /// Viewport width in pixels for cluster grid sizing.
    pub viewport_width: u32,
    /// Viewport height in pixels for cluster grid sizing.
    pub viewport_height: u32,
}

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
    /// Monotonic host frame index used by temporal effects.
    pub frame_index: u32,
    /// Packed skybox indirect-specular state for shader sampling.
    pub skybox_specular: SkyboxSpecularUniformParams,
    /// Skybox-derived SH2 irradiance coefficients for indirect diffuse.
    pub ambient_sh: [[f32; 4]; 9],
}

impl ClusterFrameParams {
    /// Coefficients for `dot(coeffs.xyz, world) + coeffs.w` → view-space Z (third row of world-to-view).
    pub fn view_space_z_coeffs(&self) -> [f32; 4] {
        FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(self.world_to_view)
    }

    /// Projection coefficients `(P[0][0], P[1][1], P[0][2], P[1][2])` for this view's projection.
    pub fn proj_params(&self) -> [f32; 4] {
        FrameGpuUniforms::proj_params_from_proj(self.proj)
    }

    /// Clip planes as the clustered compute and fragment shaders will use them for Z slicing.
    pub fn sanitized_clip_planes(&self) -> (f32, f32) {
        sanitize_cluster_clip_planes(self.near_clip, self.far_clip)
    }

    /// Maximum row length of the world-to-view linear part — the factor that converts a
    /// **world-space radius** to a **view-space radius** for this view.
    ///
    /// When the active render space has a non-unit scale `s` (e.g. a tiny avatar with `s = 0.01`)
    /// the world-to-view matrix carries `1/s` on its linear part, so any world position
    /// transformed by it is scaled by `1/s` in view space. Light positions are uploaded in world
    /// units, so the cluster compute's `light.range` must be multiplied by this factor before
    /// being compared against the view-space cluster AABB — otherwise the culling sphere appears
    /// `s ×` too small in view space and lights are bound to far fewer clusters than they cover,
    /// producing tile-shaped dark seams in the lit image. Mirrors Bevy's
    /// `view_from_world_scale.abs().max_element()` (`bevy_light/src/cluster/assign.rs`).
    ///
    /// Floored at `1e-6` to keep the multiplier finite for degenerate / zero matrices.
    pub fn world_to_view_scale_max(&self) -> f32 {
        let m = self.world_to_view;
        m.x_axis
            .truncate()
            .length()
            .max(m.y_axis.truncate().length())
            .max(m.z_axis.truncate().length())
            .max(1e-6)
    }

    /// Builds [`FrameGpuUniforms`] for clustered PBS materials (must stay in sync with compute).
    ///
    /// Right-eye fields in `params` should be the right-eye equivalents in stereo, or equal to the
    /// left/mono coefficients in desktop mode.
    pub fn frame_gpu_uniforms(&self, params: FrameGpuUniformBuildParams) -> FrameGpuUniforms {
        FrameGpuUniforms::new_clustered(ClusteredFrameGlobalsParams {
            camera_world_pos: params.camera_world_pos,
            camera_world_pos_right: params.camera_world_pos_right,
            view_space_z_coeffs: self.view_space_z_coeffs(),
            view_space_z_coeffs_right: params.right_z_coeffs,
            cluster_count_x: self.cluster_count_x,
            cluster_count_y: self.cluster_count_y,
            cluster_count_z: CLUSTER_COUNT_Z,
            near_clip: self.near_clip,
            far_clip: self.far_clip,
            light_count: params.light_count,
            viewport_width: self.viewport_width.max(1),
            viewport_height: self.viewport_height.max(1),
            proj_params_left: self.proj_params(),
            proj_params_right: params.right_proj_params,
            frame_index: params.frame_index,
            skybox_specular: params.skybox_specular,
            ambient_sh: params.ambient_sh,
        })
    }
}

/// Computes clustered-forward parameters for the current viewport and host camera (mono / desktop).
///
/// In desktop mode or when stereo views are unavailable, a single symmetric perspective projection
/// and the scene main-space view matrix are used.
pub fn cluster_frame_params(
    host_camera: &HostCameraFrame,
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
) -> Option<ClusterFrameParams> {
    if let (Some(view), Some(proj)) = (
        host_camera.cluster_view_override,
        host_camera.cluster_proj_override,
    ) {
        let (vw, vh) = viewport_px;
        if vw == 0 || vh == 0 {
            return None;
        }
        let (near_clip, far_clip) = effective_head_output_clip_planes(
            host_camera.near_clip,
            host_camera.far_clip,
            host_camera.output_device,
            scene
                .active_main_space()
                .map(|space| space.root_transform.scale),
        );
        let cluster_count_x = vw.div_ceil(TILE_SIZE);
        let cluster_count_y = vh.div_ceil(TILE_SIZE);
        return Some(ClusterFrameParams {
            near_clip,
            far_clip,
            world_to_view: view,
            proj,
            cluster_count_x,
            cluster_count_y,
            viewport_width: vw,
            viewport_height: vh,
        });
    }

    let common = CommonClusterInputs::compute(host_camera, scene, viewport_px)?;

    let world_to_view = common.scene_view;
    let proj = reverse_z_perspective(
        common.aspect,
        common.fov_rad,
        common.near_clip,
        common.far_clip,
    );

    Some(ClusterFrameParams {
        near_clip: common.near_clip,
        far_clip: common.far_clip,
        world_to_view,
        proj,
        cluster_count_x: common.cluster_count_x,
        cluster_count_y: common.cluster_count_y,
        viewport_width: common.vw,
        viewport_height: common.vh,
    })
}

/// Returns per-eye cluster params when stereo view matrices and view–projections are available.
///
/// Each eye gets its own `world_to_view` (from [`StereoViewMatrices::view_only`]) and projection
/// (decomposed as `vp * view.inverse()`). Returns `None` when stereo data is absent, falling back
/// to [`cluster_frame_params`] for mono clustering.
pub fn cluster_frame_params_stereo(
    host_camera: &HostCameraFrame,
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
) -> Option<(ClusterFrameParams, ClusterFrameParams)> {
    let common = CommonClusterInputs::compute(host_camera, scene, viewport_px)?;
    let stereo = host_camera.stereo?;
    let (sl, sr) = stereo.view_proj;
    let (view_l, view_r) = stereo.view_only;

    let proj_l = extract_proj(
        sl,
        view_l,
        common.aspect,
        common.fov_rad,
        common.near_clip,
        common.far_clip,
    );
    let proj_r = extract_proj(
        sr,
        view_r,
        common.aspect,
        common.fov_rad,
        common.near_clip,
        common.far_clip,
    );

    let left = ClusterFrameParams {
        near_clip: common.near_clip,
        far_clip: common.far_clip,
        world_to_view: view_l,
        proj: proj_l,
        cluster_count_x: common.cluster_count_x,
        cluster_count_y: common.cluster_count_y,
        viewport_width: common.vw,
        viewport_height: common.vh,
    };
    let right = ClusterFrameParams {
        near_clip: common.near_clip,
        far_clip: common.far_clip,
        world_to_view: view_r,
        proj: proj_r,
        cluster_count_x: common.cluster_count_x,
        cluster_count_y: common.cluster_count_y,
        viewport_width: common.vw,
        viewport_height: common.vh,
    };
    Some((left, right))
}

/// Shared inputs derived once for both mono and stereo paths.
struct CommonClusterInputs {
    near_clip: f32,
    far_clip: f32,
    scene_view: Mat4,
    aspect: f32,
    fov_rad: f32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    vw: u32,
    vh: u32,
}

impl CommonClusterInputs {
    fn compute(
        host_camera: &HostCameraFrame,
        scene: &SceneCoordinator,
        viewport_px: (u32, u32),
    ) -> Option<Self> {
        let (vw, vh) = viewport_px;
        if vw == 0 || vh == 0 {
            return None;
        }
        let (near_clip, far_clip) = effective_head_output_clip_planes(
            host_camera.near_clip,
            host_camera.far_clip,
            host_camera.output_device,
            scene
                .active_main_space()
                .map(|space| space.root_transform.scale),
        );
        let scene_view = scene.active_main_space().map_or(Mat4::IDENTITY, |s| {
            view_matrix_from_render_transform(&s.view_transform)
        });
        let aspect = vw as f32 / vh.max(1) as f32;
        let fov_rad = clamp_desktop_fov_degrees(host_camera.desktop_fov_degrees).to_radians();
        let cluster_count_x = vw.div_ceil(TILE_SIZE);
        let cluster_count_y = vh.div_ceil(TILE_SIZE);
        Some(Self {
            near_clip,
            far_clip,
            scene_view,
            aspect,
            fov_rad,
            cluster_count_x,
            cluster_count_y,
            vw,
            vh,
        })
    }
}

/// Decomposes projection from a combined view–projection: `proj = vp * view.inverse()`.
/// Falls back to a symmetric desktop projection if the decomposition yields non-finite values.
fn extract_proj(vp: Mat4, view: Mat4, aspect: f32, fov_rad: f32, near: f32, far: f32) -> Mat4 {
    let p = vp * view.inverse();
    if mat4_all_finite(p) {
        p
    } else {
        reverse_z_perspective(aspect, fov_rad, near, far)
    }
}

fn mat4_all_finite(m: Mat4) -> bool {
    m.to_cols_array().iter().all(|f| f.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Builds a minimal `ClusterFrameParams` with the supplied world-to-view; other fields are
    /// don't-cares for `world_to_view_scale_max`.
    fn cfp_with_view(world_to_view: Mat4) -> ClusterFrameParams {
        ClusterFrameParams {
            near_clip: 0.1,
            far_clip: 1000.0,
            world_to_view,
            proj: Mat4::IDENTITY,
            cluster_count_x: 1,
            cluster_count_y: 1,
            viewport_width: 1,
            viewport_height: 1,
        }
    }

    /// Tiny avatar (player scale `s = 0.01`) yields a view matrix with `1/s = 100` linear scale,
    /// so the multiplier that converts world radii to view-space radii is `100`.
    #[test]
    fn world_to_view_scale_max_recovers_player_inverse_scale() {
        let world = Mat4::from_scale(Vec3::splat(0.01));
        let view = world.inverse();
        let cfp = cfp_with_view(view);
        let s = cfp.world_to_view_scale_max();
        assert!((s - 100.0).abs() < 1e-3, "expected ~100, got {s}");
    }

    #[test]
    fn world_to_view_scale_max_handles_unit_scale() {
        let cfp = cfp_with_view(Mat4::IDENTITY);
        let s = cfp.world_to_view_scale_max();
        assert!((s - 1.0).abs() < 1e-6, "expected 1.0, got {s}");
    }

    /// Non-uniform scale: world `(0.01, 0.5, 1.0)` ⇒ inverse axes lengths `(100, 2, 1)` ⇒ max 100.
    #[test]
    fn world_to_view_scale_max_uses_max_axis_for_nonuniform_scale() {
        let world = Mat4::from_scale(Vec3::new(0.01, 0.5, 1.0));
        let view = world.inverse();
        let cfp = cfp_with_view(view);
        let s = cfp.world_to_view_scale_max();
        assert!((s - 100.0).abs() < 1e-3, "expected ~100, got {s}");
    }

    /// Degenerate (all-zero) view matrix must not yield `0` or `NaN` — the floor keeps it finite.
    #[test]
    fn world_to_view_scale_max_floors_degenerate_view() {
        let cfp = cfp_with_view(Mat4::ZERO);
        let s = cfp.world_to_view_scale_max();
        assert!(s.is_finite() && s >= 1e-6, "expected ≥1e-6 finite, got {s}");
    }

    #[test]
    fn cluster_z_slice_formula_matches_exponential_bounds() {
        let near = 0.1_f32;
        let far = 1000.0_f32;
        let cluster_count_z = CLUSTER_COUNT_Z;
        for k in 0..cluster_count_z {
            let t0 = k as f32 / cluster_count_z as f32;
            let t1 = (k + 1) as f32 / cluster_count_z as f32;
            let d0 = near * (far / near).powf(t0);
            let d1 = near * (far / near).powf(t1);
            let mid = 0.5 * (d0 + d1);
            let z_idx = cluster_z_slice_from_view_z(-mid, near, far, cluster_count_z);
            assert_eq!(
                z_idx, k,
                "mid-depth of slice {k} should map back to slice index (got z_idx={z_idx})"
            );
        }
    }

    /// Tiny effective near clips are lifted to the same clustered near floor used by WGSL.
    #[test]
    fn sanitize_cluster_clip_planes_matches_shader_policy_for_tiny_near() {
        let (near, far) = sanitize_cluster_clip_planes(0.00001, 100.0);
        assert_eq!(near, CLUSTER_NEAR_CLIP_MIN);
        assert_eq!(far, 100.0);
    }

    /// Far clips that collapse into near are separated by the shared minimum span.
    #[test]
    fn sanitize_cluster_clip_planes_keeps_far_above_near() {
        let (near, far) = sanitize_cluster_clip_planes(0.00001, 0.00002);
        assert_eq!(near, CLUSTER_NEAR_CLIP_MIN);
        assert_eq!(far, CLUSTER_NEAR_CLIP_MIN + CLUSTER_FAR_CLIP_MIN_SPAN);
    }

    /// `ClusterFrameParams` exposes the sanitized planes used by compute and fragment lookup.
    #[test]
    fn cluster_frame_params_exposes_sanitized_cluster_planes() {
        let cfp = ClusterFrameParams {
            near_clip: 0.00001,
            far_clip: 10.0,
            world_to_view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            cluster_count_x: 1,
            cluster_count_y: 1,
            viewport_width: 1,
            viewport_height: 1,
        };

        assert_eq!(cfp.sanitized_clip_planes(), (CLUSTER_NEAR_CLIP_MIN, 10.0));
    }

    /// The WGSL constants stay manually synchronized with the Rust constants.
    #[test]
    fn wgsl_cluster_clip_constants_match_rust() {
        let wgsl = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/source/modules/cluster_math.wgsl"
        ));

        assert!(wgsl.contains("const CLUSTER_NEAR_CLIP_MIN: f32 = 0.0001;"));
        assert!(wgsl.contains("const CLUSTER_FAR_CLIP_MIN_SPAN: f32 = 0.0001;"));
    }
}
