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
use crate::gpu::frame_globals::{ClusteredFrameGlobalsParams, FrameGpuUniforms};
use crate::render_graph::camera::{
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_perspective,
    view_matrix_from_render_transform,
};
use crate::render_graph::frame_params::HostCameraFrame;
use crate::scene::SceneCoordinator;

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

impl ClusterFrameParams {
    /// Coefficients for `dot(coeffs.xyz, world) + coeffs.w` → view-space Z (third row of world-to-view).
    pub fn view_space_z_coeffs(&self) -> [f32; 4] {
        FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(self.world_to_view)
    }

    /// Projection coefficients `(P[0][0], P[1][1], P[0][2], P[1][2])` for this view's projection.
    pub fn proj_params(&self) -> [f32; 4] {
        FrameGpuUniforms::proj_params_from_proj(self.proj)
    }

    /// Builds [`FrameGpuUniforms`] for clustered PBS materials (must stay in sync with compute).
    ///
    /// `right_z_coeffs` / `right_proj_params` should be the right-eye equivalents in stereo, or
    /// equal to the left/mono coefficients in desktop mode. `frame_index` is the monotonic host
    /// frame index used by temporal / jittered screen-space effects.
    pub fn frame_gpu_uniforms(
        &self,
        camera_world_pos: glam::Vec3,
        light_count: u32,
        right_z_coeffs: [f32; 4],
        right_proj_params: [f32; 4],
        frame_index: u32,
    ) -> FrameGpuUniforms {
        FrameGpuUniforms::new_clustered(ClusteredFrameGlobalsParams {
            camera_world_pos,
            view_space_z_coeffs: self.view_space_z_coeffs(),
            view_space_z_coeffs_right: right_z_coeffs,
            cluster_count_x: self.cluster_count_x,
            cluster_count_y: self.cluster_count_y,
            cluster_count_z: CLUSTER_COUNT_Z,
            near_clip: self.near_clip,
            far_clip: self.far_clip,
            light_count,
            viewport_width: self.viewport_width.max(1),
            viewport_height: self.viewport_height.max(1),
            proj_params_left: self.proj_params(),
            proj_params_right: right_proj_params,
            frame_index,
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
        let scene_view = scene
            .active_main_space()
            .map(|s| view_matrix_from_render_transform(&s.view_transform))
            .unwrap_or(Mat4::IDENTITY);
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
            let z_float = (mid / near).log(far / near) * cluster_count_z as f32;
            let z_idx = z_float.clamp(0.0, cluster_count_z as f32 - 1.0).floor() as u32;
            assert_eq!(
                z_idx, k,
                "mid-depth of slice {k} should map back to slice index (got z_idx={z_idx})"
            );
        }
    }
}
