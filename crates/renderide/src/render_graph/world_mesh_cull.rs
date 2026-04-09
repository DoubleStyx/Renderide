//! View–projection parameters for CPU frustum culling of world mesh draws.
//!
//! Values match [`super::passes::world_mesh_forward::WorldMeshForwardPass`] per-space `view` and
//! global projection state (`HostCameraFrame`, viewport aspect, clip planes).

use std::collections::HashMap;

use glam::Mat4;

use crate::scene::{RenderSpaceId, SceneCoordinator};

use super::camera::{
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_orthographic,
    reverse_z_perspective, view_matrix_from_render_transform,
};
use super::frame_params::HostCameraFrame;
use super::hi_z_cpu::hi_z_pyramid_dimensions;
use super::HiZCullData;

/// View and projection snapshot from the **frame that produced** the Hi-Z depth buffer (used for
/// CPU occlusion tests against the previous frame’s pyramid).
#[derive(Clone, Debug)]
pub struct HiZTemporalState {
    /// [`WorldMeshCullProjParams`] from the depth author frame (matches forward-pass cull bundle).
    pub prev_cull: WorldMeshCullProjParams,
    /// World-to-camera view matrix per render space at that frame.
    pub prev_view_by_space: HashMap<RenderSpaceId, Mat4>,
    /// Hi-Z mip0 size in texels (downscaled from full depth; see [`super::hi_z_cpu::hi_z_pyramid_dimensions`]).
    pub depth_viewport_px: (u32, u32),
}

/// Records per-space views and pyramid viewport for the next frame’s Hi-Z occlusion tests.
pub fn capture_hi_z_temporal(
    scene: &SceneCoordinator,
    prev_cull: WorldMeshCullProjParams,
    full_viewport_px: (u32, u32),
) -> HiZTemporalState {
    let mut prev_view_by_space = HashMap::new();
    for id in scene.render_space_ids() {
        if let Some(space) = scene.space(id) {
            let v = view_matrix_from_render_transform(&space.view_transform);
            prev_view_by_space.insert(id, v);
        }
    }
    let depth_viewport_px = hi_z_pyramid_dimensions(full_viewport_px.0, full_viewport_px.1);
    HiZTemporalState {
        prev_cull,
        prev_view_by_space,
        depth_viewport_px,
    }
}

/// Host camera + projection bundle for [`super::world_mesh_draw_prep::collect_and_sort_world_mesh_draws`].
pub struct WorldMeshCullInput<'a> {
    /// Shared reverse-Z projection state for the frame.
    pub proj: WorldMeshCullProjParams,
    /// Per-frame head and clip data (bone palette and overlay projection parity).
    pub host_camera: &'a HostCameraFrame,
    /// Previous-frame hierarchical depth for optional occlusion after frustum tests.
    pub hi_z: Option<HiZCullData>,
    /// View/projection from the frame that authored [`Self::hi_z`]; required for stable temporal tests.
    pub hi_z_temporal: Option<HiZTemporalState>,
}

/// Projection matrices shared by all render spaces for a frame (before multiplying per-space `view`).
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshCullProjParams {
    /// Reverse-Z perspective for the main desktop / non-stereo path.
    pub world_proj: Mat4,
    /// Orthographic overlay projection (same choice as forward pass when overlay draws exist).
    pub overlay_proj: Mat4,
    /// OpenXR per-eye view–projection when VR is active; `None` when not using stereo culling.
    pub vr_stereo: Option<(Mat4, Mat4)>,
}

/// Builds [`WorldMeshCullProjParams`] from viewport size and [`HostCameraFrame`].
pub fn build_world_mesh_cull_proj_params(
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
    hc: &HostCameraFrame,
) -> WorldMeshCullProjParams {
    let (vw, vh) = viewport_px;
    let aspect = vw as f32 / vh.max(1) as f32;
    let (near, far) = effective_head_output_clip_planes(
        hc.near_clip,
        hc.far_clip,
        hc.output_device,
        scene
            .active_main_space()
            .map(|space| space.root_transform.scale),
    );
    let fov_rad = clamp_desktop_fov_degrees(hc.desktop_fov_degrees).to_radians();
    let world_proj = reverse_z_perspective(aspect, fov_rad, near, far);

    let overlay_proj = if let Some((half_h, on, of)) = hc.primary_ortho_task {
        reverse_z_orthographic(half_h * aspect, half_h, on, of)
    } else {
        reverse_z_orthographic(1.0 * aspect, 1.0, near, far)
    };

    let vr_stereo = match (hc.vr_active, hc.stereo_view_proj) {
        (true, Some(pair)) => Some(pair),
        _ => None,
    };

    WorldMeshCullProjParams {
        world_proj,
        overlay_proj,
        vr_stereo,
    }
}
