//! View–projection parameters for CPU frustum culling of world mesh draws.
//!
//! Values match [`super::passes::world_mesh_forward::WorldMeshForwardOpaquePass`] per-space `view` and
//! global projection state (`HostCameraFrame`, viewport aspect, clip planes). When
//! [`HostCameraFrame::explicit_world_to_view`] is set, frustum and Hi-Z temporal paths use
//! that world-to-view (same as the forward pass) instead of [`view_matrix_for_world_mesh_render_space`].

use std::sync::Arc;

use hashbrown::HashMap;

use glam::Mat4;

use crate::scene::{RenderSpaceId, SceneCoordinator};

use crate::camera::{
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_orthographic,
    reverse_z_perspective, view_matrix_from_render_transform,
};
use crate::occlusion::hi_z_cpu::hi_z_pyramid_dimensions;
use crate::render_graph::HiZCullData;
use crate::render_graph::frame_params::HostCameraFrame;

/// View and projection snapshot from the **frame that produced** the Hi-Z depth buffer (used for
/// CPU occlusion tests against the previous frame’s pyramid).
///
/// The per-space view table is stored as [`Arc<HashMap<...>>`] so per-view clones are refcount
/// bumps rather than full hash table copies (important when secondary render-texture cameras fan
/// out across rayon workers).
#[derive(Clone, Debug)]
pub struct HiZTemporalState {
    /// [`WorldMeshCullProjParams`] from the depth author frame (matches forward-pass cull bundle).
    pub prev_cull: WorldMeshCullProjParams,
    /// World-to-camera view matrix per render space at that frame (shared; cloning is cheap).
    ///
    /// For views with an explicit camera pose (e.g. secondary render-texture cameras), every space
    /// stores the same [`HostCameraFrame::explicit_world_to_view`] snapshot, matching the single
    /// view used to render that pass’s depth pyramid.
    pub prev_view_by_space: Arc<HashMap<RenderSpaceId, Mat4>>,
    /// Hi-Z mip0 size in texels (downscaled from full depth; see [`crate::occlusion::hi_z_cpu::hi_z_pyramid_dimensions`]).
    pub depth_viewport_px: (u32, u32),
}

/// Records per-space views and pyramid viewport for the next frame’s Hi-Z occlusion tests.
///
/// When `explicit_world_to_view` is [`Some`], that matrix is stored for every active render
/// space so Hi-Z tests use the same view as the offscreen depth author pass (see
/// [`HostCameraFrame::explicit_world_to_view`]).
pub fn capture_hi_z_temporal(
    scene: &SceneCoordinator,
    prev_cull: WorldMeshCullProjParams,
    full_viewport_px: (u32, u32),
    explicit_world_to_view: Option<Mat4>,
) -> HiZTemporalState {
    let mut prev_view_by_space = HashMap::new();
    if let Some(override_view) = explicit_world_to_view {
        for id in scene.render_space_ids() {
            if scene.space(id).is_some() {
                prev_view_by_space.insert(id, override_view);
            }
        }
    } else {
        for id in scene.render_space_ids() {
            if let Some(space) = scene.space(id) {
                let v = view_matrix_from_render_transform(&space.view_transform);
                prev_view_by_space.insert(id, v);
            }
        }
    }
    let depth_viewport_px = hi_z_pyramid_dimensions(full_viewport_px.0, full_viewport_px.1);
    HiZTemporalState {
        prev_cull,
        prev_view_by_space: Arc::new(prev_view_by_space),
        depth_viewport_px,
    }
}

/// Host camera + projection bundle for [`super::draw_prep::collect_and_sort_world_mesh_draws`].
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

    let vr_stereo = match (hc.vr_active, hc.stereo) {
        (true, Some(stereo)) => Some(stereo.view_proj),
        _ => None,
    };

    WorldMeshCullProjParams {
        world_proj,
        overlay_proj,
        vr_stereo,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`capture_hi_z_temporal`] and [`build_world_mesh_cull_proj_params`].

    use glam::Mat4;

    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::RenderTransform;

    use super::{
        WorldMeshCullProjParams, build_world_mesh_cull_proj_params, capture_hi_z_temporal,
    };
    use crate::camera::view_matrix_from_render_transform;
    use crate::occlusion::hi_z_cpu::hi_z_pyramid_dimensions;
    use crate::render_graph::frame_params::HostCameraFrame;

    #[test]
    fn capture_hi_z_temporal_secondary_override_fills_all_spaces() {
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(
            RenderSpaceId(1),
            vec![RenderTransform::default()],
            vec![-1],
        );
        scene.test_seed_space_identity_worlds(
            RenderSpaceId(2),
            vec![RenderTransform::default()],
            vec![-1],
        );
        let prev = WorldMeshCullProjParams {
            world_proj: Mat4::IDENTITY,
            overlay_proj: Mat4::IDENTITY,
            vr_stereo: None,
        };
        let m = Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.0));
        let t = capture_hi_z_temporal(&scene, prev, (1920, 1080), Some(m));
        assert_eq!(t.prev_view_by_space.len(), 2);
        for id in scene.render_space_ids() {
            assert_eq!(t.prev_view_by_space.get(&id).copied(), Some(m));
        }
        assert_eq!(t.depth_viewport_px, hi_z_pyramid_dimensions(1920, 1080));
    }

    #[test]
    fn capture_hi_z_temporal_without_override_uses_view_per_space() {
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(
            RenderSpaceId(5),
            vec![RenderTransform::default()],
            vec![-1],
        );
        let space = scene.space(RenderSpaceId(5)).expect("space");
        let expected = view_matrix_from_render_transform(&space.view_transform);
        let prev = WorldMeshCullProjParams {
            world_proj: Mat4::IDENTITY,
            overlay_proj: Mat4::IDENTITY,
            vr_stereo: None,
        };
        let t = capture_hi_z_temporal(&scene, prev, (800, 600), None);
        assert_eq!(
            t.prev_view_by_space.get(&RenderSpaceId(5)).copied(),
            Some(expected)
        );
    }

    #[test]
    fn build_world_mesh_cull_proj_params_sets_vr_stereo_only_when_active_and_pair_present() {
        use crate::render_graph::StereoViewMatrices;
        let scene = SceneCoordinator::new();
        let stereo = Some(StereoViewMatrices {
            view_proj: (Mat4::IDENTITY, Mat4::IDENTITY),
            view_only: (Mat4::IDENTITY, Mat4::IDENTITY),
            eye_world_position: (glam::Vec3::ZERO, glam::Vec3::ZERO),
        });
        let hc = HostCameraFrame {
            vr_active: true,
            stereo,
            ..Default::default()
        };
        let p = build_world_mesh_cull_proj_params(&scene, (1280, 720), &hc);
        assert!(p.vr_stereo.is_some());

        let hc2 = HostCameraFrame {
            vr_active: false,
            stereo,
            ..Default::default()
        };
        let p2 = build_world_mesh_cull_proj_params(&scene, (1280, 720), &hc2);
        assert!(p2.vr_stereo.is_none());
    }

    #[test]
    fn build_world_mesh_cull_proj_params_overlay_uses_primary_ortho_task_when_present() {
        let scene = SceneCoordinator::new();
        let hc = HostCameraFrame::default();
        let p_no = build_world_mesh_cull_proj_params(&scene, (800, 600), &hc);

        let hc_ortho = HostCameraFrame {
            primary_ortho_task: Some((10.0, 0.01, 5000.0)),
            ..Default::default()
        };
        let p_ortho = build_world_mesh_cull_proj_params(&scene, (800, 600), &hc_ortho);

        assert_eq!(p_no.world_proj, p_ortho.world_proj);
        assert_ne!(p_no.overlay_proj, p_ortho.overlay_proj);
    }
}
