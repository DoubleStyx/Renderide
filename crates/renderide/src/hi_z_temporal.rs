//! Hi-Z temporal snapshot capture shared by occlusion and world-mesh culling.

use std::sync::Arc;

use glam::Mat4;
use hashbrown::HashMap;

use crate::camera::view_matrix_from_render_transform;
use crate::cull_contract::{HiZTemporalState, WorldMeshCullProjParams};
use crate::hi_z_cpu::hi_z_pyramid_dimensions;
use crate::scene::{RenderSpaceRead, SceneSpaceRead};

/// Records per-space views and pyramid viewport for the next frame's Hi-Z occlusion tests.
///
/// When `explicit_world_to_view` is [`Some`], that matrix is stored for every active render
/// space so Hi-Z tests use the same view as the offscreen depth author pass.
pub fn capture_hi_z_temporal(
    scene: &(impl SceneSpaceRead + ?Sized),
    prev_cull: &WorldMeshCullProjParams,
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
                let v = view_matrix_from_render_transform(space.view_transform());
                prev_view_by_space.insert(id, v);
            }
        }
    }
    let depth_viewport_px = hi_z_pyramid_dimensions(full_viewport_px.0, full_viewport_px.1);
    HiZTemporalState {
        prev_cull: *prev_cull,
        prev_view_by_space: Arc::new(prev_view_by_space),
        depth_viewport_px,
    }
}

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use crate::cull_contract::WorldMeshCullProjParams;
    use crate::hi_z_cpu::hi_z_pyramid_dimensions;
    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::RenderTransform;

    use super::capture_hi_z_temporal;
    use crate::camera::view_matrix_from_render_transform;

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
        let t = capture_hi_z_temporal(&scene, &prev, (1920, 1080), Some(m));
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
        let expected = view_matrix_from_render_transform(space.view_transform());
        let prev = WorldMeshCullProjParams {
            world_proj: Mat4::IDENTITY,
            overlay_proj: Mat4::IDENTITY,
            vr_stereo: None,
        };
        let t = capture_hi_z_temporal(&scene, &prev, (800, 600), None);
        assert_eq!(
            t.prev_view_by_space.get(&RenderSpaceId(5)).copied(),
            Some(expected)
        );
    }
}
