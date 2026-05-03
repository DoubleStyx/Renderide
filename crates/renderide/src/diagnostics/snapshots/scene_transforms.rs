//! Read-only snapshot of per-render-space world transforms for the debug HUD (no ImGui types).

use glam::{Quat, Vec3};

use crate::scene::{RenderSpaceId, SceneCoordinator};

/// One frame's world-space transform samples for every tracked render space.
#[derive(Clone, Debug, Default)]
pub struct SceneTransformsSnapshot {
    /// Sorted by [`RenderSpaceTransformsSnapshot::space_id`] for stable UI tab order.
    pub spaces: Vec<RenderSpaceTransformsSnapshot>,
}

/// Host render-space id, flags, and one row per dense transform index.
#[derive(Clone, Debug)]
pub struct RenderSpaceTransformsSnapshot {
    /// Host dictionary key for this space.
    pub space_id: i32,
    /// Mirrors [`crate::scene::RenderSpaceState::is_active`].
    pub is_active: bool,
    /// Mirrors [`crate::scene::RenderSpaceState::is_overlay`].
    pub is_overlay: bool,
    /// Mirrors [`crate::scene::RenderSpaceState::is_private`].
    pub is_private: bool,
    /// One row per dense transform node in this space.
    pub rows: Vec<TransformRow>,
}

/// Dense transform index, parent link, and hierarchy world TRS when available.
#[derive(Clone, Debug)]
pub struct TransformRow {
    /// Dense index in the host transform arena (`0..node_count`).
    pub transform_id: usize,
    /// Parent dense index, or `-1` for a hierarchy root under the space root.
    pub parent_id: i32,
    /// World matrix from the parent chain; `None` if the cache had no valid entry.
    pub world: Option<WorldTransformSample>,
}

/// Decomposed world TRS from [`SceneCoordinator::world_matrix`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldTransformSample {
    /// World-space translation.
    pub translation: Vec3,
    /// World-space rotation.
    pub rotation: Quat,
    /// World-space scale (lossy; from matrix decomposition).
    pub scale: Vec3,
}

impl SceneTransformsSnapshot {
    /// Builds a snapshot from the scene coordinator (call after [`SceneCoordinator::flush_world_caches`]
    /// when matrices must match the latest host submit).
    pub fn capture(scene: &SceneCoordinator) -> Self {
        let mut spaces: Vec<RenderSpaceTransformsSnapshot> = scene
            .render_space_ids()
            .filter_map(|id| RenderSpaceTransformsSnapshot::capture_one(scene, id))
            .collect();
        spaces.sort_by_key(|s| s.space_id);
        Self { spaces }
    }
}

impl RenderSpaceTransformsSnapshot {
    fn capture_one(scene: &SceneCoordinator, id: RenderSpaceId) -> Option<Self> {
        let space = scene.space(id)?;
        let space_id = id.0;
        let mut rows = Vec::with_capacity(space.nodes.len());
        for transform_id in 0..space.nodes.len() {
            let parent_id = space.node_parents.get(transform_id).copied().unwrap_or(-1);
            let world = scene
                .world_matrix(id, transform_id)
                .and_then(world_sample_from_mat4);
            rows.push(TransformRow {
                transform_id,
                parent_id,
                world,
            });
        }
        Some(Self {
            space_id,
            is_active: space.is_active,
            is_overlay: space.is_overlay,
            is_private: space.is_private,
            rows,
        })
    }
}

fn world_sample_from_mat4(m: glam::Mat4) -> Option<WorldTransformSample> {
    let (scale, rotation, translation) = m.to_scale_rotation_translation();
    if !translation.is_finite() || !rotation.is_finite() || !scale.is_finite() {
        return None;
    }
    Some(WorldTransformSample {
        translation,
        rotation,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_empty_scene() {
        let scene = SceneCoordinator::new();
        let snap = SceneTransformsSnapshot::capture(&scene);
        assert!(snap.spaces.is_empty());
    }

    #[test]
    fn capture_single_space_two_nodes_parent_child() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(7);
        scene.test_seed_space_identity_worlds(
            id,
            vec![
                crate::shared::RenderTransform::default(),
                crate::shared::RenderTransform::default(),
            ],
            vec![-1, 0],
        );

        let snap = SceneTransformsSnapshot::capture(&scene);
        assert_eq!(snap.spaces.len(), 1);
        assert_eq!(snap.spaces[0].space_id, 7);
        assert_eq!(snap.spaces[0].rows.len(), 2);
        assert_eq!(snap.spaces[0].rows[0].transform_id, 0);
        assert_eq!(snap.spaces[0].rows[0].parent_id, -1);
        assert_eq!(snap.spaces[0].rows[1].transform_id, 1);
        assert_eq!(snap.spaces[0].rows[1].parent_id, 0);
        assert!(
            snap.spaces[0].rows[0].world.is_some(),
            "identity world matrix should decompose"
        );
    }
}
