//! Transform-removal id fixups for static and skinned mesh renderers.
//!
//! When the host swap-removes a transform, every renderer that referenced an affected slot
//! must have its `node_id` (and skinned bone slabs) rolled forward to the new index. Both
//! sweeps share the [`crate::scene::dense_update::for_each_row_with_par_dispatch`] policy:
//! they fall out to rayon when the row count crosses
//! [`crate::scene::dense_update::FIXUP_PARALLEL_MIN`].

use crate::scene::dense_update::for_each_row_with_par_dispatch;
use crate::scene::render_space::RenderSpaceState;
use crate::scene::transforms_apply::TransformRemovalEvent;
use crate::scene::world::fixup_transform_id;

/// Rolls each [`crate::scene::mesh_renderable::StaticMeshRenderer::node_id`] forward through
/// this frame's transform swap-removals so existing entries follow their transform when it was
/// swap-moved into a freed slot. Must run before the static-mesh apply step so any new state
/// rows land on correctly reindexed entries.
pub(crate) fn fixup_static_meshes_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    if removals.is_empty() || space.static_mesh_renderers.is_empty() {
        return;
    }
    for ev in removals {
        let removed_id = ev.removed_index;
        let last_index = ev.last_index_before_swap;
        for_each_row_with_par_dispatch(&mut space.static_mesh_renderers, |m| {
            m.node_id = fixup_transform_id(m.node_id, removed_id, last_index);
        });
    }
}

/// Same as [`fixup_static_meshes_for_transform_removals`] for the skinned path: also rewrites
/// every entry's bone transform-index slab and the optional root-bone transform id.
pub(super) fn fixup_skinned_bones_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    for ev in removals {
        let removed_id = ev.removed_index;
        let last_index = ev.last_index_before_swap;
        for_each_row_with_par_dispatch(&mut space.skinned_mesh_renderers, |entry| {
            entry.base.node_id = fixup_transform_id(entry.base.node_id, removed_id, last_index);
            for id in &mut entry.bone_transform_indices {
                *id = fixup_transform_id(*id, removed_id, last_index);
            }
            if let Some(rid) = entry.root_bone_transform_id {
                entry.root_bone_transform_id =
                    Some(fixup_transform_id(rid, removed_id, last_index));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::scene::mesh_renderable::StaticMeshRenderer;
    use crate::scene::render_space::RenderSpaceState;
    use crate::scene::transforms_apply::TransformRemovalEvent;

    use super::fixup_static_meshes_for_transform_removals;

    fn space_with_static_meshes(node_ids: &[i32]) -> RenderSpaceState {
        let mut space = RenderSpaceState::default();
        for &node_id in node_ids {
            space.static_mesh_renderers.push(StaticMeshRenderer {
                node_id,
                ..Default::default()
            });
        }
        space
    }

    #[test]
    fn static_mesh_node_id_follows_swap_remove() {
        let mut space = space_with_static_meshes(&[5, 42, 7]);
        fixup_static_meshes_for_transform_removals(
            &mut space,
            &[TransformRemovalEvent {
                removed_index: 5,
                last_index_before_swap: 42,
            }],
        );
        assert_eq!(space.static_mesh_renderers[0].node_id, -1);
        assert_eq!(space.static_mesh_renderers[1].node_id, 5);
        assert_eq!(space.static_mesh_renderers[2].node_id, 7);
    }

    #[test]
    fn static_mesh_fixup_no_op_when_no_removals() {
        let mut space = space_with_static_meshes(&[1, 2, 3]);
        fixup_static_meshes_for_transform_removals(&mut space, &[]);
        assert_eq!(
            space
                .static_mesh_renderers
                .iter()
                .map(|m| m.node_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn static_mesh_node_id_cleared_when_mesh_was_on_removed_transform() {
        let mut space = space_with_static_meshes(&[1]);
        fixup_static_meshes_for_transform_removals(
            &mut space,
            &[TransformRemovalEvent {
                removed_index: 1,
                last_index_before_swap: 1,
            }],
        );
        assert_eq!(space.static_mesh_renderers[0].node_id, -1);
    }

    /// Regression for the duplicate-hides-original bug: when host swap-removes a transform whose
    /// `last_index_before_swap` is the slot referenced by an existing static mesh, the orchestrated
    /// chain must remap that mesh's `node_id` to the freed slot exactly once.
    #[test]
    fn static_mesh_survives_transform_removal_when_swapped_into_freed_slot() {
        use glam::{Quat, Vec3};

        use crate::scene::transforms_apply::{
            ExtractedTransformsUpdate, apply_transforms_update_extracted,
        };
        use crate::scene::world::WorldTransformCache;
        use crate::shared::RenderTransform;

        let identity = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        };

        let mut space = RenderSpaceState::default();
        space.nodes.push(identity);
        space.nodes.push(identity);
        space.node_parents.push(-1);
        space.node_parents.push(-1);
        space
            .static_mesh_renderers
            .push(StaticMeshRenderer::default());
        space.static_mesh_renderers[0].node_id = 1;

        let mut cache = WorldTransformCache {
            world_matrices: vec![glam::Mat4::IDENTITY; 2],
            computed: vec![false; 2],
            local_matrices: vec![glam::Mat4::IDENTITY; 2],
            local_dirty: vec![true; 2],
            degenerate_scales: vec![false; 2],
            visit_epoch: vec![0; 2],
            ..Default::default()
        };

        let extracted = ExtractedTransformsUpdate {
            removals: vec![0, -1],
            target_transform_count: 1,
            ..Default::default()
        };
        let mut removal_events = Vec::new();
        apply_transforms_update_extracted(
            &mut space,
            &mut cache,
            crate::scene::ids::RenderSpaceId(0),
            &extracted,
            &mut removal_events,
        );

        fixup_static_meshes_for_transform_removals(&mut space, &removal_events);

        assert_eq!(space.nodes.len(), 1);
        assert_eq!(
            space.static_mesh_renderers[0].node_id, 0,
            "original's transform was swap-moved from slot 1 into slot 0; mesh must follow"
        );
    }
}
