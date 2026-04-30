//! Shared-memory apply steps for transform and material override updates.
//!
//! Split into [`transforms`] (transform overrides), [`materials`] (material overrides), and
//! [`fixup`] (the generic transform-removal id sweep both call before applying their dense
//! updates). The barrel re-exports the public extraction functions, payload structs, and apply
//! entry points used by [`crate::scene::coordinator::parallel_apply`].

mod fixup;
mod materials;
mod transforms;

pub use materials::ExtractedRenderMaterialOverridesUpdate;
pub use transforms::ExtractedRenderTransformOverridesUpdate;

pub(crate) use materials::{
    apply_render_material_overrides_update_extracted, extract_render_material_overrides_update,
};
pub(crate) use transforms::{
    apply_render_transform_overrides_update_extracted, extract_render_transform_overrides_update,
};

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use crate::scene::render_overrides::types::{
        MaterialOverrideBinding, MeshRendererOverrideTarget, RenderMaterialOverrideEntry,
        RenderTransformOverrideEntry,
    };
    use crate::scene::render_space::RenderSpaceState;
    use crate::scene::transforms_apply::TransformRemovalEvent;
    use crate::shared::{
        MaterialOverrideState, RenderMaterialOverrideState, RenderTransformOverrideState,
        RenderingContext,
    };

    use super::{
        ExtractedRenderMaterialOverridesUpdate, ExtractedRenderTransformOverridesUpdate,
        apply_render_material_overrides_update_extracted,
        apply_render_transform_overrides_update_extracted,
    };

    fn removal(removed_index: i32, last_index_before_swap: usize) -> TransformRemovalEvent {
        TransformRemovalEvent {
            removed_index,
            last_index_before_swap,
        }
    }

    #[test]
    fn transform_override_apply_removes_adds_and_updates_state_rows() {
        let mut space = RenderSpaceState::default();
        space
            .render_transform_overrides
            .push(RenderTransformOverrideEntry {
                node_id: 10,
                ..Default::default()
            });
        space
            .render_transform_overrides
            .push(RenderTransformOverrideEntry {
                node_id: 20,
                ..Default::default()
            });

        let extracted = ExtractedRenderTransformOverridesUpdate {
            removals: vec![0, -1, 1],
            additions: vec![30, -1, 40],
            states: vec![
                RenderTransformOverrideState {
                    renderable_index: 0,
                    position_override: Vec3::new(1.0, 2.0, 3.0),
                    rotation_override: Quat::from_rotation_y(0.5),
                    scale_override: Vec3::new(2.0, 3.0, 4.0),
                    skinned_mesh_renderer_count: 2,
                    context: RenderingContext::ExternalView,
                    override_flags: 0b101,
                    ..Default::default()
                },
                RenderTransformOverrideState {
                    renderable_index: -1,
                    position_override: Vec3::splat(99.0),
                    ..Default::default()
                },
            ],
            skinned_mesh_renderers_indexes: vec![7, 8, 9],
        };

        apply_render_transform_overrides_update_extracted(&mut space, &extracted, &[]);

        assert_eq!(space.render_transform_overrides.len(), 2);
        let updated = &space.render_transform_overrides[0];
        assert_eq!(updated.node_id, 20);
        assert_eq!(updated.context, RenderingContext::ExternalView);
        assert_eq!(updated.position_override, Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(updated.rotation_override, None);
        assert_eq!(updated.scale_override, Some(Vec3::new(2.0, 3.0, 4.0)));
        assert_eq!(updated.skinned_mesh_renderer_indices, vec![7, 8]);
        assert_eq!(space.render_transform_overrides[1].node_id, 30);
    }

    #[test]
    fn transform_override_fixup_tracks_swap_removed_nodes() {
        let mut space = RenderSpaceState::default();
        space
            .render_transform_overrides
            .push(RenderTransformOverrideEntry {
                node_id: 5,
                ..Default::default()
            });
        space
            .render_transform_overrides
            .push(RenderTransformOverrideEntry {
                node_id: 42,
                ..Default::default()
            });
        space
            .render_transform_overrides
            .push(RenderTransformOverrideEntry {
                node_id: 7,
                ..Default::default()
            });

        apply_render_transform_overrides_update_extracted(
            &mut space,
            &ExtractedRenderTransformOverridesUpdate::default(),
            &[removal(5, 42)],
        );

        assert_eq!(space.render_transform_overrides[0].node_id, -1);
        assert_eq!(space.render_transform_overrides[1].node_id, 5);
        assert_eq!(space.render_transform_overrides[2].node_id, 7);
    }

    #[test]
    fn material_override_apply_removes_adds_decodes_target_and_rows() {
        let mut space = RenderSpaceState::default();
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 10,
                ..Default::default()
            });
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 20,
                ..Default::default()
            });

        let skinned_target = (1i32 << 30) | 12;
        let extracted = ExtractedRenderMaterialOverridesUpdate {
            removals: vec![0, -1],
            additions: vec![30, -1],
            states: vec![
                RenderMaterialOverrideState {
                    renderable_index: 0,
                    packed_mesh_renderer_index: skinned_target,
                    materrial_override_count: 2,
                    context: RenderingContext::Camera,
                    ..Default::default()
                },
                RenderMaterialOverrideState {
                    renderable_index: -1,
                    packed_mesh_renderer_index: 0,
                    ..Default::default()
                },
            ],
            material_override_states: vec![
                MaterialOverrideState {
                    material_slot_index: 0,
                    material_asset_id: 100,
                },
                MaterialOverrideState {
                    material_slot_index: 2,
                    material_asset_id: 200,
                },
                MaterialOverrideState {
                    material_slot_index: 9,
                    material_asset_id: 900,
                },
            ],
        };

        apply_render_material_overrides_update_extracted(&mut space, &extracted, &[]);

        assert_eq!(space.render_material_overrides.len(), 2);
        let updated = &space.render_material_overrides[0];
        assert_eq!(updated.node_id, 20);
        assert_eq!(updated.context, RenderingContext::Camera);
        assert_eq!(updated.target, MeshRendererOverrideTarget::Skinned(12));
        assert_eq!(
            updated.material_overrides,
            vec![
                MaterialOverrideBinding {
                    material_slot_index: 0,
                    material_asset_id: 100,
                },
                MaterialOverrideBinding {
                    material_slot_index: 2,
                    material_asset_id: 200,
                },
            ]
        );
        assert_eq!(space.render_material_overrides[1].node_id, 30);
    }

    #[test]
    fn material_override_fixup_tracks_swap_removed_nodes() {
        let mut space = RenderSpaceState::default();
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 1,
                ..Default::default()
            });
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 9,
                ..Default::default()
            });

        apply_render_material_overrides_update_extracted(
            &mut space,
            &ExtractedRenderMaterialOverridesUpdate::default(),
            &[removal(1, 9)],
        );

        assert_eq!(space.render_material_overrides[0].node_id, -1);
        assert_eq!(space.render_material_overrides[1].node_id, 1);
    }

    #[test]
    fn material_override_static_and_invalid_targets_decode() {
        let mut space = RenderSpaceState::default();
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 1,
                ..Default::default()
            });
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 2,
                ..Default::default()
            });

        let extracted = ExtractedRenderMaterialOverridesUpdate {
            states: vec![
                RenderMaterialOverrideState {
                    renderable_index: 0,
                    packed_mesh_renderer_index: 17,
                    materrial_override_count: 0,
                    context: RenderingContext::UserView,
                    ..Default::default()
                },
                RenderMaterialOverrideState {
                    renderable_index: 1,
                    packed_mesh_renderer_index: -1,
                    materrial_override_count: 0,
                    context: RenderingContext::Mirror,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        apply_render_material_overrides_update_extracted(&mut space, &extracted, &[]);

        assert_eq!(
            space.render_material_overrides[0].target,
            MeshRendererOverrideTarget::Static(17)
        );
        assert_eq!(
            space.render_material_overrides[1].target,
            MeshRendererOverrideTarget::Unknown
        );
        assert_eq!(
            space.render_material_overrides[1].context,
            RenderingContext::Mirror
        );
    }
}
