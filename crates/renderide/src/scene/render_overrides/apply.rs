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

    fn transform_entry(node_id: i32) -> RenderTransformOverrideEntry {
        RenderTransformOverrideEntry {
            node_id,
            ..Default::default()
        }
    }

    #[test]
    fn empty_extracted_override_updates_are_no_ops() {
        let mut space = RenderSpaceState::default();
        space
            .render_transform_overrides
            .push(RenderTransformOverrideEntry {
                node_id: 10,
                context: RenderingContext::Camera,
                position_override: Some(Vec3::X),
                rotation_override: Some(Quat::IDENTITY),
                scale_override: Some(Vec3::ONE),
                skinned_mesh_renderer_indices: vec![3, 4],
            });
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 20,
                context: RenderingContext::Mirror,
                target: MeshRendererOverrideTarget::Static(7),
                material_overrides: vec![MaterialOverrideBinding {
                    material_slot_index: 2,
                    material_asset_id: 99,
                }],
            });

        apply_render_transform_overrides_update_extracted(
            &mut space,
            &ExtractedRenderTransformOverridesUpdate::default(),
            &[],
        );
        apply_render_material_overrides_update_extracted(
            &mut space,
            &ExtractedRenderMaterialOverridesUpdate::default(),
            &[],
        );

        let transform = &space.render_transform_overrides[0];
        assert_eq!(transform.node_id, 10);
        assert_eq!(transform.context, RenderingContext::Camera);
        assert_eq!(transform.position_override, Some(Vec3::X));
        assert_eq!(transform.rotation_override, Some(Quat::IDENTITY));
        assert_eq!(transform.scale_override, Some(Vec3::ONE));
        assert_eq!(transform.skinned_mesh_renderer_indices, vec![3, 4]);

        let material = &space.render_material_overrides[0];
        assert_eq!(material.node_id, 20);
        assert_eq!(material.context, RenderingContext::Mirror);
        assert_eq!(material.target, MeshRendererOverrideTarget::Static(7));
        assert_eq!(
            material.material_overrides,
            vec![MaterialOverrideBinding {
                material_slot_index: 2,
                material_asset_id: 99,
            }]
        );
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
    fn transform_override_state_apply_skips_out_of_range_and_stops_at_terminator() {
        let mut space = RenderSpaceState::default();
        space.render_transform_overrides.push(transform_entry(10));

        let extracted = ExtractedRenderTransformOverridesUpdate {
            states: vec![
                RenderTransformOverrideState {
                    renderable_index: 99,
                    position_override: Vec3::new(1.0, 2.0, 3.0),
                    context: RenderingContext::Camera,
                    override_flags: 0b001,
                    ..Default::default()
                },
                RenderTransformOverrideState {
                    renderable_index: -1,
                    ..Default::default()
                },
                RenderTransformOverrideState {
                    renderable_index: 0,
                    position_override: Vec3::new(9.0, 9.0, 9.0),
                    context: RenderingContext::ExternalView,
                    override_flags: 0b001,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        apply_render_transform_overrides_update_extracted(&mut space, &extracted, &[]);

        let entry = &space.render_transform_overrides[0];
        assert_eq!(entry.node_id, 10);
        assert_eq!(entry.context, RenderingContext::default());
        assert_eq!(entry.position_override, None);
    }

    #[test]
    fn transform_override_flags_select_optional_fields() {
        let position = Vec3::new(1.0, 2.0, 3.0);
        let rotation = Quat::from_rotation_z(0.25);
        let scale = Vec3::new(2.0, 3.0, 4.0);
        let cases = [
            0b000u8, 0b001u8, 0b010u8, 0b011u8, 0b100u8, 0b101u8, 0b110u8, 0b111u8,
        ];

        for flags in cases {
            let mut space = RenderSpaceState::default();
            space
                .render_transform_overrides
                .push(RenderTransformOverrideEntry {
                    node_id: 1,
                    position_override: Some(Vec3::splat(-1.0)),
                    rotation_override: Some(Quat::IDENTITY),
                    scale_override: Some(Vec3::splat(-2.0)),
                    ..Default::default()
                });
            let extracted = ExtractedRenderTransformOverridesUpdate {
                states: vec![RenderTransformOverrideState {
                    renderable_index: 0,
                    position_override: position,
                    rotation_override: rotation,
                    scale_override: scale,
                    context: RenderingContext::Camera,
                    override_flags: flags,
                    ..Default::default()
                }],
                ..Default::default()
            };

            apply_render_transform_overrides_update_extracted(&mut space, &extracted, &[]);

            let entry = &space.render_transform_overrides[0];
            assert_eq!(entry.context, RenderingContext::Camera);
            assert_eq!(
                entry.position_override,
                ((flags & 0b001) != 0).then_some(position)
            );
            assert_eq!(
                entry.rotation_override,
                ((flags & 0b010) != 0).then_some(rotation)
            );
            assert_eq!(
                entry.scale_override,
                ((flags & 0b100) != 0).then_some(scale)
            );
        }
    }

    #[test]
    fn transform_override_skinned_index_slab_clamps_to_available_rows() {
        let mut space = RenderSpaceState::default();
        space.render_transform_overrides.push(transform_entry(10));
        space.render_transform_overrides.push(transform_entry(20));

        let extracted = ExtractedRenderTransformOverridesUpdate {
            states: vec![
                RenderTransformOverrideState {
                    renderable_index: 0,
                    skinned_mesh_renderer_count: 3,
                    ..Default::default()
                },
                RenderTransformOverrideState {
                    renderable_index: 1,
                    skinned_mesh_renderer_count: 2,
                    ..Default::default()
                },
            ],
            skinned_mesh_renderers_indexes: vec![7, 8],
            ..Default::default()
        };

        apply_render_transform_overrides_update_extracted(&mut space, &extracted, &[]);

        assert_eq!(
            space.render_transform_overrides[0].skinned_mesh_renderer_indices,
            vec![7, 8]
        );
        assert!(
            space.render_transform_overrides[1]
                .skinned_mesh_renderer_indices
                .is_empty()
        );
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
    fn material_override_state_apply_skips_out_of_range_and_stops_at_terminator() {
        let mut space = RenderSpaceState::default();
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 10,
                context: RenderingContext::Camera,
                target: MeshRendererOverrideTarget::Static(1),
                material_overrides: vec![MaterialOverrideBinding {
                    material_slot_index: 0,
                    material_asset_id: 5,
                }],
            });

        let extracted = ExtractedRenderMaterialOverridesUpdate {
            states: vec![
                RenderMaterialOverrideState {
                    renderable_index: 99,
                    packed_mesh_renderer_index: (1i32 << 30) | 6,
                    materrial_override_count: 1,
                    context: RenderingContext::Mirror,
                    ..Default::default()
                },
                RenderMaterialOverrideState {
                    renderable_index: -1,
                    ..Default::default()
                },
                RenderMaterialOverrideState {
                    renderable_index: 0,
                    packed_mesh_renderer_index: 7,
                    materrial_override_count: 1,
                    context: RenderingContext::ExternalView,
                    ..Default::default()
                },
            ],
            material_override_states: vec![MaterialOverrideState {
                material_slot_index: 9,
                material_asset_id: 900,
            }],
            ..Default::default()
        };

        apply_render_material_overrides_update_extracted(&mut space, &extracted, &[]);

        let entry = &space.render_material_overrides[0];
        assert_eq!(entry.node_id, 10);
        assert_eq!(entry.context, RenderingContext::Camera);
        assert_eq!(entry.target, MeshRendererOverrideTarget::Static(1));
        assert_eq!(
            entry.material_overrides,
            vec![MaterialOverrideBinding {
                material_slot_index: 0,
                material_asset_id: 5,
            }]
        );
    }

    #[test]
    fn material_override_rows_clear_and_clamp_to_available_slab_rows() {
        let mut space = RenderSpaceState::default();
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 10,
                material_overrides: vec![MaterialOverrideBinding {
                    material_slot_index: 7,
                    material_asset_id: 700,
                }],
                ..Default::default()
            });
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 20,
                material_overrides: vec![MaterialOverrideBinding {
                    material_slot_index: 8,
                    material_asset_id: 800,
                }],
                ..Default::default()
            });

        let extracted = ExtractedRenderMaterialOverridesUpdate {
            states: vec![
                RenderMaterialOverrideState {
                    renderable_index: 0,
                    packed_mesh_renderer_index: 3,
                    materrial_override_count: 3,
                    ..Default::default()
                },
                RenderMaterialOverrideState {
                    renderable_index: 1,
                    packed_mesh_renderer_index: 4,
                    materrial_override_count: 1,
                    ..Default::default()
                },
            ],
            material_override_states: vec![
                MaterialOverrideState {
                    material_slot_index: 0,
                    material_asset_id: 100,
                },
                MaterialOverrideState {
                    material_slot_index: 1,
                    material_asset_id: 200,
                },
            ],
            ..Default::default()
        };

        apply_render_material_overrides_update_extracted(&mut space, &extracted, &[]);

        assert_eq!(
            space.render_material_overrides[0].material_overrides,
            vec![
                MaterialOverrideBinding {
                    material_slot_index: 0,
                    material_asset_id: 100,
                },
                MaterialOverrideBinding {
                    material_slot_index: 1,
                    material_asset_id: 200,
                },
            ]
        );
        assert!(
            space.render_material_overrides[1]
                .material_overrides
                .is_empty()
        );
        assert_eq!(
            space.render_material_overrides[0].target,
            MeshRendererOverrideTarget::Static(3)
        );
        assert_eq!(
            space.render_material_overrides[1].target,
            MeshRendererOverrideTarget::Static(4)
        );
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
