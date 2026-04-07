//! Slot material and `MaterialPropertyBlock` asset ids from `mesh_materials_and_property_blocks`.
//!
//! Matches Renderite `MeshRendererManager.ApplyUpdate` consumption order:
//! for each [`MeshRendererStatePod`](super::super::pods::MeshRendererStatePod), read `materialCount`
//! ints (material asset ids), then when `materialPropertyBlockCount >= 0`, read that many property
//! block ids.
//!
//! # Consumers (inventory)
//!
//! - [`crate::scene::Drawable::material_handle`] / [`crate::scene::Drawable::mesh_renderer_property_block_slot0_id`]:
//!   legacy slot 0; kept in sync when materials are present.
//! - [`crate::scene::Drawable::material_slots`]: full slot list for multi-material rendering.
//! - [`crate::session::collect`] resolves shader / pipeline from material asset ids and optional
//!   property blocks.
//! - [`crate::render::pass::mesh_draw`] and [`crate::render::pass::material_draw_context`] bind
//!   native UI and host-unlit paths using merged property lookup.

use crate::scene::Drawable;
use crate::scene::MeshMaterialSlot;
use crate::shared::ShadowCastMode;
use crate::shared::enum_repr::EnumRepr;

use super::super::pods::MeshRendererStatePod;

/// Applies mesh renderer state to an optional drawable and advances `cursor` through `packed_ids`.
///
/// When `drawable` is `None` (e.g. invalid renderable index), mesh fields are not written but
/// packed ids are still consumed so the stream stays aligned with the host.
///
/// When `material_count < 0`, leaves existing material fields unchanged (host did not send a
/// material update for this row).
pub(super) fn apply_mesh_renderer_state_row(
    mut drawable: Option<&mut Drawable>,
    state: &MeshRendererStatePod,
    packed_ids: Option<&[i32]>,
    cursor: &mut usize,
) {
    if let Some(d) = drawable.as_mut() {
        d.mesh_handle = state.mesh_asset_id;
        d.sort_key = state.sorting_order;
        d.shadow_cast_mode = ShadowCastMode::from_i32(state.shadow_cast_mode as i32);
    }

    if state.material_count < 0 {
        return;
    }

    let packed = packed_ids.unwrap_or(&[]);
    let mc = state.material_count.max(0) as usize;

    let mat_ids: Vec<i32> = if mc > 0 {
        if *cursor + mc <= packed.len() {
            let s = packed[*cursor..*cursor + mc].to_vec();
            *cursor += mc;
            s
        } else {
            *cursor = packed.len();
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let pb_ids: Vec<i32> = if state.material_property_block_count >= 0 {
        let pbc = state.material_property_block_count.max(0) as usize;
        if pbc > 0 {
            if *cursor + pbc <= packed.len() {
                let s = packed[*cursor..*cursor + pbc].to_vec();
                *cursor += pbc;
                s
            } else {
                *cursor = packed.len();
                Vec::new()
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    if let Some(d) = drawable.as_mut() {
        d.material_slots = mat_ids
            .iter()
            .enumerate()
            .map(|(i, &material_asset_id)| MeshMaterialSlot {
                material_asset_id,
                property_block_id: pb_ids.get(i).copied(),
            })
            .collect();

        if mat_ids.is_empty() {
            d.material_handle = None;
            d.mesh_renderer_property_block_slot0_id = None;
        } else {
            d.material_handle = Some(mat_ids[0]);
            if state.material_property_block_count >= 0 {
                d.mesh_renderer_property_block_slot0_id = pb_ids.first().copied();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::LayerType;

    fn state(
        renderable_index: i32,
        mesh_id: i32,
        material_count: i32,
        property_block_count: i32,
    ) -> MeshRendererStatePod {
        MeshRendererStatePod {
            renderable_index,
            mesh_asset_id: mesh_id,
            material_count,
            material_property_block_count: property_block_count,
            sorting_order: 0,
            shadow_cast_mode: 1,
            _motion_vector_mode: 0,
            _pad: [0; 2],
        }
    }

    #[test]
    fn material_and_property_block_slot0_from_packed() {
        let packed = [10, 20, 30, 40];
        let mut d = Drawable {
            node_id: 0,
            layer: LayerType::overlay,
            mesh_handle: -1,
            material_handle: None,
            sort_key: 0,
            is_skinned: false,
            bone_transform_ids: None,
            root_bone_transform_id: None,
            blend_shape_weights: None,
            stencil_state: None,
            material_override_block_id: None,
            mesh_renderer_property_block_slot0_id: None,
            material_slots: Vec::new(),
            render_transform_override: None,
            shadow_cast_mode: ShadowCastMode::on,
        };
        let mut c = 0usize;
        apply_mesh_renderer_state_row(Some(&mut d), &state(0, 100, 2, 2), Some(&packed), &mut c);
        assert_eq!(d.mesh_handle, 100);
        assert_eq!(d.material_handle, Some(10));
        assert_eq!(d.mesh_renderer_property_block_slot0_id, Some(30));
        assert_eq!(
            d.material_slots,
            vec![
                MeshMaterialSlot {
                    material_asset_id: 10,
                    property_block_id: Some(30),
                },
                MeshMaterialSlot {
                    material_asset_id: 20,
                    property_block_id: Some(40),
                }
            ]
        );
        assert_eq!(c, 4);
    }

    #[test]
    fn three_materials_partial_property_blocks() {
        let packed = [1, 2, 3, 100, 200];
        let mut d = Drawable::default();
        let mut c = 0usize;
        apply_mesh_renderer_state_row(Some(&mut d), &state(0, 0, 3, 2), Some(&packed), &mut c);
        assert_eq!(d.material_slots.len(), 3);
        assert_eq!(d.material_slots[0].property_block_id, Some(100));
        assert_eq!(d.material_slots[1].property_block_id, Some(200));
        assert_eq!(d.material_slots[2].property_block_id, None);
        assert_eq!(c, 5);
    }

    #[test]
    fn negative_material_count_leaves_slots_unchanged() {
        let packed = [1, 2];
        let mut d = Drawable {
            material_slots: vec![MeshMaterialSlot {
                material_asset_id: 99,
                property_block_id: None,
            }],
            ..Default::default()
        };
        let mut c = 0usize;
        apply_mesh_renderer_state_row(Some(&mut d), &state(0, 5, -1, -1), Some(&packed), &mut c);
        assert_eq!(d.mesh_handle, 5);
        assert_eq!(d.material_slots.len(), 1);
        assert_eq!(d.material_slots[0].material_asset_id, 99);
        assert_eq!(c, 0);
    }

    #[test]
    fn no_property_block_stream_clears_per_slot_pb() {
        let packed = [7, 8];
        let mut d = Drawable::default();
        let mut c = 0usize;
        apply_mesh_renderer_state_row(Some(&mut d), &state(0, 0, 2, -1), Some(&packed), &mut c);
        assert_eq!(d.material_slots.len(), 2);
        assert!(
            d.material_slots
                .iter()
                .all(|s| s.property_block_id.is_none())
        );
        assert_eq!(d.mesh_renderer_property_block_slot0_id, None);
        assert_eq!(c, 2);
    }

    #[test]
    fn invalid_index_still_advances_cursor() {
        let packed = [1, 2];
        let mut c = 0usize;
        apply_mesh_renderer_state_row(None, &state(99, 0, 1, -1), Some(&packed), &mut c);
        assert_eq!(c, 1);
    }
}
