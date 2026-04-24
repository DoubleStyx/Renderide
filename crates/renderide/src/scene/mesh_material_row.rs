//! Applies one [`crate::shared::MeshRendererState`] row and packed material / property-block ids.
//!
//! Ordering matches Renderite `MeshRendererManager.ApplyUpdate`: `material_count` material asset ids,
//! then when `material_property_block_count >= 0`, that many property-block ids (possibly fewer than
//! materials).

use crate::shared::{MeshRendererState, ShadowCastMode};

use super::mesh_renderable::{MeshMaterialSlot, SkinnedMeshRenderer, StaticMeshRenderer};

/// Target for one [`MeshRendererState`] row: mesh/visual header and resolved material slots.
pub(crate) trait MeshRendererStateSink {
    /// Updates mesh asset, sort key, and shadow / motion-vector modes from the host row.
    fn set_mesh_visual_header(
        &mut self,
        mesh_asset_id: i32,
        sorting_order: i32,
        shadow: ShadowCastMode,
        motion: crate::shared::MotionVectorMode,
    );
    /// Replaces submesh [`MeshMaterialSlot`] list plus the row's primary material and property-block handles.
    fn set_material_slots_and_legacy(
        &mut self,
        slots: Vec<MeshMaterialSlot>,
        primary_material: Option<i32>,
        primary_pb: Option<i32>,
    );
}

impl MeshRendererStateSink for StaticMeshRenderer {
    fn set_mesh_visual_header(
        &mut self,
        mesh_asset_id: i32,
        sorting_order: i32,
        shadow: ShadowCastMode,
        motion: crate::shared::MotionVectorMode,
    ) {
        self.mesh_asset_id = mesh_asset_id;
        self.sorting_order = sorting_order;
        self.shadow_cast_mode = shadow;
        self.motion_vector_mode = motion;
    }

    fn set_material_slots_and_legacy(
        &mut self,
        slots: Vec<MeshMaterialSlot>,
        primary_material: Option<i32>,
        primary_pb: Option<i32>,
    ) {
        self.material_slots = slots;
        self.primary_material_asset_id = primary_material;
        self.primary_property_block_id = primary_pb;
    }
}

impl MeshRendererStateSink for SkinnedMeshRenderer {
    fn set_mesh_visual_header(
        &mut self,
        mesh_asset_id: i32,
        sorting_order: i32,
        shadow: ShadowCastMode,
        motion: crate::shared::MotionVectorMode,
    ) {
        self.base
            .set_mesh_visual_header(mesh_asset_id, sorting_order, shadow, motion);
    }

    fn set_material_slots_and_legacy(
        &mut self,
        slots: Vec<MeshMaterialSlot>,
        primary_material: Option<i32>,
        primary_pb: Option<i32>,
    ) {
        self.base
            .set_material_slots_and_legacy(slots, primary_material, primary_pb);
    }
}

/// Applies `state` to `drawable` and advances `cursor` through `packed_ids`.
///
/// When `drawable` is `None`, mesh fields are not written but packed ids are still consumed when
/// `material_count >= 0`.
///
/// When `material_count < 0`, material slots are left unchanged and the cursor is not advanced.
pub(crate) fn apply_mesh_renderer_state_row<S: MeshRendererStateSink>(
    mut drawable: Option<&mut S>,
    state: &MeshRendererState,
    packed_ids: Option<&[i32]>,
    cursor: &mut usize,
) {
    if let Some(d) = drawable.as_mut() {
        d.set_mesh_visual_header(
            state.mesh_asset_id,
            state.sorting_order,
            state.shadow_cast_mode,
            state.motion_vector_mode,
        );
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

    let slots: Vec<MeshMaterialSlot> = mat_ids
        .iter()
        .enumerate()
        .map(|(i, &material_asset_id)| MeshMaterialSlot {
            material_asset_id,
            property_block_id: pb_ids.get(i).copied(),
        })
        .collect();

    let (primary_mat, primary_pb) = if mat_ids.is_empty() {
        (None, None)
    } else {
        let pb0 = if state.material_property_block_count >= 0 {
            pb_ids.first().copied()
        } else {
            None
        };
        (Some(mat_ids[0]), pb0)
    };

    if let Some(d) = drawable {
        d.set_material_slots_and_legacy(slots, primary_mat, primary_pb);
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
    ) -> MeshRendererState {
        MeshRendererState {
            renderable_index,
            mesh_asset_id: mesh_id,
            material_count,
            material_property_block_count: property_block_count,
            sorting_order: 0,
            shadow_cast_mode: ShadowCastMode::On,
            motion_vector_mode: crate::shared::MotionVectorMode::default(),
            _padding: [0; 2],
        }
    }

    #[test]
    fn material_and_property_block_slot0_from_packed() {
        let packed = [10, 20, 30, 40];
        let mut d = StaticMeshRenderer {
            node_id: 0,
            layer: LayerType::Hidden,
            ..Default::default()
        };
        let mut c = 0usize;
        apply_mesh_renderer_state_row(Some(&mut d), &state(0, 100, 2, 2), Some(&packed), &mut c);
        assert_eq!(d.mesh_asset_id, 100);
        assert_eq!(d.primary_material_asset_id, Some(10));
        assert_eq!(d.primary_property_block_id, Some(30));
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
        let mut d = StaticMeshRenderer::default();
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
        let mut d = StaticMeshRenderer {
            material_slots: vec![MeshMaterialSlot {
                material_asset_id: 99,
                property_block_id: None,
            }],
            ..Default::default()
        };
        let mut c = 0usize;
        apply_mesh_renderer_state_row(Some(&mut d), &state(0, 5, -1, -1), Some(&packed), &mut c);
        assert_eq!(d.mesh_asset_id, 5);
        assert_eq!(d.material_slots.len(), 1);
        assert_eq!(d.material_slots[0].material_asset_id, 99);
        assert_eq!(c, 0);
    }

    #[test]
    fn no_property_block_stream_clears_per_slot_pb() {
        let packed = [7, 8];
        let mut d = StaticMeshRenderer::default();
        let mut c = 0usize;
        apply_mesh_renderer_state_row(Some(&mut d), &state(0, 0, 2, -1), Some(&packed), &mut c);
        assert_eq!(d.material_slots.len(), 2);
        assert!(d
            .material_slots
            .iter()
            .all(|s| s.property_block_id.is_none()));
        assert_eq!(d.primary_property_block_id, None);
        assert_eq!(c, 2);
    }

    #[test]
    fn invalid_index_still_advances_cursor() {
        let packed = [1, 2];
        let mut c = 0usize;
        apply_mesh_renderer_state_row::<StaticMeshRenderer>(
            None,
            &state(99, 0, 1, -1),
            Some(&packed),
            &mut c,
        );
        assert_eq!(c, 1);
    }

    #[test]
    fn skinned_delegates_to_base() {
        let packed = [11, 22];
        let mut d = SkinnedMeshRenderer::default();
        let mut c = 0usize;
        apply_mesh_renderer_state_row(Some(&mut d), &state(0, 42, 2, -1), Some(&packed), &mut c);
        assert_eq!(d.base.mesh_asset_id, 42);
        assert_eq!(d.base.material_slots.len(), 2);
        assert_eq!(c, 2);
    }
}
