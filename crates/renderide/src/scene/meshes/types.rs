//! Per-render-space mesh renderable rows and the row-apply trait.
//!
//! ## Dense tables
//!
//! Dense **`renderable_index`** from [`crate::shared::MeshRendererState`] maps to **`Vec` index**
//! after host removals (swap-with-last, buffer order). Static and skinned renderables use
//! **separate** tables, mirroring [`crate::shared::MeshRenderablesUpdate`] vs
//! [`crate::shared::SkinnedMeshRenderablesUpdate`].
//!
//! ## State row apply
//!
//! [`decode_mesh_renderer_state_plan`] consumes one [`MeshRendererState`] row plus its packed
//! material / property-block id slab. The scene apply paths merge decoded rows by renderer before
//! writing final plans in parallel. Ordering matches the host update stream: `material_count`
//! material asset ids, then when `material_property_block_count >= 0`, that many property-block
//! ids (possibly fewer than materials).

use crate::shared::{
    LayerType, MeshRendererState, MotionVectorMode, RenderBoundingBox, ShadowCastMode,
};

/// Renderer-local identity that survives dense table reindexing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MeshRendererInstanceId(
    /// Monotonic renderer-local value assigned by the owning render space.
    pub u64,
);

/// One submesh slot: material asset id and optional per-slot property block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeshMaterialSlot {
    /// `Material.assetId` from the host.
    pub material_asset_id: i32,
    /// Property block asset id for this slot when present.
    pub property_block_id: Option<i32>,
}

/// Static mesh draw instance.
#[derive(Debug, Clone)]
pub struct StaticMeshRenderer {
    /// Renderer-local identity assigned when the renderer entry is created.
    pub instance_id: MeshRendererInstanceId,
    /// Dense transform index this renderer attaches to (`node_id`).
    pub node_id: i32,
    /// Draw layer (opaque vs overlay vs hidden).
    pub layer: LayerType,
    /// Resident mesh asset id in [`crate::gpu_pools::MeshPool`].
    pub mesh_asset_id: i32,
    /// Host sorting order within the layer.
    pub sorting_order: i32,
    /// Whether this mesh casts shadows.
    pub shadow_cast_mode: ShadowCastMode,
    /// Motion vector generation mode from the host.
    pub motion_vector_mode: MotionVectorMode,
    /// Submesh order: one entry per material slot.
    pub material_slots: Vec<MeshMaterialSlot>,
    /// Legacy slot 0 material handle for single-material paths.
    pub primary_material_asset_id: Option<i32>,
    /// Legacy slot 0 property block when present.
    pub primary_property_block_id: Option<i32>,
    /// Blendshape weights by shape index (IPD path for static is reserved; skinned uses host batches).
    pub blend_shape_weights: Vec<f32>,
}

impl Default for StaticMeshRenderer {
    fn default() -> Self {
        Self {
            instance_id: MeshRendererInstanceId::default(),
            node_id: -1,
            layer: LayerType::Hidden,
            mesh_asset_id: -1,
            sorting_order: 0,
            shadow_cast_mode: ShadowCastMode::On,
            motion_vector_mode: MotionVectorMode::default(),
            material_slots: Vec::new(),
            primary_material_asset_id: None,
            primary_property_block_id: None,
            blend_shape_weights: Vec::new(),
        }
    }
}

impl StaticMeshRenderer {
    /// Returns whether this renderer should contribute to visible color rendering.
    #[inline]
    pub(crate) fn emits_visible_color_draws(&self) -> bool {
        self.shadow_cast_mode != ShadowCastMode::ShadowOnly
    }
}

/// Skinned mesh instance: [`StaticMeshRenderer`]-style header plus bone palette and root bone.
#[derive(Debug, Clone, Default)]
pub struct SkinnedMeshRenderer {
    /// Shared mesh/material/blendshape header.
    pub base: StaticMeshRenderer,
    /// Dense transform indices for each bone influence column.
    pub bone_transform_indices: Vec<i32>,
    /// Root bone transform id when the hierarchy is anchored.
    pub root_bone_transform_id: Option<i32>,
    /// Host-computed posed AABB for this skinned renderable, expressed in the space of
    /// [`Self::root_bone_transform_id`] (the renderer-root local frame the host sends to us in
    /// [`crate::shared::SkinnedMeshBoundsUpdate::local_bounds`]). `None` until the host has sent
    /// the first bounds row for this renderable -- culling falls back to the mesh bind-pose AABB
    /// transformed by the renderable's root matrix.
    pub posed_object_bounds: Option<RenderBoundingBox>,
}

/// Target for one [`MeshRendererState`] row: mesh/visual header and resolved material slots.
pub(crate) trait MeshRendererStateSink {
    /// Updates mesh asset, sort key, and shadow / motion-vector modes from the host row.
    fn set_mesh_visual_header(
        &mut self,
        mesh_asset_id: i32,
        sorting_order: i32,
        shadow: ShadowCastMode,
        motion: MotionVectorMode,
    );
    /// Replaces submesh [`MeshMaterialSlot`] list plus the row's primary material and property-block handles.
    fn set_material_slots_and_legacy(
        &mut self,
        slots: Vec<MeshMaterialSlot>,
        primary_material: Option<i32>,
        primary_pb: Option<i32>,
    );
}

#[derive(Debug, Clone)]
struct MeshRendererMaterialApplyPlan {
    slots: Vec<MeshMaterialSlot>,
    primary_material: Option<i32>,
    primary_property_block: Option<i32>,
}

/// Decoded effect of one or more [`MeshRendererState`] rows on a single mesh renderer.
#[derive(Debug, Clone)]
pub(crate) struct MeshRendererStateApplyPlan {
    mesh_asset_id: i32,
    sorting_order: i32,
    shadow_cast_mode: ShadowCastMode,
    motion_vector_mode: MotionVectorMode,
    material_update: Option<MeshRendererMaterialApplyPlan>,
}

impl MeshRendererStateApplyPlan {
    /// Merges a later decoded row into this plan using the same overwrite behavior as serial row apply.
    pub(crate) fn merge_later_row(&mut self, later: Self) {
        self.mesh_asset_id = later.mesh_asset_id;
        self.sorting_order = later.sorting_order;
        self.shadow_cast_mode = later.shadow_cast_mode;
        self.motion_vector_mode = later.motion_vector_mode;
        if later.material_update.is_some() {
            self.material_update = later.material_update;
        }
    }

    /// Applies this decoded plan to one renderer.
    pub(crate) fn apply_to<S: MeshRendererStateSink>(self, drawable: &mut S) {
        drawable.set_mesh_visual_header(
            self.mesh_asset_id,
            self.sorting_order,
            self.shadow_cast_mode,
            self.motion_vector_mode,
        );
        if let Some(materials) = self.material_update {
            drawable.set_material_slots_and_legacy(
                materials.slots,
                materials.primary_material,
                materials.primary_property_block,
            );
        }
    }
}

impl MeshRendererStateSink for StaticMeshRenderer {
    fn set_mesh_visual_header(
        &mut self,
        mesh_asset_id: i32,
        sorting_order: i32,
        shadow: ShadowCastMode,
        motion: MotionVectorMode,
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
        motion: MotionVectorMode,
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

/// Decodes one mesh renderer state row while advancing the packed-material cursor.
///
/// When `material_count < 0`, the returned plan leaves material slots unchanged and the cursor is
/// not advanced.
pub(crate) fn decode_mesh_renderer_state_plan(
    state: &MeshRendererState,
    packed_ids: Option<&[i32]>,
    cursor: &mut usize,
) -> MeshRendererStateApplyPlan {
    let material_update = if state.material_count < 0 {
        None
    } else {
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

        let (primary_material, primary_property_block) = if mat_ids.is_empty() {
            (None, None)
        } else {
            let pb0 = if state.material_property_block_count >= 0 {
                pb_ids.first().copied()
            } else {
                None
            };
            (Some(mat_ids[0]), pb0)
        };

        Some(MeshRendererMaterialApplyPlan {
            slots,
            primary_material,
            primary_property_block,
        })
    };

    MeshRendererStateApplyPlan {
        mesh_asset_id: state.mesh_asset_id,
        sorting_order: state.sorting_order,
        shadow_cast_mode: state.shadow_cast_mode,
        motion_vector_mode: state.motion_vector_mode,
        material_update,
    }
}

/// Applies `state` to `drawable` and advances `cursor` through `packed_ids`.
///
/// When `drawable` is `None`, mesh fields are not written but packed ids are still consumed when
/// `material_count >= 0`.
///
/// When `material_count < 0`, material slots are left unchanged and the cursor is not advanced.
#[cfg(test)]
pub(crate) fn apply_mesh_renderer_state_row<S: MeshRendererStateSink>(
    drawable: Option<&mut S>,
    state: &MeshRendererState,
    packed_ids: Option<&[i32]>,
    cursor: &mut usize,
) {
    let plan = decode_mesh_renderer_state_plan(state, packed_ids, cursor);
    if let Some(d) = drawable {
        plan.apply_to(d);
    }
}

#[cfg(test)]
mod renderer_tests {
    use super::*;

    #[test]
    fn shadow_only_renderers_do_not_emit_visible_color_draws() {
        let renderer = StaticMeshRenderer {
            shadow_cast_mode: ShadowCastMode::ShadowOnly,
            ..Default::default()
        };

        assert!(!renderer.emits_visible_color_draws());
    }

    #[test]
    fn non_shadow_only_modes_emit_visible_color_draws() {
        for shadow_cast_mode in [
            ShadowCastMode::Off,
            ShadowCastMode::On,
            ShadowCastMode::DoubleSided,
        ] {
            let renderer = StaticMeshRenderer {
                shadow_cast_mode,
                ..Default::default()
            };

            assert!(renderer.emits_visible_color_draws());
        }
    }
}

#[cfg(test)]
mod state_row_tests {
    use super::*;

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
            motion_vector_mode: MotionVectorMode::default(),
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
        assert!(
            d.material_slots
                .iter()
                .all(|s| s.property_block_id.is_none())
        );
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
