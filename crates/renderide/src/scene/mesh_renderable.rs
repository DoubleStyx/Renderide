//! Per–render-space mesh instances: static (`MeshRenderer`) and skinned (`SkinnedMeshRenderer`) tables.
//!
//! Dense **`renderable_index`** from [`crate::shared::MeshRendererState`] maps to **`Vec` index**
//! after host removals (swap-with-last, buffer order). Static and skinned renderables use
//! **separate** tables, mirroring [`crate::shared::MeshRenderablesUpdate`] vs
//! [`crate::shared::SkinnedMeshRenderablesUpdate`].

use crate::shared::{LayerType, MotionVectorMode, RenderBoundingBox, ShadowCastMode};

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
    /// Dense transform index this renderer attaches to (`node_id`).
    pub node_id: i32,
    /// Draw layer (opaque vs overlay vs hidden).
    pub layer: LayerType,
    /// Resident mesh asset id in [`crate::resources::MeshPool`].
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
    /// the first bounds row for this renderable — culling falls back to the mesh bind-pose AABB
    /// transformed by the renderable's root matrix.
    pub posed_object_bounds: Option<RenderBoundingBox>,
}
