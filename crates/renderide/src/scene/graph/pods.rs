//! Layout-compatible structs for shared memory access.
//!
//! These types match the host's buffer layout for zero-copy reads.

use bytemuck::{Pod, Zeroable};

/// Layout-compatible with layer_assignments buffer: (transform_id, layer_type) pairs.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(super) struct LayerAssignmentPod {
    pub transform_id: i32,
    pub layer_type: u8,
    pub _pad: [u8; 3],
}

/// Layout-compatible with MaterialOverrideState for shared memory access.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(super) struct MaterialOverrideStatePod {
    pub material_slot_index: i32,
    pub material_asset_id: i32,
}

/// Layout-compatible with MeshRendererState for shared memory access.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(super) struct MeshRendererStatePod {
    pub renderable_index: i32,
    pub mesh_asset_id: i32,
    pub material_count: i32,
    pub material_property_block_count: i32,
    pub sorting_order: i32,
    /// Host [`crate::shared::ShadowCastMode`] as `u8` (IPC layout).
    pub shadow_cast_mode: u8,
    pub _motion_vector_mode: u8,
    pub _pad: [u8; 2],
}
