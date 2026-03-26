//! Material property store for stencil and property block lookup.
//!
//! Stores values from MaterialsUpdateBatch so stencil state (comparison, operation,
//! reference, clip rect) can be read per override block when building draw entries.
//!
//! ## Buffer layout assumptions (MaterialsUpdateBatch)
//!
//! Each buffer in `material_updates` contains a sequence of (MaterialPropertyUpdate, value) pairs:
//! - **MaterialPropertyUpdate**: 8 bytes (property_id: i32, update_type: u8, padding: [u8;3])
//! - **Value** (based on update_type):
//!   - `select_target`: i32 (4 bytes) — block_id for subsequent updates
//!   - `set_float`: f32 (4 bytes)
//!   - `set_float4`: [f32; 4] (16 bytes)
//!   - `set_float4x4`: 64 bytes — skipped
//!   - `set_shader`: i32 shader asset id (4 bytes) — see [`MaterialPropertyStore::set_shader_asset`]
//!   - `set_texture`: i32 packed texture reference (4 bytes) — see [`MaterialPropertyValue::Texture`]
//!   - `set_render_queue`, `set_instancing`, `set_render_type`: i32 each (4 bytes) — consumed, not stored
//!   - `update_batch_end`: 0 bytes
//!   - Other types: skipped (value size unknown)
//!
//! Bounds checks: we stop parsing when remaining bytes are insufficient for the next record.
//!
//! ## Block id vs drawable material handle (native UI routing)
//!
//! [`MaterialPropertyStore::shader_asset_for_block`] and property lookups use the **block id** from
//! each batch’s `select_target` before `set_shader` / `set_texture` / etc. Drawables resolve that
//! block id from the active material asset id (after multi-submesh fan-out, the submesh’s slot).
//! If the host sends material updates under a **different** `select_target` than that material id,
//! the store will not find the shader or textures for native UI routing; fixing that is a host /
//! scene contract issue, not something the renderer can infer safely.

use std::collections::HashMap;

/// Single property value. Supports f32 and [f32; 4] for stencil (comparison, operation,
/// reference, clip rect). Extensible for other types.
#[derive(Clone, Debug, PartialEq)]
pub enum MaterialPropertyValue {
    /// Single float (e.g. reference, blend factor).
    Float(f32),
    /// Four floats (e.g. clip rect x, y, width, height).
    Float4([f32; 4]),
    /// Packed texture id from host `set_texture` (see Renderite Unity `MaterialUpdateReader.ReadInt`).
    Texture(i32),
}

/// Material asset id and optional per-draw property block for merged property reads.
///
/// Matches Unity’s base `Material` plus per-index `MaterialPropertyBlock`: lookups prefer
/// [`Self::mesh_property_block_slot0`] when present, then fall back to [`Self::material_asset_id`].
/// After multi-submesh fan-out, the block id is the one paired with the active submesh in
/// [`crate::scene::Drawable::material_slots`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialPropertyLookupIds {
    /// Host material asset id (`MeshRenderer.sharedMaterials[k]` after fan-out).
    pub material_asset_id: i32,
    /// Optional `MaterialPropertyBlock` asset id for this draw’s submesh (or legacy slot 0).
    pub mesh_property_block_slot0: Option<i32>,
}

/// Store of material property values per block.
///
/// `block_id -> property_id -> value`. Block IDs come from RenderMaterialOverrideState
/// (material_override_block_id on Drawable). Property IDs are host-defined (e.g. IUIX_Material
/// stencil property IDs).
pub struct MaterialPropertyStore {
    /// block_id -> (property_id -> value)
    blocks: HashMap<i32, HashMap<i32, MaterialPropertyValue>>,
    /// block_id -> shader asset id from [`MaterialPropertyUpdateType::set_shader`](crate::shared::MaterialPropertyUpdateType::set_shader).
    shader_asset_by_block: HashMap<i32, i32>,
}

impl MaterialPropertyStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            shader_asset_by_block: HashMap::new(),
        }
    }

    /// Sets a property for a block. Creates the block entry if needed.
    pub fn set(&mut self, block_id: i32, property_id: i32, value: MaterialPropertyValue) {
        self.blocks
            .entry(block_id)
            .or_default()
            .insert(property_id, value);
    }

    /// Gets a property value for a block.
    pub fn get(&self, block_id: i32, property_id: i32) -> Option<&MaterialPropertyValue> {
        self.blocks.get(&block_id)?.get(&property_id)
    }

    /// Looks up `property_id` in `mesh_property_block_slot0` first, then in `material_asset_id`.
    ///
    /// Matches Unity-style material plus per-renderer `MaterialPropertyBlock` override behavior.
    pub fn get_merged(
        &self,
        ids: MaterialPropertyLookupIds,
        property_id: i32,
    ) -> Option<&MaterialPropertyValue> {
        if let Some(pb) = ids.mesh_property_block_slot0
            && let Some(v) = self.get(pb, property_id)
        {
            return Some(v);
        }
        self.get(ids.material_asset_id, property_id)
    }

    /// Records the shader asset bound to a material property block.
    pub fn set_shader_asset(&mut self, block_id: i32, shader_asset_id: i32) {
        self.shader_asset_by_block.insert(block_id, shader_asset_id);
    }

    /// Shader asset id for `block_id` when the host sent `set_shader` for that block.
    pub fn shader_asset_for_block(&self, block_id: i32) -> Option<i32> {
        self.shader_asset_by_block.get(&block_id).copied()
    }

    /// Removes all properties for a block. Called on UnloadMaterialPropertyBlock.
    pub fn remove_block(&mut self, block_id: i32) {
        self.blocks.remove(&block_id);
        self.shader_asset_by_block.remove(&block_id);
    }

    /// Returns true if the block has any properties.
    pub fn has_block(&self, block_id: i32) -> bool {
        self.blocks.contains_key(&block_id)
    }
}

impl Default for MaterialPropertyStore {
    fn default() -> Self {
        Self::new()
    }
}
