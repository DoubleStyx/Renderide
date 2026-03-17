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
//!   - `update_batch_end`: 0 bytes
//!   - Other types: skipped (value size unknown)
//!
//! Bounds checks: we stop parsing when remaining bytes are insufficient for the next record.

use std::collections::HashMap;

/// Single property value. Supports f32 and [f32; 4] for stencil (comparison, operation,
/// reference, clip rect). Extensible for other types.
#[derive(Clone, Debug)]
pub enum MaterialPropertyValue {
    /// Single float (e.g. reference, blend factor).
    Float(f32),
    /// Four floats (e.g. clip rect x, y, width, height).
    Float4([f32; 4]),
}

/// Store of material property values per block.
///
/// `block_id -> property_id -> value`. Block IDs come from RenderMaterialOverrideState
/// (material_override_block_id on Drawable). Property IDs are host-defined (e.g. IUIX_Material
/// stencil property IDs).
pub struct MaterialPropertyStore {
    /// block_id -> (property_id -> value)
    blocks: HashMap<i32, HashMap<i32, MaterialPropertyValue>>,
}

impl MaterialPropertyStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
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

    /// Removes all properties for a block. Called on UnloadMaterialPropertyBlock.
    pub fn remove_block(&mut self, block_id: i32) {
        self.blocks.remove(&block_id);
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
