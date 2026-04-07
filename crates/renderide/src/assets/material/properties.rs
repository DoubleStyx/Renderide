//! Material property store: values from [`crate::shared::MaterialsUpdateBatch`] for per-material and
//! property-block lookups.
//!
//! Parity with FrooxEngine / Renderite `MaterialUpdateWriter` / `MaterialUpdateReader` is documented
//! in [`super::update_batch::parse_materials_update_batch_into_store`].

use std::collections::HashMap;

/// Maximum `set_float_array` elements stored when extended persistence is enabled.
pub const MATERIAL_BATCH_MAX_FLOAT_ARRAY_LEN: usize = 256;
/// Maximum `set_float4_array` vec4 elements stored when extended persistence is enabled.
pub const MATERIAL_BATCH_MAX_FLOAT4_ARRAY_LEN: usize = 64;

/// Single host material property value persisted after batch parsing.
#[derive(Clone, Debug, PartialEq)]
pub enum MaterialPropertyValue {
    /// `set_float`.
    Float(f32),
    /// `set_float4`.
    Float4([f32; 4]),
    /// Column-major `mat4` from `set_float4x4`.
    Float4x4([f32; 16]),
    /// `set_float_array` payload (capped).
    FloatArray(Vec<f32>),
    /// `set_float4_array` payload (capped).
    Float4Array(Vec<[f32; 4]>),
    /// Packed texture reference from `set_texture`.
    Texture(i32),
}

/// Host material id plus optional [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)-style override id.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialPropertyLookupIds {
    /// Material asset id (e.g. `MeshRenderer.sharedMaterials[k]`).
    pub material_asset_id: i32,
    /// Optional property block asset id for this draw.
    pub mesh_property_block_slot0: Option<i32>,
}

/// Read-only view over [`MaterialPropertyStore`] for shader and merged property queries.
pub struct MaterialDictionary<'a> {
    store: &'a MaterialPropertyStore,
}

impl<'a> MaterialDictionary<'a> {
    /// Wraps the live store from the renderer runtime.
    pub fn new(store: &'a MaterialPropertyStore) -> Self {
        Self { store }
    }

    /// Shader asset id bound to `material_id` via material-side `set_shader`.
    pub fn shader_asset_for_material(&self, material_id: i32) -> Option<i32> {
        self.store.shader_asset_for_material(material_id)
    }

    /// Property block overrides material when both define the same `property_id`.
    pub fn get_merged(
        &self,
        ids: MaterialPropertyLookupIds,
        property_id: i32,
    ) -> Option<&'a MaterialPropertyValue> {
        self.store.get_merged(ids, property_id)
    }
}

/// Stores material and property-block maps from IPC batches (separate key spaces).
#[derive(Debug, Default)]
pub struct MaterialPropertyStore {
    pub(super) material_properties: HashMap<i32, HashMap<i32, MaterialPropertyValue>>,
    pub(super) property_block_properties: HashMap<i32, HashMap<i32, MaterialPropertyValue>>,
    shader_asset_by_material: HashMap<i32, i32>,
}

impl MaterialPropertyStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            material_properties: HashMap::new(),
            property_block_properties: HashMap::new(),
            shader_asset_by_material: HashMap::new(),
        }
    }

    /// Sets a property on a host **material** asset.
    pub fn set_material(
        &mut self,
        material_id: i32,
        property_id: i32,
        value: MaterialPropertyValue,
    ) {
        self.material_properties
            .entry(material_id)
            .or_default()
            .insert(property_id, value);
    }

    /// Sets a property on a **property block** asset.
    pub fn set_property_block(
        &mut self,
        block_id: i32,
        property_id: i32,
        value: MaterialPropertyValue,
    ) {
        self.property_block_properties
            .entry(block_id)
            .or_default()
            .insert(property_id, value);
    }

    /// Gets a material-side property.
    pub fn get_material(
        &self,
        material_id: i32,
        property_id: i32,
    ) -> Option<&MaterialPropertyValue> {
        self.material_properties
            .get(&material_id)?
            .get(&property_id)
    }

    /// Gets a property-block-side property.
    pub fn get_property_block(
        &self,
        block_id: i32,
        property_id: i32,
    ) -> Option<&MaterialPropertyValue> {
        self.property_block_properties
            .get(&block_id)?
            .get(&property_id)
    }

    /// Prefer property block, then material (Unity override semantics).
    pub fn get_merged(
        &self,
        ids: MaterialPropertyLookupIds,
        property_id: i32,
    ) -> Option<&MaterialPropertyValue> {
        if let Some(pb) = ids.mesh_property_block_slot0 {
            if let Some(v) = self.get_property_block(pb, property_id) {
                return Some(v);
            }
        }
        self.get_material(ids.material_asset_id, property_id)
    }

    /// Records `set_shader` for a material (`property_id` on wire is the shader asset id).
    pub fn set_shader_asset_for_material(&mut self, material_id: i32, shader_asset_id: i32) {
        self.shader_asset_by_material
            .insert(material_id, shader_asset_id);
    }

    /// Shader asset id from the last material-side `set_shader`.
    pub fn shader_asset_for_material(&self, material_id: i32) -> Option<i32> {
        self.shader_asset_by_material.get(&material_id).copied()
    }

    /// Iterates `(material_asset_id, shader_asset_id)` for diagnostics.
    pub fn iter_material_shader_bindings(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.shader_asset_by_material
            .iter()
            .map(|(&mid, &sid)| (mid, sid))
    }

    /// Count of host materials with at least one stored property map entry.
    pub fn material_property_slot_count(&self) -> usize {
        self.material_properties.len()
    }

    /// Count of host property block assets with stored properties.
    pub fn property_block_slot_count(&self) -> usize {
        self.property_block_properties.len()
    }

    /// Count of `set_shader` bindings (`material_id` → shader asset).
    pub fn material_shader_binding_count(&self) -> usize {
        self.shader_asset_by_material.len()
    }

    /// Removes all state for a material (`UnloadMaterial`).
    pub fn remove_material(&mut self, material_id: i32) {
        self.material_properties.remove(&material_id);
        self.shader_asset_by_material.remove(&material_id);
    }

    /// Removes a property block (`UnloadMaterialPropertyBlock`).
    pub fn remove_property_block(&mut self, block_id: i32) {
        self.property_block_properties.remove(&block_id);
    }
}

#[cfg(test)]
mod material_dictionary_tests {
    use super::{MaterialDictionary, MaterialPropertyStore};

    #[test]
    fn material_dictionary_delegates_shader_binding() {
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        let d = MaterialDictionary::new(&store);
        assert_eq!(d.shader_asset_for_material(7), Some(99));
    }
}
