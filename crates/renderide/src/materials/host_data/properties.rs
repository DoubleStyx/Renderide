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

    /// Per-material mutation generation (see [`MaterialPropertyStore::material_generation`]).
    pub fn material_generation(&self, material_id: i32) -> u64 {
        self.store.material_generation(material_id)
    }

    /// Per-property-block mutation generation (see [`MaterialPropertyStore::property_block_generation`]).
    pub fn property_block_generation(&self, block_id: i32) -> u64 {
        self.store.property_block_generation(block_id)
    }

    /// Returns the two inner property maps (material-side and property-block-side) for one
    /// [`MaterialPropertyLookupIds`] in a single outer-map probe.
    ///
    /// Callers that iterate many property ids against the same lookup -- e.g.
    /// [`crate::materials::material_render_state_for_lookup`] issuing ~30 `get_merged` calls per
    /// material -- use this to hoist the two outer probes out of the inner loop, reducing per-id
    /// cost to a single inner-map lookup on each side.
    pub fn fetch_property_maps(&self, ids: MaterialPropertyLookupIds) -> PropertyMapPair<'a> {
        let mat = self.store.material_properties.get(&ids.material_asset_id);
        let pb = ids
            .mesh_property_block_slot0
            .and_then(|b| self.store.property_block_properties.get(&b));
        (mat, pb)
    }
}

/// Pair of inner `property_id -> value` maps (material-side, property-block-side) returned by
/// [`MaterialDictionary::fetch_property_maps`]. Either side may be `None` when no properties have
/// been stored for the referenced id.
pub type PropertyMapPair<'a> = (
    Option<&'a HashMap<i32, MaterialPropertyValue>>,
    Option<&'a HashMap<i32, MaterialPropertyValue>>,
);

fn bump_generation(map: &mut HashMap<i32, u64>, key: i32) {
    let g = map.entry(key).or_insert(0);
    *g = g.wrapping_add(1);
}

/// Stores material and property-block maps from IPC batches (separate key spaces).
#[derive(Debug, Default)]
pub struct MaterialPropertyStore {
    material_properties: HashMap<i32, HashMap<i32, MaterialPropertyValue>>,
    property_block_properties: HashMap<i32, HashMap<i32, MaterialPropertyValue>>,
    shader_asset_by_material: HashMap<i32, i32>,
    /// Bumped on any mutation affecting [`Self::get_merged`] for that material id (embedded bind skips).
    material_mutation_generation: HashMap<i32, u64>,
    /// Bumped on any mutation affecting [`Self::get_merged`] for that property block id.
    property_block_mutation_generation: HashMap<i32, u64>,
}

impl MaterialPropertyStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            material_properties: HashMap::new(),
            property_block_properties: HashMap::new(),
            shader_asset_by_material: HashMap::new(),
            material_mutation_generation: HashMap::new(),
            property_block_mutation_generation: HashMap::new(),
        }
    }

    /// Monotonic generation for `material_id` and optional property block, used to skip redundant GPU uniform uploads.
    pub fn mutation_generation(&self, ids: MaterialPropertyLookupIds) -> u64 {
        let m = self.material_generation(ids.material_asset_id);
        let pb = ids
            .mesh_property_block_slot0
            .map_or(0, |b| self.property_block_generation(b));
        m ^ pb.rotate_left(17)
    }

    /// Monotonic per-material generation. Bumped by every `set_material` /
    /// `set_shader_asset_for_material`. Unlike [`Self::mutation_generation`], the material and
    /// property-block generations are exposed separately so persistent caches can store both and
    /// avoid hash-collision ambiguity between pairs with the same XOR-rotated combination.
    ///
    /// Not decremented on [`Self::remove_material`] -- the counter stays in place so that a later
    /// `set_material` bumps from the old value, preserving monotonicity across unload/reload
    /// cycles. Callers who cache a snapshot of this value can then safely compare it back to the
    /// current value to detect any intervening mutation.
    pub fn material_generation(&self, material_id: i32) -> u64 {
        self.material_mutation_generation
            .get(&material_id)
            .copied()
            .unwrap_or(0)
    }

    /// Monotonic per-property-block generation. Same invariants as [`Self::material_generation`].
    pub fn property_block_generation(&self, block_id: i32) -> u64 {
        self.property_block_mutation_generation
            .get(&block_id)
            .copied()
            .unwrap_or(0)
    }

    fn bump_material_generation(&mut self, material_id: i32) {
        bump_generation(&mut self.material_mutation_generation, material_id);
    }

    fn bump_property_block_generation(&mut self, block_id: i32) {
        bump_generation(&mut self.property_block_mutation_generation, block_id);
    }

    /// Sets a property on a host **material** asset.
    pub fn set_material(
        &mut self,
        material_id: i32,
        property_id: i32,
        value: MaterialPropertyValue,
    ) {
        self.bump_material_generation(material_id);
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
        self.bump_property_block_generation(block_id);
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
        if let Some(pb) = ids.mesh_property_block_slot0
            && let Some(v) = self.get_property_block(pb, property_id)
        {
            return Some(v);
        }
        self.get_material(ids.material_asset_id, property_id)
    }

    /// Records `set_shader` for a material (`property_id` on wire is the shader asset id).
    pub fn set_shader_asset_for_material(&mut self, material_id: i32, shader_asset_id: i32) {
        self.bump_material_generation(material_id);
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

    /// Count of `set_shader` bindings (`material_id` -> shader asset).
    pub fn material_shader_binding_count(&self) -> usize {
        self.shader_asset_by_material.len()
    }

    /// Removes all state for a material (`UnloadMaterial`).
    ///
    /// Intentionally retains the `material_mutation_generation` entry and bumps it so any cached
    /// resolved-material entry keyed on this id gets invalidated on the next check. Preserving the
    /// counter across unload is what lets persistent caches compare generation snapshots safely
    /// after a later `set_material` rebinds the same id.
    pub fn remove_material(&mut self, material_id: i32) {
        self.material_properties.remove(&material_id);
        self.shader_asset_by_material.remove(&material_id);
        self.bump_material_generation(material_id);
    }

    /// Removes a property block (`UnloadMaterialPropertyBlock`).
    ///
    /// Retains and bumps `property_block_mutation_generation` for the same reason as
    /// [`Self::remove_material`].
    pub fn remove_property_block(&mut self, block_id: i32) {
        self.property_block_properties.remove(&block_id);
        self.bump_property_block_generation(block_id);
    }
}

#[cfg(test)]
mod material_dictionary_tests {
    use super::{
        MaterialDictionary, MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
    };

    #[test]
    fn material_dictionary_delegates_shader_binding() {
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        let d = MaterialDictionary::new(&store);
        assert_eq!(d.shader_asset_for_material(7), Some(99));
    }

    #[test]
    fn get_merged_prefers_property_block_over_material() {
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, 42, MaterialPropertyValue::Float(0.25));
        store.set_property_block(5, 42, MaterialPropertyValue::Float(0.75));
        let ids = MaterialPropertyLookupIds {
            material_asset_id: 1,
            mesh_property_block_slot0: Some(5),
        };
        assert_eq!(
            store.get_merged(ids, 42),
            Some(&MaterialPropertyValue::Float(0.75))
        );
    }

    #[test]
    fn get_merged_falls_through_missing_block_key_to_material() {
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, 42, MaterialPropertyValue::Float(0.25));
        let ids = MaterialPropertyLookupIds {
            material_asset_id: 1,
            mesh_property_block_slot0: Some(5),
        };
        assert_eq!(
            store.get_merged(ids, 42),
            Some(&MaterialPropertyValue::Float(0.25))
        );
        assert_eq!(store.get_merged(ids, 999), None);
    }

    #[test]
    fn set_get_material_and_property_block_are_independent() {
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, 10, MaterialPropertyValue::Float(1.0));
        store.set_property_block(1, 10, MaterialPropertyValue::Float(2.0));
        assert_eq!(
            store.get_material(1, 10),
            Some(&MaterialPropertyValue::Float(1.0))
        );
        assert_eq!(
            store.get_property_block(1, 10),
            Some(&MaterialPropertyValue::Float(2.0))
        );
    }

    #[test]
    fn mutation_generation_bumps_on_writes_only() {
        let mut store = MaterialPropertyStore::new();
        let ids = MaterialPropertyLookupIds {
            material_asset_id: 3,
            mesh_property_block_slot0: None,
        };
        let g0 = store.mutation_generation(ids);
        store.set_material(3, 7, MaterialPropertyValue::Float(1.0));
        let g1 = store.mutation_generation(ids);
        assert_ne!(g0, g1);
        // Pure reads do not bump.
        let _ = store.get_material(3, 7);
        assert_eq!(store.mutation_generation(ids), g1);
        store.set_shader_asset_for_material(3, 99);
        assert_ne!(store.mutation_generation(ids), g1);
    }

    #[test]
    fn remove_material_clears_properties_and_shader_binding() {
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, 10, MaterialPropertyValue::Float(1.0));
        store.set_shader_asset_for_material(1, 100);
        store.remove_material(1);
        assert_eq!(store.get_material(1, 10), None);
        assert_eq!(store.shader_asset_for_material(1), None);
        assert_eq!(store.material_property_slot_count(), 0);
    }

    #[test]
    fn remove_property_block_clears_its_properties() {
        let mut store = MaterialPropertyStore::new();
        store.set_property_block(4, 10, MaterialPropertyValue::Float(1.0));
        assert_eq!(store.property_block_slot_count(), 1);
        store.remove_property_block(4);
        assert_eq!(store.get_property_block(4, 10), None);
        assert_eq!(store.property_block_slot_count(), 0);
    }

    #[test]
    fn material_generation_stays_monotonic_across_remove_and_recreate() {
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, 10, MaterialPropertyValue::Float(1.0));
        let g_after_set = store.material_generation(1);
        store.remove_material(1);
        let g_after_remove = store.material_generation(1);
        // remove_material must bump (invalidate cached resolves), not reset the counter.
        assert!(g_after_remove > g_after_set);
        store.set_material(1, 10, MaterialPropertyValue::Float(2.0));
        let g_after_reset = store.material_generation(1);
        assert!(g_after_reset > g_after_remove);
    }

    #[test]
    fn property_block_generation_stays_monotonic_across_remove_and_recreate() {
        let mut store = MaterialPropertyStore::new();
        store.set_property_block(4, 10, MaterialPropertyValue::Float(1.0));
        let g0 = store.property_block_generation(4);
        store.remove_property_block(4);
        let g1 = store.property_block_generation(4);
        assert!(g1 > g0);
        store.set_property_block(4, 10, MaterialPropertyValue::Float(2.0));
        let g2 = store.property_block_generation(4);
        assert!(g2 > g1);
    }
}
