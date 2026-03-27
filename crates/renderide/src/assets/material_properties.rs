//! Material property store for stencil and property block lookup.
//!
//! Stores values from MaterialsUpdateBatch so stencil state (comparison, operation,
//! reference, clip rect) can be read per override block when building draw entries.
//!
//! ## Parity vs FrooxEngine / Renderite `MaterialUpdateWriter`
//!
//! The host sends `MaterialsUpdateBatch` opcode streams (side buffers: ints, floats, float4s, matrices).
//! Renderide’s parser is
//! [`crate::assets::material_update_batch::parse_materials_update_batch_into_store`].
//!
//! Unity’s `MaterialAssetManager` keeps **two** registries: host `Material` assets and
//! `MaterialPropertyBlock` assets, keyed separately. The batch’s `material_update_count` tells the
//! renderer how many `SelectTarget` headers refer to materials vs property blocks; this store mirrors
//! that split into separate material-side and property-block-side maps in this struct.
//!
//! | Opcode | Cursor | Persisted to store (default) | Persisted when `material_batch_persist_extended_payloads` |
//! |--------|--------|------------------------------|-----------------------------------------------------------|
//! | `set_float` / `set_float4` / `set_texture` | yes | yes | yes |
//! | `set_float4x4` | yes | no (matrix discarded) | yes → [`MaterialPropertyValue::Float4x4`] |
//! | `set_float_array` | yes | no | yes → [`MaterialPropertyValue::FloatArray`] (capped) |
//! | `set_float4_array` | yes | no | yes → [`MaterialPropertyValue::Float4Array`] (capped) |
//!
//! Optional wire counters (no persistence): [`crate::assets::material_batch_wire_metrics`].
//!
//! ## Generic PBR WGSL vs Unity Standard
//!
//! Stock forward PBR shaders read host `_Color` / `_Metallic` / `_Glossiness` and optional `_MainTex`
//! when enabled in [`crate::config::RenderConfig`]. They do **not** yet implement full Standard maps
//! (`_BumpMap`, `_OcclusionMap`, `_EmissionMap`, detail masks, etc.).
//!
//! **Skinned** meshes use separate uniform paths; host `_MainTex` / [`crate::gpu::PipelineVariant::PbrHostAlbedo`]
//! parity with non-skinned draws is not guaranteed—verify skinned layout when extending PBR host bindings.
//!
//! ## Buffer layout assumptions (MaterialsUpdateBatch)
//!
//! Each buffer in `material_updates` contains a sequence of (MaterialPropertyUpdate, value) pairs:
//! - **MaterialPropertyUpdate**: 8 bytes (property_id: i32, update_type: u8, padding: [u8;3])
//! - **Value** (based on update_type):
//!   - `select_target`: i32 (4 bytes) — block_id for subsequent updates
//!   - `set_float`: f32 (4 bytes)
//!   - `set_float4`: [f32; 4] (16 bytes)
//!   - `set_float4x4`: 64 bytes — column-major `mat4` floats — see [`MaterialPropertyValue::Float4x4`]
//!   - `set_shader`: i32 shader asset id (4 bytes) — see [`MaterialPropertyStore::set_shader_asset_for_material`]
//!   - `set_texture`: i32 packed texture reference (4 bytes) — see [`MaterialPropertyValue::Texture`]
//!   - `set_render_queue`, `set_instancing`, `set_render_type`: i32 each (4 bytes) — consumed, not stored
//!   - `update_batch_end`: 0 bytes
//!   - Array opcodes: length from int buffer then payload — see parser
//!
//! Bounds checks: we stop parsing when remaining bytes are insufficient for the next record.
//!
//! ## Block id vs drawable material handle (native UI routing)
//!
//! [`MaterialPropertyStore::shader_asset_for_material`] and material-side property lookups use the
//! **material asset id** from each batch’s `select_target` in the material section, before `set_shader` /
//! `set_texture` / etc. Drawables resolve that id from the active material asset id (after multi-submesh
//! fan-out, the submesh’s slot). If the host sends material updates under a **different** `select_target`
//! than that material id, the store will not find the shader or textures for native UI routing; fixing
//! that is a host / scene contract issue, not something the renderer can infer safely.

use std::collections::HashMap;

/// Maximum `set_float_array` / `set_float4_array` elements stored when extended persistence is on.
pub const MATERIAL_BATCH_MAX_FLOAT_ARRAY_LEN: usize = 256;
/// Maximum `set_float4_array` vec4 elements stored when extended persistence is on.
pub const MATERIAL_BATCH_MAX_FLOAT4_ARRAY_LEN: usize = 64;

/// Single property value. Supports f32 and [f32; 4] for stencil (comparison, operation,
/// reference, clip rect). Extensible for other types.
#[derive(Clone, Debug, PartialEq)]
pub enum MaterialPropertyValue {
    /// Single float (e.g. reference, blend factor).
    Float(f32),
    /// Four floats (e.g. clip rect x, y, width, height).
    Float4([f32; 4]),
    /// Column-major 4×4 matrix from `set_float4x4` (64 bytes on the wire).
    Float4x4([f32; 16]),
    /// `set_float_array` payload after the length prefix (capped).
    FloatArray(Vec<f32>),
    /// `set_float4_array` payload after the length prefix (capped).
    Float4Array(Vec<[f32; 4]>),
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

/// Store of material property values per host material asset and per `MaterialPropertyBlock` asset.
///
/// Material override / stencil paths use **material** asset ids in [`Self::material_properties`].
/// Keys are **not** merged across the two namespaces: the batch parser uses `material_update_count`
/// to route `SelectTarget` ids into the material map vs the property-block map, matching Unity’s
/// separate registries.
pub struct MaterialPropertyStore {
    /// Material asset id → (property_id → value).
    pub(crate) material_properties: HashMap<i32, HashMap<i32, MaterialPropertyValue>>,
    /// `MaterialPropertyBlock` asset id → (property_id → value).
    pub(crate) property_block_properties: HashMap<i32, HashMap<i32, MaterialPropertyValue>>,
    /// Material asset id → shader asset id from material-side `set_shader` only.
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

    /// Sets a property on a **`MaterialPropertyBlock`** asset.
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

    /// Gets a property on a material asset.
    pub fn get_material(
        &self,
        material_id: i32,
        property_id: i32,
    ) -> Option<&MaterialPropertyValue> {
        self.material_properties
            .get(&material_id)?
            .get(&property_id)
    }

    /// Gets a property on a `MaterialPropertyBlock` asset.
    pub fn get_property_block(
        &self,
        block_id: i32,
        property_id: i32,
    ) -> Option<&MaterialPropertyValue> {
        self.property_block_properties
            .get(&block_id)?
            .get(&property_id)
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
            && let Some(v) = self.get_property_block(pb, property_id)
        {
            return Some(v);
        }
        self.get_material(ids.material_asset_id, property_id)
    }

    /// Records the shader asset bound to a **material** asset (`set_shader` is invalid on property blocks).
    pub fn set_shader_asset_for_material(&mut self, material_id: i32, shader_asset_id: i32) {
        self.shader_asset_by_material
            .insert(material_id, shader_asset_id);
    }

    /// Shader asset id for `material_id` when the host sent `set_shader` for that material.
    pub fn shader_asset_for_material(&self, material_id: i32) -> Option<i32> {
        self.shader_asset_by_material.get(&material_id).copied()
    }

    /// Iterates `(material_asset_id, shader_asset_id)` for every material that received `set_shader`.
    ///
    /// Used for diagnostics such as [`crate::assets::log_world_unlit_material_inventory_if_enabled`].
    pub fn iter_material_shader_bindings(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.shader_asset_by_material
            .iter()
            .map(|(&material_id, &shader_id)| (material_id, shader_id))
    }

    /// Removes all properties and shader binding for a material asset. Used on `UnloadMaterial` IPC.
    pub fn remove_material(&mut self, material_id: i32) {
        self.material_properties.remove(&material_id);
        self.shader_asset_by_material.remove(&material_id);
    }

    /// Removes all properties for a `MaterialPropertyBlock` asset. Used on [`crate::shared::UnloadMaterialPropertyBlock`].
    pub fn remove_property_block(&mut self, block_id: i32) {
        self.property_block_properties.remove(&block_id);
    }
}

impl Default for MaterialPropertyStore {
    fn default() -> Self {
        Self::new()
    }
}
