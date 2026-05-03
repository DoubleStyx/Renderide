//! Per-opcode dispatch for the material batch parser.
//!
//! Each `apply_*` helper handles one category of opcode (scalar, array, structural). The top-level
//! [`apply_material_batch_property_opcode`] only routes by [`MaterialPropertyUpdateType`]; the
//! per-category helpers own all the side-buffer reads and `instance_changed` semantics within
//! their category.

use super::super::super::super::shared::MaterialPropertyUpdateType;
use super::super::properties::{
    MATERIAL_BATCH_MAX_FLOAT_ARRAY_LEN, MATERIAL_BATCH_MAX_FLOAT4_ARRAY_LEN, MaterialPropertyStore,
    MaterialPropertyValue,
};
use super::MaterialBatchBlobLoader;
use super::ParseMaterialBatchOptions;
use super::readers::BatchParser;
use super::wire::{MaterialBatchTarget, set_property_on_batch_target};

/// Reads a length-prefixed `f32` stream from the float side buffer and persists a capped array.
fn apply_set_float_array_from_batch<L: MaterialBatchBlobLoader + ?Sized>(
    p: &mut BatchParser<'_, L>,
    store: &mut MaterialPropertyStore,
    target: MaterialBatchTarget,
    property_id: i32,
    options: &ParseMaterialBatchOptions,
) {
    let Some(len) = p.next_int() else {
        return;
    };
    let len = len.max(0) as usize;
    let mut out: Vec<f32> = Vec::new();
    if options.persist_extended_payloads {
        out.reserve(len.min(MATERIAL_BATCH_MAX_FLOAT_ARRAY_LEN));
    }
    for _ in 0..len {
        let Some(f) = p.next_float() else {
            break;
        };
        if options.persist_extended_payloads && out.len() < MATERIAL_BATCH_MAX_FLOAT_ARRAY_LEN {
            out.push(f);
        }
    }
    if options.persist_extended_payloads && !out.is_empty() {
        set_property_on_batch_target(
            store,
            target,
            property_id,
            MaterialPropertyValue::FloatArray(out),
        );
    }
}

/// Reads a length-prefixed `float4` stream from the float4 side buffer and persists a capped array.
fn apply_set_float4_array_from_batch<L: MaterialBatchBlobLoader + ?Sized>(
    p: &mut BatchParser<'_, L>,
    store: &mut MaterialPropertyStore,
    target: MaterialBatchTarget,
    property_id: i32,
    options: &ParseMaterialBatchOptions,
) {
    let Some(len) = p.next_int() else {
        return;
    };
    let len = len.max(0) as usize;
    let mut out: Vec<[f32; 4]> = Vec::new();
    if options.persist_extended_payloads {
        out.reserve(len.min(MATERIAL_BATCH_MAX_FLOAT4_ARRAY_LEN));
    }
    for _ in 0..len {
        let Some(v) = p.next_float4() else {
            break;
        };
        if options.persist_extended_payloads && out.len() < MATERIAL_BATCH_MAX_FLOAT4_ARRAY_LEN {
            out.push(v);
        }
    }
    if options.persist_extended_payloads && !out.is_empty() {
        set_property_on_batch_target(
            store,
            target,
            property_id,
            MaterialPropertyValue::Float4Array(out),
        );
    }
}

/// Handles `SetFloat`, `SetFloat4`, `SetFloat4x4`, `SetTexture`. Returns `is_property_block` per
/// the contract in [`apply_material_batch_property_opcode`].
fn apply_scalar_opcode<L: MaterialBatchBlobLoader + ?Sized>(
    p: &mut BatchParser<'_, L>,
    store: &mut MaterialPropertyStore,
    target: MaterialBatchTarget,
    property_id: i32,
    ty: MaterialPropertyUpdateType,
    options: &ParseMaterialBatchOptions,
) -> bool {
    match ty {
        MaterialPropertyUpdateType::SetFloat => {
            if let Some(v) = p.next_float() {
                set_property_on_batch_target(
                    store,
                    target,
                    property_id,
                    MaterialPropertyValue::Float(v),
                );
            }
        }
        MaterialPropertyUpdateType::SetFloat4 => {
            if let Some(v) = p.next_float4() {
                set_property_on_batch_target(
                    store,
                    target,
                    property_id,
                    MaterialPropertyValue::Float4(v),
                );
            }
        }
        MaterialPropertyUpdateType::SetFloat4x4 => {
            if let Some(mat) = p.next_matrix()
                && options.persist_extended_payloads
            {
                set_property_on_batch_target(
                    store,
                    target,
                    property_id,
                    MaterialPropertyValue::Float4x4(mat),
                );
            }
        }
        MaterialPropertyUpdateType::SetTexture => {
            if let Some(packed) = p.next_int() {
                set_property_on_batch_target(
                    store,
                    target,
                    property_id,
                    MaterialPropertyValue::Texture(packed),
                );
            }
        }
        _ => {}
    }
    target.is_property_block()
}

/// Handles `SetFloatArray`, `SetFloat4Array`. Returns `is_property_block`.
fn apply_array_opcode<L: MaterialBatchBlobLoader + ?Sized>(
    p: &mut BatchParser<'_, L>,
    store: &mut MaterialPropertyStore,
    target: MaterialBatchTarget,
    property_id: i32,
    ty: MaterialPropertyUpdateType,
    options: &ParseMaterialBatchOptions,
) -> bool {
    match ty {
        MaterialPropertyUpdateType::SetFloatArray => {
            apply_set_float_array_from_batch(p, store, target, property_id, options);
        }
        MaterialPropertyUpdateType::SetFloat4Array => {
            apply_set_float4_array_from_batch(p, store, target, property_id, options);
        }
        _ => {}
    }
    target.is_property_block()
}

/// Handles `SetShader`, `SetInstancing`, `SetRenderQueue`, `SetRenderType`. These return `true` on
/// material targets (structural / instance-level changes) and `false` on property-block targets
/// (which always return `true` from the dispatcher anyway).
fn apply_structural_opcode(
    store: &mut MaterialPropertyStore,
    target: MaterialBatchTarget,
    property_id: i32,
    ty: MaterialPropertyUpdateType,
    options: &ParseMaterialBatchOptions,
) -> bool {
    let is_property_block = target.is_property_block();
    match ty {
        MaterialPropertyUpdateType::SetShader => match target {
            MaterialBatchTarget::Material(material_id) => {
                store.set_shader_asset_for_material(material_id, property_id);
                true
            }
            MaterialBatchTarget::PropertyBlock(_) => false,
        },
        MaterialPropertyUpdateType::SetInstancing => !is_property_block,
        MaterialPropertyUpdateType::SetRenderQueue => {
            if let Some(render_queue_pid) = options.render_queue_property_id {
                set_property_on_batch_target(
                    store,
                    target,
                    render_queue_pid,
                    MaterialPropertyValue::Float(property_id as f32),
                );
            }
            !is_property_block
        }
        MaterialPropertyUpdateType::SetRenderType => {
            if let Some(render_type_pid) = options.render_type_property_id {
                set_property_on_batch_target(
                    store,
                    target,
                    render_type_pid,
                    MaterialPropertyValue::Float(property_id as f32),
                );
            }
            !is_property_block
        }
        _ => false,
    }
}

/// Applies one material/property-block opcode after [`MaterialBatchTarget`] is active (excludes target switching).
///
/// Returns `true` when the opcode represents an **instance-level** change to the active target,
/// matching Renderite Unity `MaterialAssetManager.HandleMaterialUpdate` /
/// `HandlePropertyBlockUpdate` semantics:
/// - **Property block** ops always return `true` (per the Unity comment: "we always trigger
///   instance changed, because just changing the values doesn't seem to notify any of the mesh
///   renderers of this change"). Without this signal, the host's `MaterialAssetUpdated(false)`
///   path skips `AssetCreated()` / `Reinitialize()` and never re-emits the property block to
///   renderers -- the root cause of intermittent text-quad rendering.
/// - **Material** ops return `true` only for structural ops that stick to the material instance:
///   `SetShader`, `SetInstancing`, `SetRenderQueue`, `SetRenderType`. Per-property writes
///   (`SetFloat`, `SetFloat4`, `SetFloat4x4`, `SetTexture`, array variants) return `false`.
pub(super) fn apply_material_batch_property_opcode<L: MaterialBatchBlobLoader + ?Sized>(
    p: &mut BatchParser<'_, L>,
    store: &mut MaterialPropertyStore,
    target: MaterialBatchTarget,
    property_id: i32,
    ty: MaterialPropertyUpdateType,
    options: &ParseMaterialBatchOptions,
) -> bool {
    match ty {
        MaterialPropertyUpdateType::SelectTarget | MaterialPropertyUpdateType::UpdateBatchEnd => {
            false
        }
        MaterialPropertyUpdateType::SetShader
        | MaterialPropertyUpdateType::SetInstancing
        | MaterialPropertyUpdateType::SetRenderQueue
        | MaterialPropertyUpdateType::SetRenderType => {
            apply_structural_opcode(store, target, property_id, ty, options)
        }
        MaterialPropertyUpdateType::SetFloat
        | MaterialPropertyUpdateType::SetFloat4
        | MaterialPropertyUpdateType::SetFloat4x4
        | MaterialPropertyUpdateType::SetTexture => {
            apply_scalar_opcode(p, store, target, property_id, ty, options)
        }
        MaterialPropertyUpdateType::SetFloatArray | MaterialPropertyUpdateType::SetFloat4Array => {
            apply_array_opcode(p, store, target, property_id, ty, options)
        }
    }
}
