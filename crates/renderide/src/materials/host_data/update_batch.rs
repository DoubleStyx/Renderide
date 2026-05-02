//! Parses [`crate::shared::MaterialsUpdateBatch`] into [`super::properties::MaterialPropertyStore`].
//!
//! Layout matches FrooxEngine `MaterialUpdateWriter` and Renderite `MaterialUpdateReader`: opcode
//! stream in `material_updates` buffers; typed side buffers supply payloads in global order.

mod cursor;
mod dispatch;
mod readers;
mod wire;

#[cfg(test)]
mod tests;

use super::properties::MaterialPropertyStore;
use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::{MaterialPropertyUpdateType, MaterialsUpdateBatch};

#[cfg(test)]
use {
    super::properties::MaterialPropertyValue,
    crate::shared::packing::memory_packable::MemoryPackable,
    crate::shared::{MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES, MaterialPropertyUpdate},
};

use dispatch::apply_material_batch_property_opcode;
use readers::BatchParser;
use wire::{MaterialBatchTarget, select_target_kind};

/// Options for [`parse_materials_update_batch_into_store`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ParseMaterialBatchOptions {
    /// When true, persist `set_float4x4` and capped float / float4 arrays into the store.
    pub persist_extended_payloads: bool,
    /// Reserved for future wire-telemetry (matrix / array opcodes).
    pub record_wire_metrics: bool,
    /// Interned `_RenderType` property id. When `Some`, [`MaterialPropertyUpdateType::SetRenderType`]
    /// opcodes write the [`crate::shared::MaterialRenderType`] discriminant (`0` Opaque,
    /// `1` TransparentCutout, `2` Transparent -- matches the host's `MaterialRenderType` enum)
    /// as a synthetic [`super::MaterialPropertyValue::Float`] at this id. The keyword inference path
    /// in [`crate::materials::embedded::uniform_pack`] reads it to populate `_ALPHATEST_ON` /
    /// `_ALPHACLIP` / `_ALPHABLEND_ON` / `_ALPHAPREMULTIPLY_ON` per Unity blend mode semantics.
    /// `None` skips the capture (default for unit tests that do not exercise render-type-driven
    /// inference).
    pub render_type_property_id: Option<i32>,
    /// Interned `_RenderQueue` property id. When `Some`,
    /// [`MaterialPropertyUpdateType::SetRenderQueue`] opcodes write the queue value (Unity
    /// convention: `[1000, 2450)` opaque, `[2450, 3000)` alpha-test, `[3000, inf)` transparent)
    /// as a synthetic [`super::MaterialPropertyValue::Float`] at this id. PBS material providers
    /// (`PBS_DualSidedMaterial.cs`, `PBS_DisplaceMaterial.cs`, ...) bypass `MaterialProvider.SetBlendMode`
    /// entirely and route their `AlphaHandling` enum through this opcode plus the
    /// `_ALPHACLIP` shader keyword. The keyword bitmask is not on the wire, so the queue
    /// range is the only signal the renderer can use to infer alpha-test for those
    /// materials. `None` skips the capture (default for unit tests).
    pub render_queue_property_id: Option<i32>,
}

/// Loads a blob for a [`SharedMemoryBufferDescriptor`] (production: shared-memory mmap).
pub trait MaterialBatchBlobLoader {
    /// Returns a copy of the region described by `descriptor`, or `None` on failure / empty.
    fn load_blob(&mut self, descriptor: &SharedMemoryBufferDescriptor) -> Option<Vec<u8>>;
}

impl MaterialBatchBlobLoader for crate::ipc::SharedMemoryAccessor {
    fn load_blob(&mut self, descriptor: &SharedMemoryBufferDescriptor) -> Option<Vec<u8>> {
        self.access_copy::<u8>(descriptor)
    }
}

/// Applies all material updates in `batch` into `store` using `loader`.
///
/// See [`parse_materials_update_batch_into_store_with_instance_changed`] for the variant that
/// also reports per-target instance-changed bits required by the host's `MaterialAssetUpdated`
/// dispatch.
pub fn parse_materials_update_batch_into_store(
    loader: &mut impl MaterialBatchBlobLoader,
    batch: &MaterialsUpdateBatch,
    store: &mut MaterialPropertyStore,
    options: &ParseMaterialBatchOptions,
) {
    parse_materials_update_batch_into_store_with_instance_changed(
        loader,
        batch,
        store,
        options,
        &mut [],
    );
}

/// Same as [`parse_materials_update_batch_into_store`] but writes per-target instance-changed
/// flags into `instance_changed_out`.
///
/// `instance_changed_out` is indexed by `SelectTarget` order: bit `i` corresponds to the `i`-th
/// `SelectTarget` opcode encountered (materials first, then property blocks, matching Unity
/// `MaterialAssetManager.ApplyUpdate`). When the slice is shorter than the number of
/// `SelectTarget` ops in the batch, extra targets are silently dropped -- the parser still
/// processes the payload so cursors stay aligned.
///
/// Per-target initial value:
/// - **Material**: `false` -- Unity does not call `EnsureInstance` on materials, only OR's per-op
///   results.
/// - **Property block**: `true` -- mirrors the effect of Unity's
///   `MaterialPropertyBlockAsset.EnsureInstance()` plus the comment in
///   `MaterialAssetManager.HandlePropertyBlockUpdate` that says property-block updates always
///   trigger instance-changed. Without this, the host's `MaterialAssetUpdated(false)` path skips
///   the `AssetCreated()` re-emission needed for property blocks (e.g. font atlases) to be
///   re-bound on renderers.
pub fn parse_materials_update_batch_into_store_with_instance_changed(
    loader: &mut impl MaterialBatchBlobLoader,
    batch: &MaterialsUpdateBatch,
    store: &mut MaterialPropertyStore,
    options: &ParseMaterialBatchOptions,
    instance_changed_out: &mut [bool],
) {
    profiling::scope!("material::parse_update_batch");
    let _ = options.record_wire_metrics;
    let mut p = BatchParser::new(loader, batch);

    let material_update_count = batch.material_update_count.max(0) as usize;
    let mut select_target_index: usize = 0;
    let mut current: Option<MaterialBatchTarget> = None;
    // Index into `instance_changed_out` for the active target. Lags `select_target_index` by one
    // because `select_target_index` is incremented by `select_target_kind` *before* we've finished
    // accumulating bits for the previous target.
    let mut active_bit_index: Option<usize> = None;

    while let Some(update) = p.next_update() {
        if update.update_type == MaterialPropertyUpdateType::UpdateBatchEnd {
            break;
        }

        let Some(target) = current else {
            if update.update_type == MaterialPropertyUpdateType::SelectTarget {
                let bit_index = select_target_index;
                let kind = select_target_kind(
                    update.property_id,
                    &mut select_target_index,
                    material_update_count,
                );
                begin_target_bit(kind, bit_index, instance_changed_out);
                active_bit_index = Some(bit_index);
                current = Some(kind);
            }
            continue;
        };

        match update.update_type {
            MaterialPropertyUpdateType::SelectTarget => {
                let bit_index = select_target_index;
                let kind = select_target_kind(
                    update.property_id,
                    &mut select_target_index,
                    material_update_count,
                );
                begin_target_bit(kind, bit_index, instance_changed_out);
                active_bit_index = Some(bit_index);
                current = Some(kind);
            }
            MaterialPropertyUpdateType::UpdateBatchEnd => break,
            other => {
                let instance_changed = apply_material_batch_property_opcode(
                    &mut p,
                    store,
                    target,
                    update.property_id,
                    other,
                    options,
                );
                if instance_changed
                    && let Some(bit_index) = active_bit_index
                    && let Some(slot) = instance_changed_out.get_mut(bit_index)
                {
                    *slot = true;
                }
            }
        }
    }
}

fn begin_target_bit(
    target: MaterialBatchTarget,
    bit_index: usize,
    instance_changed_out: &mut [bool],
) {
    if let Some(slot) = instance_changed_out.get_mut(bit_index) {
        *slot = target.is_property_block();
    }
}
