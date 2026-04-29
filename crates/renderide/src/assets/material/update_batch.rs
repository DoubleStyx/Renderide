//! Parses [`crate::shared::MaterialsUpdateBatch`] into [`super::properties::MaterialPropertyStore`].
//!
//! Layout matches FrooxEngine `MaterialUpdateWriter` and Renderite `MaterialUpdateReader`: opcode
//! stream in `material_updates` buffers; typed side buffers supply payloads in global order.

use bytemuck::{Pod, Zeroable};

use super::properties::{
    MaterialPropertyStore, MaterialPropertyValue, MATERIAL_BATCH_MAX_FLOAT4_ARRAY_LEN,
    MATERIAL_BATCH_MAX_FLOAT_ARRAY_LEN,
};
use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::packing::default_entity_pool::DefaultEntityPool;
use crate::shared::packing::memory_packable::MemoryPackable;
use crate::shared::packing::memory_unpacker::MemoryUnpacker;
use crate::shared::{
    MaterialPropertyUpdate, MaterialPropertyUpdateType, MaterialsUpdateBatch,
    MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES,
};

/// Options for [`parse_materials_update_batch_into_store`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ParseMaterialBatchOptions {
    /// When true, persist `set_float4x4` and capped float / float4 arrays into the store.
    pub persist_extended_payloads: bool,
    /// Reserved for future wire-telemetry (matrix / array opcodes).
    pub record_wire_metrics: bool,
    /// Interned `_RenderType` property id. When `Some`, [`MaterialPropertyUpdateType::SetRenderType`]
    /// opcodes write the [`crate::shared::MaterialRenderType`] discriminant (`0` Opaque,
    /// `1` TransparentCutout, `2` Transparent — matches the host's `MaterialRenderType` enum)
    /// as a synthetic [`MaterialPropertyValue::Float`] at this id. The keyword inference path
    /// in [`crate::backend::embedded::uniform_pack`] reads it to populate `_ALPHATEST_ON` /
    /// `_ALPHACLIP` / `_ALPHABLEND_ON` / `_ALPHAPREMULTIPLY_ON` per Unity blend mode semantics.
    /// `None` skips the capture (default for unit tests that do not exercise render-type-driven
    /// inference).
    pub render_type_property_id: Option<i32>,
    /// Interned `_RenderQueue` property id. When `Some`,
    /// [`MaterialPropertyUpdateType::SetRenderQueue`] opcodes write the queue value (Unity
    /// convention: `[1000, 2450)` opaque, `[2450, 3000)` alpha-test, `[3000, ∞)` transparent)
    /// as a synthetic [`MaterialPropertyValue::Float`] at this id. PBS material providers
    /// (`PBS_DualSidedMaterial.cs`, `PBS_DisplaceMaterial.cs`, …) bypass `MaterialProvider.SetBlendMode`
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

/// Host material vs property-block target for one `select_target` row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaterialBatchTarget {
    Material(i32),
    PropertyBlock(i32),
}

/// Routes a parsed [`MaterialPropertyValue`] to the active material or property block.
fn set_property_on_batch_target(
    store: &mut MaterialPropertyStore,
    target: MaterialBatchTarget,
    property_id: i32,
    value: MaterialPropertyValue,
) {
    match target {
        MaterialBatchTarget::Material(id) => store.set_material(id, property_id, value),
        MaterialBatchTarget::PropertyBlock(id) => store.set_property_block(id, property_id, value),
    }
}

fn select_target_kind(
    property_id: i32,
    select_target_index: &mut usize,
    material_update_count: usize,
) -> MaterialBatchTarget {
    let is_material = *select_target_index < material_update_count;
    *select_target_index += 1;
    if is_material {
        MaterialBatchTarget::Material(property_id)
    } else {
        MaterialBatchTarget::PropertyBlock(property_id)
    }
}

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

/// Applies one material/property-block opcode after [`MaterialBatchTarget`] is active (excludes target switching).
///
/// Returns `true` when the opcode represents an **instance-level** change to the active target,
/// matching Renderite Unity `MaterialAssetManager.HandleMaterialUpdate` /
/// `HandlePropertyBlockUpdate` semantics:
/// - **Property block** ops always return `true` (per the Unity comment: "we always trigger
///   instance changed, because just changing the values doesn't seem to notify any of the mesh
///   renderers of this change"). Without this signal, the host's `MaterialAssetUpdated(false)`
///   path skips `AssetCreated()` / `Reinitialize()` and never re-emits the property block to
///   renderers — the root cause of intermittent text-quad rendering.
/// - **Material** ops return `true` only for structural ops that stick to the material instance:
///   `SetShader`, `SetInstancing`, `SetRenderQueue`, `SetRenderType`. Per-property writes
///   (`SetFloat`, `SetFloat4`, `SetFloat4x4`, `SetTexture`, array variants) return `false`.
fn apply_material_batch_property_opcode<L: MaterialBatchBlobLoader + ?Sized>(
    p: &mut BatchParser<'_, L>,
    store: &mut MaterialPropertyStore,
    target: MaterialBatchTarget,
    property_id: i32,
    ty: MaterialPropertyUpdateType,
    options: &ParseMaterialBatchOptions,
) -> bool {
    let is_property_block = matches!(target, MaterialBatchTarget::PropertyBlock(_));
    match ty {
        MaterialPropertyUpdateType::SelectTarget | MaterialPropertyUpdateType::UpdateBatchEnd => {
            false
        }
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
        MaterialPropertyUpdateType::SetFloat => {
            if let Some(v) = p.next_float() {
                set_property_on_batch_target(
                    store,
                    target,
                    property_id,
                    MaterialPropertyValue::Float(v),
                );
            }
            is_property_block
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
            is_property_block
        }
        MaterialPropertyUpdateType::SetFloat4x4 => {
            if let Some(mat) = p.next_matrix() {
                if options.persist_extended_payloads {
                    set_property_on_batch_target(
                        store,
                        target,
                        property_id,
                        MaterialPropertyValue::Float4x4(mat),
                    );
                }
            }
            is_property_block
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
            is_property_block
        }
        MaterialPropertyUpdateType::SetFloatArray => {
            apply_set_float_array_from_batch(p, store, target, property_id, options);
            is_property_block
        }
        MaterialPropertyUpdateType::SetFloat4Array => {
            apply_set_float4_array_from_batch(p, store, target, property_id, options);
            is_property_block
        }
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
/// `SelectTarget` ops in the batch, extra targets are silently dropped — the parser still
/// processes the payload so cursors stay aligned.
///
/// Per-target initial value:
/// - **Material**: `false` — Unity does not call `EnsureInstance` on materials, only OR's per-op
///   results.
/// - **Property block**: `true` — mirrors the effect of Unity's
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
    let mut p = BatchParser {
        loader,
        updates: ChainCursor::new(&batch.material_updates),
        ints: ChainCursor::new(&batch.int_buffers),
        floats: ChainCursor::new(&batch.float_buffers),
        float4s: ChainCursor::new(&batch.float4_buffers),
        matrices: ChainCursor::new(&batch.matrix_buffers),
    };

    let material_update_count = batch.material_update_count.max(0) as usize;
    let mut select_target_index: usize = 0;
    let mut current: Option<MaterialBatchTarget> = None;
    // Index into `instance_changed_out` for the active target. Lags `select_target_index` by one
    // because `select_target_index` is incremented by `select_target_kind` *before* we've finished
    // accumulating bits for the previous target.
    let mut active_bit_index: Option<usize> = None;

    fn begin_target_bit(
        target: MaterialBatchTarget,
        bit_index: usize,
        instance_changed_out: &mut [bool],
    ) {
        if let Some(slot) = instance_changed_out.get_mut(bit_index) {
            *slot = matches!(target, MaterialBatchTarget::PropertyBlock(_));
        }
    }

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
                if instance_changed {
                    if let Some(bit_index) = active_bit_index {
                        if let Some(slot) = instance_changed_out.get_mut(bit_index) {
                            *slot = true;
                        }
                    }
                }
            }
        }
    }
}

struct ChainCursor<'a> {
    descriptors: &'a [SharedMemoryBufferDescriptor],
    descriptor_index: usize,
    data: Vec<u8>,
    offset: usize,
}

impl<'a> ChainCursor<'a> {
    fn new(descriptors: &'a [SharedMemoryBufferDescriptor]) -> Self {
        Self {
            descriptors,
            descriptor_index: 0,
            data: Vec::new(),
            offset: 0,
        }
    }

    fn advance<L: MaterialBatchBlobLoader + ?Sized>(&mut self, loader: &mut L) -> bool {
        profiling::scope!("material::batch_blob_advance");
        while self.descriptor_index < self.descriptors.len() {
            let desc = &self.descriptors[self.descriptor_index];
            self.descriptor_index += 1;
            if desc.length <= 0 {
                continue;
            }
            if let Some(bytes) = loader.load_blob(desc) {
                self.data = bytes;
                self.offset = 0;
                return !self.data.is_empty();
            }
        }
        self.data.clear();
        self.offset = 0;
        false
    }

    fn ensure_capacity<L: MaterialBatchBlobLoader + ?Sized>(
        &mut self,
        loader: &mut L,
        elem_size: usize,
    ) -> bool {
        loop {
            if self.offset + elem_size <= self.data.len() {
                return true;
            }
            if !self.advance(loader) {
                return false;
            }
        }
    }

    fn next<T: Pod + Zeroable, L: MaterialBatchBlobLoader + ?Sized>(
        &mut self,
        loader: &mut L,
    ) -> Option<T> {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Some(T::zeroed());
        }
        if !self.ensure_capacity(loader, elem_size) {
            return None;
        }
        let slice = &self.data[self.offset..self.offset + elem_size];
        let v = bytemuck::pod_read_unaligned(slice);
        self.offset += elem_size;
        Some(v)
    }

    fn next_packable<T: MemoryPackable + Default, L: MaterialBatchBlobLoader + ?Sized>(
        &mut self,
        loader: &mut L,
        host_row_bytes: usize,
    ) -> Option<T> {
        if host_row_bytes == 0 {
            return Some(T::default());
        }
        if !self.ensure_capacity(loader, host_row_bytes) {
            return None;
        }
        let slice = &self.data[self.offset..self.offset + host_row_bytes];
        let mut pool = DefaultEntityPool;
        let mut unpacker = MemoryUnpacker::new(slice, &mut pool);
        let mut out = T::default();
        if out.unpack(&mut unpacker).is_err() {
            return None;
        }
        self.offset += host_row_bytes;
        Some(out)
    }
}

struct BatchParser<'a, L: MaterialBatchBlobLoader + ?Sized> {
    loader: &'a mut L,
    updates: ChainCursor<'a>,
    ints: ChainCursor<'a>,
    floats: ChainCursor<'a>,
    float4s: ChainCursor<'a>,
    matrices: ChainCursor<'a>,
}

impl<L: MaterialBatchBlobLoader + ?Sized> BatchParser<'_, L> {
    fn next_update(&mut self) -> Option<MaterialPropertyUpdate> {
        self.updates
            .next_packable(self.loader, MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES)
    }

    fn next_int(&mut self) -> Option<i32> {
        self.ints.next(self.loader)
    }

    fn next_float(&mut self) -> Option<f32> {
        self.floats.next(self.loader)
    }

    fn next_float4(&mut self) -> Option<[f32; 4]> {
        self.float4s.next(self.loader)
    }

    fn next_matrix(&mut self) -> Option<[f32; 16]> {
        self.matrices.next(self.loader)
    }
}

#[cfg(test)]
mod tests;
