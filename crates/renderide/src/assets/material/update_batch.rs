//! Parses [`crate::shared::MaterialsUpdateBatch`] using the same layout as FrooxEngine
//! `MaterialUpdateWriter` and Renderite `MaterialUpdateReader`:
//! `MaterialPropertyUpdate` records live in `material_updates` buffers; payload values live in
//! separate `int_buffers`, `float_buffers`, `float4_buffers`, and `matrix_buffers`, consumed in
//! global order across each list.

use bytemuck::{Pod, Zeroable};

use super::material_batch_wire_metrics::{MaterialBatchWireKind, record_material_batch_wire};
use super::material_properties::{
    MATERIAL_BATCH_MAX_FLOAT_ARRAY_LEN, MATERIAL_BATCH_MAX_FLOAT4_ARRAY_LEN, MaterialPropertyStore,
    MaterialPropertyValue,
};
use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::{MaterialPropertyUpdate, MaterialPropertyUpdateType, MaterialsUpdateBatch};

/// Options for [`parse_materials_update_batch_into_store`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ParseMaterialBatchOptions {
    /// When true, persist `set_float4x4` and bounded float / float4 arrays into [`MaterialPropertyStore`].
    pub persist_extended_payloads: bool,
    /// When true, increment [`crate::assets::material_batch_wire_metrics`] for matrix/array opcodes.
    pub record_wire_metrics: bool,
}

/// Copies the bytes for a material batch descriptor (production: shared-memory mmap).
pub trait MaterialBatchBlobLoader {
    /// Returns a copy of the region described by `descriptor`, or `None` on failure / empty.
    fn load_blob(&mut self, descriptor: &SharedMemoryBufferDescriptor) -> Option<Vec<u8>>;
}

impl MaterialBatchBlobLoader for SharedMemoryAccessor {
    fn load_blob(&mut self, descriptor: &SharedMemoryBufferDescriptor) -> Option<Vec<u8>> {
        self.access_copy::<u8>(descriptor)
    }
}

/// Host material vs `MaterialPropertyBlock` target for one `SelectTarget` header.
///
/// Matches Unity `MaterialAssetManager.ApplyUpdate`: the first `material_update_count` `SelectTarget`
/// rows apply to [`MaterialBatchTarget::Material`], then remaining rows apply to
/// [`MaterialBatchTarget::PropertyBlock`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaterialBatchTarget {
    /// Host material asset id (Unity `Materials.GetAsset`).
    Material(i32),
    /// `MaterialPropertyBlock` asset id (Unity `PropertyBlocks.GetAsset`).
    PropertyBlock(i32),
}

/// Interprets the next `SelectTarget` using `material_update_count` (Unity `MaterialAssetManager` rules).
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

/// Applies all material updates in `batch` into `store` using `loader`.
///
/// Chains every `material_updates` descriptor into one logical update stream (matching Renderite’s
/// reader). Typed side buffers are consumed in order for payloads.
///
/// Uses `batch.material_update_count` to route each `SelectTarget` to either material-side or
/// property-block-side storage, matching FrooxEngine / Unity `MaterialAssetManager`.
pub fn parse_materials_update_batch_into_store(
    loader: &mut impl MaterialBatchBlobLoader,
    batch: &MaterialsUpdateBatch,
    store: &mut MaterialPropertyStore,
    options: &ParseMaterialBatchOptions,
) {
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

    loop {
        let Some(update) = p.next_update() else {
            break;
        };
        if update.update_type == MaterialPropertyUpdateType::update_batch_end {
            break;
        }

        let Some(target) = current else {
            if update.update_type == MaterialPropertyUpdateType::select_target {
                current = Some(select_target_kind(
                    update.property_id,
                    &mut select_target_index,
                    material_update_count,
                ));
            }
            continue;
        };

        match update.update_type {
            MaterialPropertyUpdateType::select_target => {
                current = Some(select_target_kind(
                    update.property_id,
                    &mut select_target_index,
                    material_update_count,
                ));
            }
            MaterialPropertyUpdateType::set_shader => match target {
                MaterialBatchTarget::Material(material_id) => {
                    store.set_shader_asset_for_material(material_id, update.property_id);
                }
                MaterialBatchTarget::PropertyBlock(_) => {
                    // Unity `HandlePropertyBlockUpdate` rejects `set_shader` on property blocks.
                }
            },
            MaterialPropertyUpdateType::set_render_queue
            | MaterialPropertyUpdateType::set_instancing
            | MaterialPropertyUpdateType::set_render_type => {}
            MaterialPropertyUpdateType::set_float => {
                if let Some(v) = p.next_float() {
                    match target {
                        MaterialBatchTarget::Material(id) => {
                            store.set_material(
                                id,
                                update.property_id,
                                MaterialPropertyValue::Float(v),
                            );
                        }
                        MaterialBatchTarget::PropertyBlock(id) => {
                            store.set_property_block(
                                id,
                                update.property_id,
                                MaterialPropertyValue::Float(v),
                            );
                        }
                    }
                }
            }
            MaterialPropertyUpdateType::set_float4 => {
                if let Some(v) = p.next_float4() {
                    match target {
                        MaterialBatchTarget::Material(id) => {
                            store.set_material(
                                id,
                                update.property_id,
                                MaterialPropertyValue::Float4(v),
                            );
                        }
                        MaterialBatchTarget::PropertyBlock(id) => {
                            store.set_property_block(
                                id,
                                update.property_id,
                                MaterialPropertyValue::Float4(v),
                            );
                        }
                    }
                }
            }
            MaterialPropertyUpdateType::set_float4x4 => {
                if options.record_wire_metrics {
                    record_material_batch_wire(true, MaterialBatchWireKind::SetFloat4x4);
                }
                if let Some(mat) = p.next_matrix()
                    && options.persist_extended_payloads
                {
                    let v = MaterialPropertyValue::Float4x4(mat);
                    match target {
                        MaterialBatchTarget::Material(id) => {
                            store.set_material(id, update.property_id, v);
                        }
                        MaterialBatchTarget::PropertyBlock(id) => {
                            store.set_property_block(id, update.property_id, v);
                        }
                    }
                }
            }
            MaterialPropertyUpdateType::set_texture => {
                if let Some(packed) = p.next_int() {
                    let v = MaterialPropertyValue::Texture(packed);
                    match target {
                        MaterialBatchTarget::Material(id) => {
                            store.set_material(id, update.property_id, v);
                        }
                        MaterialBatchTarget::PropertyBlock(id) => {
                            store.set_property_block(id, update.property_id, v);
                        }
                    }
                }
            }
            MaterialPropertyUpdateType::set_float_array => {
                if options.record_wire_metrics {
                    record_material_batch_wire(true, MaterialBatchWireKind::SetFloatArray);
                }
                let Some(len) = p.next_int() else {
                    continue;
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
                    if options.persist_extended_payloads
                        && out.len() < MATERIAL_BATCH_MAX_FLOAT_ARRAY_LEN
                    {
                        out.push(f);
                    }
                }
                if options.persist_extended_payloads && !out.is_empty() {
                    let v = MaterialPropertyValue::FloatArray(out);
                    match target {
                        MaterialBatchTarget::Material(id) => {
                            store.set_material(id, update.property_id, v);
                        }
                        MaterialBatchTarget::PropertyBlock(id) => {
                            store.set_property_block(id, update.property_id, v);
                        }
                    }
                }
            }
            MaterialPropertyUpdateType::set_float4_array => {
                if options.record_wire_metrics {
                    record_material_batch_wire(true, MaterialBatchWireKind::SetFloat4Array);
                }
                let Some(len) = p.next_int() else {
                    continue;
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
                    if options.persist_extended_payloads
                        && out.len() < MATERIAL_BATCH_MAX_FLOAT4_ARRAY_LEN
                    {
                        out.push(v);
                    }
                }
                if options.persist_extended_payloads && !out.is_empty() {
                    let v = MaterialPropertyValue::Float4Array(out);
                    match target {
                        MaterialBatchTarget::Material(id) => {
                            store.set_material(id, update.property_id, v);
                        }
                        MaterialBatchTarget::PropertyBlock(id) => {
                            store.set_property_block(id, update.property_id, v);
                        }
                    }
                }
            }
            MaterialPropertyUpdateType::update_batch_end => break,
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
}

struct BatchParser<'a, L: MaterialBatchBlobLoader + ?Sized> {
    loader: &'a mut L,
    updates: ChainCursor<'a>,
    ints: ChainCursor<'a>,
    floats: ChainCursor<'a>,
    float4s: ChainCursor<'a>,
    matrices: ChainCursor<'a>,
}

impl<'a, L: MaterialBatchBlobLoader + ?Sized> BatchParser<'a, L> {
    fn next_update(&mut self) -> Option<MaterialPropertyUpdate> {
        self.updates.next(self.loader)
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
mod tests {
    use super::*;
    use crate::shared::buffer::SharedMemoryBufferDescriptor;

    /// Test loader: `buffer_id` indexes into `blobs`.
    struct TestLoader {
        blobs: Vec<Vec<u8>>,
    }

    impl MaterialBatchBlobLoader for TestLoader {
        fn load_blob(&mut self, descriptor: &SharedMemoryBufferDescriptor) -> Option<Vec<u8>> {
            let i = descriptor.buffer_id.max(0) as usize;
            self.blobs.get(i).cloned()
        }
    }

    fn desc(blob_idx: i32, bytes: &[u8]) -> SharedMemoryBufferDescriptor {
        SharedMemoryBufferDescriptor {
            buffer_id: blob_idx,
            buffer_capacity: bytes.len() as i32,
            offset: 0,
            length: bytes.len() as i32,
        }
    }

    fn write_update(property_id: i32, ty: MaterialPropertyUpdateType) -> MaterialPropertyUpdate {
        MaterialPropertyUpdate {
            property_id,
            update_type: ty,
            _padding: [0; 3],
        }
    }

    #[test]
    fn select_target_uses_property_id_set_shader_in_property_id() {
        let b0 = bytemuck::bytes_of(&write_update(42, MaterialPropertyUpdateType::select_target))
            .to_vec();
        let b1 =
            bytemuck::bytes_of(&write_update(7, MaterialPropertyUpdateType::set_shader)).to_vec();
        let b2 = bytemuck::bytes_of(&write_update(
            0,
            MaterialPropertyUpdateType::update_batch_end,
        ))
        .to_vec();
        let mut loader = TestLoader {
            blobs: vec![b0.clone(), b1.clone(), b2.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &b0), desc(1, &b1), desc(2, &b2)],
            material_update_count: 1,
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(
            &mut loader,
            &batch,
            &mut store,
            &ParseMaterialBatchOptions::default(),
        );
        assert_eq!(store.shader_asset_for_material(42), Some(7));
    }

    #[test]
    fn set_texture_reads_packed_from_int_buffer() {
        let stream: Vec<u8> =
            bytemuck::bytes_of(&write_update(99, MaterialPropertyUpdateType::select_target))
                .iter()
                .chain(bytemuck::bytes_of(&write_update(
                    1,
                    MaterialPropertyUpdateType::set_texture,
                )))
                .chain(bytemuck::bytes_of(&write_update(
                    0,
                    MaterialPropertyUpdateType::update_batch_end,
                )))
                .copied()
                .collect();
        let packed: i32 = 0x00AB_CD01;
        let int_bytes = bytemuck::bytes_of(&packed).to_vec();

        let mut loader = TestLoader {
            blobs: vec![stream.clone(), int_bytes.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &stream)],
            int_buffers: vec![desc(1, &int_bytes)],
            material_update_count: 1,
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(
            &mut loader,
            &batch,
            &mut store,
            &ParseMaterialBatchOptions::default(),
        );
        assert_eq!(
            store.get_material(99, 1),
            Some(&MaterialPropertyValue::Texture(0x00AB_CD01))
        );
    }

    #[test]
    fn set_float_and_float4_from_typed_buffers() {
        let stream: Vec<u8> =
            bytemuck::bytes_of(&write_update(10, MaterialPropertyUpdateType::select_target))
                .iter()
                .chain(bytemuck::bytes_of(&write_update(
                    2,
                    MaterialPropertyUpdateType::set_float,
                )))
                .chain(bytemuck::bytes_of(&write_update(
                    3,
                    MaterialPropertyUpdateType::set_float4,
                )))
                .chain(bytemuck::bytes_of(&write_update(
                    0,
                    MaterialPropertyUpdateType::update_batch_end,
                )))
                .copied()
                .collect();
        let fv: f32 = 2.5;
        let v4 = [1.0f32, 2.0, 3.0, 4.0];

        let fbytes = bytemuck::bytes_of(&fv).to_vec();
        let v4bytes = bytemuck::cast_slice(&v4).to_vec();
        let mut loader = TestLoader {
            blobs: vec![stream.clone(), fbytes.clone(), v4bytes.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &stream)],
            float_buffers: vec![desc(1, &fbytes)],
            float4_buffers: vec![desc(2, &v4bytes)],
            material_update_count: 1,
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(
            &mut loader,
            &batch,
            &mut store,
            &ParseMaterialBatchOptions::default(),
        );
        assert_eq!(
            store.get_material(10, 2),
            Some(&MaterialPropertyValue::Float(2.5))
        );
        assert_eq!(
            store.get_material(10, 3),
            Some(&MaterialPropertyValue::Float4([1.0, 2.0, 3.0, 4.0]))
        );
    }

    #[test]
    fn chained_material_update_buffers() {
        let b0 = bytemuck::bytes_of(&write_update(5, MaterialPropertyUpdateType::select_target))
            .to_vec();
        let b1 =
            bytemuck::bytes_of(&write_update(9, MaterialPropertyUpdateType::set_shader)).to_vec();
        let mut loader = TestLoader {
            blobs: vec![b0.clone(), b1.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &b0), desc(1, &b1)],
            material_update_count: 1,
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(
            &mut loader,
            &batch,
            &mut store,
            &ParseMaterialBatchOptions::default(),
        );
        assert_eq!(store.shader_asset_for_material(5), Some(9));
    }

    #[test]
    fn set_float4x4_persisted_when_option_on() {
        let stream: Vec<u8> =
            bytemuck::bytes_of(&write_update(20, MaterialPropertyUpdateType::select_target))
                .iter()
                .chain(bytemuck::bytes_of(&write_update(
                    3,
                    MaterialPropertyUpdateType::set_float4x4,
                )))
                .chain(bytemuck::bytes_of(&write_update(
                    0,
                    MaterialPropertyUpdateType::update_batch_end,
                )))
                .copied()
                .collect();
        let mat: [f32; 16] = std::array::from_fn(|i| i as f32 + 1.0);
        let matrix_bytes = bytemuck::cast_slice(&mat).to_vec();
        let mut loader = TestLoader {
            blobs: vec![stream.clone(), matrix_bytes.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &stream)],
            matrix_buffers: vec![desc(1, &matrix_bytes)],
            material_update_count: 1,
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        let opts = ParseMaterialBatchOptions {
            persist_extended_payloads: true,
            ..Default::default()
        };
        parse_materials_update_batch_into_store(&mut loader, &batch, &mut store, &opts);
        assert_eq!(
            store.get_material(20, 3),
            Some(&MaterialPropertyValue::Float4x4(mat))
        );
    }

    #[test]
    fn set_float_array_persisted_when_option_on() {
        let stream: Vec<u8> =
            bytemuck::bytes_of(&write_update(21, MaterialPropertyUpdateType::select_target))
                .iter()
                .chain(bytemuck::bytes_of(&write_update(
                    4,
                    MaterialPropertyUpdateType::set_float_array,
                )))
                .chain(bytemuck::bytes_of(&write_update(
                    0,
                    MaterialPropertyUpdateType::update_batch_end,
                )))
                .copied()
                .collect();
        let len: i32 = 2;
        let f0: f32 = 0.25;
        let f1: f32 = 0.75;
        let int_bytes = bytemuck::bytes_of(&len).to_vec();
        let fbytes = bytemuck::bytes_of(&f0)
            .iter()
            .chain(bytemuck::bytes_of(&f1))
            .copied()
            .collect::<Vec<u8>>();
        let mut loader = TestLoader {
            blobs: vec![stream.clone(), int_bytes.clone(), fbytes.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &stream)],
            int_buffers: vec![desc(1, &int_bytes)],
            float_buffers: vec![desc(2, &fbytes)],
            material_update_count: 1,
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        let opts = ParseMaterialBatchOptions {
            persist_extended_payloads: true,
            ..Default::default()
        };
        parse_materials_update_batch_into_store(&mut loader, &batch, &mut store, &opts);
        assert_eq!(
            store.get_material(21, 4),
            Some(&MaterialPropertyValue::FloatArray(vec![0.25, 0.75]))
        );
    }

    #[test]
    fn material_update_count_zero_targets_property_blocks_only() {
        let stream: Vec<u8> =
            bytemuck::bytes_of(&write_update(10, MaterialPropertyUpdateType::select_target))
                .iter()
                .chain(bytemuck::bytes_of(&write_update(
                    2,
                    MaterialPropertyUpdateType::set_float,
                )))
                .chain(bytemuck::bytes_of(&write_update(
                    0,
                    MaterialPropertyUpdateType::update_batch_end,
                )))
                .copied()
                .collect();
        let fv: f32 = 3.0;
        let fbytes = bytemuck::bytes_of(&fv).to_vec();
        let mut loader = TestLoader {
            blobs: vec![stream.clone(), fbytes.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &stream)],
            float_buffers: vec![desc(1, &fbytes)],
            material_update_count: 0,
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(
            &mut loader,
            &batch,
            &mut store,
            &ParseMaterialBatchOptions::default(),
        );
        assert_eq!(
            store.get_property_block(10, 2),
            Some(&MaterialPropertyValue::Float(3.0))
        );
        assert_eq!(store.get_material(10, 2), None);
    }

    #[test]
    fn same_numeric_id_material_and_property_block_do_not_collide() {
        let stream: Vec<u8> = bytemuck::bytes_of(&write_update(
            100,
            MaterialPropertyUpdateType::select_target,
        ))
        .iter()
        .chain(bytemuck::bytes_of(&write_update(
            1,
            MaterialPropertyUpdateType::set_float,
        )))
        .chain(bytemuck::bytes_of(&write_update(
            100,
            MaterialPropertyUpdateType::select_target,
        )))
        .chain(bytemuck::bytes_of(&write_update(
            1,
            MaterialPropertyUpdateType::set_float,
        )))
        .chain(bytemuck::bytes_of(&write_update(
            0,
            MaterialPropertyUpdateType::update_batch_end,
        )))
        .copied()
        .collect();
        let fbytes = bytemuck::bytes_of(&1.0f32)
            .iter()
            .chain(bytemuck::bytes_of(&2.0f32))
            .copied()
            .collect::<Vec<u8>>();
        let mut loader = TestLoader {
            blobs: vec![stream.clone(), fbytes.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &stream)],
            float_buffers: vec![desc(1, &fbytes)],
            material_update_count: 1,
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(
            &mut loader,
            &batch,
            &mut store,
            &ParseMaterialBatchOptions::default(),
        );
        assert_eq!(
            store.get_material(100, 1),
            Some(&MaterialPropertyValue::Float(1.0))
        );
        assert_eq!(
            store.get_property_block(100, 1),
            Some(&MaterialPropertyValue::Float(2.0))
        );
    }
}
