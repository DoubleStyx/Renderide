//! Unit tests for material-update batch parsing.

use super::*;
use crate::shared::buffer::SharedMemoryBufferDescriptor;

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

fn update_bytes(property_id: i32, ty: MaterialPropertyUpdateType) -> Vec<u8> {
    let mut row = write_update(property_id, ty);
    let mut buf = vec![0u8; MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES];
    let mut packer = crate::shared::packing::memory_packer::MemoryPacker::new(&mut buf);
    row.pack(&mut packer);
    buf
}

#[test]
fn select_target_uses_property_id_set_shader_in_property_id() {
    let b0 = update_bytes(42, MaterialPropertyUpdateType::SelectTarget);
    let b1 = update_bytes(7, MaterialPropertyUpdateType::SetShader);
    let b2 = update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd);
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
    let stream: Vec<u8> = [
        update_bytes(99, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(1, MaterialPropertyUpdateType::SetTexture),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
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
    let stream: Vec<u8> = [
        update_bytes(10, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(2, MaterialPropertyUpdateType::SetFloat),
        update_bytes(3, MaterialPropertyUpdateType::SetFloat4),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
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
    let b0 = update_bytes(5, MaterialPropertyUpdateType::SelectTarget);
    let b1 = update_bytes(9, MaterialPropertyUpdateType::SetShader);
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
    let stream: Vec<u8> = [
        update_bytes(20, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(3, MaterialPropertyUpdateType::SetFloat4x4),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
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
    let stream: Vec<u8> = [
        update_bytes(21, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(4, MaterialPropertyUpdateType::SetFloatArray),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
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
    let stream: Vec<u8> = [
        update_bytes(10, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(2, MaterialPropertyUpdateType::SetFloat),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
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
    let stream: Vec<u8> = [
        update_bytes(100, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(1, MaterialPropertyUpdateType::SetFloat),
        update_bytes(100, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(1, MaterialPropertyUpdateType::SetFloat),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
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

/// `SetRenderType` opcodes carry the [`crate::shared::MaterialRenderType`] discriminant in
/// `property_id` (`0` Opaque / `1` TransparentCutout / `2` Transparent). When
/// [`ParseMaterialBatchOptions::render_type_property_id`] is set, the parser writes that
/// discriminant as a synthetic float on the active material so the keyword inference path
/// can read it back.
#[test]
fn set_render_type_writes_synthetic_render_type_property_when_enabled() {
    let stream: Vec<u8> = [
        update_bytes(50, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(1, MaterialPropertyUpdateType::SetRenderType),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let mut loader = TestLoader {
        blobs: vec![stream.clone()],
    };
    let batch = MaterialsUpdateBatch {
        material_updates: vec![desc(0, &stream)],
        material_update_count: 1,
        ..Default::default()
    };
    let mut store = MaterialPropertyStore::new();
    let render_type_pid: i32 = 9999;
    let opts = ParseMaterialBatchOptions {
        render_type_property_id: Some(render_type_pid),
        ..ParseMaterialBatchOptions::default()
    };
    parse_materials_update_batch_into_store(&mut loader, &batch, &mut store, &opts);
    assert_eq!(
        store.get_material(50, render_type_pid),
        Some(&MaterialPropertyValue::Float(1.0))
    );
}

/// `SetRenderQueue` opcodes carry the queue value in `property_id` (Unity convention:
/// 2000 Opaque, 2450 AlphaTest, 3000 Transparent). When
/// [`ParseMaterialBatchOptions::render_queue_property_id`] is set the parser writes that
/// value as a synthetic float on the active material so the keyword inference path can
/// drive `_ALPHACLIP` / `_ALPHATEST_ON` for PBS materials whose `AlphaHandling` enum
/// only appears on the wire as a queue value.
#[test]
fn set_render_queue_writes_synthetic_render_queue_property_when_enabled() {
    let stream: Vec<u8> = [
        update_bytes(70, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(2450, MaterialPropertyUpdateType::SetRenderQueue),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let mut loader = TestLoader {
        blobs: vec![stream.clone()],
    };
    let batch = MaterialsUpdateBatch {
        material_updates: vec![desc(0, &stream)],
        material_update_count: 1,
        ..Default::default()
    };
    let mut store = MaterialPropertyStore::new();
    let render_queue_pid: i32 = 8888;
    let opts = ParseMaterialBatchOptions {
        render_queue_property_id: Some(render_queue_pid),
        ..ParseMaterialBatchOptions::default()
    };
    parse_materials_update_batch_into_store(&mut loader, &batch, &mut store, &opts);
    assert_eq!(
        store.get_material(70, render_queue_pid),
        Some(&MaterialPropertyValue::Float(2450.0))
    );
}

/// When the synthetic id is `None` (default options) the parser must skip the SetRenderType
/// opcode so it does not contaminate the property store with a wire-encoded enum.
#[test]
fn set_render_type_is_dropped_when_property_id_unset() {
    let stream: Vec<u8> = [
        update_bytes(60, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(2, MaterialPropertyUpdateType::SetRenderType),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let mut loader = TestLoader {
        blobs: vec![stream.clone()],
    };
    let batch = MaterialsUpdateBatch {
        material_updates: vec![desc(0, &stream)],
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
    assert_eq!(store.material_property_slot_count(), 0);
}

/// Helper: build a one-buffer batch from a script, parse it with instance-changed reporting,
/// and return the populated bit slab plus the resulting store.
fn parse_with_bits(
    material_count: i32,
    script: Vec<u8>,
    side_blobs: Vec<(i32, Vec<u8>)>,
    bit_slab_len: usize,
) -> (Vec<bool>, MaterialPropertyStore) {
    let mut blobs: Vec<Vec<u8>> = vec![script.clone()];
    for (_, bytes) in &side_blobs {
        blobs.push(bytes.clone());
    }
    let mut loader = TestLoader { blobs };

    let mut int_buffers: Vec<SharedMemoryBufferDescriptor> = Vec::new();
    let mut float_buffers: Vec<SharedMemoryBufferDescriptor> = Vec::new();
    let mut float4_buffers: Vec<SharedMemoryBufferDescriptor> = Vec::new();
    let mut matrix_buffers: Vec<SharedMemoryBufferDescriptor> = Vec::new();
    for (blob_idx, (kind, bytes)) in (1i32..).zip(side_blobs.iter()) {
        let d = desc(blob_idx, bytes);
        match *kind {
            0 => int_buffers.push(d),
            1 => float_buffers.push(d),
            2 => float4_buffers.push(d),
            3 => matrix_buffers.push(d),
            _ => unreachable!("invalid side-blob kind"),
        }
    }
    let batch = MaterialsUpdateBatch {
        material_updates: vec![desc(0, &script)],
        material_update_count: material_count,
        int_buffers,
        float_buffers,
        float4_buffers,
        matrix_buffers,
        ..Default::default()
    };
    let mut store = MaterialPropertyStore::new();
    let mut bits = vec![false; bit_slab_len];
    parse_materials_update_batch_into_store_with_instance_changed(
        &mut loader,
        &batch,
        &mut store,
        &ParseMaterialBatchOptions {
            render_type_property_id: Some(9999),
            render_queue_property_id: Some(8888),
            ..ParseMaterialBatchOptions::default()
        },
        &mut bits,
    );
    (bits, store)
}

/// Property-block targets must report instance-changed=true for every kind of payload, since
/// the host's `MaterialAssetUpdated(true)` path is what triggers `AssetCreated()` and the
/// re-emission of property block bindings to the renderers using them. Without this, font
/// atlas glyph updates do not propagate to text mesh renderers.
#[test]
fn instance_changed_property_block_set_float_is_true() {
    let stream: Vec<u8> = [
        update_bytes(10, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(2, MaterialPropertyUpdateType::SetFloat),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let fbytes = bytemuck::bytes_of(&3.0f32).to_vec();
    let (bits, _) = parse_with_bits(0, stream, vec![(1, fbytes)], 8);
    assert!(bits[0], "PB SetFloat must report instance_changed=true");
}

#[test]
fn instance_changed_property_block_set_texture_is_true() {
    let stream: Vec<u8> = [
        update_bytes(11, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(1, MaterialPropertyUpdateType::SetTexture),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let int_bytes = bytemuck::bytes_of(&0x00AB_CD01i32).to_vec();
    let (bits, _) = parse_with_bits(0, stream, vec![(0, int_bytes)], 8);
    assert!(bits[0], "PB SetTexture must report instance_changed=true");
}

/// Material targets only flip the bit on structural ops (`SetShader` / `SetInstancing` /
/// `SetRenderQueue` / `SetRenderType`); per-property writes must not.
#[test]
fn instance_changed_material_set_float_only_is_false() {
    let stream: Vec<u8> = [
        update_bytes(20, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(7, MaterialPropertyUpdateType::SetFloat),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let fbytes = bytemuck::bytes_of(&1.0f32).to_vec();
    let (bits, _) = parse_with_bits(1, stream, vec![(1, fbytes)], 8);
    assert!(
        !bits[0],
        "material SetFloat alone must not report instance_changed"
    );
}

#[test]
fn instance_changed_material_set_texture_only_is_false() {
    let stream: Vec<u8> = [
        update_bytes(21, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(8, MaterialPropertyUpdateType::SetTexture),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let int_bytes = bytemuck::bytes_of(&5i32).to_vec();
    let (bits, _) = parse_with_bits(1, stream, vec![(0, int_bytes)], 8);
    assert!(!bits[0], "material SetTexture alone must not flip the bit");
}

#[test]
fn instance_changed_material_set_shader_is_true() {
    let stream: Vec<u8> = [
        update_bytes(30, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(99, MaterialPropertyUpdateType::SetShader),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let (bits, _) = parse_with_bits(1, stream, vec![], 8);
    assert!(bits[0], "material SetShader must report instance_changed");
}

#[test]
fn instance_changed_material_set_render_queue_is_true() {
    let stream: Vec<u8> = [
        update_bytes(31, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(2450, MaterialPropertyUpdateType::SetRenderQueue),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let (bits, _) = parse_with_bits(1, stream, vec![], 8);
    assert!(
        bits[0],
        "material SetRenderQueue must report instance_changed"
    );
}

#[test]
fn instance_changed_material_set_render_type_is_true() {
    let stream: Vec<u8> = [
        update_bytes(32, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(1, MaterialPropertyUpdateType::SetRenderType),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let (bits, _) = parse_with_bits(1, stream, vec![], 8);
    assert!(
        bits[0],
        "material SetRenderType must report instance_changed"
    );
}

#[test]
fn instance_changed_material_set_instancing_is_true() {
    let stream: Vec<u8> = [
        update_bytes(33, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(0, MaterialPropertyUpdateType::SetInstancing),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    let (bits, _) = parse_with_bits(1, stream, vec![], 8);
    assert!(
        bits[0],
        "material SetInstancing must report instance_changed"
    );
}

/// Mixing material and PB targets in one batch: bit ordering is materials-first then
/// property-blocks, mirroring `MaterialUpdateData.RunCompleted` indexing.
#[test]
fn instance_changed_mixed_targets_indexed_materials_first_then_pbs() {
    let stream: Vec<u8> = [
        // Material #0 with only SetFloat → bit 0 false
        update_bytes(40, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(1, MaterialPropertyUpdateType::SetFloat),
        // Material #1 with SetShader → bit 1 true
        update_bytes(41, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(99, MaterialPropertyUpdateType::SetShader),
        // PB #0 with SetFloat → bit 2 true
        update_bytes(50, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(2, MaterialPropertyUpdateType::SetFloat),
        // PB #1 with no payload after select → bit 3 still true (PB initial)
        update_bytes(51, MaterialPropertyUpdateType::SelectTarget),
        update_bytes(0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]
    .concat();
    // Two SetFloat payloads in float buffer.
    let fbytes: Vec<u8> = [bytemuck::bytes_of(&1.0f32), bytemuck::bytes_of(&2.0f32)].concat();
    let (bits, _) = parse_with_bits(2, stream, vec![(1, fbytes)], 8);
    assert_eq!(bits[..4], [false, true, true, true]);
}

/// Bit indexing into the BitSpanMut packing must land in the right element/bit slot for
/// boundary positions across two `u32` elements. Mirrors `Renderite.Shared.BitSpan` exactly.
#[test]
fn instance_changed_bitspan_packing_at_word_boundaries() {
    use crate::shared::bit_span::BitSpanMut;
    let mut data = [0u32; 3];
    let bools: [bool; 65] = std::array::from_fn(|i| matches!(i, 0 | 31 | 32 | 33 | 63 | 64));
    {
        let mut bits = BitSpanMut::new(&mut data);
        for (i, &v) in bools.iter().enumerate() {
            if v {
                bits.set(i, true);
            }
        }
    }
    assert_eq!(data[0], 1u32 | (1u32 << 31));
    assert_eq!(data[1], 1u32 | (1u32 << 1) | (1u32 << 31));
    assert_eq!(data[2], 1u32);
}
