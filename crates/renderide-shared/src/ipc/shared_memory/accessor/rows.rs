//! Fixed-stride [`MemoryPackable`](crate::packing::memory_packable::MemoryPackable) row helpers.

use crate::packing::default_entity_pool::DefaultEntityPool;
use crate::packing::memory_packable::MemoryPackable;
use crate::packing::memory_packer::MemoryPacker;
use crate::packing::memory_unpacker::MemoryUnpacker;
use crate::packing::wire_decode_error::WireDecodeError;

/// Decodes one fixed-stride host row using the same `MemoryPackable` contract as full row copies.
pub(super) fn unpack_memory_packable_row<T: MemoryPackable + Default>(
    chunk: &[u8],
    element_stride: usize,
    prefix_err: &impl Fn(&str) -> String,
) -> Result<T, String> {
    let mut pool = DefaultEntityPool;
    let mut unpacker = MemoryUnpacker::new(chunk, &mut pool);
    let mut row = T::default();
    row.unpack(&mut unpacker)
        .map_err(|e: WireDecodeError| prefix_err(&format!("MemoryPackable::unpack: {e}")))?;
    if unpacker.remaining_data() != 0 {
        return Err(prefix_err(&format!(
            "unpack left {} bytes unconsumed (stride {element_stride})",
            unpacker.remaining_data()
        )));
    }
    Ok(row)
}

/// Encodes one fixed-stride host row through [`MemoryPackable`].
pub(super) fn pack_memory_packable_row<T: MemoryPackable>(
    row: &mut T,
    chunk: &mut [u8],
    element_stride: usize,
    prefix_err: &impl Fn(&str) -> String,
) -> Result<(), String> {
    let mut packer = MemoryPacker::new(chunk);
    row.pack(&mut packer);
    if let Some(err) = packer.overflow_error() {
        return Err(prefix_err(&format!("MemoryPackable::pack: {err}")));
    }
    if packer.remaining_len() != 0 {
        return Err(prefix_err(&format!(
            "pack left {} bytes unwritten (stride {element_stride})",
            packer.remaining_len()
        )));
    }
    Ok(())
}
