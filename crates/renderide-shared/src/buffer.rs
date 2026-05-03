//! Shared-memory region descriptor for IPC.
//!
//! Describes a byte range inside a host-managed shared mapping (file mapping on Windows, POSIX shared memory, etc.).
//! The renderer uses `buffer_id`, `offset`, and `length` to locate packed payloads.

use bytemuck::{Pod, Zeroable};

use super::packing::memory_packable::MemoryPackable;
use super::packing::memory_packer::MemoryPacker;
use super::packing::memory_packer_entity_pool::MemoryPackerEntityPool;
use super::packing::memory_unpacker::MemoryUnpacker;
use super::packing::wire_decode_error::WireDecodeError;

/// Identifies a subrange of a shared buffer: which mapping, capacity hint, start offset, and span length (all in bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct SharedMemoryBufferDescriptor {
    /// Identifier for the shared memory object on the host.
    pub buffer_id: i32,
    /// Total capacity of that buffer (bytes), as known to the host.
    pub buffer_capacity: i32,
    /// Byte offset from the start of the mapping where this message's data begins.
    pub offset: i32,
    /// Length of the useful region in bytes.
    pub length: i32,
}

impl SharedMemoryBufferDescriptor {
    /// Returns `true` when the descriptor refers to no data (`length == 0`).
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }
}

impl MemoryPackable for SharedMemoryBufferDescriptor {
    fn pack(&mut self, packer: &mut MemoryPacker<'_>) {
        packer.write(&self.buffer_id);
        packer.write(&self.buffer_capacity);
        packer.write(&self.offset);
        packer.write(&self.length);
    }

    fn unpack<P: MemoryPackerEntityPool>(
        &mut self,
        unpacker: &mut MemoryUnpacker<'_, '_, P>,
    ) -> Result<(), WireDecodeError> {
        self.buffer_id = unpacker.read()?;
        self.buffer_capacity = unpacker.read()?;
        self.offset = unpacker.read()?;
        self.length = unpacker.read()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::SharedMemoryBufferDescriptor;
    use crate::packing::default_entity_pool::DefaultEntityPool;
    use crate::packing::memory_packable::MemoryPackable;
    use crate::packing::memory_packer::MemoryPacker;
    use crate::packing::memory_unpacker::MemoryUnpacker;

    #[test]
    fn is_empty_when_length_zero() {
        assert!(
            SharedMemoryBufferDescriptor {
                buffer_id: 1,
                buffer_capacity: 100,
                offset: 0,
                length: 0,
            }
            .is_empty()
        );
        assert!(
            !SharedMemoryBufferDescriptor {
                buffer_id: 1,
                buffer_capacity: 100,
                offset: 0,
                length: 1,
            }
            .is_empty()
        );
    }

    #[test]
    fn memory_packable_roundtrip() {
        let original = SharedMemoryBufferDescriptor {
            buffer_id: 7,
            buffer_capacity: 65_536,
            offset: 128,
            length: 256,
        };
        let mut packed = original;
        let mut buf = [0u8; 32];
        let full = buf.len();
        let written = {
            let mut p = MemoryPacker::new(&mut buf);
            packed.pack(&mut p);
            full - p.remaining_len()
        };
        assert_eq!(written, 16, "four i32 fields");

        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&buf[..written], &mut pool);
        let mut decoded = SharedMemoryBufferDescriptor::default();
        decoded.unpack(&mut u).expect("unpack");
        assert_eq!(decoded, original);
    }
}
