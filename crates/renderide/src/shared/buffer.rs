//! Shared memory buffer descriptor for IPC.
//!
//! Mirrors `Renderite.Shared.SharedMemoryBufferDescriptor`—identifies a region
//! within a shared memory file by buffer ID, offset, and length.

use bytemuck::{Pod, Zeroable};

use super::memory_packable::MemoryPackable;
use super::memory_packer::MemoryPacker;
use super::memory_unpacker::MemoryUnpacker;
use super::memory_packer_entity_pool::MemoryPackerEntityPool;

/// Descriptor for a region within a shared memory buffer.
/// Used by the host to tell the renderer where to find packed data in `/dev/shm`.
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct SharedMemoryBufferDescriptor {
    pub buffer_id: i32,
    pub buffer_capacity: i32,
    pub offset: i32,
    pub length: i32,
}

impl SharedMemoryBufferDescriptor {
    /// Returns true if the descriptor has no data (length is zero).
    /// Matches C# `SharedMemoryBufferDescriptor.IsEmpty`.
    #[inline]
    pub fn is_empty(&self) -> bool {
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

    fn unpack<P: MemoryPackerEntityPool>(&mut self, unpacker: &mut MemoryUnpacker<'_, '_, P>) {
        self.buffer_id = unpacker.read();
        self.buffer_capacity = unpacker.read();
        self.offset = unpacker.read();
        self.length = unpacker.read();
    }
}
