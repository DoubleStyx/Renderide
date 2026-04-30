//! Streaming cursor over chained [`SharedMemoryBufferDescriptor`]s used by the batch parser.
//!
//! Each side buffer (ints, floats, float4s, matrices, packed updates) is a sequence of descriptor
//! rows; [`ChainCursor`] hands those out to the typed `next_*` calls in [`super::readers`] without
//! ever holding more than one descriptor's worth of bytes at a time.

use bytemuck::{Pod, Zeroable};

use super::super::super::super::shared::buffer::SharedMemoryBufferDescriptor;
use super::super::super::super::shared::packing::default_entity_pool::DefaultEntityPool;
use super::super::super::super::shared::packing::memory_packable::MemoryPackable;
use super::super::super::super::shared::packing::memory_unpacker::MemoryUnpacker;
use super::MaterialBatchBlobLoader;

/// Cursor that advances through `descriptors` lazily, materializing one descriptor's bytes at a
/// time via the supplied [`MaterialBatchBlobLoader`].
pub(super) struct ChainCursor<'a> {
    descriptors: &'a [SharedMemoryBufferDescriptor],
    descriptor_index: usize,
    data: Vec<u8>,
    offset: usize,
}

impl<'a> ChainCursor<'a> {
    /// Creates a cursor positioned before the first descriptor in `descriptors`.
    pub(super) fn new(descriptors: &'a [SharedMemoryBufferDescriptor]) -> Self {
        Self {
            descriptors,
            descriptor_index: 0,
            data: Vec::new(),
            offset: 0,
        }
    }

    /// Loads the next non-empty descriptor's bytes into `self.data`. Returns `false` when exhausted.
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

    /// Drains and reloads buffers until at least `elem_size` bytes are addressable.
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

    /// Reads the next `T` from the stream, advancing the cursor on success.
    pub(super) fn next<T: Pod + Zeroable, L: MaterialBatchBlobLoader + ?Sized>(
        &mut self,
        loader: &mut L,
    ) -> Option<T> {
        let elem_size = size_of::<T>();
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

    /// Reads the next [`MemoryPackable`] row of `host_row_bytes` size.
    pub(super) fn next_packable<
        T: MemoryPackable + Default,
        L: MaterialBatchBlobLoader + ?Sized,
    >(
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
