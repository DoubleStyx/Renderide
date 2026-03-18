use core::mem::size_of;

use bytemuck::Pod;

use super::enum_repr::EnumRepr;
use super::memory_packable::MemoryPackable;
use super::memory_packer_entity_pool::MemoryPackerEntityPool;
use super::packed_bools::PackedBools;

/// Unpacks data from a byte buffer for IPC.
pub struct MemoryUnpacker<'a, 'pool, P: MemoryPackerEntityPool> {
    buffer: &'a [u8],
    pool: &'pool mut P,
}

impl<'a, 'pool, P: MemoryPackerEntityPool> MemoryUnpacker<'a, 'pool, P> {
    /// Creates a new unpacker over the given buffer and entity pool.
    pub fn new(buffer: &'a [u8], pool: &'pool mut P) -> Self {
        Self { buffer, pool }
    }

    /// Returns the number of bytes remaining in the buffer.
    pub fn remaining_data(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `count` elements of type `T`, advancing the internal buffer.
    /// Uses unaligned reads so buffers from IPC/shared memory work regardless of alignment.
    pub fn access<T: Pod>(&mut self, count: usize) -> Vec<T> {
        let byte_len = count * size_of::<T>();
        assert!(
            byte_len <= self.buffer.len(),
            "buffer too small for {} elements of type {} (need {} bytes, have {} remaining)",
            count,
            std::any::type_name::<T>(),
            byte_len,
            self.buffer.len()
        );
        let (consumed, remaining) = self.buffer.split_at(byte_len);
        self.buffer = remaining;
        (0..count)
            .map(|i| {
                let start = i * size_of::<T>();
                bytemuck::pod_read_unaligned::<T>(&consumed[start..start + size_of::<T>()])
            })
            .collect()
    }

    /// Reads a single bool (one byte, 0 = false, non-zero = true).
    pub fn read_bool(&mut self) -> bool {
        self.read::<u8>() != 0
    }

    /// Reads a single `Pod` value.
    pub fn read<T: Pod>(&mut self) -> T {
        self.access::<T>(1)[0]
    }

    /// Reads an optional `Pod` value: 0 byte if None, 1 byte + value if Some.
    pub fn read_option<T: Pod>(&mut self) -> Option<T> {
        if self.read::<u8>() == 0 {
            None
        } else {
            Some(self.read())
        }
    }

    /// Reads a string in C#-compatible format: length (i32) then UTF-16 code units.
    /// Returns `None` if length is -1 (null).
    pub fn read_str(&mut self) -> Option<String> {
        let len = self.read::<i32>();
        if len < 0 {
            return None;
        }
        if len == 0 {
            return Some(String::new());
        }
        let utf16: Vec<u16> = self.access::<u16>(len as usize);
        Some(String::from_utf16(&utf16).unwrap_or_default())
    }

    /// Reads up to 8 bools packed into a single byte (bit0 = LSB).
    pub fn read_packed_bools(&mut self) -> PackedBools {
        PackedBools::from_byte(self.read::<u8>())
    }

    /// Reads a required `MemoryPackable` object into the given mutable reference.
    pub fn read_object_required<T: MemoryPackable>(&mut self, obj: &mut T) {
        obj.unpack(self);
    }

    /// Reads an optional `MemoryPackable` object: 0 byte if None, 1 byte + unpacked object if Some.
    pub fn read_object<T: MemoryPackable + Default>(&mut self) -> Option<T> {
        if self.read::<u8>() == 0 {
            return None;
        }
        let mut obj = self.pool.borrow::<T>();
        obj.unpack(self);
        Some(obj)
    }

    /// Reads a list of `MemoryPackable` objects.
    pub fn read_object_list<T: MemoryPackable + Default>(&mut self) -> Vec<T> {
        let count = self.read::<i32>() as usize;
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            let mut obj = self.pool.borrow::<T>();
            obj.unpack(self);
            list.push(obj);
        }
        list
    }

    /// Reads a polymorphic list using the given decode callback for each element.
    pub fn read_polymorphic_list<F, T>(&mut self, mut decode: F) -> Vec<T>
    where
        F: FnMut(&mut MemoryUnpacker<'a, 'pool, P>) -> T,
    {
        let count = self.read::<i32>() as usize;
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            list.push(decode(self));
        }
        list
    }

    /// Reads a list of `Pod` values.
    pub fn read_value_list<T: Pod>(&mut self) -> Vec<T> {
        let count = self.read::<i32>() as usize;
        self.access::<T>(count)
    }

    /// Reads a list of enum values from their underlying i32 representation.
    pub fn read_enum_value_list<E: EnumRepr>(&mut self) -> Vec<E> {
        let count = self.read::<i32>() as usize;
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            list.push(E::from_i32(self.read::<i32>()));
        }
        list
    }

    /// Reads a list of strings (each element can be `None` for null).
    pub fn read_string_list(&mut self) -> Vec<Option<String>> {
        let count = self.read::<i32>() as usize;
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            list.push(self.read_str());
        }
        list
    }

    /// Reads a nested list of value lists (e.g. `Vec<Vec<T>>`).
    pub fn read_nested_value_list<T: Pod>(&mut self) -> Vec<Vec<T>> {
        self.read_nested_list(|unpacker| unpacker.read_value_list())
    }

    /// Reads a nested list using a custom reader for each sublist.
    pub fn read_nested_list<F, T>(&mut self, mut sublist_reader: F) -> Vec<T>
    where
        F: FnMut(&mut MemoryUnpacker<'a, 'pool, P>) -> T,
    {
        let count = self.read::<i32>() as usize;
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            list.push(sublist_reader(self));
        }
        list
    }
}
