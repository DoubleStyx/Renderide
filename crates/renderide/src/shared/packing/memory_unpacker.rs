//! [`MemoryUnpacker`]: host-compatible reads from a byte slice with an entity pool.

use core::mem::size_of;

use bytemuck::Pod;

use super::enum_repr::EnumRepr;
use super::memory_packable::MemoryPackable;
use super::memory_packer_entity_pool::MemoryPackerEntityPool;
use super::packed_bools::PackedBools;

/// Cursor over read-only IPC bytes, using `pool` when unpacking optional heap types.
pub struct MemoryUnpacker<'a, 'pool, P: MemoryPackerEntityPool> {
    buffer: &'a [u8],
    pool: &'pool mut P,
}

impl<'a, 'pool, P: MemoryPackerEntityPool> MemoryUnpacker<'a, 'pool, P> {
    /// Starts at the beginning of `buffer`.
    pub fn new(buffer: &'a [u8], pool: &'pool mut P) -> Self {
        Self { buffer, pool }
    }

    /// Bytes not yet read.
    pub fn remaining_data(&self) -> usize {
        self.buffer.len()
    }

    /// Consumes `count` contiguous `T` values (unaligned-safe).
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
        let elem_size = size_of::<T>();
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * elem_size;
            out.push(bytemuck::pod_read_unaligned::<T>(
                &consumed[start..start + elem_size],
            ));
        }
        out
    }

    /// One-byte boolean (any non-zero is true).
    pub fn read_bool(&mut self) -> bool {
        self.read::<u8>() != 0
    }

    /// Single POD value.
    pub fn read<T: Pod>(&mut self) -> T {
        self.access::<T>(1)[0]
    }

    /// Optional POD with `u8` discriminant.
    pub fn read_option<T: Pod>(&mut self) -> Option<T> {
        if self.read::<u8>() == 0 {
            None
        } else {
            Some(self.read())
        }
    }

    /// Host string: UTF-16 LE code units with `i32` length. `-1` → [`None`]. Surrogate halves or
    /// invalid sequences decode to the empty string (defensive; the host typically sends valid UTF-16).
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

    /// Eight booleans from one byte.
    pub fn read_packed_bools(&mut self) -> PackedBools {
        PackedBools::from_byte(self.read::<u8>())
    }

    /// Fills an existing `MemoryPackable` (no presence byte).
    pub fn read_object_required<T: MemoryPackable>(&mut self, obj: &mut T) {
        obj.unpack(self);
    }

    /// Optional object with `u8` discriminant, allocated from `pool` when present.
    pub fn read_object<T: MemoryPackable + Default>(&mut self) -> Option<T> {
        if self.read::<u8>() == 0 {
            return None;
        }
        let mut obj = self.pool.borrow::<T>();
        obj.unpack(self);
        Some(obj)
    }

    /// Object list; negative outer count is treated as empty (defensive).
    pub fn read_object_list<T: MemoryPackable + Default>(&mut self) -> Vec<T> {
        let count = self.read::<i32>();
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            let mut obj = self.pool.borrow::<T>();
            obj.unpack(self);
            list.push(obj);
        }
        list
    }

    /// Polymorphic list: `decode` reads discriminator and payload per element.
    pub fn read_polymorphic_list<F, T>(&mut self, mut decode: F) -> Vec<T>
    where
        F: FnMut(&mut MemoryUnpacker<'a, 'pool, P>) -> T,
    {
        let count = self.read::<i32>();
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            list.push(decode(self));
        }
        list
    }

    /// POD list.
    pub fn read_value_list<T: Pod>(&mut self) -> Vec<T> {
        let count = self.read::<i32>();
        let count = if count < 0 { 0 } else { count as usize };
        self.access::<T>(count)
    }

    /// Enum list stored as `i32` discriminants.
    pub fn read_enum_value_list<E: EnumRepr>(&mut self) -> Vec<E> {
        let count = self.read::<i32>();
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            list.push(E::from_i32(self.read::<i32>()));
        }
        list
    }

    /// List of nullable strings.
    pub fn read_string_list(&mut self) -> Vec<Option<String>> {
        let count = self.read::<i32>();
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            list.push(self.read_str());
        }
        list
    }

    /// Nested value lists.
    pub fn read_nested_value_list<T: Pod>(&mut self) -> Vec<Vec<T>> {
        self.read_nested_list(|unpacker| unpacker.read_value_list())
    }

    /// Nested list with custom inner reader.
    pub fn read_nested_list<F, T>(&mut self, mut sublist_reader: F) -> Vec<T>
    where
        F: FnMut(&mut MemoryUnpacker<'a, 'pool, P>) -> T,
    {
        let count = self.read::<i32>();
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            list.push(sublist_reader(self));
        }
        list
    }
}
