//! [`MemoryPacker`]: host-compatible sequential writes into a byte slice.

use core::mem::size_of;

use bytemuck::Pod;

use super::enum_repr::EnumRepr;
use super::memory_packable::MemoryPackable;
use super::polymorphic_memory_packable_entity::PolymorphicEncode;

/// Sequential binary writer for IPC buffers (writes `Pod` values as byte slices; works for unaligned buffers).
pub struct MemoryPacker<'a> {
    buffer: &'a mut [u8],
}

impl<'a> MemoryPacker<'a> {
    /// Wraps `buffer`; writing advances an internal cursor toward the end of the slice.
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self { buffer }
    }

    /// Returns how many bytes were written relative to the original full slice length.
    pub fn compute_length(&self, original_buffer: &[u8]) -> i32 {
        (original_buffer.len() - self.buffer.len()) as i32
    }

    /// Bytes not yet consumed at the front of the backing slice.
    pub fn remaining_len(&self) -> usize {
        self.buffer.len()
    }

    /// Writes one byte: `1` for true, `0` for false.
    pub fn write_bool(&mut self, value: bool) {
        self.write(&(value as u8));
    }

    /// Writes a plain data value with potentially unaligned storage (safe for shared-memory views).
    ///
    /// Uses [`std::mem::replace`] so the slice can be split after [`bytemuck::bytes_of`] without
    /// borrowing `value` for the lifetime of the backing buffer.
    pub fn write<T: Pod>(&mut self, value: &T) {
        let byte_len = size_of::<T>();
        assert!(
            byte_len <= self.buffer.len(),
            "MemoryPacker::write: need {} bytes for {}, {} remaining",
            byte_len,
            std::any::type_name::<T>(),
            self.buffer.len()
        );
        let bytes = bytemuck::bytes_of(value);
        let empty_tail: &mut [u8] = &mut [];
        let buf = std::mem::replace(&mut self.buffer, empty_tail);
        let (head, tail) = buf.split_at_mut(byte_len);
        head.copy_from_slice(bytes);
        self.buffer = tail;
    }

    /// UTF‑16 code units (two-byte wchar layout): `i32` length, then each `u16`. Length `-1` means null.
    pub fn write_str(&mut self, s: Option<&str>) {
        match s {
            None => self.write(&(-1i32)),
            Some(str) => {
                let utf16: Vec<u16> = str.encode_utf16().collect();
                let len = utf16.len() as i32;
                self.write(&len);
                for c in &utf16 {
                    self.write(c);
                }
            }
        }
    }

    /// Optional POD: `0` prefix absent, `1` prefix then value.
    pub fn write_option<T: Pod>(&mut self, value: Option<&T>) {
        match value {
            None => self.write(&0u8),
            Some(v) => {
                self.write(&1u8);
                self.write(v);
            }
        }
    }

    /// Packs eight booleans into one byte (bit0 = LSB).
    ///
    /// SharedTypeGenerator emits `packer.write_packed_bools_array([...])` for packed-bool fields in the generated shared types.
    pub fn write_packed_bools_array(&mut self, bits: [bool; 8]) {
        let byte = (bits[0] as u8)
            | (bits[1] as u8) << 1
            | (bits[2] as u8) << 2
            | (bits[3] as u8) << 3
            | (bits[4] as u8) << 4
            | (bits[5] as u8) << 5
            | (bits[6] as u8) << 6
            | (bits[7] as u8) << 7;
        self.write(&byte);
    }

    /// Inlines packing without an optional presence byte.
    pub fn write_object_required<T: MemoryPackable>(&mut self, obj: &mut T) {
        obj.pack(self);
    }

    /// Optional object: `0` absent, `1` then nested pack.
    pub fn write_object<T: MemoryPackable>(&mut self, obj: Option<&mut T>) {
        match obj {
            None => self.write(&0u8),
            Some(o) => {
                self.write(&1u8);
                o.pack(self);
            }
        }
    }

    /// `Vec<Vec<T>>`-style structure: outer count, then each inner value-list.
    pub fn write_nested_value_list<T: Pod>(&mut self, list: Option<&[Vec<T>]>) {
        self.write_nested_list(list, |packer, sublist| {
            packer.write_value_list(Some(sublist))
        });
    }

    /// Outer list length plus custom writer per element.
    pub fn write_nested_list<T, F>(&mut self, list: Option<&[T]>, mut sublist_writer: F)
    where
        F: FnMut(&mut MemoryPacker<'a>, &T),
    {
        let count = list.map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for item in list.iter() {
                sublist_writer(self, item);
            }
        }
    }

    /// Object list: count then each element packed in order.
    pub fn write_object_list<T: MemoryPackable>(&mut self, list: Option<&mut [T]>) {
        let count = list.as_deref().map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for item in list.iter_mut() {
                item.pack(self);
            }
        }
    }

    /// Polymorphic list: count then each element’s `encode`.
    pub fn write_polymorphic_list<T: PolymorphicEncode>(&mut self, list: Option<&mut [T]>) {
        let count = list.as_deref().map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for item in list.iter_mut() {
                item.encode(self);
            }
        }
    }

    /// Homogeneous POD list: count then each element.
    pub fn write_value_list<T: Pod>(&mut self, list: Option<&[T]>) {
        let count = list.map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for item in list.iter() {
                self.write(item);
            }
        }
    }

    /// Like [`Self::write_value_list`] but each item is an enum stored as `i32`.
    pub fn write_enum_value_list<E: EnumRepr>(&mut self, list: Option<&[E]>) {
        let count = list.map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for e in list.iter() {
                self.write(&e.as_i32());
            }
        }
    }

    /// List of nullable strings in host format.
    pub fn write_string_list(&mut self, list: Option<&[Option<&str>]>) {
        let count = list.map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for s in list.iter() {
                self.write_str(*s);
            }
        }
    }
}
