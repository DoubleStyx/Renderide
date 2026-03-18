use core::mem::size_of;
use std::ptr;

use bytemuck::Pod;

use super::enum_repr::EnumRepr;
use super::memory_packable::MemoryPackable;
use super::polymorphic_memory_packable_entity::PolymorphicEncode;

/// Packs data into a byte buffer for IPC.
pub struct MemoryPacker<'a> {
    buffer: &'a mut [u8],
}

impl<'a> MemoryPacker<'a> {
    /// Creates a new packer over the given buffer.
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self { buffer }
    }

    /// Computes the number of bytes written to the buffer.
    pub fn compute_length(&self, original_buffer: &[u8]) -> i32 {
        (original_buffer.len() - self.buffer.len()) as i32
    }

    /// Returns the number of bytes remaining in the buffer (not yet written).
    pub fn remaining_len(&self) -> usize {
        self.buffer.len()
    }

    /// Writes a single bool as one byte (0 or 1).
    pub fn write_bool(&mut self, value: bool) {
        self.write(&(value as u8));
    }

    /// Writes a single `Pod` value. Uses unaligned write for buffers that may not be aligned.
    pub fn write<T: Pod>(&mut self, value: &T) {
        let byte_len = size_of::<T>();
        assert!(byte_len <= self.buffer.len());
        let buf_ptr = self.buffer.as_mut_ptr();
        unsafe {
            ptr::write_unaligned(buf_ptr as *mut T, *value);
            self.buffer = core::slice::from_raw_parts_mut(
                buf_ptr.add(byte_len),
                self.buffer.len() - byte_len,
            );
        }
    }

    /// Writes a string in C#-compatible format: length (i32) then UTF-16 code units.
    /// `None` writes -1 as the length.
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

    /// Writes an optional `Pod` value: 0 byte if None, 1 byte + value if Some.
    pub fn write_option<T: Pod>(&mut self, value: Option<&T>) {
        match value {
            None => self.write(&0u8),
            Some(v) => {
                self.write(&1u8);
                self.write(v);
            }
        }
    }

    /// Writes up to 8 bools packed into a single byte (bit0 = LSB).
    pub fn write_packed_bools(
        &mut self,
        bit0: bool,
        bit1: bool,
        bit2: bool,
        bit3: bool,
        bit4: bool,
        bit5: bool,
        bit6: bool,
        bit7: bool,
    ) {
        let byte = (bit0 as u8)
            | (bit1 as u8) << 1
            | (bit2 as u8) << 2
            | (bit3 as u8) << 3
            | (bit4 as u8) << 4
            | (bit5 as u8) << 5
            | (bit6 as u8) << 6
            | (bit7 as u8) << 7;
        self.write(&byte);
    }

    /// Writes a required `MemoryPackable` object (no null byte prefix).
    pub fn write_object_required<T: MemoryPackable>(&mut self, obj: &mut T) {
        obj.pack(self);
    }

    /// Writes an optional `MemoryPackable` object: 0 byte if None, 1 byte + packed object if Some.
    pub fn write_object<T: MemoryPackable>(&mut self, obj: Option<&mut T>) {
        match obj {
            None => self.write(&0u8),
            Some(o) => {
                self.write(&1u8);
                o.pack(self);
            }
        }
    }

    /// Writes a nested list of value lists (e.g. `Vec<Vec<T>>`).
    pub fn write_nested_value_list<T: Pod>(&mut self, list: Option<&[Vec<T>]>) {
        self.write_nested_list(list, |packer, sublist| {
            packer.write_value_list(Some(sublist))
        });
    }

    /// Writes a nested list using a custom writer for each sublist.
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

    /// Writes a list of `MemoryPackable` objects.
    pub fn write_object_list<T: MemoryPackable>(&mut self, list: Option<&mut [T]>) {
        let count = list.as_deref().map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for i in 0..list.len() {
                list[i].pack(self);
            }
        }
    }

    /// Writes a list of polymorphic entities. Each item writes its type index then packs.
    pub fn write_polymorphic_list<T: PolymorphicEncode>(&mut self, list: Option<&mut [T]>) {
        let count = list.as_deref().map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for i in 0..list.len() {
                list[i].encode(self);
            }
        }
    }

    /// Writes a list of `Pod` values.
    pub fn write_value_list<T: Pod>(&mut self, list: Option<&[T]>) {
        let count = list.map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for item in list.iter() {
                self.write(item);
            }
        }
    }

    /// Writes a list of enum values as their underlying i32 representation.
    pub fn write_enum_value_list<E: EnumRepr>(&mut self, list: Option<&[E]>) {
        let count = list.map(|l| l.len()).unwrap_or(0) as i32;
        self.write(&count);
        if let Some(list) = list {
            for e in list.iter() {
                self.write(&e.as_i32());
            }
        }
    }

    /// Writes a list of strings (each element can be `None` for null).
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
