//! [`MemoryPacker`]: host-compatible sequential writes into a byte slice.

use core::mem::size_of;

use bytemuck::Pod;

use super::enum_repr::EnumRepr;
use super::memory_pack_error::MemoryPackError;
use super::memory_packable::MemoryPackable;
use super::polymorphic_memory_packable_entity::PolymorphicEncode;

/// Sequential binary writer for IPC buffers (writes `Pod` values as byte slices; works for unaligned buffers).
///
/// Buffer overflow used to `panic!` via `assert!`; CLAUDE.md forbids panics in library/runtime
/// paths. The packer now tracks the overflow as state: subsequent writes silently no-op so
/// the trait-shaped `pack(&mut self, packer)` API used by both hand-written code and the
/// `SharedTypeGenerator`-emitted `pack` methods does not need to thread `Result` through every
/// write site. After encoding, callers must invoke [`MemoryPacker::into_result`] (or check
/// [`MemoryPacker::had_overflow`]) to determine whether the message is complete; an incomplete
/// message must be discarded rather than transmitted because the trailing fields are missing.
pub struct MemoryPacker<'a> {
    /// Remaining unwritten tail of the backing slice.
    buffer: &'a mut [u8],
    /// Overflow state: `None` until the first failed write, then the captured error.
    overflow: Option<MemoryPackError>,
}

impl<'a> MemoryPacker<'a> {
    /// Wraps `buffer`; writing advances an internal cursor toward the end of the slice.
    pub const fn new(buffer: &'a mut [u8]) -> Self {
        Self {
            buffer,
            overflow: None,
        }
    }

    /// Returns how many bytes were written relative to the original full slice length.
    pub const fn compute_length(&self, original_buffer: &[u8]) -> i32 {
        (original_buffer.len() - self.buffer.len()) as i32
    }

    /// Bytes not yet consumed at the front of the backing slice.
    pub const fn remaining_len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` once any write has failed because the buffer ran out of space.
    pub const fn had_overflow(&self) -> bool {
        self.overflow.is_some()
    }

    /// Returns the captured overflow as an `Err`, or `Ok(bytes_written)` on a clean encode.
    ///
    /// The byte count is computed against `original_buffer` (the same slice handed to
    /// [`MemoryPacker::new`]); pass that through if the caller needs to send the prefix length.
    pub const fn into_result(self, original_buffer: &[u8]) -> Result<i32, MemoryPackError> {
        if let Some(err) = self.overflow {
            return Err(err);
        }
        Ok((original_buffer.len() - self.buffer.len()) as i32)
    }

    /// Returns the captured [`MemoryPackError`] if the encode overflowed.
    pub const fn overflow_error(&self) -> Option<MemoryPackError> {
        self.overflow
    }

    /// Writes one byte: `1` for true, `0` for false.
    pub fn write_bool(&mut self, value: bool) {
        self.write(&u8::from(value));
    }

    /// Writes a plain data value with potentially unaligned storage (safe for shared-memory views).
    ///
    /// Uses [`std::mem::replace`] so the slice can be split after [`bytemuck::bytes_of`] without
    /// borrowing `value` for the lifetime of the backing buffer.
    ///
    /// On buffer overflow this records the failure on the packer (see
    /// [`MemoryPacker::had_overflow`] / [`MemoryPacker::into_result`]) and silently skips the
    /// write so the cursor remains at the prior position; callers must verify the result after
    /// encoding instead of relying on a panic.
    pub fn write<T: Pod>(&mut self, value: &T) {
        if self.overflow.is_some() {
            return;
        }
        let byte_len = size_of::<T>();
        if byte_len > self.buffer.len() {
            self.overflow = Some(MemoryPackError::BufferTooSmall {
                ty: short_type_name::<T>(),
                needed: byte_len,
                remaining: self.buffer.len(),
            });
            return;
        }
        let bytes = bytemuck::bytes_of(value);
        let empty_tail: &mut [u8] = &mut [];
        let buf = std::mem::replace(&mut self.buffer, empty_tail);
        let (head, tail) = buf.split_at_mut(byte_len);
        head.copy_from_slice(bytes);
        self.buffer = tail;
    }

    /// UTF-16 code units (two-byte wchar layout): `i32` length, then each `u16`. Length `-1` means null.
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
    /// `SharedTypeGenerator` emits `packer.write_packed_bools_array([...])` for packed-bool fields in the generated shared types.
    pub fn write_packed_bools_array(&mut self, bits: [bool; 8]) {
        let byte = u8::from(bits[0])
            | u8::from(bits[1]) << 1
            | u8::from(bits[2]) << 2
            | u8::from(bits[3]) << 3
            | u8::from(bits[4]) << 4
            | u8::from(bits[5]) << 5
            | u8::from(bits[6]) << 6
            | u8::from(bits[7]) << 7;
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
            packer.write_value_list(Some(sublist));
        });
    }

    /// Outer list length plus custom writer per element.
    pub fn write_nested_list<T, F>(&mut self, list: Option<&[T]>, mut sublist_writer: F)
    where
        F: FnMut(&mut MemoryPacker<'a>, &T),
    {
        let count = list.map_or(0, <[T]>::len) as i32;
        self.write(&count);
        if let Some(list) = list {
            for item in list {
                sublist_writer(self, item);
            }
        }
    }

    /// Object list: count then each element packed in order.
    pub fn write_object_list<T: MemoryPackable>(&mut self, list: Option<&mut [T]>) {
        let count = list.as_deref().map_or(0, <[T]>::len) as i32;
        self.write(&count);
        if let Some(list) = list {
            for item in list.iter_mut() {
                item.pack(self);
            }
        }
    }

    /// Polymorphic list: count then each element's `encode`.
    pub fn write_polymorphic_list<T: PolymorphicEncode>(&mut self, list: Option<&mut [T]>) {
        let count = list.as_deref().map_or(0, <[T]>::len) as i32;
        self.write(&count);
        if let Some(list) = list {
            for item in list.iter_mut() {
                item.encode(self);
            }
        }
    }

    /// Homogeneous POD list: count then each element.
    pub fn write_value_list<T: Pod>(&mut self, list: Option<&[T]>) {
        let count = list.map_or(0, <[T]>::len) as i32;
        self.write(&count);
        if let Some(list) = list {
            for item in list {
                self.write(item);
            }
        }
    }

    /// Like [`Self::write_value_list`] but each item is an enum stored as `i32`.
    pub fn write_enum_value_list<E: EnumRepr>(&mut self, list: Option<&[E]>) {
        let count = list.map_or(0, <[E]>::len) as i32;
        self.write(&count);
        if let Some(list) = list {
            for e in list {
                self.write(&e.as_i32());
            }
        }
    }

    /// List of nullable strings in host format.
    pub fn write_string_list(&mut self, list: Option<&[Option<&str>]>) {
        let count = list.map_or(0, <[Option<&str>]>::len) as i32;
        self.write(&count);
        if let Some(list) = list {
            for s in list {
                self.write_str(*s);
            }
        }
    }
}

/// Returns the unqualified Rust type name of `T` for diagnostics.
fn short_type_name<T>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_overflow_does_not_panic_and_records_error() {
        let mut buf = [0u8; 3];
        let mut packer = MemoryPacker::new(&mut buf);
        // 4 bytes does not fit in 3.
        packer.write(&0x1122_3344u32);
        assert!(packer.had_overflow(), "overflow flag should be set");
        let err = packer.overflow_error().expect("captured error");
        match err {
            MemoryPackError::BufferTooSmall {
                needed, remaining, ..
            } => {
                assert_eq!(needed, 4);
                assert_eq!(remaining, 3);
            }
        }
    }

    #[test]
    fn writes_after_overflow_are_no_ops() {
        let mut buf = [0u8; 1];
        let mut packer = MemoryPacker::new(&mut buf);
        packer.write(&0u32); // overflow: needs 4, has 1
        assert!(packer.had_overflow());
        // A subsequent legitimate-size write must be skipped so the cursor stays put.
        packer.write(&0u8);
        assert!(packer.had_overflow());
        // Buffer was untouched (no partial corruption).
        assert_eq!(buf, [0u8; 1]);
    }

    #[test]
    fn into_result_reports_byte_count_on_clean_encode() {
        let mut backing = vec![0u8; 8];
        let original = backing.clone();
        let mut packer = MemoryPacker::new(&mut backing);
        packer.write(&0x1122_3344u32);
        packer.write(&0x55u8);
        let written = packer.into_result(&original).expect("clean encode");
        assert_eq!(written, 5);
    }

    #[test]
    fn into_result_returns_err_after_overflow() {
        let mut backing = vec![0u8; 2];
        let original = backing.clone();
        let mut packer = MemoryPacker::new(&mut backing);
        packer.write(&0u32);
        let err = packer.into_result(&original).expect_err("should overflow");
        assert!(matches!(err, MemoryPackError::BufferTooSmall { .. }));
    }
}
