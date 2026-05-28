//! [`MemoryPacker`]: host-compatible sequential writes into a byte slice.

use core::mem::size_of;

use bytemuck::Pod;

use super::enum_repr::EnumRepr;
use super::memory_pack_error::MemoryPackError;
use super::memory_packable::MemoryPackable;
use super::polymorphic_memory_packable_entity::PolymorphicEncode;
use super::type_name::short_type_name;

/// Sequential binary writer for IPC buffers.
///
/// Values are appended to the backing byte slice in host wire order. Overflow is captured as
/// packer state, and later writes are skipped so generated and hand-written `pack` implementations
/// can keep their infallible signature. Callers must finish an encode by checking
/// [`MemoryPacker::into_result`] or [`MemoryPacker::had_overflow`] and discard incomplete messages.
pub struct MemoryPacker<'a> {
    /// Remaining unwritten tail of the backing slice.
    buffer: &'a mut [u8],
    /// First failed write, retained until the caller inspects the encode result.
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

    /// Writes a plain data value with potentially unaligned storage.
    ///
    /// On overflow this records the first failed reservation and leaves the write cursor
    /// unchanged. Later writes are skipped until the caller inspects the result.
    pub fn write<T: Pod>(&mut self, value: &T) {
        if let Some(dst) = self.reserve_pod_bytes::<T>() {
            dst.copy_from_slice(bytemuck::bytes_of(value));
        }
    }

    /// Reserves space for one plain data value and advances the cursor on success.
    fn reserve_pod_bytes<T: Pod>(&mut self) -> Option<&mut [u8]> {
        let byte_len = size_of::<T>();
        self.reserve_bytes(byte_len, short_type_name::<T>())
    }

    /// Reserves `byte_len` bytes and advances the cursor on success.
    fn reserve_bytes(&mut self, byte_len: usize, ty: &'static str) -> Option<&mut [u8]> {
        if self.overflow.is_some() {
            return None;
        }
        if byte_len > self.buffer.len() {
            self.record_overflow(ty, byte_len);
            return None;
        }
        Some(self.take_prefix(byte_len))
    }

    /// Records the first failed write reservation.
    fn record_overflow(&mut self, ty: &'static str, needed: usize) {
        self.overflow = Some(MemoryPackError::BufferTooSmall {
            ty,
            needed,
            remaining: self.buffer.len(),
        });
    }

    /// Splits `byte_len` bytes from the front of the remaining buffer.
    fn take_prefix(&mut self, byte_len: usize) -> &mut [u8] {
        let empty_tail: &mut [u8] = &mut [];
        let buf = std::mem::replace(&mut self.buffer, empty_tail);
        let (head, tail) = buf.split_at_mut(byte_len);
        self.buffer = tail;
        head
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
    /// Generated shared types call this for fields stored as packed boolean bytes.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    enum TestEnum {
        A,
        B,
    }

    impl EnumRepr for TestEnum {
        fn as_i32(self) -> i32 {
            match self {
                Self::A => 11,
                Self::B => 22,
            }
        }

        fn from_i32(i: i32) -> Self {
            if i == 22 { Self::B } else { Self::A }
        }
    }

    fn read_i32(bytes: &[u8], offset: usize) -> i32 {
        i32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("i32 bytes"))
    }

    fn read_u16(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("u16 bytes"))
    }

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
        packer.write(&0u32);
        assert!(packer.had_overflow());
        packer.write(&0u8);
        assert!(packer.had_overflow());
        assert_eq!(buf, [0u8; 1]);
    }

    #[test]
    fn overflow_preserves_first_failed_reservation() {
        let mut buf = [0u8; 2];
        let mut packer = MemoryPacker::new(&mut buf);

        packer.write(&0u32);
        packer.write(&0u64);

        let err = packer.overflow_error().expect("captured error");
        match err {
            MemoryPackError::BufferTooSmall {
                needed, remaining, ..
            } => {
                assert_eq!(needed, 4);
                assert_eq!(remaining, 2);
            }
        }
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
    fn compute_length_and_remaining_track_cursor_progress() {
        let mut backing = vec![0u8; 8];
        let original = backing.clone();
        let mut packer = MemoryPacker::new(&mut backing);

        packer.write(&0x1122u16);
        packer.write_bool(true);

        assert_eq!(packer.compute_length(&original), 3);
        assert_eq!(packer.remaining_len(), 5);
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

    #[test]
    fn write_bool_and_packed_bools_use_expected_bit_layout() {
        let mut backing = [0u8; 3];
        let mut packer = MemoryPacker::new(&mut backing);

        packer.write_bool(false);
        packer.write_bool(true);
        packer.write_packed_bools_array([true, false, true, true, false, false, true, false]);

        assert_eq!(backing, [0, 1, 0b0100_1101]);
    }

    #[test]
    fn write_str_encodes_none_and_utf16_code_units() {
        let mut backing = [0u8; 16];
        let mut packer = MemoryPacker::new(&mut backing);

        packer.write_str(None);
        packer.write_str(Some("Az"));

        assert_eq!(read_i32(&backing, 0), -1);
        assert_eq!(read_i32(&backing, 4), 2);
        assert_eq!(read_u16(&backing, 8), b'A' as u16);
        assert_eq!(read_u16(&backing, 10), b'z' as u16);
    }

    #[test]
    fn write_option_encodes_presence_byte_before_value() {
        let mut backing = [0u8; 12];
        let mut packer = MemoryPacker::new(&mut backing);

        packer.write_option::<u32>(None);
        packer.write_option(Some(&0x1122_3344u32));
        let written = packer.compute_length(&[0u8; 12]);

        assert_eq!(backing[0], 0);
        assert_eq!(backing[1], 1);
        assert_eq!(
            u32::from_le_bytes(backing[2..6].try_into().expect("u32 bytes")),
            0x1122_3344
        );
        assert_eq!(written, 6);
    }

    #[test]
    fn write_value_list_encodes_count_and_values() {
        let mut backing = [0u8; 16];
        let mut packer = MemoryPacker::new(&mut backing);

        packer.write_value_list(Some(&[5u16, 9u16]));
        packer.write_value_list::<u16>(None);

        assert_eq!(read_i32(&backing, 0), 2);
        assert_eq!(read_u16(&backing, 4), 5);
        assert_eq!(read_u16(&backing, 6), 9);
        assert_eq!(read_i32(&backing, 8), 0);
    }

    #[test]
    fn write_nested_value_list_encodes_outer_and_inner_counts() {
        let nested = vec![vec![1u16, 2u16], vec![3u16]];
        let mut backing = [0u8; 32];
        let mut packer = MemoryPacker::new(&mut backing);

        packer.write_nested_value_list(Some(&nested));

        assert_eq!(read_i32(&backing, 0), 2);
        assert_eq!(read_i32(&backing, 4), 2);
        assert_eq!(read_u16(&backing, 8), 1);
        assert_eq!(read_u16(&backing, 10), 2);
        assert_eq!(read_i32(&backing, 12), 1);
        assert_eq!(read_u16(&backing, 16), 3);
    }

    #[test]
    fn write_enum_value_list_stores_i32_discriminants() {
        let mut backing = [0u8; 16];
        let mut packer = MemoryPacker::new(&mut backing);

        packer.write_enum_value_list(Some(&[TestEnum::A, TestEnum::B]));

        assert_eq!(read_i32(&backing, 0), 2);
        assert_eq!(read_i32(&backing, 4), 11);
        assert_eq!(read_i32(&backing, 8), 22);
    }

    #[test]
    fn write_string_list_encodes_nullable_entries() {
        let entries = [Some("x"), None];
        let mut backing = [0u8; 24];
        let mut packer = MemoryPacker::new(&mut backing);

        packer.write_string_list(Some(&entries));

        assert_eq!(read_i32(&backing, 0), 2);
        assert_eq!(read_i32(&backing, 4), 1);
        assert_eq!(read_u16(&backing, 8), b'x' as u16);
        assert_eq!(read_i32(&backing, 10), -1);
    }

    #[test]
    fn short_type_name_strips_module_prefix() {
        assert_eq!(short_type_name::<Option<u32>>(), "Option<u32>");
        assert_eq!(short_type_name::<MemoryPacker<'_>>(), "MemoryPacker<'_>");
    }
}
