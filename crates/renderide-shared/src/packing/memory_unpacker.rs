//! [`MemoryUnpacker`]: host-compatible reads from a byte slice with an entity pool.

use core::mem::size_of;

use bytemuck::Pod;

use super::enum_repr::EnumRepr;
use super::memory_packable::MemoryPackable;
use super::memory_packer_entity_pool::MemoryPackerEntityPool;
use super::memory_unpack_error::MemoryUnpackError;
use super::packed_bools::PackedBools;
use super::wire_decode_error::WireDecodeError;

/// Cursor over read-only IPC bytes, using `pool` when unpacking optional heap types.
pub struct MemoryUnpacker<'a, 'pool, P: MemoryPackerEntityPool> {
    buffer: &'a [u8],
    pool: &'pool mut P,
}

/// Maximum UTF-16 code units accepted by [`MemoryUnpacker::read_str`].
///
/// Caps speculative allocation when an attacker-influenced length prefix would otherwise
/// drive a multi-megabyte `String` allocation per field. `1 << 20` (one mebi) code units is
/// two megabytes of UTF-16 -- comfortably above any legitimate IPC string.
pub const MAX_STRING_LEN: usize = 1 << 20;

/// Returns a `Vec::with_capacity` hint that does not exceed the unread buffer length.
///
/// Each element of any list reader consumes at least one wire byte, so a `count` larger than
/// the remaining buffer cannot decode successfully -- the per-element loop will surface
/// [`MemoryUnpackError::Underrun`]. This helper keeps the speculative pre-allocation bounded
/// by the input size so a malicious `i32::MAX` count cannot reserve gigabytes ahead of the
/// real underrun error.
fn alloc_hint(count: usize, remaining_bytes: usize) -> usize {
    count.min(remaining_bytes)
}

impl<'a, 'pool, P: MemoryPackerEntityPool> MemoryUnpacker<'a, 'pool, P> {
    /// Starts at the beginning of `buffer`.
    pub const fn new(buffer: &'a [u8], pool: &'pool mut P) -> Self {
        Self { buffer, pool }
    }

    /// Bytes not yet read.
    pub const fn remaining_data(&self) -> usize {
        self.buffer.len()
    }

    /// Consumes `count` contiguous `T` values (unaligned-safe).
    ///
    /// On the IPC decode hot path the per-element [`bytemuck::pod_read_unaligned`] loop dominates
    /// large vertex / transform / index payloads. For non-zero-sized POD types we instead do a
    /// single `ptr::copy_nonoverlapping` into a freshly-allocated [`Vec<T>`] of capacity `count`
    /// and then `set_len(count)`. `T: Pod` guarantees any byte pattern is a valid `T`, the source
    /// length was bounds-checked above, and `Vec::with_capacity` allocates `count * size_of::<T>()`
    /// bytes of properly-aligned destination storage.
    pub fn access<T: Pod>(&mut self, count: usize) -> Result<Vec<T>, MemoryUnpackError> {
        let elem_size = size_of::<T>();
        let byte_len = count
            .checked_mul(elem_size)
            .ok_or(MemoryUnpackError::LengthOverflow)?;
        if byte_len > self.buffer.len() {
            return Err(MemoryUnpackError::pod_underrun::<T>(
                byte_len,
                self.buffer.len(),
            ));
        }
        let (consumed, remaining) = self.buffer.split_at(byte_len);
        self.buffer = remaining;
        if count == 0 || elem_size == 0 {
            return Ok(Vec::new());
        }
        let mut out: Vec<T> = Vec::with_capacity(count);
        // SAFETY:
        // - `consumed` has exactly `byte_len = count * elem_size` bytes (bounds-checked above).
        // - `out` has capacity `count` allocated through `Vec::with_capacity`, so its backing
        //   storage holds `count * elem_size` bytes of properly-aligned writable memory and the
        //   destination range does not overlap `consumed` (different allocations).
        // - `T: Pod` permits any byte pattern, so the copied bytes form a valid `T` for every
        //   index in `0..count`. After the copy every slot is initialized; `set_len(count)` is
        //   sound.
        // - `ptr::copy_nonoverlapping` accepts unaligned source / aligned destination via byte
        //   pointers.
        unsafe {
            core::ptr::copy_nonoverlapping(
                consumed.as_ptr(),
                out.as_mut_ptr().cast::<u8>(),
                byte_len,
            );
            out.set_len(count);
        };
        Ok(out)
    }

    /// One-byte boolean (any non-zero is true).
    pub fn read_bool(&mut self) -> Result<bool, MemoryUnpackError> {
        Ok(self.read::<u8>()? != 0)
    }

    /// Single POD value.
    pub fn read<T: Pod>(&mut self) -> Result<T, MemoryUnpackError> {
        let elem_size = size_of::<T>();
        if elem_size > self.buffer.len() {
            return Err(MemoryUnpackError::pod_underrun::<T>(
                elem_size,
                self.buffer.len(),
            ));
        }
        let (chunk, rest) = self.buffer.split_at(elem_size);
        self.buffer = rest;
        Ok(bytemuck::pod_read_unaligned(chunk))
    }

    /// Optional POD with `u8` discriminant.
    pub fn read_option<T: Pod>(&mut self) -> Result<Option<T>, MemoryUnpackError> {
        if self.read::<u8>()? == 0 {
            Ok(None)
        } else {
            Ok(Some(self.read()?))
        }
    }

    /// Host string: UTF-16 LE code units with `i32` length. `-1` -> [`None`]. Surrogate halves or
    /// invalid sequences decode to the empty string (defensive; the host typically sends valid UTF-16).
    /// Lengths above [`MAX_STRING_LEN`] are rejected with [`MemoryUnpackError::StringTooLong`].
    pub fn read_str(&mut self) -> Result<Option<String>, MemoryUnpackError> {
        let len = self.read::<i32>()?;
        if len < 0 {
            return Ok(None);
        }
        if len == 0 {
            return Ok(Some(String::new()));
        }
        let len = len as usize;
        if len > MAX_STRING_LEN {
            return Err(MemoryUnpackError::StringTooLong {
                requested: len,
                max: MAX_STRING_LEN,
            });
        }
        let utf16: Vec<u16> = self.access::<u16>(len)?;
        Ok(Some(String::from_utf16(&utf16).unwrap_or_default()))
    }

    /// Eight booleans from one byte.
    pub fn read_packed_bools(&mut self) -> Result<PackedBools, MemoryUnpackError> {
        Ok(PackedBools::from_byte(self.read::<u8>()?))
    }

    /// Fills an existing `MemoryPackable` (no presence byte).
    pub fn read_object_required<T: MemoryPackable>(
        &mut self,
        obj: &mut T,
    ) -> Result<(), WireDecodeError> {
        obj.unpack(self)
    }

    /// Optional object with `u8` discriminant, allocated from `pool` when present.
    pub fn read_object<T: MemoryPackable + Default>(
        &mut self,
    ) -> Result<Option<T>, WireDecodeError> {
        if self.read::<u8>()? == 0 {
            return Ok(None);
        }
        let mut obj = self.pool.borrow::<T>();
        obj.unpack(self)?;
        Ok(Some(obj))
    }

    /// Object list; negative outer count is treated as empty (defensive).
    pub fn read_object_list<T: MemoryPackable + Default>(
        &mut self,
    ) -> Result<Vec<T>, WireDecodeError> {
        let count = self.read::<i32>()?;
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(alloc_hint(count, self.buffer.len()));
        for _ in 0..count {
            let mut obj = self.pool.borrow::<T>();
            obj.unpack(self)?;
            list.push(obj);
        }
        Ok(list)
    }

    /// Polymorphic list: `decode` reads discriminator and payload per element.
    pub fn read_polymorphic_list<F, T>(&mut self, mut decode: F) -> Result<Vec<T>, WireDecodeError>
    where
        F: FnMut(&mut MemoryUnpacker<'a, 'pool, P>) -> Result<T, WireDecodeError>,
    {
        let count = self.read::<i32>()?;
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(alloc_hint(count, self.buffer.len()));
        for _ in 0..count {
            list.push(decode(self)?);
        }
        Ok(list)
    }

    /// POD list.
    pub fn read_value_list<T: Pod>(&mut self) -> Result<Vec<T>, MemoryUnpackError> {
        let count = self.read::<i32>()?;
        let count = if count < 0 { 0 } else { count as usize };
        self.access::<T>(count)
    }

    /// Enum list stored as `i32` discriminants.
    pub fn read_enum_value_list<E: EnumRepr>(&mut self) -> Result<Vec<E>, MemoryUnpackError> {
        let count = self.read::<i32>()?;
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(alloc_hint(count, self.buffer.len()));
        for _ in 0..count {
            list.push(E::from_i32(self.read::<i32>()?));
        }
        Ok(list)
    }

    /// List of nullable strings.
    pub fn read_string_list(&mut self) -> Result<Vec<Option<String>>, MemoryUnpackError> {
        let count = self.read::<i32>()?;
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(alloc_hint(count, self.buffer.len()));
        for _ in 0..count {
            list.push(self.read_str()?);
        }
        Ok(list)
    }

    /// Nested value lists.
    pub fn read_nested_value_list<T: Pod>(&mut self) -> Result<Vec<Vec<T>>, MemoryUnpackError> {
        self.read_nested_list(MemoryUnpacker::read_value_list)
    }

    /// Nested list with custom inner reader.
    pub fn read_nested_list<F, T>(
        &mut self,
        mut sublist_reader: F,
    ) -> Result<Vec<T>, MemoryUnpackError>
    where
        F: FnMut(&mut MemoryUnpacker<'a, 'pool, P>) -> Result<T, MemoryUnpackError>,
    {
        let count = self.read::<i32>()?;
        let count = if count < 0 { 0 } else { count as usize };
        let mut list = Vec::with_capacity(alloc_hint(count, self.buffer.len()));
        for _ in 0..count {
            list.push(sublist_reader(self)?);
        }
        Ok(list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packing::default_entity_pool::DefaultEntityPool;
    use crate::packing::memory_packer::MemoryPacker;
    use crate::packing::polymorphic_memory_packable_entity::PolymorphicEncode;
    use crate::packing::wire_decode_error::WireDecodeError;

    fn pack(write: impl FnOnce(&mut MemoryPacker<'_>)) -> Vec<u8> {
        let mut buf = vec![0u8; 4096];
        let cap = buf.len();
        let written = {
            let mut p = MemoryPacker::new(&mut buf);
            write(&mut p);
            cap - p.remaining_len()
        };
        buf.truncate(written);
        buf
    }

    #[test]
    fn access_returns_empty_vec_for_zero_count() {
        let bytes = [1u8, 2, 3, 4];
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let out: Vec<u32> = u.access::<u32>(0).expect("zero count");
        assert!(out.is_empty());
        assert_eq!(u.remaining_data(), 4, "no bytes should be consumed");
    }

    #[test]
    fn access_returns_underrun_when_buffer_too_small() {
        let bytes = [0u8; 3];
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let err = u.access::<u32>(1).expect_err("should underrun");
        match err {
            MemoryUnpackError::Underrun {
                needed, remaining, ..
            } => {
                assert_eq!(needed, 4);
                assert_eq!(remaining, 3);
            }
            other => panic!("expected Underrun, got {other:?}"),
        }
    }

    #[test]
    fn access_returns_length_overflow_when_count_size_overflows_usize() {
        let bytes = [0u8; 16];
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let err = u
            .access::<u64>(usize::MAX)
            .expect_err("count * size_of::<u64>() must overflow");
        assert!(matches!(err, MemoryUnpackError::LengthOverflow));
    }

    #[test]
    fn access_handles_unaligned_source_for_pod() {
        let mut backing = [0u8; 9];
        backing[1..].copy_from_slice(&[0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55]);
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&backing[1..], &mut pool);
        let values: Vec<u32> = u.access::<u32>(2).expect("two u32s");
        assert_eq!(values, vec![0x1122_3344, 0x5566_7788]);
        assert_eq!(u.remaining_data(), 0);
    }

    #[test]
    fn read_str_negative_length_returns_none() {
        let bytes = pack(|p| p.write_str(None));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        assert_eq!(u.read_str().expect("str"), None);
    }

    #[test]
    fn read_str_zero_length_returns_some_empty() {
        let bytes = pack(|p| p.write_str(Some("")));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        assert_eq!(u.read_str().expect("str").as_deref(), Some(""));
    }

    #[test]
    fn read_str_above_max_returns_string_too_long() {
        let too_long = (MAX_STRING_LEN + 1) as i32;
        let bytes = pack(|p| p.write(&too_long));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let err = u.read_str().expect_err("must reject oversize string");
        match err {
            MemoryUnpackError::StringTooLong { requested, max } => {
                assert_eq!(requested, MAX_STRING_LEN + 1);
                assert_eq!(max, MAX_STRING_LEN);
            }
            other => panic!("expected StringTooLong, got {other:?}"),
        }
    }

    #[test]
    fn read_str_invalid_utf16_decodes_to_empty_string_defensively() {
        let mut bytes = pack(|p| p.write(&1i32));
        bytes.extend_from_slice(&[0x00, 0xD8]);
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        assert_eq!(
            u.read_str().expect("invalid utf16 must not error"),
            Some(String::new())
        );
    }

    #[test]
    fn read_value_list_negative_count_returns_empty() {
        let bytes = pack(|p| p.write(&-7i32));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let out: Vec<i32> = u.read_value_list().expect("decoded");
        assert!(out.is_empty());
    }

    #[test]
    fn read_object_list_negative_count_returns_empty() {
        let bytes = pack(|p| p.write(&-1i32));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let out: Vec<DummyObj> = u.read_object_list().expect("decoded");
        assert!(out.is_empty());
    }

    #[test]
    fn read_polymorphic_list_negative_count_returns_empty() {
        let bytes = pack(|p| p.write(&-3i32));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let out: Vec<i32> = u
            .read_polymorphic_list(|_| panic!("decode closure must not run for negative count"))
            .expect("decoded");
        assert!(out.is_empty());
    }

    #[test]
    fn read_enum_value_list_negative_count_returns_empty() {
        let bytes = pack(|p| p.write(&-5i32));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let out: Vec<DummyEnum> = u.read_enum_value_list().expect("decoded");
        assert!(out.is_empty());
    }

    #[test]
    fn read_string_list_negative_count_returns_empty() {
        let bytes = pack(|p| p.write(&-2i32));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let out = u.read_string_list().expect("decoded");
        assert!(out.is_empty());
    }

    #[test]
    fn read_nested_value_list_negative_count_returns_empty() {
        let bytes = pack(|p| p.write(&-9i32));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let out: Vec<Vec<u32>> = u.read_nested_value_list().expect("decoded");
        assert!(out.is_empty());
    }

    #[test]
    fn read_value_list_caps_preallocation_to_remaining_bytes_then_underruns() {
        let mut bytes = pack(|p| p.write(&i32::MAX));
        bytes.extend_from_slice(&[0u8; 2]);
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let err = u
            .read_value_list::<u32>()
            .expect_err("must underrun, not allocate gigabytes");
        assert!(matches!(err, MemoryUnpackError::Underrun { .. }));
    }

    #[test]
    fn read_object_list_caps_preallocation_to_remaining_bytes_then_underruns() {
        let bytes = pack(|p| p.write(&i32::MAX));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let err = u
            .read_object_list::<DummyObj>()
            .expect_err("must underrun, not allocate gigabytes");
        assert!(matches!(
            err,
            WireDecodeError::Unpack(MemoryUnpackError::Underrun { .. })
        ));
    }

    #[test]
    fn read_polymorphic_list_caps_preallocation_to_remaining_bytes_then_underruns() {
        let bytes = pack(|p| p.write(&i32::MAX));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let err = u
            .read_polymorphic_list::<_, i32>(|reader| {
                let v: i32 = reader.read().map_err(WireDecodeError::from)?;
                Ok(v)
            })
            .expect_err("must underrun, not allocate gigabytes");
        assert!(matches!(
            err,
            WireDecodeError::Unpack(MemoryUnpackError::Underrun { .. })
        ));
    }

    #[test]
    fn read_option_zero_discriminant_returns_none() {
        let bytes = pack(|p| p.write_option::<i32>(None));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        assert_eq!(u.read_option::<i32>().expect("opt"), None);
    }

    #[test]
    fn read_option_nonzero_discriminant_reads_value() {
        let bytes = pack(|p| p.write_option(Some(&0x55aau16)));
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        assert_eq!(u.read_option::<u16>().expect("opt"), Some(0x55aa));
    }

    #[test]
    fn read_packed_bools_round_trips_byte() {
        let bytes = pack(|p| {
            p.write_packed_bools_array([true, false, true, true, false, true, false, true]);
        });
        let mut pool = DefaultEntityPool;
        let mut u = MemoryUnpacker::new(&bytes, &mut pool);
        let pb = u.read_packed_bools().expect("packed bools");
        assert_eq!(
            (
                pb.bit0, pb.bit1, pb.bit2, pb.bit3, pb.bit4, pb.bit5, pb.bit6, pb.bit7,
            ),
            (true, false, true, true, false, true, false, true),
        );
    }

    #[derive(Debug, Default)]
    struct DummyObj {
        v: i32,
    }

    impl MemoryPackable for DummyObj {
        fn pack(&mut self, packer: &mut MemoryPacker<'_>) {
            packer.write(&self.v);
        }

        fn unpack<P: MemoryPackerEntityPool>(
            &mut self,
            unpacker: &mut MemoryUnpacker<'_, '_, P>,
        ) -> Result<(), WireDecodeError> {
            self.v = unpacker.read::<i32>()?;
            Ok(())
        }
    }

    impl PolymorphicEncode for DummyObj {
        fn encode(&mut self, packer: &mut MemoryPacker<'_>) {
            packer.write(&self.v);
        }
    }

    #[derive(Clone, Copy)]
    struct DummyEnum(i32);

    impl EnumRepr for DummyEnum {
        fn as_i32(self) -> i32 {
            self.0
        }
        fn from_i32(i: i32) -> Self {
            Self(i)
        }
    }
}
