//! Unit tests for [`super::memory_packer::MemoryPacker`] and [`super::memory_unpacker::MemoryUnpacker`]
//! wire primitives (host-compatible layout).

use super::default_entity_pool::DefaultEntityPool;
use super::memory_packer::MemoryPacker;
use super::memory_unpack_error::MemoryUnpackError;
use super::memory_unpacker::MemoryUnpacker;
use super::packed_bools::PackedBools;

#[test]
fn pack_unpack_bool_roundtrip() {
    let mut buf = [0u8; 8];
    let full = buf.len();
    {
        let mut p = MemoryPacker::new(&mut buf);
        p.write_bool(true);
        p.write_bool(false);
        assert_eq!(full - p.remaining_len(), 2);
    }
    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf[..2], &mut pool);
    assert!(u.read_bool().expect("bool"));
    assert!(!u.read_bool().expect("bool"));
}

#[test]
fn pack_unpack_i32_and_u8_roundtrip() {
    let mut buf = [0u8; 16];
    {
        let mut p = MemoryPacker::new(&mut buf);
        p.write(&42i32);
        p.write(&0xabu8);
    }
    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf, &mut pool);
    assert_eq!(u.read::<i32>().expect("i32"), 42);
    assert_eq!(u.read::<u8>().expect("u8"), 0xab);
}

#[test]
fn pack_unpack_option_roundtrip() {
    let mut buf = [0u8; 32];
    {
        let mut p = MemoryPacker::new(&mut buf);
        p.write_option::<i32>(None);
        p.write_option(Some(&7i32));
    }
    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf, &mut pool);
    assert_eq!(u.read_option::<i32>().expect("opt"), None);
    assert_eq!(u.read_option::<i32>().expect("opt"), Some(7));
}

#[test]
fn pack_unpack_str_none_empty_ascii_and_utf16() {
    const CAP: usize = 256;
    let mut buf = [0u8; CAP];
    let hi = "こんにちは";
    let written = {
        let mut p = MemoryPacker::new(&mut buf);
        p.write_str(None);
        p.write_str(Some(""));
        p.write_str(Some("ascii"));
        p.write_str(Some(hi));
        CAP - p.remaining_len()
    };
    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf[..written], &mut pool);
    assert_eq!(u.read_str().expect("s"), None);
    assert_eq!(u.read_str().expect("s").as_deref(), Some(""));
    assert_eq!(u.read_str().expect("s").as_deref(), Some("ascii"));
    assert_eq!(u.read_str().expect("s").as_deref(), Some(hi));
}

#[test]
fn read_truncated_mid_i32_returns_underrun() {
    let buf = [0x2a, 0x00];
    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf, &mut pool);
    let err = u.read::<i32>().unwrap_err();
    assert_eq!(err, MemoryUnpackError::pod_underrun::<i32>(4, 2));
}

#[test]
fn access_count_times_elem_size_overflow_returns_length_overflow() {
    let buf = [0u8; 4];
    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf, &mut pool);
    let err = u.access::<u16>(usize::MAX).unwrap_err();
    assert_eq!(err, MemoryUnpackError::LengthOverflow);
}

#[test]
fn packed_bools_from_byte_matches_bit_positions() {
    assert_eq!(
        PackedBools::from_byte(0x00),
        PackedBools {
            bit0: false,
            bit1: false,
            bit2: false,
            bit3: false,
            bit4: false,
            bit5: false,
            bit6: false,
            bit7: false,
        }
    );
    assert_eq!(
        PackedBools::from_byte(0xFF).eight(),
        (true, true, true, true, true, true, true, true)
    );
    assert_eq!(
        PackedBools::from_byte(0xAA).eight(),
        (false, true, false, true, false, true, false, true)
    );
}

#[test]
fn write_packed_bools_array_roundtrips_through_unpacker() {
    let mut buf = [0u8; 4];
    let pattern = [true, false, true, false, true, false, true, false];
    {
        let mut p = MemoryPacker::new(&mut buf);
        p.write_packed_bools_array(pattern);
    }
    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf[..1], &mut pool);
    let got = u.read_packed_bools().expect("packed");
    assert_eq!(
        got.eight(),
        (true, false, true, false, true, false, true, false)
    );
    assert_eq!(PackedBools::from_byte(buf[0]).eight(), got.eight());
}
