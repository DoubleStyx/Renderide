//! Integration test: IPC wire primitives round-trip through the `renderide::shared` re-exports.
//!
//! Exercises the `MemoryPacker` / `MemoryUnpacker` boundary that every IPC-touching module depends
//! on, proving the re-exports in `crate::shared` stay ergonomic for external consumers.

use renderide::shared::default_entity_pool::DefaultEntityPool;
use renderide::shared::memory_packer::MemoryPacker;
use renderide::shared::memory_unpack_error::MemoryUnpackError;
use renderide::shared::memory_unpacker::MemoryUnpacker;

/// Packs and unpacks a small mixed sequence (`i32`, `u8`, `bool`, `Option<i32>`, several strings)
/// and asserts every value round-trips. Mirrors the in-crate pattern but routed through
/// `renderide::shared::*` so the re-export surface is covered.
#[test]
fn mixed_primitives_roundtrip() {
    let mut buf = [0u8; 256];
    let greeting = "hola";
    let utf16_like = "こんにちは";

    let written = {
        let mut p = MemoryPacker::new(&mut buf);
        p.write(&0x0102_0304_i32);
        p.write(&0xabu8);
        p.write_bool(true);
        p.write_bool(false);
        p.write_option::<i32>(None);
        p.write_option(Some(&-7_i32));
        p.write_str(None);
        p.write_str(Some(""));
        p.write_str(Some(greeting));
        p.write_str(Some(utf16_like));
        256 - p.remaining_len()
    };

    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf[..written], &mut pool);
    assert_eq!(u.read::<i32>().expect("i32"), 0x0102_0304);
    assert_eq!(u.read::<u8>().expect("u8"), 0xab);
    assert!(u.read_bool().expect("bool true"));
    assert!(!u.read_bool().expect("bool false"));
    assert_eq!(u.read_option::<i32>().expect("opt none"), None);
    assert_eq!(u.read_option::<i32>().expect("opt some"), Some(-7));
    assert_eq!(u.read_str().expect("str none"), None);
    assert_eq!(u.read_str().expect("str empty").as_deref(), Some(""));
    assert_eq!(u.read_str().expect("str ascii").as_deref(), Some(greeting));
    assert_eq!(
        u.read_str().expect("str utf16").as_deref(),
        Some(utf16_like)
    );
}

/// A buffer shorter than the next `i32` read must surface an `Underrun` error from the unpacker.
/// This is the single user-facing error type for truncated wire payloads, so embedders can match
/// on it directly.
#[test]
fn underrun_reports_underrun_variant() {
    // Two bytes cannot fit an `i32`; the pack side intentionally left them this short.
    let buf = [0x01_u8, 0x02];
    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf, &mut pool);
    let err = u.read::<i32>().expect_err("truncated read must fail");
    match err {
        MemoryUnpackError::Underrun {
            ty,
            needed,
            remaining,
        } => {
            assert_eq!(needed, 4);
            assert_eq!(remaining, 2);
            assert!(
                ty == "i32" || ty.ends_with("i32"),
                "unexpected type name in error: {ty:?}"
            );
        }
        other @ MemoryUnpackError::LengthOverflow => panic!("expected Underrun, got {other:?}"),
    }
}

/// A sequence of `write_bool` calls produces exactly one byte per bool and consumes the exact same
/// number of bytes on the unpack side; useful regression guard for wire compatibility.
#[test]
fn bool_packing_uses_fixed_byte_budget() {
    let mut buf = [0u8; 8];
    let full = buf.len();
    let written = {
        let mut p = MemoryPacker::new(&mut buf);
        p.write_bool(true);
        p.write_bool(false);
        p.write_bool(true);
        full - p.remaining_len()
    };
    assert_eq!(written, 3);

    let mut pool = DefaultEntityPool;
    let mut u = MemoryUnpacker::new(&buf[..written], &mut pool);
    assert!(u.read_bool().expect("bool 0"));
    assert!(!u.read_bool().expect("bool 1"));
    assert!(u.read_bool().expect("bool 2"));
}
