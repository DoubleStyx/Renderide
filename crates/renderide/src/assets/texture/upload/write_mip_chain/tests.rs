//! Tests for 2D mip-chain upload conversion and state helpers.

use super::super::super::layout::mip_byte_len;
use super::super::mip_write_common::{upload_uses_storage_v_inversion, MipUploadFormatCtx};
use super::conversion::mip_src_to_upload_pixels;
use crate::shared::TextureFormat;
fn upload_ctx(fmt_format: TextureFormat, wgpu_format: wgpu::TextureFormat) -> MipUploadFormatCtx {
    MipUploadFormatCtx {
        asset_id: 77,
        fmt_format,
        wgpu_format,
        needs_rgba8_decode: false,
    }
}

#[test]
fn bc7_flip_y_uploads_bytes_unchanged_with_storage_orientation_hint() {
    let raw: Vec<u8> = (0..64).collect();
    let pixels = mip_src_to_upload_pixels(
        upload_ctx(TextureFormat::BC7, wgpu::TextureFormat::Bc7RgbaUnorm),
        8,
        8,
        true,
        &raw,
        0,
    )
    .expect("bc7 upload");

    assert_eq!(pixels.bytes, raw);
    assert!(pixels.storage_v_inverted);
}

#[test]
fn affected_native_compressed_flip_y_uses_storage_orientation_hint() {
    for (host_format, wgpu_format) in [
        (TextureFormat::BC6H, wgpu::TextureFormat::Bc6hRgbUfloat),
        (TextureFormat::BC7, wgpu::TextureFormat::Bc7RgbaUnorm),
        (TextureFormat::ETC2RGB, wgpu::TextureFormat::Etc2Rgb8Unorm),
        (
            TextureFormat::ETC2RGBA1,
            wgpu::TextureFormat::Etc2Rgb8A1Unorm,
        ),
        (
            TextureFormat::ETC2RGBA8,
            wgpu::TextureFormat::Etc2Rgba8Unorm,
        ),
    ] {
        let len = mip_byte_len(host_format, 8, 8).expect("compressed mip byte length");
        let raw: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let pixels =
            mip_src_to_upload_pixels(upload_ctx(host_format, wgpu_format), 8, 8, true, &raw, 0)
                .expect("affected native compressed upload");

        assert_eq!(
            pixels.bytes, raw,
            "{host_format:?} bytes should stay intact"
        );
        assert!(
            pixels.storage_v_inverted,
            "{host_format:?} should use shader-side storage compensation"
        );
    }
}

#[test]
fn bc1_flip_y_keeps_exact_compressed_flip_path() {
    let mut raw = vec![0u8; 32];
    raw[..16].fill(0x11);
    raw[16..].fill(0x22);
    let pixels = mip_src_to_upload_pixels(
        upload_ctx(TextureFormat::BC1, wgpu::TextureFormat::Bc1RgbaUnorm),
        8,
        8,
        true,
        &raw,
        0,
    )
    .expect("bc1 upload");

    assert_ne!(pixels.bytes, raw);
    assert!(!pixels.storage_v_inverted);
}

#[test]
fn bc3_flip_y_keeps_exact_compressed_flip_path() {
    let mut raw = vec![0u8; 64];
    raw[..32].fill(0x11);
    raw[32..].fill(0x22);
    let pixels = mip_src_to_upload_pixels(
        upload_ctx(TextureFormat::BC3, wgpu::TextureFormat::Bc3RgbaUnorm),
        8,
        8,
        true,
        &raw,
        0,
    )
    .expect("bc3 upload");

    assert_ne!(pixels.bytes, raw);
    assert!(!pixels.storage_v_inverted);
}

#[test]
fn storage_orientation_helper_only_applies_to_native_affected_compression() {
    assert!(upload_uses_storage_v_inversion(
        TextureFormat::BC7,
        wgpu::TextureFormat::Bc7RgbaUnorm,
        true
    ));
    assert!(!upload_uses_storage_v_inversion(
        TextureFormat::BC7,
        wgpu::TextureFormat::Rgba8Unorm,
        true
    ));
    assert!(!upload_uses_storage_v_inversion(
        TextureFormat::BC1,
        wgpu::TextureFormat::Bc1RgbaUnorm,
        true
    ));
    assert!(!upload_uses_storage_v_inversion(
        TextureFormat::BC7,
        wgpu::TextureFormat::Bc7RgbaUnorm,
        false
    ));
}
