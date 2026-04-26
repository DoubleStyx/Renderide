//! Unit tests for [`super::layout`] (mip byte sizing and compressed mip row flips).

use glam::IVec2;

use super::layout::{
    block_extent, bytes_per_compressed_block, flip_compressed_mip_block_rows_y,
    flip_compressed_mip_block_rows_y_supported, mip_byte_len, mip_dimensions_at_level,
    mip_tight_bytes_per_texel, validate_mip_upload_layout,
};
use crate::shared::{SetTexture2DData, TextureFormat};

#[test]
fn validate_mip_layout_accepts_contiguous_payload() {
    let mut d = SetTexture2DData::default();
    d.data.length = 4 * 4 * 4 + 2 * 2 * 4; // rgba32 mip0 + mip1
    d.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
    d.mip_starts = vec![0, 64];
    assert!(validate_mip_upload_layout(TextureFormat::RGBA32, &d).is_ok());
}

#[test]
fn validate_mip_layout_rejects_overflow() {
    let mut d = SetTexture2DData::default();
    d.data.length = 10;
    d.mip_map_sizes = vec![IVec2::new(4, 4)];
    d.mip_starts = vec![0];
    assert!(validate_mip_upload_layout(TextureFormat::RGBA32, &d).is_err());
}

#[test]
fn bc1_mip0_128_bytes_for_32x32() {
    let b = mip_byte_len(TextureFormat::BC1, 32, 32).expect("bc1");
    assert_eq!(b, (32 / 4) * (32 / 4) * 8);
}

#[test]
fn rgba32_mip0_byte_len() {
    assert_eq!(
        mip_byte_len(TextureFormat::RGBA32, 16, 16).unwrap(),
        16 * 16 * 4
    );
}

#[test]
fn rgba_float_matches_rgba32_float_texel_size() {
    assert_eq!(mip_byte_len(TextureFormat::RGBAFloat, 1, 1).unwrap(), 16);
    assert_eq!(mip_tight_bytes_per_texel(16 * 4 * 4, 4, 4), Some(16));
}

#[test]
fn mip_dimensions_at_level_halves_each_step() {
    assert_eq!(mip_dimensions_at_level(114, 200, 0), (114, 200));
    assert_eq!(mip_dimensions_at_level(114, 200, 1), (57, 100));
    assert_eq!(mip_dimensions_at_level(114, 200, 2), (28, 50));
    assert_eq!(mip_dimensions_at_level(1, 1, 5), (1, 1));
}

#[test]
fn flip_bc1_block_rows_swaps_horizontal_block_bands() {
    let w = 8u32;
    let h = 8u32;
    let mut mip = vec![0u8; mip_byte_len(TextureFormat::BC1, w, h).unwrap() as usize];
    let row_b = (w.div_ceil(4) * 8) as usize;
    mip[..row_b].fill(0x10);
    mip[row_b..].fill(0x20);
    let flipped = flip_compressed_mip_block_rows_y(TextureFormat::BC1, w, h, &mip).expect("flip");
    assert!(flipped[..row_b].iter().all(|&b| b == 0x20));
    assert!(flipped[row_b..].iter().all(|&b| b == 0x10));
}

#[test]
fn flip_bc1_single_block_row_is_identity() {
    let mip = vec![0xabu8; 8];
    let out = flip_compressed_mip_block_rows_y(TextureFormat::BC1, 4, 4, &mip).expect("flip");
    assert_eq!(out, mip);
}

#[test]
fn flip_compressed_wrong_len_returns_none() {
    assert!(flip_compressed_mip_block_rows_y(TextureFormat::BC1, 4, 4, &[0u8; 4]).is_none());
}

#[test]
fn flip_bc1_intra_block_swaps_selector_rows() {
    let w = 4u32;
    let h = 4u32;
    let mut mip = vec![0u8; mip_byte_len(TextureFormat::BC1, w, h).unwrap() as usize];
    mip[4] = 0x01;
    mip[5] = 0x02;
    mip[6] = 0x03;
    mip[7] = 0x04;
    let flipped = flip_compressed_mip_block_rows_y(TextureFormat::BC1, w, h, &mip).expect("flip");
    assert_eq!(&flipped[4..8], &[0x04, 0x03, 0x02, 0x01]);
}

#[test]
fn flip_bc2_intra_block_swaps_alpha_and_color_rows() {
    let w = 4u32;
    let h = 4u32;
    let mut mip = vec![0u8; mip_byte_len(TextureFormat::BC2, w, h).unwrap() as usize];
    mip[0] = 0xa0;
    mip[1] = 0xa1;
    mip[6] = 0xb0;
    mip[7] = 0xb1;
    mip[12] = 0xc0;
    mip[15] = 0xc3;
    let flipped = flip_compressed_mip_block_rows_y(TextureFormat::BC2, w, h, &mip).expect("flip");
    assert_eq!(flipped[0], 0xb0);
    assert_eq!(flipped[1], 0xb1);
    assert_eq!(flipped[6], 0xa0);
    assert_eq!(flipped[7], 0xa1);
    assert_eq!(flipped[12], 0xc3);
    assert_eq!(flipped[15], 0xc0);
}

#[test]
fn flip_bc3_double_flip_restores_mip() {
    let w = 4u32;
    let h = 4u32;
    let mut mip = vec![0u8; mip_byte_len(TextureFormat::BC3, w, h).unwrap() as usize];
    for (i, b) in mip.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(17).wrapping_add(3);
    }
    let once = flip_compressed_mip_block_rows_y(TextureFormat::BC3, w, h, &mip).expect("flip");
    let twice = flip_compressed_mip_block_rows_y(TextureFormat::BC3, w, h, &once).expect("flip");
    assert_eq!(twice, mip);
}

#[test]
fn flip_bc4_double_flip_restores_mip() {
    let w = 4u32;
    let h = 4u32;
    let mut mip = vec![0u8; mip_byte_len(TextureFormat::BC4, w, h).unwrap() as usize];
    for (i, b) in mip.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(19).wrapping_add(5);
    }
    let once = flip_compressed_mip_block_rows_y(TextureFormat::BC4, w, h, &mip).expect("flip");
    let twice = flip_compressed_mip_block_rows_y(TextureFormat::BC4, w, h, &once).expect("flip");
    assert_eq!(twice, mip);
}

#[test]
fn flip_bc5_double_flip_restores_mip() {
    let w = 4u32;
    let h = 4u32;
    let mut mip = vec![0u8; mip_byte_len(TextureFormat::BC5, w, h).unwrap() as usize];
    for (i, b) in mip.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(23).wrapping_add(7);
    }
    let once = flip_compressed_mip_block_rows_y(TextureFormat::BC5, w, h, &mip).expect("flip");
    let twice = flip_compressed_mip_block_rows_y(TextureFormat::BC5, w, h, &once).expect("flip");
    assert_eq!(twice, mip);
}

#[test]
fn flip_compressed_bc6h_block_row_reorder_is_some() {
    // BC6H is now flipped at block-row granularity (no in-block flip; documented zigzag).
    let len = mip_byte_len(TextureFormat::BC6H, 4, 4).unwrap() as usize;
    let mip = vec![0u8; len];
    assert!(flip_compressed_mip_block_rows_y(TextureFormat::BC6H, 4, 4, &mip).is_some());
}

#[test]
fn flip_bc7_reorders_block_rows_without_inblock_flip() {
    // 2×2 grid of BC7 blocks (8×8 texels). Each block is 16 bytes; tag every byte in a block
    // with the block-row index so we can observe the reorder. The bytes inside each block must
    // be untouched (no in-block flip available for BC7).
    let bpb = bytes_per_compressed_block(TextureFormat::BC7).unwrap() as usize;
    let (bw, bh) = block_extent(TextureFormat::BC7);
    let blocks_x: u32 = 2;
    let blocks_y: u32 = 2;
    let w = blocks_x * bw;
    let h = blocks_y * bh;
    let mut src = vec![0u8; (blocks_x as usize) * (blocks_y as usize) * bpb];
    for byi in 0..blocks_y {
        for bxi in 0..blocks_x {
            let off = ((byi as usize) * (blocks_x as usize) + (bxi as usize)) * bpb;
            for (i, b) in src[off..off + bpb].iter_mut().enumerate() {
                *b = ((byi as u8) << 4) | (bxi as u8) | ((i as u8) << 1);
            }
        }
    }
    let dst = flip_compressed_mip_block_rows_y(TextureFormat::BC7, w, h, &src).expect("flips BC7");
    let row_stride = (blocks_x as usize) * bpb;
    // Block-row 0 of dst must equal block-row (blocks_y-1) of src, byte-for-byte.
    assert_eq!(&dst[0..row_stride], &src[row_stride..2 * row_stride]);
    assert_eq!(&dst[row_stride..2 * row_stride], &src[0..row_stride]);
}

#[test]
fn flip_compressed_astc_returns_none() {
    // ASTC stays unsupported (block sizes up to 12×12 → too-visible zigzag without a real decoder).
    let len = mip_byte_len(TextureFormat::ASTC8x8, 16, 16).unwrap() as usize;
    let mip = vec![0u8; len];
    assert!(flip_compressed_mip_block_rows_y(TextureFormat::ASTC8x8, 16, 16, &mip).is_none());
}

#[test]
fn flip_compressed_mip_block_rows_y_supported_4x4_blocks_only() {
    assert!(flip_compressed_mip_block_rows_y_supported(
        TextureFormat::BC1
    ));
    assert!(flip_compressed_mip_block_rows_y_supported(
        TextureFormat::BC5
    ));
    assert!(flip_compressed_mip_block_rows_y_supported(
        TextureFormat::BC6H
    ));
    assert!(flip_compressed_mip_block_rows_y_supported(
        TextureFormat::BC7
    ));
    assert!(flip_compressed_mip_block_rows_y_supported(
        TextureFormat::ETC2RGB
    ));
    assert!(flip_compressed_mip_block_rows_y_supported(
        TextureFormat::ETC2RGBA8
    ));
    assert!(!flip_compressed_mip_block_rows_y_supported(
        TextureFormat::ASTC4x4
    ));
    assert!(!flip_compressed_mip_block_rows_y_supported(
        TextureFormat::ASTC8x8
    ));
}
