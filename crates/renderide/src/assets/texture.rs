//! Texture asset: CPU-side mip0 after host `SetTexture2D*` commands.
//!
//! GPU [`wgpu::TextureView`] creation lives in [`crate::gpu::GpuState::ensure_texture2d_gpu`].

use crate::shared::TextureFormat;

use super::Asset;
use super::AssetId;

/// Stored 2D texture data for GPU upload (mip level 0, RGBA8).
pub struct TextureAsset {
    /// Unique identifier for this texture.
    pub id: AssetId,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// Decoded mip0: `width * height * 4` bytes, sRGB-ready RGBA8.
    pub rgba8_mip0: Vec<u8>,
}

impl TextureAsset {
    /// Returns true when [`crate::gpu::GpuState::ensure_texture2d_gpu`] can build a GPU texture.
    pub fn ready_for_gpu(&self) -> bool {
        let expected = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        self.width > 0 && self.height > 0 && self.rgba8_mip0.len() >= expected
    }
}

impl Asset for TextureAsset {
    fn id(&self) -> AssetId {
        self.id
    }
}

/// Converts raw mip0 bytes from the host into tight RGBA8 (row-major, top row first after optional flip).
pub fn decode_texture_mip0_to_rgba8(
    format: TextureFormat,
    width: u32,
    height: u32,
    flip_y: bool,
    raw: &[u8],
) -> Option<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let count = w.checked_mul(h)?;
    match format {
        TextureFormat::rgba32 => {
            let need = count.checked_mul(4)?;
            if raw.len() < need {
                return None;
            }
            let mut out = raw[..need].to_vec();
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::bgra32 => {
            let need = count.checked_mul(4)?;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(need);
            for p in raw[..need].chunks_exact(4) {
                out.push(p[2]);
                out.push(p[1]);
                out.push(p[0]);
                out.push(p[3]);
            }
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::r8 => {
            let need = count;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(count * 4);
            for &g in &raw[..need] {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::alpha8 => {
            let need = count;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(count * 4);
            for &a in &raw[..need] {
                out.extend_from_slice(&[255, 255, 255, a]);
            }
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        _ => None,
    }
}

fn flip_rgba_image_rows(buf: &mut [u8], width: usize, height: usize) {
    let row = width.saturating_mul(4);
    if row == 0 || height < 2 {
        return;
    }
    let mut tmp = vec![0u8; row];
    for y in 0..height / 2 {
        let a = y * row;
        let b = (height - 1 - y) * row;
        let (before, after) = buf.split_at_mut(b);
        let row_a = &mut before[a..a + row];
        let row_b = &mut after[..row];
        tmp.copy_from_slice(row_a);
        row_a.copy_from_slice(row_b);
        row_b.copy_from_slice(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_texture_mip0_to_rgba8, flip_rgba_image_rows};
    use crate::shared::TextureFormat;

    #[test]
    fn rgba32_roundtrip_size() {
        let raw: Vec<u8> = vec![255, 0, 0, 255, 0, 255, 0, 255];
        let out =
            decode_texture_mip0_to_rgba8(TextureFormat::rgba32, 2, 1, false, &raw).expect("ok");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn flip_y_swaps_rows() {
        let mut v = vec![
            1, 0, 0, 0, 2, 0, 0, 0, //
            3, 0, 0, 0, 4, 0, 0, 0,
        ];
        flip_rgba_image_rows(&mut v, 2, 2);
        assert_eq!(v[0..4], [3, 0, 0, 0]);
        assert_eq!(v[4..8], [4, 0, 0, 0]);
    }
}
