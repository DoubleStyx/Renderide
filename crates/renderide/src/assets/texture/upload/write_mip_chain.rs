//! Full mip chain path: decode, optional flip, [`super::mip_write_common::write_one_mip`] per level.

use crate::assets::texture::layout::host_mip_payload_byte_offset;
use crate::shared::{SetTexture2DData, SetTexture2DFormat};

use super::super::decode::{decode_mip_to_rgba8, flip_mip_rows, needs_rgba8_decode_before_upload};
use super::super::layout::{
    host_format_is_compressed, mip_byte_len, mip_dimensions_at_level, mip_tight_bytes_per_texel,
};
use super::mip_write_common::{
    choose_mip_start_bias, is_rgba8_family, uncompressed_row_bytes, write_one_mip,
};
use super::subregion::{hint_region_is_empty, try_write_texture2d_subregion};

/// Incremental full mip-chain upload: call [`Self::upload_next_mip`] until [`MipChainAdvance::Finished`].
#[derive(Debug)]
pub struct TextureMipChainUploader {
    next_i: usize,
    uploaded_mips: u32,
    start_bias: usize,
    start_base: u32,
    mipmap_count: u32,
    tex_extent: wgpu::Extent3d,
    flip: bool,
    stopped: bool,
}

/// Result of one [`TextureMipChainUploader::upload_next_mip`] step.
#[derive(Debug)]
pub enum MipChainAdvance {
    /// Uploaded a single mip; call again for the next level (same `payload` slice).
    UploadedOne,
    /// Chain complete (`total_uploaded` mips in this chain).
    Finished {
        /// Total mips successfully written in this chain.
        total_uploaded: u32,
    },
}

impl TextureMipChainUploader {
    /// Validates `raw` / `upload` / `fmt` against `texture` and prepares chain state (no GPU work).
    pub fn new(
        texture: &wgpu::Texture,
        fmt: &SetTexture2DFormat,
        upload: &SetTexture2DData,
        raw: &[u8],
    ) -> Result<Self, String> {
        let want = upload.data.length.max(0) as usize;
        if raw.len() < want {
            return Err(format!(
                "raw shorter than descriptor (need {want}, got {})",
                raw.len()
            ));
        }

        let start_base = upload.start_mip_level.max(0) as u32;
        let mipmap_count = fmt.mipmap_count.max(1) as u32;
        if start_base >= mipmap_count {
            return Err(format!(
                "start_mip_level {start_base} >= mipmap_count {mipmap_count}"
            ));
        }

        let flip = upload.flip_y;

        let tex_extent = texture.size();
        let fmt_w = fmt.width.max(0) as u32;
        let fmt_h = fmt.height.max(0) as u32;
        if tex_extent.width != fmt_w || tex_extent.height != fmt_h {
            return Err(format!(
                "GPU texture {}x{} does not match SetTexture2DFormat {}x{} for asset {}",
                tex_extent.width, tex_extent.height, fmt_w, fmt_h, upload.asset_id
            ));
        }

        if upload.mip_map_sizes.len() != upload.mip_starts.len() {
            return Err("mip_map_sizes and mip_starts length mismatch".into());
        }
        if upload.mip_map_sizes.is_empty() {
            return Err("no mips in upload".into());
        }

        let payload_len = want;
        let (start_bias, _valid_prefix_mips) =
            choose_mip_start_bias(fmt.format, upload, payload_len)?;
        if start_bias != 0 {
            logger::debug!(
                "texture {}: rebasing mip_starts by descriptor offset {}",
                upload.asset_id,
                start_bias
            );
        }

        Ok(Self {
            next_i: 0,
            uploaded_mips: 0,
            start_bias,
            start_base,
            mipmap_count,
            tex_extent,
            flip,
            stopped: false,
        })
    }

    /// Writes at most one mip level. `payload` must be `&raw[..upload.data.length]` for the same mapping as `new`.
    pub fn upload_next_mip(
        &mut self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        fmt: &SetTexture2DFormat,
        wgpu_format: wgpu::TextureFormat,
        upload: &SetTexture2DData,
        payload: &[u8],
    ) -> Result<MipChainAdvance, String> {
        if self.stopped {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        let flip = self.flip;
        if flip && host_format_is_compressed(fmt.format) && !is_rgba8_family(wgpu_format) {
            logger::warn!(
                "texture {}: flip_y unsupported for compressed GPU texture {:?}; mips may look upside-down",
                upload.asset_id,
                wgpu_format
            );
        }

        let tex_extent = self.tex_extent;
        let start_base = self.start_base;
        let mipmap_count = self.mipmap_count;
        let start_bias = self.start_bias;
        let (_bias_check, valid_prefix_mips) =
            choose_mip_start_bias(fmt.format, upload, payload.len())?;
        debug_assert_eq!(start_bias, _bias_check);

        let i = self.next_i;
        if i >= upload.mip_map_sizes.len() {
            if self.uploaded_mips == 0 {
                return Err("no mip levels uploaded".into());
            }
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        let sz = upload.mip_map_sizes[i];
        let w = sz.x.max(0) as u32;
        let h = sz.y.max(0) as u32;
        let mip_level = start_base + i as u32;
        if mip_level >= mipmap_count {
            return Err(format!(
                "upload mip {mip_level} exceeds texture mips {mipmap_count}"
            ));
        }

        let (gw, gh) = mip_dimensions_at_level(tex_extent.width, tex_extent.height, mip_level);
        if w != gw || h != gh {
            logger::debug!(
                "texture {} mip {mip_level}: mip_map_sizes {w}x{h} != GPU {gw}x{gh} (using GPU dimensions; base {}x{})",
                upload.asset_id,
                tex_extent.width,
                tex_extent.height
            );
        }

        let start_raw = upload.mip_starts[i];
        if start_raw < 0 {
            if self.uploaded_mips == 0 {
                return Err("negative mip_starts".into());
            }
            logger::warn!(
                "texture {}: uploaded {}/{} mips; stopping at mip {} because mip_starts is negative",
                upload.asset_id,
                self.uploaded_mips,
                upload.mip_map_sizes.len(),
                i
            );
            self.stopped = true;
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }
        let start_abs = start_raw as usize;
        if start_abs < start_bias {
            if self.uploaded_mips == 0 {
                return Err(format!(
                    "mip 0 start {} is before descriptor offset {}",
                    start_abs, start_bias
                ));
            }
            logger::warn!(
                "texture {}: uploaded {}/{} mips; stopping at mip {} because start {} is before descriptor offset {}",
                upload.asset_id,
                self.uploaded_mips,
                upload.mip_map_sizes.len(),
                i,
                start_abs,
                start_bias
            );
            self.stopped = true;
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }
        let start_rel = start_abs - start_bias;
        let start = host_mip_payload_byte_offset(fmt.format, start_rel).ok_or_else(|| {
            format!(
                "texture {} mip {mip_level}: mip start offset unsupported for {:?}",
                upload.asset_id, fmt.format
            )
        })?;
        let host_len = mip_byte_len(fmt.format, w, h)
            .ok_or_else(|| format!("mip byte size unsupported for {:?}", fmt.format))?
            as usize;
        let Some(mip_src) = payload.get(start..start + host_len) else {
            if self.uploaded_mips == 0 {
                return Err(format!(
                    "mip 0 slice out of range after rebasing by {start_bias} (payload_len={}, valid_prefix_mips={valid_prefix_mips})",
                    payload.len()
                ));
            }
            logger::warn!(
                "texture {}: uploaded {}/{} mips; stopping at mip {} because payload_len={} does not cover start={} len={} after rebasing by {}",
                upload.asset_id,
                self.uploaded_mips,
                upload.mip_map_sizes.len(),
                i,
                payload.len(),
                start,
                host_len,
                start_bias
            );
            self.stopped = true;
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        };

        let pixels: std::borrow::Cow<'_, [u8]> = if is_rgba8_family(wgpu_format) {
            if needs_rgba8_decode_before_upload(fmt.format) || host_format_is_compressed(fmt.format)
            {
                std::borrow::Cow::Owned(
                    decode_mip_to_rgba8(fmt.format, gw, gh, flip, mip_src).ok_or_else(|| {
                        format!("RGBA decode failed for mip {i} ({:?})", fmt.format)
                    })?,
                )
            } else if flip {
                let mut v = mip_src.to_vec();
                let bpp = mip_tight_bytes_per_texel(v.len(), gw, gh).ok_or_else(|| {
                    format!(
                        "mip {i}: RGBA8 upload len {} not divisible by {}×{} texels",
                        v.len(),
                        gw,
                        gh
                    )
                })?;
                if bpp != 4 {
                    return Err(format!(
                        "mip {i}: RGBA8 family expects 4 bytes per texel, got {bpp}"
                    ));
                }
                flip_mip_rows(&mut v, gw, gh, bpp);
                std::borrow::Cow::Owned(v)
            } else {
                std::borrow::Cow::Borrowed(mip_src)
            }
        } else {
            if needs_rgba8_decode_before_upload(fmt.format) {
                return Err(format!(
                    "host {:?} must use RGBA decode but GPU format is {:?}",
                    fmt.format, wgpu_format
                ));
            }
            if flip && !host_format_is_compressed(fmt.format) {
                let mut v = mip_src.to_vec();
                let bpp_host = mip_tight_bytes_per_texel(v.len(), gw, gh).ok_or_else(|| {
                    format!(
                        "mip {i}: len {} not divisible by {}×{} texels (cannot infer row stride for flip_y)",
                        v.len(),
                        gw,
                        gh
                    )
                })?;
                if let Ok(bpp_gpu) = uncompressed_row_bytes(wgpu_format) {
                    if bpp_host != bpp_gpu {
                        logger::warn!(
                            "texture {} mip {i}: host texel stride {} B != GPU {:?} stride {} B; flip_y uses host packing",
                            upload.asset_id,
                            bpp_host,
                            wgpu_format,
                            bpp_gpu
                        );
                    }
                }
                flip_mip_rows(&mut v, gw, gh, bpp_host);
                std::borrow::Cow::Owned(v)
            } else {
                if flip && host_format_is_compressed(fmt.format) {
                    logger::warn!(
                        "texture {} mip {i}: flip_y skipped for compressed {:?} GPU upload",
                        upload.asset_id,
                        wgpu_format
                    );
                }
                std::borrow::Cow::Borrowed(mip_src)
            }
        };

        write_one_mip(
            queue,
            texture,
            mip_level,
            gw,
            gh,
            wgpu_format,
            pixels.as_ref(),
        )?;
        self.uploaded_mips += 1;
        self.next_i += 1;

        if self.next_i >= upload.mip_map_sizes.len() {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        Ok(MipChainAdvance::UploadedOne)
    }
}

/// Result of [`texture_upload_start`]: either sub-region finished in one step or a mip-chain uploader is needed.
#[derive(Debug)]
pub enum TextureDataStart {
    /// Sub-region path completed (`n` is the mip-equivalent count from the subregion helper).
    SubregionComplete(u32),
    /// Full mip chain; call [`TextureMipChainUploader::upload_next_mip`] until [`MipChainAdvance::Finished`].
    MipChain(TextureMipChainUploader),
}

/// Classifies sub-region vs full mip chain and runs the sub-region upload when applicable.
pub fn texture_upload_start(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Result<TextureDataStart, String> {
    if upload.hint.has_region != 0 {
        if hint_region_is_empty(&upload.hint) {
            logger::trace!(
                "texture {}: TextureUploadHint.has_region set but region empty; skipping upload",
                upload.asset_id
            );
            return Ok(TextureDataStart::SubregionComplete(0));
        }
        match try_write_texture2d_subregion(queue, texture, fmt, wgpu_format, upload, raw) {
            Some(Ok(n)) => {
                logger::trace!(
                    "texture {}: sub-region texture upload ({} mips equivalent)",
                    upload.asset_id,
                    n
                );
                return Ok(TextureDataStart::SubregionComplete(n));
            }
            Some(Err(e)) => return Err(e),
            None => {
                logger::trace!(
                    "texture {}: TextureUploadHint.has_region set; using full mip upload path",
                    upload.asset_id
                );
            }
        }
    }
    Ok(TextureDataStart::MipChain(TextureMipChainUploader::new(
        texture, fmt, upload, raw,
    )?))
}

/// Uploads mips from `raw` (exact shared-memory descriptor window) into `texture` using `wgpu_format`.
pub fn write_texture2d_mips(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Result<u32, String> {
    let want = upload.data.length.max(0) as usize;
    if raw.len() < want {
        return Err(format!(
            "raw shorter than descriptor (need {want}, got {})",
            raw.len()
        ));
    }
    let payload = &raw[..want];

    match texture_upload_start(queue, texture, fmt, wgpu_format, upload, raw)? {
        TextureDataStart::SubregionComplete(n) => Ok(n),
        TextureDataStart::MipChain(mut uploader) => loop {
            match uploader.upload_next_mip(queue, texture, fmt, wgpu_format, upload, payload)? {
                MipChainAdvance::UploadedOne => {}
                MipChainAdvance::Finished { total_uploaded } => {
                    return Ok(total_uploaded);
                }
            }
        },
    }
}
