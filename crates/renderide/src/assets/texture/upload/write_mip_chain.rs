//! Full mip chain path: decode, optional flip, [`super::mip_write_common::write_one_mip`] per level.

use std::sync::Arc;

use crate::assets::texture::layout::host_mip_payload_byte_offset;
use crate::shared::{SetTexture2DData, SetTexture2DFormat};

use super::super::decode::{decode_mip_to_rgba8, flip_mip_rows, needs_rgba8_decode_before_upload};
use super::super::layout::{
    flip_compressed_mip_block_rows_y, host_format_is_compressed, mip_byte_len,
    mip_dimensions_at_level, mip_tight_bytes_per_texel,
};
use super::error::TextureUploadError;
use super::mip_write_common::{
    choose_mip_start_bias, is_rgba8_family, uncompressed_row_bytes, write_one_mip,
    MipUploadFormatCtx,
};
use super::subregion::{hint_region_is_empty, try_write_texture2d_subregion};

/// Shared device, host format, and payload window for walking a 2D mip chain.
struct MipChainWalkState<'a> {
    fmt: &'a SetTexture2DFormat,
    upload: &'a SetTexture2DData,
    payload: &'a [u8],
    start_bias: usize,
}

/// Converts host mip bytes into a buffer suitable for [`write_one_mip`] (decode, optional row flip).
fn mip_src_to_upload_pixels(
    ctx: MipUploadFormatCtx,
    gw: u32,
    gh: u32,
    flip: bool,
    mip_src: &[u8],
    mip_index: usize,
) -> Result<Vec<u8>, TextureUploadError> {
    let MipUploadFormatCtx {
        asset_id,
        fmt_format,
        wgpu_format,
        needs_rgba8_decode,
    } = ctx;
    if is_rgba8_family(wgpu_format) {
        if needs_rgba8_decode || host_format_is_compressed(fmt_format) {
            decode_mip_to_rgba8(fmt_format, gw, gh, flip, mip_src).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "RGBA decode failed for mip {} ({:?})",
                    mip_index, fmt_format
                ))
            })
        } else if flip {
            let mut v = mip_src.to_vec();
            let bpp = mip_tight_bytes_per_texel(v.len(), gw, gh).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "mip {}: RGBA8 upload len {} not divisible by {}×{} texels",
                    mip_index,
                    v.len(),
                    gw,
                    gh
                ))
            })?;
            if bpp != 4 {
                return Err(TextureUploadError::from(format!(
                    "mip {}: RGBA8 family expects 4 bytes per texel, got {bpp}",
                    mip_index
                )));
            }
            flip_mip_rows(&mut v, gw, gh, bpp);
            Ok(v)
        } else {
            Ok(mip_src.to_vec())
        }
    } else if needs_rgba8_decode {
        Err(TextureUploadError::from(format!(
            "host {:?} must use RGBA decode but GPU format is {:?}",
            fmt_format, wgpu_format
        )))
    } else if flip && !host_format_is_compressed(fmt_format) {
        let mut v = mip_src.to_vec();
        let bpp_host = mip_tight_bytes_per_texel(v.len(), gw, gh).ok_or_else(|| {
            TextureUploadError::from(format!(
                "mip {}: len {} not divisible by {}×{} texels (cannot infer row stride for flip_y)",
                mip_index,
                v.len(),
                gw,
                gh
            ))
        })?;
        if let Ok(bpp_gpu) = uncompressed_row_bytes(wgpu_format) {
            if bpp_host != bpp_gpu {
                logger::warn!(
                    "texture {} mip {}: host texel stride {} B != GPU {:?} stride {} B; flip_y uses host packing",
                    asset_id,
                    mip_index,
                    bpp_host,
                    wgpu_format,
                    bpp_gpu
                );
            }
        }
        flip_mip_rows(&mut v, gw, gh, bpp_host);
        Ok(v)
    } else if flip && host_format_is_compressed(fmt_format) {
        let expected_len = mip_byte_len(fmt_format, gw, gh).ok_or_else(|| {
            TextureUploadError::from(format!(
                "texture {asset_id} mip {}: mip byte size unknown for {:?}",
                mip_index, fmt_format
            ))
        })? as usize;
        if mip_src.len() != expected_len {
            return Err(TextureUploadError::from(format!(
                "texture {asset_id} mip {}: mip len {} != expected {} for {:?}",
                mip_index,
                mip_src.len(),
                expected_len,
                fmt_format
            )));
        }
        if let Some(v) = flip_compressed_mip_block_rows_y(fmt_format, gw, gh, mip_src) {
            Ok(v)
        } else {
            logger::warn!(
                "texture {asset_id} mip {}: flip_y skipped for compressed {:?} (vertical flip not implemented for this block-compressed format)",
                mip_index,
                fmt_format
            );
            Ok(mip_src.to_vec())
        }
    } else {
        Ok(mip_src.to_vec())
    }
}

/// Outcome of [`validate_and_resolve_next_mip_slice`] for one [`TextureMipChainUploader`] step.
#[expect(
    variant_size_differences,
    reason = "short-lived per-mip outcome; boxing `Ready` would add an allocation per mip"
)]
enum NextMipUploadSlice<'a> {
    /// Stop iteration: chain finished normally (`uploaded_mips` may be zero only when no mip was ever uploaded — caller treats as error in that case).
    ChainDone { total_uploaded: u32 },
    /// Stop iteration: truncated payload or negative offset (`stopped` flag should be set on the uploader).
    ChainStopped { total_uploaded: u32 },
    /// GPU dimensions and source bytes for this mip level.
    Ready {
        mip_level: u32,
        gw: u32,
        gh: u32,
        mip_index: usize,
        mip_src: &'a [u8],
    },
}

/// Resolved host payload for one mip level before GPU dimensions from [`mip_dimensions_at_level`] are merged in.
enum HostMipPayloadResolved<'a> {
    /// Stop iteration: truncated payload or negative offset (`stopped` flag should be set on the uploader).
    Stopped { total_uploaded: u32 },
    /// Host payload subslice for this mip (dimensions come from [`validate_and_resolve_next_mip_slice`]).
    Slice { mip_src: &'a [u8] },
}

/// Per-mip indices for [`resolve_mip_host_payload_slice`].
struct MipHostPayloadResolveStep {
    uploaded_mips: u32,
    next_i: usize,
    mip_level: u32,
    w: u32,
    h: u32,
    valid_prefix_mips: usize,
}

/// Resolves host `mip_starts` (relative to descriptor), rebasing, and payload bounds to a mip subslice.
fn resolve_mip_host_payload_slice<'a>(
    chain: &MipChainWalkState<'a>,
    step: MipHostPayloadResolveStep,
) -> Result<HostMipPayloadResolved<'a>, TextureUploadError> {
    let fmt = chain.fmt;
    let upload = chain.upload;
    let payload = chain.payload;
    let start_bias = chain.start_bias;
    let MipHostPayloadResolveStep {
        uploaded_mips,
        next_i,
        mip_level,
        w,
        h,
        valid_prefix_mips,
    } = step;
    let start_raw = upload.mip_starts[next_i];
    if start_raw < 0 {
        if uploaded_mips == 0 {
            return Err("negative mip_starts".into());
        }
        logger::warn!(
            "texture {}: uploaded {}/{} mips; stopping at mip {} because mip_starts is negative",
            upload.asset_id,
            uploaded_mips,
            upload.mip_map_sizes.len(),
            next_i
        );
        return Ok(HostMipPayloadResolved::Stopped {
            total_uploaded: uploaded_mips,
        });
    }
    let start_abs = start_raw as usize;
    if start_abs < start_bias {
        if uploaded_mips == 0 {
            return Err(TextureUploadError::from(format!(
                "mip 0 start {start_abs} is before descriptor offset {start_bias}"
            )));
        }
        logger::warn!(
            "texture {}: uploaded {}/{} mips; stopping at mip {} because start {start_abs} is before descriptor offset {}",
            upload.asset_id,
            uploaded_mips,
            upload.mip_map_sizes.len(),
            next_i,
            start_bias
        );
        return Ok(HostMipPayloadResolved::Stopped {
            total_uploaded: uploaded_mips,
        });
    }
    let start_rel = start_abs - start_bias;
    let start = host_mip_payload_byte_offset(fmt.format, start_rel).ok_or_else(|| {
        TextureUploadError::from(format!(
            "texture {} mip {mip_level}: mip start offset unsupported for {:?}",
            upload.asset_id, fmt.format
        ))
    })?;
    let host_len = mip_byte_len(fmt.format, w, h).ok_or_else(|| {
        TextureUploadError::from(format!("mip byte size unsupported for {:?}", fmt.format))
    })? as usize;
    let Some(mip_src) = payload.get(start..start + host_len) else {
        if uploaded_mips == 0 {
            return Err(TextureUploadError::from(format!(
                "mip 0 slice out of range after rebasing by {start_bias} (payload_len={}, valid_prefix_mips={valid_prefix_mips})",
                payload.len()
            )));
        }
        logger::warn!(
            "texture {}: uploaded {}/{} mips; stopping at mip {} because payload_len={} does not cover start={} len={} after rebasing by {}",
            upload.asset_id,
            uploaded_mips,
            upload.mip_map_sizes.len(),
            next_i,
            payload.len(),
            start,
            host_len,
            start_bias
        );
        return Ok(HostMipPayloadResolved::Stopped {
            total_uploaded: uploaded_mips,
        });
    };

    Ok(HostMipPayloadResolved::Slice { mip_src })
}

/// Validates mip metadata, descriptor-relative offsets, and payload bounds for the current mip index.
fn validate_and_resolve_next_mip_slice<'a>(
    chain: &MipChainWalkState<'a>,
    uploaded_mips: u32,
    next_i: usize,
    start_base: u32,
    mipmap_count: u32,
    tex_extent: wgpu::Extent3d,
) -> Result<NextMipUploadSlice<'a>, TextureUploadError> {
    let fmt = chain.fmt;
    let upload = chain.upload;
    let payload = chain.payload;
    let start_bias = chain.start_bias;
    let (_bias_check, valid_prefix_mips) =
        choose_mip_start_bias(fmt.format, upload, payload.len())?;
    debug_assert_eq!(start_bias, _bias_check);

    if next_i >= upload.mip_map_sizes.len() {
        if uploaded_mips == 0 {
            return Err("no mip levels uploaded".into());
        }
        return Ok(NextMipUploadSlice::ChainDone {
            total_uploaded: uploaded_mips,
        });
    }

    let sz = upload.mip_map_sizes[next_i];
    let w = sz.x.max(0) as u32;
    let h = sz.y.max(0) as u32;
    let mip_level = start_base + next_i as u32;
    if mip_level >= mipmap_count {
        return Err(TextureUploadError::from(format!(
            "upload mip {mip_level} exceeds texture mips {mipmap_count}"
        )));
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

    match resolve_mip_host_payload_slice(
        chain,
        MipHostPayloadResolveStep {
            uploaded_mips,
            next_i,
            mip_level,
            w,
            h,
            valid_prefix_mips,
        },
    )? {
        HostMipPayloadResolved::Stopped { total_uploaded } => {
            Ok(NextMipUploadSlice::ChainStopped { total_uploaded })
        }
        HostMipPayloadResolved::Slice { mip_src } => Ok(NextMipUploadSlice::Ready {
            mip_level,
            gw,
            gh,
            mip_index: next_i,
            mip_src,
        }),
    }
}

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
    generating_tail: bool,
    last_rgba8_mip: Option<Rgba8Mip>,
    background_rx: Option<crossbeam_channel::Receiver<Result<Vec<u8>, TextureUploadError>>>,
    pending_mip: Option<(u32, u32, u32)>, // mip_level, gw, gh
}

/// Result of one [`TextureMipChainUploader::upload_next_mip`] step.
#[derive(Debug)]
pub enum MipChainAdvance {
    /// Uploaded or generated a single mip; call again for the next level (same `payload` slice).
    UploadedOne {
        /// Total mips successfully written in this chain.
        total_uploaded: u32,
    },
    /// Chain complete (`total_uploaded` mips in this chain).
    Finished {
        /// Total mips successfully written in this chain.
        total_uploaded: u32,
    },
    /// Waiting on background decoding thread. Call again next tick.
    YieldBackground,
}

/// GPU device, queue, and host upload view for one [`TextureMipChainUploader::upload_next_mip`] step.
pub struct TextureMipUploadStep<'a> {
    /// Device for decode paths.
    pub device: &'a wgpu::Device,
    /// Queue for [`write_one_mip`].
    pub queue: &'a wgpu::Queue,
    /// Destination texture.
    pub texture: &'a wgpu::Texture,
    /// Host format.
    pub fmt: &'a SetTexture2DFormat,
    /// Resolved GPU format.
    pub wgpu_format: wgpu::TextureFormat,
    /// Upload record.
    pub upload: &'a SetTexture2DData,
    /// Payload (`&raw[..upload.data.length]`).
    pub payload: &'a Arc<[u8]>,
}

#[derive(Clone, Debug)]
struct Rgba8Mip {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl TextureMipChainUploader {
    /// Validates `raw` / `upload` / `fmt` against `texture` and prepares chain state (no GPU work).
    pub fn new(
        texture: &wgpu::Texture,
        fmt: &SetTexture2DFormat,
        upload: &SetTexture2DData,
        raw: &[u8],
    ) -> Result<Self, TextureUploadError> {
        let want = upload.data.length.max(0) as usize;
        if raw.len() < want {
            return Err(TextureUploadError::from(format!(
                "raw shorter than descriptor (need {want}, got {})",
                raw.len()
            )));
        }

        let start_base = upload.start_mip_level.max(0) as u32;
        let mipmap_count = fmt.mipmap_count.max(1) as u32;
        if start_base >= mipmap_count {
            return Err(TextureUploadError::from(format!(
                "start_mip_level {start_base} >= mipmap_count {mipmap_count}"
            )));
        }

        let flip = upload.flip_y;

        let tex_extent = texture.size();
        let fmt_w = fmt.width.max(0) as u32;
        let fmt_h = fmt.height.max(0) as u32;
        if tex_extent.width != fmt_w || tex_extent.height != fmt_h {
            return Err(TextureUploadError::from(format!(
                "GPU texture {}x{} does not match SetTexture2DFormat {}x{} for asset {}",
                tex_extent.width, tex_extent.height, fmt_w, fmt_h, upload.asset_id
            )));
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
            generating_tail: false,
            last_rgba8_mip: None,
            background_rx: None,
            pending_mip: None,
        })
    }

    /// Writes at most one mip level. `payload` must be `&raw[..upload.data.length]` for the same mapping as `new`.
    pub fn upload_next_mip(
        &mut self,
        step: TextureMipUploadStep<'_>,
    ) -> Result<MipChainAdvance, TextureUploadError> {
        let TextureMipUploadStep {
            device,
            queue,
            texture,
            fmt,
            wgpu_format,
            upload,
            payload,
        } = step;
        if self.stopped {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        if let Some(rx) = &self.background_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.background_rx = None;
                    let pixels = res?;
                    let (mip_level, gw, gh) = self.pending_mip.take().ok_or_else(|| {
                        TextureUploadError::from(
                            "write_mip_chain: background decode completed without a pending mip slot; state machine desync",
                        )
                    })?;

                    write_one_mip(queue, texture, mip_level, gw, gh, wgpu_format, &pixels)?;

                    if is_rgba8_family(wgpu_format) {
                        self.last_rgba8_mip = Some(Rgba8Mip {
                            width: gw,
                            height: gh,
                            pixels,
                        });
                    }
                    self.uploaded_mips += 1;
                    self.next_i += 1;

                    if self.start_base + self.next_i as u32 >= self.mipmap_count {
                        self.stopped = true;
                        return Ok(MipChainAdvance::Finished {
                            total_uploaded: self.uploaded_mips,
                        });
                    }
                    return Ok(MipChainAdvance::UploadedOne {
                        total_uploaded: self.uploaded_mips,
                    });
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    return Ok(MipChainAdvance::YieldBackground);
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    return Err(TextureUploadError::from(
                        "Background decode thread panicked",
                    ));
                }
            }
        }

        let flip = self.flip;

        let tex_extent = self.tex_extent;
        let start_base = self.start_base;
        let mipmap_count = self.mipmap_count;
        let start_bias = self.start_bias;
        let i = self.next_i;

        let chain = MipChainWalkState {
            fmt,
            upload,
            payload,
            start_bias,
        };
        let slice = validate_and_resolve_next_mip_slice(
            &chain,
            self.uploaded_mips,
            i,
            start_base,
            mipmap_count,
            tex_extent,
        )?;
        let (mip_level, gw, gh, mip_index, mip_src_range) = match slice {
            NextMipUploadSlice::ChainDone { total_uploaded } => {
                if self.start_base + (self.next_i as u32) < self.mipmap_count {
                    //review: stopping here leaves undefined mips on some Unity uploads; synthesize the tail when we can.
                    return self.spawn_generated_tail_mip(wgpu_format, upload);
                }
                self.stopped = true;
                return Ok(MipChainAdvance::Finished { total_uploaded });
            }
            NextMipUploadSlice::ChainStopped { total_uploaded } => {
                if self.start_base + (self.next_i as u32) < self.mipmap_count
                    && self.last_rgba8_mip.is_some()
                {
                    return self.spawn_generated_tail_mip(wgpu_format, upload);
                }
                self.stopped = true;
                return Ok(MipChainAdvance::Finished { total_uploaded });
            }
            NextMipUploadSlice::Ready {
                mip_level,
                gw,
                gh,
                mip_index,
                mip_src,
            } => {
                let offset = mip_src.as_ptr() as usize - payload.as_ptr() as usize;
                let len = mip_src.len();
                (mip_level, gw, gh, mip_index, offset..offset + len)
            }
        };

        self.pending_mip = Some((mip_level, gw, gh));

        let (tx, rx) = crossbeam_channel::bounded(1);
        self.background_rx = Some(rx);

        let fmt_format = fmt.format;
        let needs_rgba8_decode = needs_rgba8_decode_before_upload(device, fmt_format);
        let payload_arc = Arc::clone(payload);
        let asset_id = upload.asset_id;

        let ctx = MipUploadFormatCtx {
            asset_id,
            fmt_format,
            wgpu_format,
            needs_rgba8_decode,
        };
        rayon::spawn(move || {
            let mip_src = &payload_arc[mip_src_range];
            let res = mip_src_to_upload_pixels(ctx, gw, gh, flip, mip_src, mip_index);
            let _ = tx.send(res);
        });

        Ok(MipChainAdvance::YieldBackground)
    }

    fn spawn_generated_tail_mip(
        &mut self,
        wgpu_format: wgpu::TextureFormat,
        upload: &SetTexture2DData,
    ) -> Result<MipChainAdvance, TextureUploadError> {
        let mip_level = self.start_base + self.next_i as u32;
        if mip_level >= self.mipmap_count {
            self.stopped = true;
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        if !is_rgba8_family(wgpu_format) {
            self.stopped = true;
            logger::warn!(
                "texture {}: uploaded {}/{} mips; cannot synthesize remaining tail for GPU format {:?}",
                upload.asset_id,
                self.uploaded_mips,
                self.mipmap_count.saturating_sub(self.start_base),
                wgpu_format
            );
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        let Some(source) = self.last_rgba8_mip.clone() else {
            self.stopped = true;
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        };

        if !self.generating_tail {
            logger::warn!(
                "texture {}: uploaded {}/{} host mips; synthesizing missing RGBA8 tail mips",
                upload.asset_id,
                self.uploaded_mips,
                self.mipmap_count.saturating_sub(self.start_base)
            );
            self.generating_tail = true;
        }

        let (w, h) =
            mip_dimensions_at_level(self.tex_extent.width, self.tex_extent.height, mip_level);

        self.pending_mip = Some((mip_level, w, h));

        let (tx, rx) = crossbeam_channel::bounded(1);
        self.background_rx = Some(rx);

        rayon::spawn(move || {
            let res = downsample_rgba8_box(&source.pixels, source.width, source.height, w, h);
            let _ = tx.send(res);
        });

        Ok(MipChainAdvance::YieldBackground)
    }
}

fn downsample_rgba8_box(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, TextureUploadError> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Err("zero-sized RGBA8 mip".into());
    }
    let expected = (src_w as usize)
        .checked_mul(src_h as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| TextureUploadError::from("RGBA8 mip byte size overflow"))?;
    if src.len() != expected {
        return Err(TextureUploadError::from(format!(
            "RGBA8 mip len {} != expected {} ({}x{})",
            src.len(),
            expected,
            src_w,
            src_h
        )));
    }

    let dst_len = (dst_w as usize)
        .checked_mul(dst_h as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| TextureUploadError::from("RGBA8 target mip byte size overflow"))?;
    let mut out = vec![0u8; dst_len];
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;

    for dy in 0..dh {
        let y0 = dy * sh / dh;
        let y1 = ((dy + 1) * sh).div_ceil(dh).max(y0 + 1).min(sh);
        for dx in 0..dw {
            let x0 = dx * sw / dw;
            let x1 = ((dx + 1) * sw).div_ceil(dw).max(x0 + 1).min(sw);
            let mut sum = [0u32; 4];
            let mut count = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let si = (sy * sw + sx) * 4;
                    sum[0] += u32::from(src[si]);
                    sum[1] += u32::from(src[si + 1]);
                    sum[2] += u32::from(src[si + 2]);
                    sum[3] += u32::from(src[si + 3]);
                    count += 1;
                }
            }
            let di = (dy * dw + dx) * 4;
            out[di] = ((sum[0] + count / 2) / count) as u8;
            out[di + 1] = ((sum[1] + count / 2) / count) as u8;
            out[di + 2] = ((sum[2] + count / 2) / count) as u8;
            out[di + 3] = ((sum[3] + count / 2) / count) as u8;
        }
    }

    Ok(out)
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
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Result<TextureDataStart, TextureUploadError> {
    if upload.hint.has_region != 0 {
        if hint_region_is_empty(&upload.hint) {
            logger::trace!(
                "texture {}: TextureUploadHint.has_region set but region empty; skipping upload",
                upload.asset_id
            );
            return Ok(TextureDataStart::SubregionComplete(0));
        }
        match try_write_texture2d_subregion(device, queue, texture, fmt, wgpu_format, upload, raw) {
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
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Result<u32, TextureUploadError> {
    let want = upload.data.length.max(0) as usize;
    if raw.len() < want {
        return Err(TextureUploadError::from(format!(
            "raw shorter than descriptor (need {want}, got {})",
            raw.len()
        )));
    }
    let payload = Arc::from(&raw[..want]);

    match texture_upload_start(device, queue, texture, fmt, wgpu_format, upload, raw)? {
        TextureDataStart::SubregionComplete(n) => Ok(n),
        TextureDataStart::MipChain(mut uploader) => loop {
            match uploader.upload_next_mip(TextureMipUploadStep {
                device,
                queue,
                texture,
                fmt,
                wgpu_format,
                upload,
                payload: &payload,
            })? {
                MipChainAdvance::UploadedOne { .. } => {}
                MipChainAdvance::Finished { total_uploaded } => {
                    return Ok(total_uploaded);
                }
                MipChainAdvance::YieldBackground => {
                    std::thread::yield_now();
                }
            }
        },
    }
}
