//! [`SetCubemapData`](crate::shared::SetCubemapData) → cubemap array layers ([`super::mip_write_common::write_cubemap_face_mip`]).

use crate::shared::{SetCubemapData, SetCubemapFormat};

use super::super::decode::{decode_mip_to_rgba8, flip_mip_rows, needs_rgba8_decode_before_upload};
use super::super::layout::{
    flip_compressed_mip_block_rows_y, host_format_is_compressed, mip_byte_len,
    mip_dimensions_at_level, mip_tight_bytes_per_texel,
};
use super::error::TextureUploadError;
use super::mip_write_common::{
    is_rgba8_family, uncompressed_row_bytes, write_cubemap_face_mip, CubemapFaceMipWrite,
};
use super::write_mip_chain::MipChainAdvance;

/// Shared device, format, upload record, payload window, and mip start bias for cubemap chain walks.
struct CubemapMipChainState<'a> {
    fmt: &'a SetCubemapFormat,
    upload: &'a SetCubemapData,
    payload: &'a [u8],
    start_bias: usize,
}

/// Face index and mip dimensions for [`resolve_cubemap_face_mip_slice`].
struct CubemapFaceMipSliceStep {
    face: usize,
    mip_i: usize,
    w: u32,
    h: u32,
}

/// Host payload subslice for one cubemap face × mip after bias and length checks.
fn resolve_cubemap_face_mip_slice<'a>(
    chain: &CubemapMipChainState<'a>,
    step: CubemapFaceMipSliceStep,
) -> Result<&'a [u8], TextureUploadError> {
    let fmt = chain.fmt;
    let upload = chain.upload;
    let payload = chain.payload;
    let start_bias = chain.start_bias;
    let CubemapFaceMipSliceStep { face, mip_i, w, h } = step;
    let start_raw = upload.mip_starts[face][mip_i];
    if start_raw < 0 {
        return Err("negative mip_starts".into());
    }
    let start_abs = start_raw as usize;
    if start_abs < start_bias {
        return Err(TextureUploadError::from(format!(
            "mip start {} is before descriptor offset {}",
            start_abs, start_bias
        )));
    }
    let start = start_abs - start_bias;

    let host_len = mip_byte_len(fmt.format, w, h).ok_or_else(|| {
        TextureUploadError::from(format!(
            "cubemap mip byte size unsupported for {:?}",
            fmt.format
        ))
    })? as usize;

    payload
        .get(start..start + host_len)
        .ok_or_else(|| {
            TextureUploadError::from(format!(
                "cubemap {} face {} mip {mip_i}: slice out of range (start {start} len {host_len}, payload {})",
                upload.asset_id, face, payload.len()
            ))
        })
}

/// Converts host face mip bytes for [`write_cubemap_face_mip`] (decode, optional row flip).
fn cubemap_mip_src_to_upload_pixels(
    asset_id: i32,
    fmt_format: crate::shared::TextureFormat,
    wgpu_format: wgpu::TextureFormat,
    needs_rgba8_decode: bool,
    w: u32,
    h: u32,
    flip: bool,
    mip_i: usize,
    face: u32,
    mip_src: &[u8],
) -> Result<Vec<u8>, TextureUploadError> {
    let pixels = if is_rgba8_family(wgpu_format) {
        if needs_rgba8_decode || host_format_is_compressed(fmt_format) {
            decode_mip_to_rgba8(fmt_format, w, h, flip, mip_src).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "RGBA decode failed for cubemap face {face} mip {mip_i}"
                ))
            })?
        } else if flip {
            let mut v = mip_src.to_vec();
            let bpp = mip_tight_bytes_per_texel(v.len(), w, h).ok_or_else(|| {
                format!(
                    "cubemap mip {mip_i}: RGBA8 upload len {} not divisible by {}×{} texels",
                    v.len(),
                    w,
                    h
                )
            })?;
            if bpp != 4 {
                return Err(TextureUploadError::from(format!(
                    "cubemap mip {mip_i}: RGBA8 family expects 4 bytes per texel, got {bpp}"
                )));
            }
            flip_mip_rows(&mut v, w, h, bpp);
            v
        } else {
            mip_src.to_vec()
        }
    } else {
        if needs_rgba8_decode {
            return Err(TextureUploadError::from(format!(
                "host {:?} must use RGBA decode but GPU format is {:?}",
                fmt_format, wgpu_format
            )));
        }
        if flip && !host_format_is_compressed(fmt_format) {
            let mut v = mip_src.to_vec();
            let bpp_host = mip_tight_bytes_per_texel(v.len(), w, h).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "cubemap mip {mip_i}: len {} not divisible by {}×{} texels (flip_y)",
                    v.len(),
                    w,
                    h
                ))
            })?;
            if let Ok(bpp_gpu) = uncompressed_row_bytes(wgpu_format) {
                if bpp_host != bpp_gpu {
                    logger::warn!(
                        "cubemap {} face {face} mip {mip_i}: host texel stride {} B != GPU {:?} stride {} B",
                        asset_id,
                        bpp_host,
                        wgpu_format,
                        bpp_gpu
                    );
                }
            }
            flip_mip_rows(&mut v, w, h, bpp_host);
            v
        } else if flip && host_format_is_compressed(fmt_format) {
            let expected_len = mip_byte_len(fmt_format, w, h).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "cubemap {asset_id} face {face} mip {mip_i}: mip byte size unknown for {:?}",
                    fmt_format
                ))
            })? as usize;
            if mip_src.len() != expected_len {
                return Err(TextureUploadError::from(format!(
                    "cubemap {asset_id} face {face} mip {mip_i}: mip len {} != expected {} for {:?}",
                    mip_src.len(),
                    expected_len,
                    fmt_format
                )));
            }
            if let Some(v) = flip_compressed_mip_block_rows_y(fmt_format, w, h, mip_src) {
                v
            } else {
                logger::warn!(
                    "cubemap {asset_id} face {face} mip {mip_i}: flip_y skipped for compressed {:?} (vertical flip not implemented for this block-compressed format)",
                    fmt_format
                );
                mip_src.to_vec()
            }
        } else {
            mip_src.to_vec()
        }
    };
    Ok(pixels)
}

/// GPU and host view for one [`CubemapMipChainUploader::upload_next_face_mip`] step.
pub struct CubemapFaceMipUploadStep<'a> {
    /// Device for decode paths.
    pub device: &'a wgpu::Device,
    /// Queue for the face mip write.
    pub queue: &'a wgpu::Queue,
    /// Destination cubemap texture.
    pub texture: &'a wgpu::Texture,
    /// Host format.
    pub fmt: &'a SetCubemapFormat,
    /// GPU storage format.
    pub wgpu_format: wgpu::TextureFormat,
    /// Upload record.
    pub upload: &'a SetCubemapData,
    /// Payload (`&raw[..upload.data.length]`).
    pub payload: &'a std::sync::Arc<[u8]>,
}

/// Incremental cubemap upload: one face × one mip per step.
#[derive(Debug)]
pub struct CubemapMipChainUploader {
    face: u32,
    mip_i: usize,
    uploaded: u32,
    start_bias: usize,
    start_base: u32,
    mipmap_count: u32,
    face_size: u32,
    flip: bool,
    background_rx: Option<crossbeam_channel::Receiver<Result<Vec<u8>, TextureUploadError>>>,
    pending_mip: Option<(u32, u32, u32, u32)>, // face, mip_level, w, h
}

impl CubemapMipChainUploader {
    /// Validates `raw` / `upload` / `fmt` (no GPU work).
    pub fn new(
        texture: &wgpu::Texture,
        fmt: &SetCubemapFormat,
        upload: &SetCubemapData,
        raw: &[u8],
    ) -> Result<Self, TextureUploadError> {
        let want = upload.data.length.max(0) as usize;
        if raw.len() < want {
            return Err(TextureUploadError::from(format!(
                "raw shorter than descriptor (need {want}, got {})",
                raw.len()
            )));
        }

        if upload.mip_map_sizes.is_empty() {
            return Err("cubemap: no mips in upload".into());
        }
        if upload.mip_starts.len() != 6 {
            return Err(TextureUploadError::from(format!(
                "cubemap: expected mip_starts len 6 (faces), got {}",
                upload.mip_starts.len()
            )));
        }
        for (fi, starts) in upload.mip_starts.iter().enumerate() {
            if starts.len() != upload.mip_map_sizes.len() {
                return Err(TextureUploadError::from(format!(
                    "cubemap: face {fi} mip_starts len {} != mip_map_sizes len {}",
                    starts.len(),
                    upload.mip_map_sizes.len()
                )));
            }
        }

        let start_base = upload.start_mip_level.max(0) as u32;
        let mipmap_count = fmt.mipmap_count.max(1) as u32;
        if start_base >= mipmap_count {
            return Err(TextureUploadError::from(format!(
                "start_mip_level {start_base} >= mipmap_count {mipmap_count}"
            )));
        }

        let tex_extent = texture.size();
        let face_size = fmt.size.max(0) as u32;
        if tex_extent.width != face_size
            || tex_extent.height != face_size
            || tex_extent.depth_or_array_layers != 6
        {
            return Err(TextureUploadError::from(format!(
                "GPU cubemap {}×{}×{} does not match format face {} (asset {})",
                tex_extent.width,
                tex_extent.height,
                tex_extent.depth_or_array_layers,
                face_size,
                upload.asset_id
            )));
        }

        let payload_len = want;
        let (start_bias, _prefix) = choose_mip_start_bias_cubemap(fmt.format, upload, payload_len)?;

        Ok(Self {
            face: 0,
            mip_i: 0,
            uploaded: 0,
            start_bias,
            start_base,
            mipmap_count,
            face_size,
            flip: upload.flip_y,
            background_rx: None,
            pending_mip: None,
        })
    }

    /// Writes at most one face mip. `payload` is `&raw[..upload.data.length]`.
    pub fn upload_next_face_mip(
        &mut self,
        step: CubemapFaceMipUploadStep<'_>,
    ) -> Result<MipChainAdvance, TextureUploadError> {
        let CubemapFaceMipUploadStep {
            device,
            queue,
            texture,
            fmt,
            wgpu_format,
            upload,
            payload,
        } = step;
        if self.face >= 6 {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded,
            });
        }

        if let Some(rx) = &self.background_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.background_rx = None;
                    let pixels = res?;
                    let (face, mip_level, w, h) = self.pending_mip.take().unwrap();

                    write_cubemap_face_mip(&CubemapFaceMipWrite {
                        queue,
                        texture,
                        mip_level,
                        face_layer: face,
                        width: w,
                        height: h,
                        format: wgpu_format,
                        bytes: &pixels,
                    })?;

                    self.uploaded += 1;
                    self.mip_i += 1;
                    if self.mip_i >= upload.mip_map_sizes.len() {
                        self.face += 1;
                        self.mip_i = 0;
                    }

                    if self.face >= 6 {
                        return Ok(MipChainAdvance::Finished {
                            total_uploaded: self.uploaded,
                        });
                    }

                    return Ok(MipChainAdvance::UploadedOne {
                        total_uploaded: self.uploaded,
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

        let mip_i = self.mip_i;
        debug_assert!(mip_i < upload.mip_map_sizes.len());

        let sz = upload.mip_map_sizes[mip_i];
        let w = sz.x.max(0) as u32;
        let h = sz.y.max(0) as u32;
        let mip_level = self.start_base + mip_i as u32;
        if mip_level >= self.mipmap_count {
            return Err(TextureUploadError::from(format!(
                "cubemap mip_level {mip_level} exceeds format mipmap_count {}",
                self.mipmap_count
            )));
        }

        let (gw, gh) = mip_dimensions_at_level(self.face_size, self.face_size, mip_level);
        if w != gw || h != gh {
            return Err(TextureUploadError::from(format!(
                "cubemap {} mip {mip_level}: upload says {w}×{h} but GPU mip is {gw}×{gh}",
                upload.asset_id
            )));
        }

        let chain = CubemapMipChainState {
            fmt,
            upload,
            payload,
            start_bias: self.start_bias,
        };

        let mip_src = resolve_cubemap_face_mip_slice(
            &chain,
            CubemapFaceMipSliceStep {
                face: self.face as usize,
                mip_i,
                w,
                h,
            },
        )?;

        self.pending_mip = Some((self.face, mip_level, w, h));
        let offset = mip_src.as_ptr() as usize - payload.as_ptr() as usize;
        let len = mip_src.len();
        let mip_src_range = offset..offset + len;

        let (tx, rx) = crossbeam_channel::bounded(1);
        self.background_rx = Some(rx);

        let asset_id = upload.asset_id;
        let fmt_format = fmt.format;
        let needs_rgba8_decode = needs_rgba8_decode_before_upload(device, fmt_format);
        let payload_arc = std::sync::Arc::clone(payload);
        let flip = self.flip;
        let face = self.face;

        rayon::spawn(move || {
            let mip_src = &payload_arc[mip_src_range];
            let res = cubemap_mip_src_to_upload_pixels(
                asset_id,
                fmt_format,
                wgpu_format,
                needs_rgba8_decode,
                w,
                h,
                flip,
                mip_i,
                face,
                mip_src,
            );
            let _ = tx.send(res);
        });

        Ok(MipChainAdvance::YieldBackground)
    }
}

fn choose_mip_start_bias_cubemap(
    format: crate::shared::TextureFormat,
    upload: &SetCubemapData,
    payload_len: usize,
) -> Result<(usize, usize), TextureUploadError> {
    let offset_bias = upload.data.offset.max(0) as usize;
    let candidates = if offset_bias > 0 {
        [0usize, offset_bias]
    } else {
        [0usize, 0usize]
    };
    let mut best_bias = 0usize;
    let mut best_prefix = 0usize;
    for bias in candidates {
        let prefix = valid_cubemap_mip_prefix_len(format, upload, payload_len, bias)?;
        if prefix > best_prefix {
            best_prefix = prefix;
            best_bias = bias;
        }
    }
    if best_prefix == 0 {
        return Err(TextureUploadError::from(format!(
            "cubemap mip region exceeds shared memory descriptor (payload_len={}, descriptor_offset={})",
            payload_len, offset_bias
        )));
    }
    Ok((best_bias, best_prefix))
}

fn valid_cubemap_mip_prefix_len(
    format: crate::shared::TextureFormat,
    upload: &SetCubemapData,
    payload_len: usize,
    bias: usize,
) -> Result<usize, TextureUploadError> {
    let mut count = 0usize;
    'outer: for face in 0..6usize {
        for (i, sz) in upload.mip_map_sizes.iter().enumerate() {
            if sz.x <= 0 || sz.y <= 0 {
                return Err("non-positive mip dimensions".into());
            }
            let w = sz.x as u32;
            let h = sz.y as u32;
            let host_len = mip_byte_len(format, w, h).ok_or_else(|| {
                TextureUploadError::from(format!("mip byte size unsupported for {:?}", format))
            })? as usize;
            let starts = upload
                .mip_starts
                .get(face)
                .ok_or_else(|| TextureUploadError::from("cubemap mip_starts face missing"))?;
            let start_raw = *starts
                .get(i)
                .ok_or_else(|| TextureUploadError::from("cubemap mip_starts index"))?;
            if start_raw < 0 {
                break 'outer;
            }
            let start_abs = start_raw as usize;
            if start_abs < bias {
                break 'outer;
            }
            let start = start_abs - bias;
            let end = start
                .checked_add(host_len)
                .ok_or_else(|| TextureUploadError::from("mip end overflow"))?;
            if end > payload_len {
                break 'outer;
            }
            count += 1;
        }
    }
    Ok(count)
}
