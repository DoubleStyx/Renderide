//! [`SetTexture3DData`](crate::shared::SetTexture3DData) → [`wgpu::Queue::write_texture`] for [`wgpu::TextureDimension::D3`].

use std::sync::Arc;

use crate::shared::{SetTexture3DData, SetTexture3DFormat};

use super::super::decode::{decode_mip_to_rgba8, needs_rgba8_decode_before_upload};
use super::super::layout::{host_format_is_compressed, mip_byte_len, mip_dimensions_at_level_3d};
use super::error::TextureUploadError;
use super::mip_write_common::{
    is_rgba8_family, write_texture3d_volume_mip, MipUploadFormatCtx, Texture3dVolumeMipWrite,
};

/// Per-level 3D geometry bundle: volume dimensions plus the tight slice / volume byte sizes for
/// one mip level.
#[derive(Copy, Clone)]
struct Texture3dMipGeom {
    /// Width of the mip in texels.
    w: u32,
    /// Height of the mip in texels.
    h: u32,
    /// Depth of the mip in texels.
    d: u32,
    /// Mip level index for diagnostics.
    level_idx: u32,
    /// Tight byte length of one depth slice (stride × height).
    slice_bytes: usize,
    /// Tight byte length of the full volume (`slice_bytes × d`).
    vol_bytes: usize,
}

/// Host mip dimensions, flat payload slice, and slice/volume byte sizes for one 3D level.
type Texture3dMipPayload<'a> = (u32, u32, u32, &'a [u8], usize, usize);

/// Byte offset in a tight mip chain for `level` (sum of prior level volumes).
fn texture3d_chain_byte_offset_to_level(
    base_w: u32,
    base_h: u32,
    base_d: u32,
    level: u32,
    format: crate::shared::TextureFormat,
    asset_id: i32,
) -> Result<usize, TextureUploadError> {
    let mut offset = 0usize;
    for l in 0..level {
        let (lw, lh, ld) = mip_dimensions_at_level_3d(base_w, base_h, base_d, l);
        let slice = mip_byte_len(format, lw, lh).ok_or_else(|| {
            TextureUploadError::from(format!(
                "texture3d {}: mip byte size unsupported for {:?}",
                asset_id, format
            ))
        })? as usize;
        let vol = slice
            .checked_mul(ld as usize)
            .ok_or_else(|| TextureUploadError::from("texture3d offset overflow"))?;
        offset = offset
            .checked_add(vol)
            .ok_or_else(|| TextureUploadError::from("texture3d offset overflow"))?;
    }
    Ok(offset)
}

/// Host payload subslice for one 3D mip level (full volume), with flat sizes for decode/upload.
fn texture3d_mip_volume_payload_slice<'a>(
    base_w: u32,
    base_h: u32,
    base_d: u32,
    level: u32,
    fmt: &SetTexture3DFormat,
    upload: &SetTexture3DData,
    payload: &'a [u8],
) -> Result<Texture3dMipPayload<'a>, TextureUploadError> {
    let (w, h, d) = mip_dimensions_at_level_3d(base_w, base_h, base_d, level);

    let offset = texture3d_chain_byte_offset_to_level(
        base_w,
        base_h,
        base_d,
        level,
        fmt.format,
        upload.asset_id,
    )?;

    let slice_bytes = mip_byte_len(fmt.format, w, h).ok_or_else(|| {
        TextureUploadError::from(format!(
            "texture3d {}: mip byte size unsupported for {:?}",
            upload.asset_id, fmt.format
        ))
    })? as usize;
    let vol_bytes = slice_bytes
        .checked_mul(d as usize)
        .ok_or_else(|| TextureUploadError::from("texture3d volume bytes overflow"))?;

    let mip_src = payload.get(offset..offset + vol_bytes).ok_or_else(|| {
        TextureUploadError::from(format!(
            "texture3d {}: mip {level} slice out of range (offset {offset} len {vol_bytes} payload {})",
            upload.asset_id,
            payload.len()
        ))
    })?;
    Ok((w, h, d, mip_src, slice_bytes, vol_bytes))
}

/// Prepares decoded RGBA8 slab or passes raw host bytes through for 3D volume upload.
fn texture3d_mip_to_upload_pixels(
    ctx: MipUploadFormatCtx,
    geom: Texture3dMipGeom,
    mip_src: &[u8],
) -> Result<Vec<u8>, TextureUploadError> {
    let MipUploadFormatCtx {
        asset_id,
        fmt_format,
        wgpu_format,
        needs_rgba8_decode,
    } = ctx;
    let Texture3dMipGeom {
        w,
        h,
        d,
        level_idx,
        slice_bytes,
        vol_bytes,
    } = geom;
    let pixels = if is_rgba8_family(wgpu_format) {
        if needs_rgba8_decode || host_format_is_compressed(fmt_format) {
            let mut out = Vec::with_capacity(vol_bytes);
            let mut z_off = 0usize;
            for _z in 0..d {
                let slice_raw = mip_src
                    .get(z_off..z_off + slice_bytes)
                    .ok_or_else(|| TextureUploadError::from("texture3d slice bounds"))?;
                let decoded =
                    decode_mip_to_rgba8(fmt_format, w, h, false, slice_raw).ok_or_else(|| {
                        TextureUploadError::from(format!(
                            "texture3d {}: RGBA decode failed mip {level_idx}",
                            asset_id
                        ))
                    })?;
                out.extend_from_slice(&decoded);
                z_off += slice_bytes;
            }
            out
        } else {
            mip_src.to_vec()
        }
    } else {
        if needs_rgba8_decode {
            return Err(TextureUploadError::from(format!(
                "texture3d {}: host {:?} must decode to RGBA but GPU format is {:?}",
                asset_id, fmt_format, wgpu_format
            )));
        }
        mip_src.to_vec()
    };
    Ok(pixels)
}

/// GPU device, queue, and host upload view for one [`Texture3dMipChainUploader::upload_next_mip`] step.
pub struct Texture3dMipUploadStep<'a> {
    /// Device for format capability checks during decode.
    pub device: &'a wgpu::Device,
    /// Queue for [`write_texture3d_volume_mip`].
    pub queue: &'a wgpu::Queue,
    /// Shared ABBA gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::WriteTextureSubmitGate`].
    pub write_texture_submit_gate: &'a crate::gpu::WriteTextureSubmitGate,
    /// Destination volume texture.
    pub texture: &'a wgpu::Texture,
    /// Host format descriptor.
    pub fmt: &'a SetTexture3DFormat,
    /// Resolved GPU storage format.
    pub wgpu_format: wgpu::TextureFormat,
    /// Upload record (asset id, descriptor length, etc.).
    pub upload: &'a SetTexture3DData,
    /// Payload bytes (`&raw[..upload.data.length]`).
    pub payload: &'a Arc<[u8]>,
}

/// Incremental 3D mip upload: one mip level per [`Texture3dMipChainUploader::upload_next_mip`] call.
#[derive(Debug)]
pub struct Texture3dMipChainUploader {
    next_mip: u32,
    uploaded_mips: u32,
    base_w: u32,
    base_h: u32,
    base_d: u32,
    mipmap_count: u32,
    background_rx: Option<crossbeam_channel::Receiver<Result<Vec<u8>, TextureUploadError>>>,
    pending_mip: Option<(u32, u32, u32, u32)>, // level, w, h, d
}

/// Result of one [`Texture3dMipChainUploader::upload_next_mip`] step.
#[derive(Debug)]
pub enum Texture3dMipAdvance {
    /// Uploaded a single mip; call again.
    UploadedOne,
    /// Chain complete.
    Finished {
        /// Total mips successfully written.
        total_uploaded: u32,
    },
    /// Waiting on background decoding thread. Call again next tick.
    YieldBackground,
}

impl Texture3dMipChainUploader {
    /// Validates `raw` against `fmt` and prepares chain state (no GPU work).
    pub fn new(
        texture: &wgpu::Texture,
        fmt: &SetTexture3DFormat,
        upload: &SetTexture3DData,
        raw: &[u8],
    ) -> Result<Self, TextureUploadError> {
        let want = upload.data.length.max(0) as usize;
        if raw.len() < want {
            return Err(TextureUploadError::from(format!(
                "raw shorter than descriptor (need {want}, got {})",
                raw.len()
            )));
        }

        let base_w = fmt.width.max(0) as u32;
        let base_h = fmt.height.max(0) as u32;
        let base_d = fmt.depth.max(0) as u32;
        let mipmap_count = fmt.mipmap_count.max(1) as u32;

        let tex_extent = texture.size();
        if tex_extent.width != base_w
            || tex_extent.height != base_h
            || tex_extent.depth_or_array_layers != base_d
        {
            return Err(TextureUploadError::from(format!(
                "GPU texture {}×{}×{} does not match SetTexture3DFormat {}×{}×{} for asset {}",
                tex_extent.width,
                tex_extent.height,
                tex_extent.depth_or_array_layers,
                base_w,
                base_h,
                base_d,
                upload.asset_id
            )));
        }

        let mut total_need = 0usize;
        for level in 0..mipmap_count {
            let (w, h, d) = mip_dimensions_at_level_3d(base_w, base_h, base_d, level);
            let slice = mip_byte_len(fmt.format, w, h).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "texture3d {}: mip byte size unsupported for {:?}",
                    upload.asset_id, fmt.format
                ))
            })? as usize;
            let vol = slice
                .checked_mul(d as usize)
                .ok_or_else(|| TextureUploadError::from("texture3d mip volume byte overflow"))?;
            total_need = total_need
                .checked_add(vol)
                .ok_or_else(|| TextureUploadError::from("texture3d mip chain total overflow"))?;
        }

        if total_need > want {
            return Err(TextureUploadError::from(format!(
                "texture3d {}: mip chain needs {total_need} B but descriptor length is {want}",
                upload.asset_id
            )));
        }

        Ok(Self {
            next_mip: 0,
            uploaded_mips: 0,
            base_w,
            base_h,
            base_d,
            mipmap_count,
            background_rx: None,
            pending_mip: None,
        })
    }

    /// Writes at most one mip level. `payload` is `&raw[..upload.data.length]`.
    pub fn upload_next_mip(
        &mut self,
        step: Texture3dMipUploadStep<'_>,
    ) -> Result<Texture3dMipAdvance, TextureUploadError> {
        let Texture3dMipUploadStep {
            device,
            queue,
            write_texture_submit_gate,
            texture,
            fmt,
            wgpu_format,
            upload,
            payload,
        } = step;
        let level = self.next_mip;
        if level >= self.mipmap_count {
            return Ok(Texture3dMipAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        if let Some(rx) = &self.background_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.background_rx = None;
                    let pixels = res?;
                    let (level, w, h, d) = self.pending_mip.take().ok_or_else(|| {
                        TextureUploadError::from(
                            "texture3d_write: background decode completed without a pending mip slot; state machine desync",
                        )
                    })?;

                    write_texture3d_volume_mip(&Texture3dVolumeMipWrite {
                        queue,
                        write_texture_submit_gate,
                        texture,
                        mip_level: level,
                        width: w,
                        height: h,
                        depth: d,
                        format: wgpu_format,
                        bytes: &pixels,
                    })?;

                    self.uploaded_mips += 1;
                    self.next_mip += 1;

                    if self.next_mip >= self.mipmap_count {
                        return Ok(Texture3dMipAdvance::Finished {
                            total_uploaded: self.uploaded_mips,
                        });
                    }
                    return Ok(Texture3dMipAdvance::UploadedOne);
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    return Ok(Texture3dMipAdvance::YieldBackground);
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    return Err(TextureUploadError::from(
                        "Background decode thread panicked",
                    ));
                }
            }
        }

        let (w, h, d, mip_src, slice_bytes, vol_bytes) = texture3d_mip_volume_payload_slice(
            self.base_w,
            self.base_h,
            self.base_d,
            level,
            fmt,
            upload,
            payload,
        )?;

        self.pending_mip = Some((level, w, h, d));
        let offset = mip_src.as_ptr() as usize - payload.as_ptr() as usize;
        let len = mip_src.len();
        let mip_src_range = offset..offset + len;

        let (tx, rx) = crossbeam_channel::bounded(1);
        self.background_rx = Some(rx);

        let asset_id = upload.asset_id;
        let fmt_format = fmt.format;
        let needs_rgba8_decode = needs_rgba8_decode_before_upload(device, fmt_format);
        let payload_arc = std::sync::Arc::clone(payload);

        let ctx = MipUploadFormatCtx {
            asset_id,
            fmt_format,
            wgpu_format,
            needs_rgba8_decode,
        };
        let geom = Texture3dMipGeom {
            w,
            h,
            d,
            level_idx: level,
            slice_bytes,
            vol_bytes,
        };
        rayon::spawn(move || {
            let mip_src = &payload_arc[mip_src_range];
            let res = texture3d_mip_to_upload_pixels(ctx, geom, mip_src);
            let _ = tx.send(res);
        });

        Ok(Texture3dMipAdvance::YieldBackground)
    }
}

/// GPU target, host format, and raw payload for one [`write_texture3d_mips`] call.
pub struct Texture3dUploadContext<'a> {
    /// Device for format capability checks during decode.
    pub device: &'a wgpu::Device,
    /// Queue for volume mip writes.
    pub queue: &'a wgpu::Queue,
    /// Shared ABBA gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::WriteTextureSubmitGate`].
    pub write_texture_submit_gate: &'a crate::gpu::WriteTextureSubmitGate,
    /// Destination volume texture.
    pub texture: &'a wgpu::Texture,
    /// Host format descriptor (dimensions, mip count, texel format).
    pub fmt: &'a SetTexture3DFormat,
    /// Resolved GPU storage format.
    pub wgpu_format: wgpu::TextureFormat,
    /// Upload record (asset id, descriptor length, etc.).
    pub upload: &'a SetTexture3DData,
    /// Raw shared-memory bytes covering the descriptor window.
    pub raw: &'a [u8],
}

/// Runs the full mip chain upload for 3D data (non-cooperative path).
pub fn write_texture3d_mips(ctx: &Texture3dUploadContext<'_>) -> Result<u32, TextureUploadError> {
    let want = ctx.upload.data.length.max(0) as usize;
    if ctx.raw.len() < want {
        return Err(TextureUploadError::from(format!(
            "raw shorter than descriptor (need {want}, got {})",
            ctx.raw.len()
        )));
    }
    let payload = std::sync::Arc::from(&ctx.raw[..want]);
    let mut uploader = Texture3dMipChainUploader::new(ctx.texture, ctx.fmt, ctx.upload, ctx.raw)?;
    loop {
        match uploader.upload_next_mip(Texture3dMipUploadStep {
            device: ctx.device,
            queue: ctx.queue,
            write_texture_submit_gate: ctx.write_texture_submit_gate,
            texture: ctx.texture,
            fmt: ctx.fmt,
            wgpu_format: ctx.wgpu_format,
            upload: ctx.upload,
            payload: &payload,
        })? {
            Texture3dMipAdvance::UploadedOne => {}
            Texture3dMipAdvance::Finished { total_uploaded } => {
                return Ok(total_uploaded);
            }
            Texture3dMipAdvance::YieldBackground => {
                std::thread::yield_now();
            }
        }
    }
}
