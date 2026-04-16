//! [`SetTexture3DData`](crate::shared::SetTexture3DData) → [`wgpu::Queue::write_texture`] for [`wgpu::TextureDimension::D3`].

use crate::shared::{SetTexture3DData, SetTexture3DFormat};

use super::super::decode::{decode_mip_to_rgba8, needs_rgba8_decode_before_upload};
use super::super::layout::{host_format_is_compressed, mip_byte_len, mip_dimensions_at_level_3d};
use super::error::TextureUploadError;
use super::mip_write_common::{
    is_rgba8_family, write_texture3d_volume_mip, Texture3dVolumeMipWrite,
};

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

/// Device and format descriptors shared across 3D mip uploads.
struct Texture3dMipChainState<'a> {
    device: &'a wgpu::Device,
    fmt: &'a SetTexture3DFormat,
    upload: &'a SetTexture3DData,
}

/// Dimensions and byte layout for one 3D mip level decode ([`texture3d_mip_to_upload_pixels`]).
struct Texture3dMipLevelDecode<'a> {
    wgpu_format: wgpu::TextureFormat,
    w: u32,
    h: u32,
    d: u32,
    level: u32,
    slice_bytes: usize,
    vol_bytes: usize,
    mip_src: &'a [u8],
}

/// Prepares decoded RGBA8 slab or passes raw host bytes through for 3D volume upload.
fn texture3d_mip_to_upload_pixels<'a>(
    chain: &Texture3dMipChainState<'a>,
    level: Texture3dMipLevelDecode<'a>,
) -> Result<std::borrow::Cow<'a, [u8]>, TextureUploadError> {
    let device = chain.device;
    let fmt = chain.fmt;
    let upload = chain.upload;
    let wgpu_format = level.wgpu_format;
    let w = level.w;
    let h = level.h;
    let d = level.d;
    let level_idx = level.level;
    let slice_bytes = level.slice_bytes;
    let vol_bytes = level.vol_bytes;
    let mip_src = level.mip_src;
    let pixels: std::borrow::Cow<'a, [u8]> = if is_rgba8_family(wgpu_format) {
        if needs_rgba8_decode_before_upload(device, fmt.format)
            || host_format_is_compressed(fmt.format)
        {
            let mut out = Vec::with_capacity(vol_bytes);
            let mut z_off = 0usize;
            for _z in 0..d {
                let slice_raw = mip_src
                    .get(z_off..z_off + slice_bytes)
                    .ok_or_else(|| TextureUploadError::from("texture3d slice bounds"))?;
                let decoded =
                    decode_mip_to_rgba8(fmt.format, w, h, false, slice_raw).ok_or_else(|| {
                        TextureUploadError::from(format!(
                            "texture3d {}: RGBA decode failed mip {level_idx}",
                            upload.asset_id
                        ))
                    })?;
                out.extend_from_slice(&decoded);
                z_off += slice_bytes;
            }
            std::borrow::Cow::Owned(out)
        } else {
            std::borrow::Cow::Borrowed(mip_src)
        }
    } else {
        if needs_rgba8_decode_before_upload(device, fmt.format) {
            return Err(TextureUploadError::from(format!(
                "texture3d {}: host {:?} must decode to RGBA but GPU format is {:?}",
                upload.asset_id, fmt.format, wgpu_format
            )));
        }
        std::borrow::Cow::Borrowed(mip_src)
    };
    Ok(pixels)
}

/// GPU device, queue, and host upload view for one [`Texture3dMipChainUploader::upload_next_mip`] step.
pub struct Texture3dMipUploadStep<'a> {
    /// Device for format capability checks during decode.
    pub device: &'a wgpu::Device,
    /// Queue for [`write_texture3d_volume_mip`].
    pub queue: &'a wgpu::Queue,
    /// Destination volume texture.
    pub texture: &'a wgpu::Texture,
    /// Host format descriptor.
    pub fmt: &'a SetTexture3DFormat,
    /// Resolved GPU storage format.
    pub wgpu_format: wgpu::TextureFormat,
    /// Upload record (asset id, descriptor length, etc.).
    pub upload: &'a SetTexture3DData,
    /// Payload bytes (`&raw[..upload.data.length]`).
    pub payload: &'a [u8],
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

        let (w, h, d, mip_src, slice_bytes, vol_bytes) = texture3d_mip_volume_payload_slice(
            self.base_w,
            self.base_h,
            self.base_d,
            level,
            fmt,
            upload,
            payload,
        )?;

        let chain = Texture3dMipChainState {
            device,
            fmt,
            upload,
        };
        let pixels = texture3d_mip_to_upload_pixels(
            &chain,
            Texture3dMipLevelDecode {
                wgpu_format,
                w,
                h,
                d,
                level,
                slice_bytes,
                vol_bytes,
                mip_src,
            },
        )?;

        write_texture3d_volume_mip(&Texture3dVolumeMipWrite {
            queue,
            texture,
            mip_level: level,
            width: w,
            height: h,
            depth: d,
            format: wgpu_format,
            bytes: pixels.as_ref(),
        })?;

        self.uploaded_mips += 1;
        self.next_mip += 1;

        if self.next_mip >= self.mipmap_count {
            return Ok(Texture3dMipAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }
        Ok(Texture3dMipAdvance::UploadedOne)
    }
}

/// Runs the full mip chain upload for 3D data (non-cooperative path).
pub fn write_texture3d_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture3DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture3DData,
    raw: &[u8],
) -> Result<u32, TextureUploadError> {
    let want = upload.data.length.max(0) as usize;
    if raw.len() < want {
        return Err(TextureUploadError::from(format!(
            "raw shorter than descriptor (need {want}, got {})",
            raw.len()
        )));
    }
    let payload = &raw[..want];
    let mut uploader = Texture3dMipChainUploader::new(texture, fmt, upload, raw)?;
    loop {
        match uploader.upload_next_mip(Texture3dMipUploadStep {
            device,
            queue,
            texture,
            fmt,
            wgpu_format,
            upload,
            payload,
        })? {
            Texture3dMipAdvance::UploadedOne => {}
            Texture3dMipAdvance::Finished { total_uploaded } => {
                return Ok(total_uploaded);
            }
        }
    }
}
