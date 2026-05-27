//! [`SetTexture3DData`](crate::shared::SetTexture3DData) -> [`wgpu::Queue::write_texture`] for [`wgpu::TextureDimension::D3`].

use std::sync::Arc;

use crate::gpu::GpuQueueAccessMode;
use crate::shared::{SetTexture3DData, SetTexture3DFormat};

use super::super::decode::{decode_mip_to_rgba8, needs_rgba8_decode_before_upload};
use super::super::layout::{
    clamp_host_texture_mip_count, host_format_is_compressed, mip_byte_len,
    mip_dimensions_at_level_3d,
};
use super::error::TextureUploadError;
use super::mip_write_common::{
    MipUploadFormatCtx, Texture3dVolumeMipWrite, is_rgba8_family, write_texture3d_volume_mip,
    write_texture3d_volume_mip_with_gate,
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
    /// Tight byte length of one depth slice (stride x height).
    slice_bytes: usize,
    /// Tight byte length of the full volume (`slice_bytes x d`).
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
                "texture3d {asset_id}: mip byte size unsupported for {format:?}"
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
    profiling::scope!("asset::texture3d_convert_mip_pixels");
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
                            "texture3d {asset_id}: RGBA decode failed mip {level_idx}"
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
                "texture3d {asset_id}: host {fmt_format:?} must decode to RGBA but GPU format is {wgpu_format:?}"
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
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this upload step.
    pub queue_access_mode: GpuQueueAccessMode,
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
        profiling::scope!("asset::texture3d_mip_chain_new");
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
        let mipmap_count =
            clamp_host_texture_mip_count(fmt.mipmap_count, texture.mip_level_count());

        let tex_extent = texture.size();
        if tex_extent.width != base_w
            || tex_extent.height != base_h
            || tex_extent.depth_or_array_layers != base_d
        {
            return Err(TextureUploadError::from(format!(
                "GPU texture {}x{}x{} does not match SetTexture3DFormat {}x{}x{} for asset {}",
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
        profiling::scope!("asset::texture3d_mip_chain_step");
        let Texture3dMipUploadStep {
            device,
            queue,
            gpu_queue_access_gate,
            queue_access_mode,
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

        if matches!(queue_access_mode, GpuQueueAccessMode::NonBlocking)
            && self.background_rx.is_some()
        {
            let Some(gate) = gpu_queue_access_gate.try_lock() else {
                return Ok(Texture3dMipAdvance::YieldBackground);
            };
            return self.poll_background_mip_with_gate(
                queue,
                gpu_queue_access_gate,
                queue_access_mode,
                texture,
                wgpu_format,
                &gate,
            );
        }

        if let Some(rx) = &self.background_rx {
            profiling::scope!("asset::texture3d_poll_decoded_mip");
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
                        gpu_queue_access_gate,
                        queue_access_mode,
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

        self.spawn_next_mip_decode(device, fmt, wgpu_format, upload, payload, level)
    }

    fn spawn_next_mip_decode(
        &mut self,
        device: &wgpu::Device,
        fmt: &SetTexture3DFormat,
        wgpu_format: wgpu::TextureFormat,
        upload: &SetTexture3DData,
        payload: &Arc<[u8]>,
        level: u32,
    ) -> Result<Texture3dMipAdvance, TextureUploadError> {
        profiling::scope!("asset::texture3d_spawn_mip_decode");
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
        let payload_arc = Arc::clone(payload);

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
            profiling::scope!("asset::texture3d_decode_mip");
            let mip_src = &payload_arc[mip_src_range];
            let res = texture3d_mip_to_upload_pixels(ctx, geom, mip_src);
            let _ = tx.send(res);
        });

        Ok(Texture3dMipAdvance::YieldBackground)
    }

    fn poll_background_mip_with_gate(
        &mut self,
        queue: &wgpu::Queue,
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
        queue_access_mode: GpuQueueAccessMode,
        texture: &wgpu::Texture,
        wgpu_format: wgpu::TextureFormat,
        gate: &parking_lot::MutexGuard<'_, ()>,
    ) -> Result<Texture3dMipAdvance, TextureUploadError> {
        profiling::scope!("asset::texture3d_poll_decoded_mip_locked");
        let rx = self.background_rx.as_ref().ok_or_else(|| {
            TextureUploadError::from(
                "texture3d_write: locked background poll without a decode receiver; state machine desync",
            )
        })?;
        match rx.try_recv() {
            Ok(res) => {
                self.background_rx = None;
                let pixels = res?;
                let (level, w, h, d) = self.pending_mip.take().ok_or_else(|| {
                    TextureUploadError::from(
                        "texture3d_write: background decode completed without a pending mip slot; state machine desync",
                    )
                })?;

                write_texture3d_volume_mip_with_gate(
                    &Texture3dVolumeMipWrite {
                        queue,
                        gpu_queue_access_gate,
                        queue_access_mode,
                        texture,
                        mip_level: level,
                        width: w,
                        height: h,
                        depth: d,
                        format: wgpu_format,
                        bytes: &pixels,
                    },
                    gate,
                )?;

                self.uploaded_mips += 1;
                self.next_mip += 1;

                if self.next_mip >= self.mipmap_count {
                    return Ok(Texture3dMipAdvance::Finished {
                        total_uploaded: self.uploaded_mips,
                    });
                }
                Ok(Texture3dMipAdvance::UploadedOne)
            }
            Err(crossbeam_channel::TryRecvError::Empty) => Ok(Texture3dMipAdvance::YieldBackground),
            Err(crossbeam_channel::TryRecvError::Disconnected) => Err(TextureUploadError::from(
                "Background texture3d decode thread panicked",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::shared::{ColorProfile, TextureFormat};

    /// Builds a Texture3D format record for local upload-layout tests.
    fn texture3d_format(
        asset_id: i32,
        width: i32,
        height: i32,
        depth: i32,
        mipmap_count: i32,
    ) -> SetTexture3DFormat {
        SetTexture3DFormat {
            width,
            height,
            depth,
            mipmap_count,
            format: TextureFormat::RGBA32,
            profile: ColorProfile::Linear,
            asset_id,
        }
    }

    /// Builds a Texture3D upload record with a descriptor length matching `payload_len`.
    fn texture3d_upload(asset_id: i32, payload_len: usize) -> SetTexture3DData {
        let mut upload = SetTexture3DData {
            asset_id,
            ..SetTexture3DData::default()
        };
        upload.data.length = i32::try_from(payload_len).unwrap();
        upload
    }

    /// Encodes one RGBA8 marker texel for an authored `(x, y, z)` coordinate.
    fn marker_texel(x: u8, y: u8, z: u8) -> [u8; 4] {
        [x, y, z, 255]
    }

    /// Builds a tight `Bitmap3D`-order RGBA8 volume: X fastest, Y next, Z as full slices.
    fn marker_volume(width: u32, height: u32, depth: u32) -> Vec<u8> {
        let mut payload = Vec::with_capacity((width * height * depth * 4) as usize);
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    payload.extend_from_slice(&marker_texel(x as u8, y as u8, z as u8));
                }
            }
        }
        payload
    }

    /// Reads one marker texel using the same tight `Bitmap3D` linearization.
    fn marker_at(payload: &[u8], width: u32, height: u32, x: u32, y: u32, z: u32) -> [u8; 4] {
        let texel = ((z * width * height + y * width + x) * 4) as usize;
        payload[texel..texel + 4].try_into().unwrap()
    }

    /// Verifies that tight Texture3D uploads keep authored `Bitmap3D` X/Y/Z texel order intact.
    #[test]
    fn texture3d_payload_preserves_bitmap3d_xyz_order() {
        let fmt = texture3d_format(7, 2, 2, 2, 1);
        let payload = marker_volume(2, 2, 2);
        let upload = texture3d_upload(fmt.asset_id, payload.len());
        let (w, h, d, mip_src, slice_bytes, vol_bytes) =
            texture3d_mip_volume_payload_slice(2, 2, 2, 0, &fmt, &upload, &payload).unwrap();

        assert_eq!((w, h, d), (2, 2, 2));
        assert_eq!(slice_bytes, 16);
        assert_eq!(vol_bytes, 32);
        assert_eq!(mip_src, payload.as_slice());

        let pixels = texture3d_mip_to_upload_pixels(
            MipUploadFormatCtx {
                asset_id: fmt.asset_id,
                fmt_format: TextureFormat::RGBA32,
                wgpu_format: wgpu::TextureFormat::Rgba8Unorm,
                needs_rgba8_decode: false,
            },
            Texture3dMipGeom {
                w,
                h,
                d,
                level_idx: 0,
                slice_bytes,
                vol_bytes,
            },
            mip_src,
        )
        .unwrap();

        assert_eq!(marker_at(&pixels, 2, 2, 0, 0, 0), marker_texel(0, 0, 0));
        assert_eq!(marker_at(&pixels, 2, 2, 1, 0, 0), marker_texel(1, 0, 0));
        assert_eq!(marker_at(&pixels, 2, 2, 0, 1, 0), marker_texel(0, 1, 0));
        assert_eq!(marker_at(&pixels, 2, 2, 0, 0, 1), marker_texel(0, 0, 1));
    }

    /// Verifies mip offsets advance by whole 3D volumes rather than by one 2D slice.
    #[test]
    fn texture3d_mip_offsets_walk_complete_z_slices() {
        let fmt = texture3d_format(9, 4, 2, 2, 2);
        let mut payload = marker_volume(4, 2, 2);
        let mip1 = marker_volume(2, 1, 1);
        payload.extend_from_slice(&mip1);
        let upload = texture3d_upload(fmt.asset_id, payload.len());

        let (w, h, d, mip_src, slice_bytes, vol_bytes) =
            texture3d_mip_volume_payload_slice(4, 2, 2, 1, &fmt, &upload, &payload).unwrap();

        assert_eq!((w, h, d), (2, 1, 1));
        assert_eq!(slice_bytes, 8);
        assert_eq!(vol_bytes, 8);
        assert_eq!(mip_src, mip1.as_slice());
        assert_eq!(marker_at(mip_src, 2, 1, 0, 0, 0), marker_texel(0, 0, 0));
        assert_eq!(marker_at(mip_src, 2, 1, 1, 0, 0), marker_texel(1, 0, 0));
    }
}
