//! [`SetCubemapData`](crate::shared::SetCubemapData) -> cubemap array layers ([`super::mip_write_common::write_cubemap_face_mip`]).

use crate::gpu::GpuQueueAccessMode;
use crate::shared::{SetCubemapData, SetCubemapFormat};

use super::super::decode::needs_rgba8_decode_before_upload;
use super::super::layout::{
    clamp_host_texture_mip_count, host_mip_payload_byte_offset, mip_byte_len,
    mip_dimensions_at_level,
};
use super::error::TextureUploadError;
use super::mip_chain_walk::{MipChainStop, resolve_mip_payload_slot};
use super::mip_write_common::{
    CubemapFaceMipWrite, MipUploadFormatCtx, MipUploadLabel, MipUploadPixels,
    mip_src_to_upload_pixels as shared_mip_src_to_upload_pixels, write_cubemap_face_mip,
    write_cubemap_face_mip_with_gate,
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

/// Host payload subslice for one cubemap face x mip after bias and length checks.
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
            "mip start {start_abs} is before descriptor offset {start_bias}"
        )));
    }
    let start_rel = start_abs - start_bias;
    let start = host_mip_payload_byte_offset(fmt.format, start_rel).ok_or_else(|| {
        TextureUploadError::from(format!(
            "cubemap {} face {face} mip {mip_i}: mip start offset unsupported for {:?}",
            upload.asset_id, fmt.format
        ))
    })?;

    let host_len = mip_byte_len(fmt.format, w, h).ok_or_else(|| {
        TextureUploadError::from(format!(
            "cubemap mip byte size unsupported for {:?}",
            fmt.format
        ))
    })? as usize;
    let end = start
        .checked_add(host_len)
        .ok_or_else(|| TextureUploadError::from("cubemap mip end overflow"))?;

    payload
        .get(start..end)
        .ok_or_else(|| {
            TextureUploadError::from(format!(
                "cubemap {} face {} mip {mip_i}: slice out of range (raw_start {start_abs} byte_start {start} len {host_len}, payload {})",
                upload.asset_id,
                face,
                payload.len()
            ))
        })
}

/// Converts host face mip bytes for [`write_cubemap_face_mip`].
fn cubemap_mip_src_to_upload_pixels(
    ctx: MipUploadFormatCtx,
    w: u32,
    h: u32,
    flip: bool,
    mip_i: usize,
    face: u32,
    mip_src: &[u8],
) -> Result<MipUploadPixels, TextureUploadError> {
    profiling::scope!("asset::cubemap_convert_mip_pixels");
    shared_mip_src_to_upload_pixels(
        ctx,
        w,
        h,
        flip,
        mip_src,
        MipUploadLabel::cubemap(face, mip_i),
    )
    .map(|pixels| pixels.with_storage_v_inverted(false))
}

/// GPU and host view for one [`CubemapMipChainUploader::upload_next_face_mip`] step.
pub struct CubemapFaceMipUploadStep<'a> {
    /// Device for decode paths.
    pub device: &'a wgpu::Device,
    /// Queue for the face mip write.
    pub queue: &'a wgpu::Queue,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this upload step.
    pub queue_access_mode: GpuQueueAccessMode,
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

/// Incremental cubemap upload: one face x one mip per step.
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
    storage_v_inverted: bool,
    background_rx: Option<crossbeam_channel::Receiver<Result<MipUploadPixels, TextureUploadError>>>,
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
        profiling::scope!("asset::cubemap_mip_chain_new");
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
        let mipmap_count =
            clamp_host_texture_mip_count(fmt.mipmap_count, texture.mip_level_count());
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
                "GPU cubemap {}x{}x{} does not match format face {} (asset {})",
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
            storage_v_inverted: false,
            background_rx: None,
            pending_mip: None,
        })
    }

    /// Writes at most one face mip. `payload` is `&raw[..upload.data.length]`.
    pub fn upload_next_face_mip(
        &mut self,
        step: CubemapFaceMipUploadStep<'_>,
    ) -> Result<MipChainAdvance, TextureUploadError> {
        profiling::scope!("asset::cubemap_mip_chain_step");
        if self.face >= 6 {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded,
                storage_v_inverted: self.storage_v_inverted,
            });
        }

        if let Some(advance) = self.poll_background_decoded_face_mip(&step)? {
            return Ok(advance);
        }

        self.spawn_upload_next_face_mip(&step)
    }

    /// Drains a completed background face-mip decode into a `Queue::write_texture`, or yields if pending.
    ///
    /// Returns `None` when no background decode is in flight (caller should start one).
    fn poll_background_decoded_face_mip(
        &mut self,
        step: &CubemapFaceMipUploadStep<'_>,
    ) -> Result<Option<MipChainAdvance>, TextureUploadError> {
        let Some(rx) = &self.background_rx else {
            return Ok(None);
        };
        profiling::scope!("asset::cubemap_poll_decoded_mip");
        if matches!(step.queue_access_mode, GpuQueueAccessMode::NonBlocking) {
            let Some(gate) = step.gpu_queue_access_gate.try_lock() else {
                return Ok(Some(MipChainAdvance::YieldBackground));
            };
            return self.poll_background_decoded_mip_with_gate(step, &gate);
        }
        match rx.try_recv() {
            Ok(res) => {
                self.background_rx = None;
                let pixels = res?;
                let (face, mip_level, w, h) = self.pending_mip.take().ok_or_else(|| {
                    TextureUploadError::from(
                        "cubemap_write: background decode completed without a pending mip slot; state machine desync",
                    )
                })?;

                write_cubemap_face_mip(&CubemapFaceMipWrite {
                    queue: step.queue,
                    gpu_queue_access_gate: step.gpu_queue_access_gate,
                    queue_access_mode: step.queue_access_mode,
                    texture: step.texture,
                    mip_level,
                    face_layer: face,
                    width: w,
                    height: h,
                    format: step.wgpu_format,
                    bytes: &pixels.bytes,
                })?;

                self.storage_v_inverted |= pixels.storage_v_inverted;
                self.uploaded += 1;
                self.mip_i += 1;
                self.advance_face_if_mip_limit_reached(step.upload);

                if self.face >= 6 {
                    return Ok(Some(MipChainAdvance::Finished {
                        total_uploaded: self.uploaded,
                        storage_v_inverted: self.storage_v_inverted,
                    }));
                }

                Ok(Some(MipChainAdvance::UploadedOne {
                    total_uploaded: self.uploaded,
                    storage_v_inverted: self.storage_v_inverted,
                }))
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {
                Ok(Some(MipChainAdvance::YieldBackground))
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => Err(TextureUploadError::from(
                "Background decode thread panicked",
            )),
        }
    }

    fn poll_background_decoded_mip_with_gate(
        &mut self,
        step: &CubemapFaceMipUploadStep<'_>,
        gate: &parking_lot::MutexGuard<'_, ()>,
    ) -> Result<Option<MipChainAdvance>, TextureUploadError> {
        let rx = self.background_rx.as_ref().ok_or_else(|| {
            TextureUploadError::from(
                "cubemap_write: locked background poll without a decode receiver; state machine desync",
            )
        })?;
        match rx.try_recv() {
            Ok(res) => {
                self.background_rx = None;
                let pixels = res?;
                let (face, mip_level, w, h) = self.pending_mip.take().ok_or_else(|| {
                    TextureUploadError::from(
                        "cubemap_write: background decode completed without a pending mip slot; state machine desync",
                    )
                })?;

                write_cubemap_face_mip_with_gate(
                    &CubemapFaceMipWrite {
                        queue: step.queue,
                        gpu_queue_access_gate: step.gpu_queue_access_gate,
                        queue_access_mode: step.queue_access_mode,
                        texture: step.texture,
                        mip_level,
                        face_layer: face,
                        width: w,
                        height: h,
                        format: step.wgpu_format,
                        bytes: &pixels.bytes,
                    },
                    gate,
                )?;

                self.storage_v_inverted |= pixels.storage_v_inverted;
                self.uploaded += 1;
                self.mip_i += 1;
                self.advance_face_if_mip_limit_reached(step.upload);

                if self.face >= 6 {
                    return Ok(Some(MipChainAdvance::Finished {
                        total_uploaded: self.uploaded,
                        storage_v_inverted: self.storage_v_inverted,
                    }));
                }
                Ok(Some(MipChainAdvance::UploadedOne {
                    total_uploaded: self.uploaded,
                    storage_v_inverted: self.storage_v_inverted,
                }))
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {
                Ok(Some(MipChainAdvance::YieldBackground))
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => Err(TextureUploadError::from(
                "Background cubemap decode thread panicked",
            )),
        }
    }

    /// Resolves the next face/mip slice and spawns a rayon decode.
    fn spawn_upload_next_face_mip(
        &mut self,
        step: &CubemapFaceMipUploadStep<'_>,
    ) -> Result<MipChainAdvance, TextureUploadError> {
        profiling::scope!("asset::cubemap_spawn_mip_decode");
        let mip_i = self.mip_i;
        if mip_i >= self.face_mip_limit(step.upload) {
            self.advance_face_if_mip_limit_reached(step.upload);
            return if self.face >= 6 {
                Ok(MipChainAdvance::Finished {
                    total_uploaded: self.uploaded,
                    storage_v_inverted: self.storage_v_inverted,
                })
            } else {
                Ok(MipChainAdvance::UploadedOne {
                    total_uploaded: self.uploaded,
                    storage_v_inverted: self.storage_v_inverted,
                })
            };
        }
        debug_assert!(mip_i < step.upload.mip_map_sizes.len());

        let sz = step.upload.mip_map_sizes[mip_i];
        let w = sz.x.max(0) as u32;
        let h = sz.y.max(0) as u32;
        let mip_level = self.start_base + mip_i as u32;
        if mip_level >= self.mipmap_count {
            self.advance_face_if_mip_limit_reached(step.upload);
            return Ok(MipChainAdvance::UploadedOne {
                total_uploaded: self.uploaded,
                storage_v_inverted: self.storage_v_inverted,
            });
        }

        let (gw, gh) = mip_dimensions_at_level(self.face_size, self.face_size, mip_level);
        if w != gw || h != gh {
            return Err(TextureUploadError::from(format!(
                "cubemap {} mip {mip_level}: upload says {w}x{h} but GPU mip is {gw}x{gh}",
                step.upload.asset_id
            )));
        }

        let chain = CubemapMipChainState {
            fmt: step.fmt,
            upload: step.upload,
            payload: step.payload,
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
        let offset = mip_src.as_ptr() as usize - step.payload.as_ptr() as usize;
        let len = mip_src.len();
        let mip_src_range = offset..offset + len;

        let (tx, rx) = crossbeam_channel::bounded(1);
        self.background_rx = Some(rx);

        let ctx = MipUploadFormatCtx {
            asset_id: step.upload.asset_id,
            fmt_format: step.fmt.format,
            wgpu_format: step.wgpu_format,
            needs_rgba8_decode: needs_rgba8_decode_before_upload(step.device, step.fmt.format),
        };
        let payload_arc = std::sync::Arc::clone(step.payload);
        let flip = self.flip;
        let face = self.face;
        crate::assets::worker::spawn_asset_job(move || {
            profiling::scope!("asset::cubemap_decode_mip");
            let mip_src = &payload_arc[mip_src_range];
            let res = cubemap_mip_src_to_upload_pixels(ctx, w, h, flip, mip_i, face, mip_src);
            let _ = tx.send(res);
        });

        Ok(MipChainAdvance::YieldBackground)
    }

    fn face_mip_limit(&self, upload: &SetCubemapData) -> usize {
        upload
            .mip_map_sizes
            .len()
            .min(self.mipmap_count.saturating_sub(self.start_base) as usize)
    }

    fn advance_face_if_mip_limit_reached(&mut self, upload: &SetCubemapData) {
        if self.mip_i >= self.face_mip_limit(upload) {
            self.face += 1;
            self.mip_i = 0;
        }
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
            "cubemap mip region exceeds shared memory descriptor (payload_len={payload_len}, descriptor_offset={offset_bias})"
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
                TextureUploadError::from(format!("mip byte size unsupported for {format:?}"))
            })? as usize;
            let starts = upload
                .mip_starts
                .get(face)
                .ok_or_else(|| TextureUploadError::from("cubemap mip_starts face missing"))?;
            let start_raw = *starts
                .get(i)
                .ok_or_else(|| TextureUploadError::from("cubemap mip_starts index"))?;
            match resolve_mip_payload_slot(format, host_len, start_raw, bias, payload_len, || {
                format!("cubemap face {face} mip {i}")
            })? {
                Ok(()) => count += 1,
                Err(
                    MipChainStop::NegativeStart
                    | MipChainStop::BeforeBias
                    | MipChainStop::OutOfPayload,
                ) => break 'outer,
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec2;

    use crate::shared::TextureFormat;
    use crate::shared::buffer::SharedMemoryBufferDescriptor;

    fn upload_ctx(
        fmt_format: TextureFormat,
        wgpu_format: wgpu::TextureFormat,
    ) -> MipUploadFormatCtx {
        MipUploadFormatCtx {
            asset_id: 91,
            fmt_format,
            wgpu_format,
            needs_rgba8_decode: false,
        }
    }

    fn bc1_single_mip_format() -> SetCubemapFormat {
        SetCubemapFormat {
            asset_id: 5,
            size: 512,
            mipmap_count: 1,
            format: TextureFormat::BC1,
            ..Default::default()
        }
    }

    fn bc1_single_mip_upload() -> SetCubemapData {
        SetCubemapData {
            asset_id: 5,
            data: SharedMemoryBufferDescriptor {
                length: 6 * 131_072,
                ..Default::default()
            },
            mip_map_sizes: vec![IVec2::new(512, 512)],
            mip_starts: (0_i32..6).map(|face| vec![face * 512 * 512]).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn cubemap_bc1_face_starts_are_linear_texel_offsets() {
        let fmt = bc1_single_mip_format();
        let upload = bc1_single_mip_upload();
        let payload = vec![0_u8; upload.data.length as usize];
        let chain = CubemapMipChainState {
            fmt: &fmt,
            upload: &upload,
            payload: &payload,
            start_bias: 0,
        };

        let mip_src = resolve_cubemap_face_mip_slice(
            &chain,
            CubemapFaceMipSliceStep {
                face: 3,
                mip_i: 0,
                w: 512,
                h: 512,
            },
        )
        .expect("face 3 mip 0 slice");
        let offset = mip_src.as_ptr() as usize - payload.as_ptr() as usize;

        assert_eq!(offset, 3 * 131_072);
        assert_eq!(mip_src.len(), 131_072);
    }

    #[test]
    fn cubemap_bc1_prefix_validation_uses_converted_byte_offsets() {
        let upload = bc1_single_mip_upload();

        let prefix = valid_cubemap_mip_prefix_len(
            TextureFormat::BC1,
            &upload,
            upload.data.length as usize,
            0,
        )
        .expect("cubemap prefix");

        assert_eq!(prefix, 6);
    }

    #[test]
    fn cubemap_bc7_flip_y_uploads_bytes_unchanged_with_native_storage_orientation() {
        let raw: Vec<u8> = (0..64).collect();
        let pixels = cubemap_mip_src_to_upload_pixels(
            upload_ctx(TextureFormat::BC7, wgpu::TextureFormat::Bc7RgbaUnorm),
            8,
            8,
            true,
            0,
            3,
            &raw,
        )
        .expect("bc7 cubemap upload");

        assert_eq!(pixels.bytes, raw);
        assert!(!pixels.storage_v_inverted);
    }

    #[test]
    fn face_mip_limit_caps_to_allocated_mips_after_start_base() {
        let upload = SetCubemapData {
            mip_map_sizes: vec![
                IVec2::new(64, 64),
                IVec2::new(32, 32),
                IVec2::new(16, 16),
                IVec2::new(8, 8),
            ],
            ..Default::default()
        };
        let uploader = CubemapMipChainUploader {
            face: 0,
            mip_i: 0,
            uploaded: 0,
            start_bias: 0,
            start_base: 2,
            mipmap_count: 4,
            face_size: 64,
            flip: false,
            storage_v_inverted: false,
            background_rx: None,
            pending_mip: None,
        };

        assert_eq!(uploader.face_mip_limit(&upload), 2);
    }
}
