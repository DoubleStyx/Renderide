//! Incremental 2D mip-chain upload state machine.

use std::sync::Arc;

use crate::shared::{SetTexture2DData, SetTexture2DFormat};

use super::super::super::decode::needs_rgba8_decode_before_upload;
use super::super::super::layout::mip_dimensions_at_level;
use super::super::mip_write_common::{
    choose_mip_start_bias, is_rgba8_family, write_one_mip, MipUploadFormatCtx, MipUploadPixels,
    Texture2dMipWrite,
};
use super::super::subregion::{hint_region_is_empty, try_write_texture2d_subregion};
use super::super::TextureUploadError;
use super::conversion::{downsample_rgba8_box, mip_src_to_upload_pixels};
use super::payload::{validate_and_resolve_next_mip_slice, MipChainWalkState, NextMipUploadSlice};

/// Incremental full mip-chain upload: call [`Self::upload_next_mip`] until [`MipChainAdvance::Finished`].
#[derive(Debug)]
pub struct TextureMipChainUploader {
    /// Next host mip-array index to upload.
    next_i: usize,
    /// Number of mip levels successfully written.
    uploaded_mips: u32,
    /// Descriptor-relative byte offset used to rebase host `mip_starts`.
    start_bias: usize,
    /// First destination mip level in the GPU texture.
    start_base: u32,
    /// Total mip levels allocated on the GPU texture.
    mipmap_count: u32,
    /// Destination texture extent.
    tex_extent: wgpu::Extent3d,
    /// Whether host rows should be flipped before upload when the format supports it.
    flip: bool,
    /// Whether the chain has reached a terminal state.
    stopped: bool,
    /// Whether the uploader has already logged missing-tail mip synthesis for this chain.
    generating_tail: bool,
    /// Whether any written mip used host-V-inverted storage.
    storage_v_inverted: bool,
    /// Last written RGBA8 mip used to synthesize missing tail mips.
    last_rgba8_mip: Option<Rgba8Mip>,
    /// Background decode/downsample result pending for the current mip.
    background_rx: Option<crossbeam_channel::Receiver<Result<MipUploadPixels, TextureUploadError>>>,
    /// Destination `(mip_level, width, height)` paired with [`Self::background_rx`].
    pending_mip: Option<(u32, u32, u32)>,
}

/// Result of one [`TextureMipChainUploader::upload_next_mip`] step.
#[derive(Debug)]
pub enum MipChainAdvance {
    /// Uploaded or generated a single mip; call again for the next level (same `payload` slice).
    UploadedOne {
        /// Total mips successfully written in this chain.
        total_uploaded: u32,
        /// Whether any uploaded mip in this chain uses V-inverted storage.
        storage_v_inverted: bool,
    },
    /// Chain complete (`total_uploaded` mips in this chain).
    Finished {
        /// Total mips successfully written in this chain.
        total_uploaded: u32,
        /// Whether any uploaded mip in this chain uses V-inverted storage.
        storage_v_inverted: bool,
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
    /// Shared ABBA gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::WriteTextureSubmitGate`].
    pub write_texture_submit_gate: &'a crate::gpu::WriteTextureSubmitGate,
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
    /// Mip width in texels.
    width: u32,
    /// Mip height in texels.
    height: u32,
    /// RGBA8 texel bytes.
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
            storage_v_inverted: false,
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
        if self.stopped {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
                storage_v_inverted: self.storage_v_inverted,
            });
        }

        if let Some(advance) = self.poll_background_decoded_mip(&step)? {
            return Ok(advance);
        }

        self.spawn_upload_next_host_mip(&step)
    }

    /// Drains a completed background decode into a `Queue::write_texture`, or yields if still pending.
    ///
    /// Returns `None` when no background decode is in flight (caller should start one).
    fn poll_background_decoded_mip(
        &mut self,
        step: &TextureMipUploadStep<'_>,
    ) -> Result<Option<MipChainAdvance>, TextureUploadError> {
        let Some(rx) = &self.background_rx else {
            return Ok(None);
        };
        match rx.try_recv() {
            Ok(res) => {
                self.background_rx = None;
                let pixels = res?;
                let (mip_level, gw, gh) = self.pending_mip.take().ok_or_else(|| {
                    TextureUploadError::from(
                        "write_mip_chain: background decode completed without a pending mip slot; state machine desync",
                    )
                })?;

                write_one_mip(&Texture2dMipWrite {
                    queue: step.queue,
                    write_texture_submit_gate: step.write_texture_submit_gate,
                    texture: step.texture,
                    mip_level,
                    width: gw,
                    height: gh,
                    format: step.wgpu_format,
                    bytes: &pixels.bytes,
                })?;

                if is_rgba8_family(step.wgpu_format) {
                    self.last_rgba8_mip = Some(Rgba8Mip {
                        width: gw,
                        height: gh,
                        pixels: pixels.bytes,
                    });
                }
                self.storage_v_inverted |= pixels.storage_v_inverted;
                self.uploaded_mips += 1;
                self.next_i += 1;

                if self.start_base + self.next_i as u32 >= self.mipmap_count {
                    self.stopped = true;
                    return Ok(Some(MipChainAdvance::Finished {
                        total_uploaded: self.uploaded_mips,
                        storage_v_inverted: self.storage_v_inverted,
                    }));
                }
                Ok(Some(MipChainAdvance::UploadedOne {
                    total_uploaded: self.uploaded_mips,
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

    /// Resolves the next host mip slice and spawns a rayon decode; yields or finishes when none remain.
    fn spawn_upload_next_host_mip(
        &mut self,
        step: &TextureMipUploadStep<'_>,
    ) -> Result<MipChainAdvance, TextureUploadError> {
        let chain = MipChainWalkState {
            fmt: step.fmt,
            upload: step.upload,
            payload: step.payload,
            start_bias: self.start_bias,
        };
        let slice = validate_and_resolve_next_mip_slice(
            &chain,
            self.uploaded_mips,
            self.next_i,
            self.start_base,
            self.mipmap_count,
            self.tex_extent,
        )?;
        let (mip_level, gw, gh, mip_index, mip_src_range) = match slice {
            NextMipUploadSlice::ChainDone { total_uploaded } => {
                if self.start_base + (self.next_i as u32) < self.mipmap_count {
                    // Stopping here leaves undefined mips on some Unity uploads; synthesize the tail when we can.
                    return self.spawn_generated_tail_mip(step.wgpu_format, step.upload);
                }
                self.stopped = true;
                return Ok(MipChainAdvance::Finished {
                    total_uploaded,
                    storage_v_inverted: self.storage_v_inverted,
                });
            }
            NextMipUploadSlice::ChainStopped { total_uploaded } => {
                if self.start_base + (self.next_i as u32) < self.mipmap_count
                    && self.last_rgba8_mip.is_some()
                {
                    return self.spawn_generated_tail_mip(step.wgpu_format, step.upload);
                }
                self.stopped = true;
                return Ok(MipChainAdvance::Finished {
                    total_uploaded,
                    storage_v_inverted: self.storage_v_inverted,
                });
            }
            NextMipUploadSlice::Ready {
                mip_level,
                gw,
                gh,
                mip_index,
                mip_src,
            } => {
                let offset = mip_src.as_ptr() as usize - step.payload.as_ptr() as usize;
                let len = mip_src.len();
                (mip_level, gw, gh, mip_index, offset..offset + len)
            }
        };

        self.pending_mip = Some((mip_level, gw, gh));

        let (tx, rx) = crossbeam_channel::bounded(1);
        self.background_rx = Some(rx);

        let ctx = MipUploadFormatCtx {
            asset_id: step.upload.asset_id,
            fmt_format: step.fmt.format,
            wgpu_format: step.wgpu_format,
            needs_rgba8_decode: needs_rgba8_decode_before_upload(step.device, step.fmt.format),
        };
        let flip = self.flip;
        let payload_arc = Arc::clone(step.payload);
        rayon::spawn(move || {
            let mip_src = &payload_arc[mip_src_range];
            let res = mip_src_to_upload_pixels(ctx, gw, gh, flip, mip_src, mip_index);
            let _ = tx.send(res);
        });

        Ok(MipChainAdvance::YieldBackground)
    }

    /// Spawns background generation for one missing RGBA8 tail mip when the host payload ends early.
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
                storage_v_inverted: self.storage_v_inverted,
            });
        }

        if !is_rgba8_family(wgpu_format) {
            self.stopped = true;
            logger::trace!(
                "texture {}: uploaded {}/{} mips; cannot synthesize remaining tail for GPU format {:?}",
                upload.asset_id,
                self.uploaded_mips,
                self.mipmap_count.saturating_sub(self.start_base),
                wgpu_format
            );
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
                storage_v_inverted: self.storage_v_inverted,
            });
        }

        let Some(source) = self.last_rgba8_mip.clone() else {
            self.stopped = true;
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
                storage_v_inverted: self.storage_v_inverted,
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
            let res = downsample_rgba8_box(&source.pixels, source.width, source.height, w, h)
                .map(MipUploadPixels::normal);
            let _ = tx.send(res);
        });

        Ok(MipChainAdvance::YieldBackground)
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

/// GPU target, host format, and raw payload for one [`texture_upload_start`] / [`write_texture2d_mips`] call.
pub struct Texture2dUploadContext<'a> {
    /// Device for decode-path capability checks.
    pub device: &'a wgpu::Device,
    /// Queue for texel copies.
    pub queue: &'a wgpu::Queue,
    /// Shared ABBA gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::WriteTextureSubmitGate`].
    pub write_texture_submit_gate: &'a crate::gpu::WriteTextureSubmitGate,
    /// Destination texture (must match `fmt` dimensions).
    pub texture: &'a wgpu::Texture,
    /// Host-side format descriptor (dimensions, mip count, texel format).
    pub fmt: &'a SetTexture2DFormat,
    /// Resolved GPU storage format.
    pub wgpu_format: wgpu::TextureFormat,
    /// Upload record (mip starts, region hint, descriptor length).
    pub upload: &'a SetTexture2DData,
    /// Raw shared-memory bytes covering the descriptor window.
    pub raw: &'a [u8],
}

/// Classifies sub-region vs full mip chain and runs the sub-region upload when applicable.
pub fn texture_upload_start(
    ctx: &Texture2dUploadContext<'_>,
) -> Result<TextureDataStart, TextureUploadError> {
    if ctx.upload.hint.has_region != 0 {
        if hint_region_is_empty(&ctx.upload.hint) {
            logger::trace!(
                "texture {}: TextureUploadHint.has_region set but region empty; skipping upload",
                ctx.upload.asset_id
            );
            return Ok(TextureDataStart::SubregionComplete(0));
        }
        match try_write_texture2d_subregion(ctx) {
            Some(Ok(n)) => {
                logger::trace!(
                    "texture {}: sub-region texture upload ({} mips equivalent)",
                    ctx.upload.asset_id,
                    n
                );
                return Ok(TextureDataStart::SubregionComplete(n));
            }
            Some(Err(e)) => return Err(e),
            None => {
                logger::trace!(
                    "texture {}: TextureUploadHint.has_region set; using full mip upload path",
                    ctx.upload.asset_id
                );
            }
        }
    }
    Ok(TextureDataStart::MipChain(TextureMipChainUploader::new(
        ctx.texture,
        ctx.fmt,
        ctx.upload,
        ctx.raw,
    )?))
}

/// Uploads mips from `ctx.raw` (exact shared-memory descriptor window) into `ctx.texture` using `ctx.wgpu_format`.
pub fn write_texture2d_mips(ctx: &Texture2dUploadContext<'_>) -> Result<u32, TextureUploadError> {
    let want = ctx.upload.data.length.max(0) as usize;
    if ctx.raw.len() < want {
        return Err(TextureUploadError::from(format!(
            "raw shorter than descriptor (need {want}, got {})",
            ctx.raw.len()
        )));
    }
    let payload = Arc::from(&ctx.raw[..want]);

    match texture_upload_start(ctx)? {
        TextureDataStart::SubregionComplete(n) => Ok(n),
        TextureDataStart::MipChain(mut uploader) => loop {
            match uploader.upload_next_mip(TextureMipUploadStep {
                device: ctx.device,
                queue: ctx.queue,
                write_texture_submit_gate: ctx.write_texture_submit_gate,
                texture: ctx.texture,
                fmt: ctx.fmt,
                wgpu_format: ctx.wgpu_format,
                upload: ctx.upload,
                payload: &payload,
            })? {
                MipChainAdvance::UploadedOne { .. } => {}
                MipChainAdvance::Finished { total_uploaded, .. } => {
                    return Ok(total_uploaded);
                }
                MipChainAdvance::YieldBackground => {
                    std::thread::yield_now();
                }
            }
        },
    }
}
