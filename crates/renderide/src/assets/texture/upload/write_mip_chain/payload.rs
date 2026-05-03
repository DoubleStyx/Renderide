//! Host mip metadata validation and payload-slice resolution for 2D mip-chain uploads.

use crate::assets::texture::layout::host_mip_payload_byte_offset;
use crate::shared::{SetTexture2DData, SetTexture2DFormat};

use super::super::super::layout::{mip_byte_len, mip_dimensions_at_level};
use super::super::TextureUploadError;
use super::super::mip_write_common::choose_mip_start_bias;

/// Shared device, host format, and payload window for walking a 2D mip chain.
pub(super) struct MipChainWalkState<'a> {
    /// Host-side texture format and mip count.
    pub fmt: &'a SetTexture2DFormat,
    /// Upload metadata including mip starts and descriptor window.
    pub upload: &'a SetTexture2DData,
    /// Owned shared-memory descriptor window used across integration ticks.
    pub payload: &'a [u8],
    /// Descriptor-relative byte offset used to rebase host `mip_starts`.
    pub start_bias: usize,
}

/// Outcome of [`validate_and_resolve_next_mip_slice`] for one [`TextureMipChainUploader`] step.
#[expect(
    variant_size_differences,
    reason = "short-lived per-mip outcome; boxing `Ready` would add an allocation per mip"
)]
pub(super) enum NextMipUploadSlice<'a> {
    /// Stop iteration: chain finished normally (`uploaded_mips` may be zero only when no mip was ever uploaded -- caller treats as error in that case).
    ChainDone {
        /// Number of mip levels successfully uploaded before stopping.
        total_uploaded: u32,
    },
    /// Stop iteration: truncated payload or negative offset (`stopped` flag should be set on the uploader).
    ChainStopped {
        /// Number of mip levels successfully uploaded before stopping.
        total_uploaded: u32,
    },
    /// GPU dimensions and source bytes for this mip level.
    Ready {
        /// Destination mip level in the GPU texture.
        mip_level: u32,
        /// GPU mip width in texels.
        gw: u32,
        /// GPU mip height in texels.
        gh: u32,
        /// Host mip-array index.
        mip_index: usize,
        /// Host payload slice for this mip.
        mip_src: &'a [u8],
    },
}

/// Resolved host payload for one mip level before GPU dimensions from [`mip_dimensions_at_level`] are merged in.
enum HostMipPayloadResolved<'a> {
    /// Stop iteration: truncated payload or negative offset (`stopped` flag should be set on the uploader).
    Stopped {
        /// Number of mip levels successfully uploaded before stopping.
        total_uploaded: u32,
    },
    /// Host payload subslice for this mip (dimensions come from [`validate_and_resolve_next_mip_slice`]).
    Slice {
        /// Host payload slice for the mip.
        mip_src: &'a [u8],
    },
}

/// Per-mip indices for [`resolve_mip_host_payload_slice`].
struct MipHostPayloadResolveStep {
    /// Number of mips already written before this step.
    uploaded_mips: u32,
    /// Host mip-array index to resolve.
    next_i: usize,
    /// Destination mip level in the GPU texture.
    mip_level: u32,
    /// Host-declared mip width in texels.
    w: u32,
    /// Host-declared mip height in texels.
    h: u32,
    /// Count of mips whose offsets fit inside the descriptor window before rebasing.
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
pub(super) fn validate_and_resolve_next_mip_slice<'a>(
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
