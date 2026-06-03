//! Shared mip offset validation and [`wgpu::Queue::write_texture`] layout for full mip and subregion paths.

use crate::shared::SetTexture2DData;

mod cubemap_face;
mod region;
mod texture2d;
mod volume;

pub(super) use cubemap_face::{
    CubemapFaceMipWrite, write_cubemap_face_mip, write_cubemap_face_mip_with_gate,
};
pub(super) use region::{TextureRegionWrite, write_texture_region};
#[cfg(test)]
pub(super) use region::{copy_extent_for_mip, copy_layout_for_mip};
pub(super) use texture2d::{Texture2dMipWrite, write_one_mip, write_one_mip_with_gate};
pub(super) use volume::{
    Texture3dVolumeMipWrite, write_texture3d_volume_mip, write_texture3d_volume_mip_with_gate,
};

use super::super::decode::decode_mip_to_rgba8;
use super::super::layout::{host_format_is_compressed, mip_byte_len};
use super::error::TextureUploadError;
use super::mip_chain_walk::{MipChainStop, resolve_mip_payload_slot};

/// Format-side context shared by every mip in one texture upload (2D, cubemap, 3D).
///
/// Bundled so the per-mip decode functions don't take the same four handles on every call.
/// Fields are [`Copy`] so the context can be captured into an asset-worker closure by value.
#[derive(Copy, Clone)]
pub(super) struct MipUploadFormatCtx {
    /// Host asset id for logging and diagnostics.
    pub asset_id: i32,
    /// Host-side texel format from the upload descriptor.
    pub fmt_format: crate::shared::TextureFormat,
    /// GPU-facing texel format the material system expects.
    pub wgpu_format: wgpu::TextureFormat,
    /// Whether host bytes must be decoded to RGBA8 before upload.
    pub needs_rgba8_decode: bool,
}

/// CPU-side bytes for one mip plus the storage-orientation flag.
///
/// The renderer uploads host texture bytes as-is (Unity V=0 bottom). For host-uploaded 2D
/// textures, `storage_v_inverted` is `true` because the bytes are in Unity orientation. Cubemap
/// upload paths override the flag because their face layout is already native cube orientation.
/// For renderer-baked sources it is `false`.
#[derive(Debug)]
pub(super) struct MipUploadPixels {
    /// Bytes ready for [`wgpu::Queue::write_texture`].
    pub bytes: Vec<u8>,
    /// Whether the bytes need shader-side storage-orientation compensation.
    pub storage_v_inverted: bool,
}

impl MipUploadPixels {
    /// Builds an upload from host bytes (Unity V=0 bottom).
    pub fn host(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            storage_v_inverted: true,
        }
    }

    /// Backwards-compatible alias for [`Self::host`] retained for renderer-baked tail synthesis.
    pub fn normal(bytes: Vec<u8>) -> Self {
        Self::host(bytes)
    }

    /// Returns this upload with an explicit storage-orientation flag.
    pub fn with_storage_v_inverted(mut self, storage_v_inverted: bool) -> Self {
        self.storage_v_inverted = storage_v_inverted;
        self
    }
}

/// Texture family being converted for upload diagnostics.
#[derive(Copy, Clone, Debug)]
pub(super) enum MipUploadKind {
    /// Texture2D mip-chain upload.
    Texture2d,
    /// Cubemap face upload.
    Cubemap {
        /// Cubemap face index.
        face: u32,
    },
}

/// Per-mip label used to keep shared conversion diagnostics clear.
#[derive(Copy, Clone, Debug)]
pub(super) struct MipUploadLabel {
    /// Upload family.
    pub kind: MipUploadKind,
    /// Mip index inside the texture or cubemap face.
    pub mip_index: usize,
}

impl MipUploadLabel {
    /// Builds a label for a Texture2D mip.
    pub fn texture2d(mip_index: usize) -> Self {
        Self {
            kind: MipUploadKind::Texture2d,
            mip_index,
        }
    }

    /// Builds a label for one cubemap face mip.
    pub fn cubemap(face: u32, mip_index: usize) -> Self {
        Self {
            kind: MipUploadKind::Cubemap { face },
            mip_index,
        }
    }

    /// Asset-qualified diagnostic label.
    fn asset_mip(self, asset_id: i32) -> String {
        match self.kind {
            MipUploadKind::Texture2d => format!("texture {asset_id} mip {}", self.mip_index),
            MipUploadKind::Cubemap { face } => {
                format!("cubemap {asset_id} face {face} mip {}", self.mip_index)
            }
        }
    }
}

/// Whether a sampled Texture2D upload keeps host bytes in Unity orientation.
pub(crate) fn upload_uses_storage_v_inversion(
    _host_format: crate::shared::TextureFormat,
    _wgpu_format: wgpu::TextureFormat,
    _flip_y: bool,
) -> bool {
    true
}

/// Picks the descriptor offset bias that maximizes how many mips fit in the SHM payload.
pub(super) fn choose_mip_start_bias(
    format: crate::shared::TextureFormat,
    upload: &SetTexture2DData,
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
        let prefix = valid_mip_prefix_len(format, upload, payload_len, bias)?;
        if prefix > best_prefix {
            best_prefix = prefix;
            best_bias = bias;
        }
    }
    if best_prefix == 0 {
        return Err(TextureUploadError::from(format!(
            "mip region exceeds shared memory descriptor (payload_len={payload_len}, descriptor_offset={offset_bias})"
        )));
    }
    Ok((best_bias, best_prefix))
}

/// Counts how many descriptor mips fit inside `payload_len` after applying `bias`.
pub(super) fn valid_mip_prefix_len(
    format: crate::shared::TextureFormat,
    upload: &SetTexture2DData,
    payload_len: usize,
    bias: usize,
) -> Result<usize, TextureUploadError> {
    let mut count = 0usize;
    for (i, sz) in upload.mip_map_sizes.iter().enumerate() {
        if sz.x <= 0 || sz.y <= 0 {
            return Err("non-positive mip dimensions".into());
        }
        let w = sz.x as u32;
        let h = sz.y as u32;
        let host_len = mip_byte_len(format, w, h).ok_or_else(|| {
            TextureUploadError::from(format!("mip byte size unsupported for {format:?}"))
        })? as usize;
        let start_raw = upload.mip_starts[i];
        match resolve_mip_payload_slot(format, host_len, start_raw, bias, payload_len, || {
            format!("mip {i}")
        })? {
            Ok(()) => count += 1,
            Err(
                MipChainStop::NegativeStart | MipChainStop::BeforeBias | MipChainStop::OutOfPayload,
            ) => break,
        }
    }
    Ok(count)
}

/// Returns whether `gpu` is an RGBA8 texture format accepted by the direct upload path.
pub(super) fn is_rgba8_family(gpu: wgpu::TextureFormat) -> bool {
    matches!(
        gpu,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
    )
}

/// Validates host mip bytes and produces a buffer for a full-mip texture upload.
///
/// The renderer no longer flips host data on ingestion. Sampled textures use Unity (V=0 bottom)
/// orientation throughout: host bytes are stored as-is, mesh UVs match, and shaders apply no V flip.
/// Cubemap face orientation is the only remaining storage-orientation concern; that is handled in
/// the cubemap pool and the projection360 sampling helpers, not here.
///
/// `_flip_y` is currently ignored: the host contract is to send Unity-oriented bytes regardless.
/// The parameter is retained for IPC compatibility and to keep call sites stable while the host
/// continues to set it.
pub(super) fn mip_src_to_upload_pixels(
    ctx: MipUploadFormatCtx,
    width: u32,
    height: u32,
    _flip_y: bool,
    mip_src: &[u8],
    label: MipUploadLabel,
) -> Result<MipUploadPixels, TextureUploadError> {
    profiling::scope!("asset::texture_convert_mip_pixels");
    let MipUploadFormatCtx {
        asset_id,
        fmt_format,
        wgpu_format,
        needs_rgba8_decode,
    } = ctx;
    let bytes_result: Result<Vec<u8>, TextureUploadError> = if is_rgba8_family(wgpu_format) {
        if needs_rgba8_decode || host_format_is_compressed(fmt_format) {
            decode_mip_to_rgba8(fmt_format, width, height, false, mip_src).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "RGBA decode failed for {} ({:?})",
                    label.asset_mip(asset_id),
                    fmt_format
                ))
            })
        } else {
            Ok(mip_src.to_vec())
        }
    } else if needs_rgba8_decode {
        Err(TextureUploadError::from(format!(
            "host {fmt_format:?} must use RGBA decode but GPU format is {wgpu_format:?}"
        )))
    } else if host_format_is_compressed(fmt_format) {
        let expected_len = mip_byte_len(fmt_format, width, height).ok_or_else(|| {
            TextureUploadError::from(format!(
                "{}: mip byte size unknown for {:?}",
                label.asset_mip(asset_id),
                fmt_format
            ))
        })? as usize;
        if mip_src.len() != expected_len {
            return Err(TextureUploadError::from(format!(
                "{}: mip len {} != expected {} for {:?}",
                label.asset_mip(asset_id),
                mip_src.len(),
                expected_len,
                fmt_format
            )));
        }
        Ok(mip_src.to_vec())
    } else {
        Ok(mip_src.to_vec())
    };
    bytes_result.map(MipUploadPixels::host)
}

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use super::{
        MipUploadFormatCtx, MipUploadLabel, choose_mip_start_bias, copy_extent_for_mip,
        copy_layout_for_mip, is_rgba8_family, mip_src_to_upload_pixels,
        upload_uses_storage_v_inversion, valid_mip_prefix_len,
    };
    use crate::shared::{SetTexture2DData, TextureFormat};

    #[test]
    fn relative_mip_starts_need_no_rebase() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 80;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        // `mip_starts` are linear texel indices into the chain; texel 16 begins the 2x2 mip (byte 64).
        upload.mip_starts = vec![0, 16];

        let (bias, prefix) = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 80).unwrap();
        assert_eq!(bias, 0);
        assert_eq!(prefix, 2);
    }

    #[test]
    fn absolute_mip_starts_rebase_to_descriptor_offset() {
        let mut upload = SetTexture2DData::default();
        upload.data.offset = 128;
        upload.data.length = 80;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        // Absolute SHM indices: base mip at descriptor offset; second mip at texel 144 (= 128 + 16).
        upload.mip_starts = vec![128, 144];

        let (bias, prefix) = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 80).unwrap();
        assert_eq!(bias, 128);
        assert_eq!(prefix, 2);
    }

    #[test]
    fn valid_prefix_len_stops_when_later_mip_exceeds_payload() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 68;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![0, 64];

        let prefix = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 68, 0).unwrap();
        assert_eq!(prefix, 1);
    }

    #[test]
    fn choose_mip_start_bias_rejects_uploads_with_no_fitting_mips() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 4;
        upload.mip_map_sizes = vec![IVec2::new(4, 4)];
        upload.mip_starts = vec![0];

        let err = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 4)
            .expect_err("no mip fits payload");

        assert!(
            err.to_string()
                .contains("mip region exceeds shared memory descriptor"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn valid_prefix_len_errors_on_non_positive_dimensions() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 64;
        upload.mip_map_sizes = vec![IVec2::new(4, 0)];
        upload.mip_starts = vec![0];

        let err = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 64, 0)
            .expect_err("bad mip dimensions");

        assert!(err.to_string().contains("non-positive mip dimensions"));
    }

    #[test]
    fn valid_prefix_len_stops_at_negative_tail_start() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 64;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![0, -1];

        let prefix = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 64, 0).unwrap();
        assert_eq!(prefix, 1);
    }

    #[test]
    fn valid_prefix_len_stops_when_start_is_before_bias() {
        let mut upload = SetTexture2DData::default();
        upload.data.offset = 128;
        upload.data.length = 64;
        upload.mip_map_sizes = vec![IVec2::new(4, 4)];
        upload.mip_starts = vec![64];

        let prefix = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 64, 128).unwrap();

        assert_eq!(prefix, 0);
    }

    #[test]
    fn valid_prefix_len_uses_compressed_byte_offsets() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 8;
        upload.mip_map_sizes = vec![IVec2::new(4, 4)];
        upload.mip_starts = vec![0];
        assert_eq!(
            valid_mip_prefix_len(TextureFormat::BC1, &upload, 8, 0).unwrap(),
            1
        );

        upload.mip_starts = vec![16];
        assert_eq!(
            valid_mip_prefix_len(TextureFormat::BC1, &upload, 8, 0).unwrap(),
            0
        );
    }

    #[test]
    fn copy_extent_aligns_block_compressed_mips() {
        let extent = copy_extent_for_mip(wgpu::TextureFormat::Bc1RgbaUnorm, 7, 5, 1);

        assert_eq!(extent.width, 8);
        assert_eq!(extent.height, 8);
        assert_eq!(extent.depth_or_array_layers, 1);
    }

    #[test]
    fn copy_extent_keeps_uncompressed_mips_tight() {
        let extent = copy_extent_for_mip(wgpu::TextureFormat::Rgba8Unorm, 7, 5, 3);

        assert_eq!(extent.width, 7);
        assert_eq!(extent.height, 5);
        assert_eq!(extent.depth_or_array_layers, 3);
    }

    #[test]
    fn copy_layout_for_mip_reports_tight_uncompressed_layout() {
        let (layout, expected) =
            copy_layout_for_mip(wgpu::TextureFormat::Rgba8Unorm, 3, 2).expect("layout");

        assert_eq!(layout.bytes_per_row, Some(12));
        assert_eq!(layout.rows_per_image, Some(2));
        assert_eq!(expected, 24);
    }

    #[test]
    fn copy_layout_for_mip_reports_block_compressed_layout() {
        let (layout, expected) =
            copy_layout_for_mip(wgpu::TextureFormat::Bc1RgbaUnorm, 5, 5).expect("layout");

        assert_eq!(layout.bytes_per_row, Some(16));
        assert_eq!(layout.rows_per_image, Some(2));
        assert_eq!(expected, 32);
    }

    #[test]
    fn rgba8_family_and_storage_orientation_contract_are_stable() {
        assert!(is_rgba8_family(wgpu::TextureFormat::Rgba8Unorm));
        assert!(is_rgba8_family(wgpu::TextureFormat::Rgba8UnormSrgb));
        assert!(!is_rgba8_family(wgpu::TextureFormat::Bgra8Unorm));
        assert!(upload_uses_storage_v_inversion(
            TextureFormat::RGBA32,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        ));
    }

    #[test]
    fn mip_src_to_upload_pixels_direct_rgba8_keeps_bytes_and_marks_host_orientation() {
        let ctx = MipUploadFormatCtx {
            asset_id: 7,
            fmt_format: TextureFormat::RGBA32,
            wgpu_format: wgpu::TextureFormat::Rgba8Unorm,
            needs_rgba8_decode: false,
        };

        let pixels = mip_src_to_upload_pixels(
            ctx,
            1,
            1,
            false,
            &[1, 2, 3, 4],
            MipUploadLabel::texture2d(0),
        )
        .expect("direct rgba upload");

        assert_eq!(pixels.bytes, vec![1, 2, 3, 4]);
        assert!(pixels.storage_v_inverted);
    }
}
