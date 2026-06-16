//! Hi-Z pyramid copy-to-staging.
//!
//! Copies one history-texture array layer's full mip chain into the active write-slot staging
//! ring entry for asynchronous CPU readback.

use crate::hi_z_cpu::pyramid::mip_dimensions;

use super::EncodeSession;

/// Copies the active write-slot pyramid for `array_layer` into its staging ring entry.
///
/// `right_eye` selects the stereo-right staging ring; the desktop / stereo-left layer always
/// targets `staging_desktop`.
pub(super) fn copy_layer(
    session: &mut EncodeSession<'_>,
    history_texture: &wgpu::Texture,
    ws: usize,
    array_layer: u32,
    right_eye: bool,
) {
    let label = if right_eye {
        "hi_z::copy_pyramid_to_staging.right"
    } else {
        "hi_z::copy_pyramid_to_staging.left"
    };
    let copy_query = session
        .profiler
        .map(|p| p.begin_query(label, session.encoder));
    let (bw, bh) = session.scratch.extent;
    let mip_levels = session.scratch.mip_levels;
    let staging = if right_eye {
        let Some(staging_r) = session.scratch.staging_right() else {
            if let (Some(p), Some(q)) = (session.profiler, copy_query) {
                p.end_query(session.encoder, q);
            }
            return;
        };
        &staging_r[ws]
    } else {
        &session.scratch.staging_desktop[ws]
    };
    copy_pyramid_to_staging(
        session.encoder,
        history_texture,
        array_layer,
        bw,
        bh,
        mip_levels,
        staging,
    );
    if let (Some(p), Some(q)) = (session.profiler, copy_query) {
        p.end_query(session.encoder, q);
    }
}

/// Copies all mips for one history texture array layer into the selected readback staging buffer.
fn copy_pyramid_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    array_layer: u32,
    base_w: u32,
    base_h: u32,
    mip_levels: u32,
    staging: &wgpu::Buffer,
) {
    let mut offset = 0u64;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap_or((1, 1));
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: array_layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        offset += u64::from(row_pitch) * u64::from(h);
    }
}
