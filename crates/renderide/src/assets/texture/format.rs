//! Maps host [`TextureFormat`] + [`ColorProfile`] to [`wgpu::TextureFormat`] when the device reports required compression features.

use crate::shared::{ColorProfile, TextureFormat};

/// Picks a [`wgpu::TextureFormat`] for `host` if this device advertises the needed [`wgpu::Features`].
///
/// Returns [`None`] when the combination is unknown or compression features are missing (caller may decode to `Rgba8UnormSrgb`).
pub fn pick_wgpu_storage_format(
    device: &wgpu::Device,
    host: TextureFormat,
    profile: ColorProfile,
) -> Option<wgpu::TextureFormat> {
    let f = map_host_format(host, profile)?;
    if texture_format_supported(device, f) {
        Some(f)
    } else {
        None
    }
}

/// Maps host format without feature checks (for estimating sizes or documentation).
pub fn map_host_format(host: TextureFormat, profile: ColorProfile) -> Option<wgpu::TextureFormat> {
    use ColorProfile::{s_rgb, s_rgb_alpha};
    use TextureFormat::*;

    let srgb = matches!(profile, s_rgb | s_rgb_alpha);

    Some(match host {
        unknown => return None,
        alpha8 | r8 => wgpu::TextureFormat::R8Unorm,
        rgb24 | rgb565 | bgr565 => return None, // decode path
        rgba32 => {
            if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            }
        }
        argb32 | bgra32 => {
            if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            }
        }
        rgba_half | argb_half => wgpu::TextureFormat::Rgba16Float,
        r_half => wgpu::TextureFormat::R16Float,
        rg_half => wgpu::TextureFormat::Rg16Float,
        rgba_float | argb_float => wgpu::TextureFormat::Rgba32Float,
        r_float => wgpu::TextureFormat::R32Float,
        rg_float => wgpu::TextureFormat::Rg32Float,
        bc1 => {
            if srgb {
                wgpu::TextureFormat::Bc1RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc1RgbaUnorm
            }
        }
        bc2 => {
            if srgb {
                wgpu::TextureFormat::Bc2RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc2RgbaUnorm
            }
        }
        bc3 => {
            if srgb {
                wgpu::TextureFormat::Bc3RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc3RgbaUnorm
            }
        }
        bc4 => wgpu::TextureFormat::Bc4RUnorm,
        bc5 => wgpu::TextureFormat::Bc5RgUnorm,
        bc6_h => wgpu::TextureFormat::Bc6hRgbUfloat,
        bc7 => {
            if srgb {
                wgpu::TextureFormat::Bc7RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc7RgbaUnorm
            }
        }
        etc2_rgb => {
            if srgb {
                wgpu::TextureFormat::Etc2Rgb8UnormSrgb
            } else {
                wgpu::TextureFormat::Etc2Rgb8Unorm
            }
        }
        etc2_rgba1 => {
            if srgb {
                wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb
            } else {
                wgpu::TextureFormat::Etc2Rgb8A1Unorm
            }
        }
        etc2_rgba8 => {
            if srgb {
                wgpu::TextureFormat::Etc2Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Etc2Rgba8Unorm
            }
        }
        astc_4x4 => astc_wgpu(wgpu::AstcBlock::B4x4, srgb),
        astc_5x5 => astc_wgpu(wgpu::AstcBlock::B5x5, srgb),
        astc_6x6 => astc_wgpu(wgpu::AstcBlock::B6x6, srgb),
        astc_8x8 => astc_wgpu(wgpu::AstcBlock::B8x8, srgb),
        astc_10x10 => astc_wgpu(wgpu::AstcBlock::B10x10, srgb),
        astc_12x12 => astc_wgpu(wgpu::AstcBlock::B12x12, srgb),
    })
}

fn astc_wgpu(block: wgpu::AstcBlock, srgb: bool) -> wgpu::TextureFormat {
    let channel = if srgb {
        wgpu::AstcChannel::UnormSrgb
    } else {
        wgpu::AstcChannel::Unorm
    };
    wgpu::TextureFormat::Astc { block, channel }
}

fn texture_format_supported(device: &wgpu::Device, format: wgpu::TextureFormat) -> bool {
    if !format.is_compressed() {
        return true;
    }
    let feats = device.features();
    if format_required_bc(format) && !feats.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        return false;
    }
    if format_required_etc2(format) && !feats.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
        return false;
    }
    if format_required_astc(format) && !feats.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
        return false;
    }
    true
}

fn format_required_bc(f: wgpu::TextureFormat) -> bool {
    matches!(
        f,
        wgpu::TextureFormat::Bc1RgbaUnorm
            | wgpu::TextureFormat::Bc1RgbaUnormSrgb
            | wgpu::TextureFormat::Bc2RgbaUnorm
            | wgpu::TextureFormat::Bc2RgbaUnormSrgb
            | wgpu::TextureFormat::Bc3RgbaUnorm
            | wgpu::TextureFormat::Bc3RgbaUnormSrgb
            | wgpu::TextureFormat::Bc4RUnorm
            | wgpu::TextureFormat::Bc4RSnorm
            | wgpu::TextureFormat::Bc5RgUnorm
            | wgpu::TextureFormat::Bc5RgSnorm
            | wgpu::TextureFormat::Bc6hRgbUfloat
            | wgpu::TextureFormat::Bc6hRgbFloat
            | wgpu::TextureFormat::Bc7RgbaUnorm
            | wgpu::TextureFormat::Bc7RgbaUnormSrgb
    )
}

fn format_required_etc2(f: wgpu::TextureFormat) -> bool {
    matches!(
        f,
        wgpu::TextureFormat::Etc2Rgb8Unorm
            | wgpu::TextureFormat::Etc2Rgb8UnormSrgb
            | wgpu::TextureFormat::Etc2Rgb8A1Unorm
            | wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb
            | wgpu::TextureFormat::Etc2Rgba8Unorm
            | wgpu::TextureFormat::Etc2Rgba8UnormSrgb
    )
}

fn format_required_astc(f: wgpu::TextureFormat) -> bool {
    matches!(
        f,
        wgpu::TextureFormat::Astc {
            channel: wgpu::AstcChannel::Unorm | wgpu::AstcChannel::UnormSrgb,
            ..
        }
    )
}

/// Formats we can accept via GPU-native storage or transient RGBA8 decode (advertised to the host).
pub fn supported_host_formats_for_init() -> Vec<TextureFormat> {
    use TextureFormat::*;
    vec![
        alpha8, r8, rgb24, rgba32, argb32, bgra32, rgb565, bgr565, rgba_half, argb_half, r_half,
        rg_half, rgba_float, argb_float, r_float, rg_float, bc1, bc2, bc3, bc4, bc5, bc6_h, bc7,
        etc2_rgb, etc2_rgba1, etc2_rgba8, astc_4x4, astc_5x5, astc_6x6, astc_8x8, astc_10x10,
        astc_12x12,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba32_linear_maps() {
        assert_eq!(
            map_host_format(TextureFormat::rgba32, ColorProfile::linear),
            Some(wgpu::TextureFormat::Rgba8Unorm)
        );
    }
}
