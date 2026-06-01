//! Placeholder textures used as default bindings for unset or nonresident `@group(1)` texture slots.

use std::sync::Arc;

use super::super::bind_kind::TextureBindKind;

/// Width and height of the generated checkerboard fallback texture.
pub(super) const CHECKERBOARD_TEXTURE_SIZE: u32 = 128;
/// Width and height of each checkerboard cell.
pub(super) const CHECKERBOARD_CELL_SIZE: u32 = 16;
/// Dark texel used by the nonresident texture checkerboard.
pub(super) const CHECKERBOARD_DARK_RGBA: [u8; 4] = [0, 0, 0, 255];
/// Light texel used by the nonresident texture checkerboard.
pub(super) const CHECKERBOARD_LIGHT_RGBA: [u8; 4] = [13, 13, 13, 255];

/// Placeholder 1x1 texture for one [`TextureBindKind`].
pub(super) struct PlaceholderTexture {
    /// Underlying device texture.
    pub(super) texture: Arc<wgpu::Texture>,
    /// Default texture view used at binding time.
    pub(super) view: Arc<wgpu::TextureView>,
}

/// Texel color uploaded into a placeholder texture.
#[derive(Clone, Copy)]
pub(super) enum PlaceholderTextureColor {
    /// Opaque white texel.
    White,
    /// Opaque black texel.
    Black,
    /// Opaque middle-gray texel sampled as a linear value.
    Gray,
    /// Opaque middle-gray texel sampled through sRGB decode.
    SrgbGray,
    /// Opaque red texel.
    Red,
    /// Unity bump placeholder texel `(0.5, 0.5, 1.0, 0.5)`.
    FlatNormal,
}

impl PlaceholderTextureColor {
    fn rgba(self) -> [u8; 4] {
        match self {
            Self::White => [255, 255, 255, 255],
            Self::Black => [0, 0, 0, 255],
            Self::Gray | Self::SrgbGray => [128, 128, 128, 255],
            Self::Red => [255, 0, 0, 255],
            Self::FlatNormal => [128, 128, 255, 128],
        }
    }

    /// Texture format whose sampler decode preserves [`Self::rgba`] as the intended linear value.
    ///
    /// `White`, `Black`, and `Red` keep the sRGB format the color-slot placeholders shipped with
    /// so 0 and 1 round-trip identically. `SrgbGray` is reserved for Unity gray color-texture
    /// defaults such as detail albedo. `Gray` and `FlatNormal` must stay linear (`Rgba8Unorm`):
    /// a component of 0.5 is stored as byte 128, and the sRGB EOTF would decode that as about
    /// 0.216 instead of about 0.502.
    fn format(self) -> wgpu::TextureFormat {
        match self {
            Self::White | Self::Black | Self::SrgbGray | Self::Red => {
                wgpu::TextureFormat::Rgba8UnormSrgb
            }
            Self::Gray | Self::FlatNormal => wgpu::TextureFormat::Rgba8Unorm,
        }
    }
}

impl TextureBindKind {
    /// Texture descriptor parameters (label, dimension, layer count, view dimension, format).
    fn placeholder_descriptor(self, color: PlaceholderTextureColor) -> PlaceholderDescriptor {
        let format = color.format();
        match self {
            TextureBindKind::Tex2D => PlaceholderDescriptor {
                label: tex2d_placeholder_label(color),
                view_label: None,
                dimension: wgpu::TextureDimension::D2,
                view_dimension: None,
                depth_or_array_layers: 1,
                format,
            },
            TextureBindKind::Tex3D => {
                let labels = tex3d_placeholder_labels(color);
                PlaceholderDescriptor {
                    label: labels.label,
                    view_label: Some(labels.view_label),
                    dimension: wgpu::TextureDimension::D3,
                    view_dimension: Some(wgpu::TextureViewDimension::D3),
                    depth_or_array_layers: 1,
                    format,
                }
            }
            TextureBindKind::Cube => {
                let labels = cube_placeholder_labels(color);
                PlaceholderDescriptor {
                    label: labels.label,
                    view_label: Some(labels.view_label),
                    dimension: wgpu::TextureDimension::D2,
                    view_dimension: Some(wgpu::TextureViewDimension::Cube),
                    depth_or_array_layers: 6,
                    format,
                }
            }
        }
    }
}

struct PlaceholderLabels {
    label: &'static str,
    view_label: &'static str,
}

fn tex2d_placeholder_label(color: PlaceholderTextureColor) -> &'static str {
    match color {
        PlaceholderTextureColor::White => "embedded_default_white",
        PlaceholderTextureColor::Black => "embedded_default_black",
        PlaceholderTextureColor::Gray => "embedded_default_gray",
        PlaceholderTextureColor::SrgbGray => "embedded_default_gray_srgb",
        PlaceholderTextureColor::Red => "embedded_default_red",
        PlaceholderTextureColor::FlatNormal => "embedded_default_flat_normal",
    }
}

fn tex3d_placeholder_labels(color: PlaceholderTextureColor) -> PlaceholderLabels {
    match color {
        PlaceholderTextureColor::White => PlaceholderLabels {
            label: "embedded_default_white_3d",
            view_label: "embedded_default_white_3d_view",
        },
        PlaceholderTextureColor::Black => PlaceholderLabels {
            label: "embedded_default_black_3d",
            view_label: "embedded_default_black_3d_view",
        },
        PlaceholderTextureColor::Gray => PlaceholderLabels {
            label: "embedded_default_gray_3d",
            view_label: "embedded_default_gray_3d_view",
        },
        PlaceholderTextureColor::SrgbGray => PlaceholderLabels {
            label: "embedded_default_gray_srgb_3d",
            view_label: "embedded_default_gray_srgb_3d_view",
        },
        PlaceholderTextureColor::Red => PlaceholderLabels {
            label: "embedded_default_red_3d",
            view_label: "embedded_default_red_3d_view",
        },
        PlaceholderTextureColor::FlatNormal => PlaceholderLabels {
            label: "embedded_default_flat_normal_3d",
            view_label: "embedded_default_flat_normal_3d_view",
        },
    }
}

fn cube_placeholder_labels(color: PlaceholderTextureColor) -> PlaceholderLabels {
    match color {
        PlaceholderTextureColor::White => PlaceholderLabels {
            label: "embedded_default_white_cube",
            view_label: "embedded_default_white_cube_view",
        },
        PlaceholderTextureColor::Black => PlaceholderLabels {
            label: "embedded_default_black_cube",
            view_label: "embedded_default_black_cube_view",
        },
        PlaceholderTextureColor::Gray => PlaceholderLabels {
            label: "embedded_default_gray_cube",
            view_label: "embedded_default_gray_cube_view",
        },
        PlaceholderTextureColor::SrgbGray => PlaceholderLabels {
            label: "embedded_default_gray_srgb_cube",
            view_label: "embedded_default_gray_srgb_cube_view",
        },
        PlaceholderTextureColor::Red => PlaceholderLabels {
            label: "embedded_default_red_cube",
            view_label: "embedded_default_red_cube_view",
        },
        PlaceholderTextureColor::FlatNormal => PlaceholderLabels {
            label: "embedded_default_flat_normal_cube",
            view_label: "embedded_default_flat_normal_cube_view",
        },
    }
}

struct PlaceholderDescriptor {
    label: &'static str,
    view_label: Option<&'static str>,
    dimension: wgpu::TextureDimension,
    view_dimension: Option<wgpu::TextureViewDimension>,
    depth_or_array_layers: u32,
    format: wgpu::TextureFormat,
}

fn create_placeholder(
    device: &wgpu::Device,
    kind: TextureBindKind,
    color: PlaceholderTextureColor,
) -> PlaceholderTexture {
    let desc = kind.placeholder_descriptor(color);
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some(desc.label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: desc.depth_or_array_layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: desc.dimension,
        format: desc.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }));
    let view_descriptor = wgpu::TextureViewDescriptor {
        label: desc.view_label,
        dimension: desc.view_dimension,
        ..Default::default()
    };
    let view = Arc::new(texture.create_view(&view_descriptor));
    crate::profiling::note_resource_churn!(
        TextureView,
        "materials::embedded_placeholder_texture_view"
    );
    PlaceholderTexture { texture, view }
}

/// Allocates a 1x1 white texture and a default view for `kind`.
pub(super) fn create_white(device: &wgpu::Device, kind: TextureBindKind) -> PlaceholderTexture {
    create_placeholder(device, kind, PlaceholderTextureColor::White)
}

/// Allocates a 1x1 black texture and a default view for `kind`.
pub(super) fn create_black(device: &wgpu::Device, kind: TextureBindKind) -> PlaceholderTexture {
    create_placeholder(device, kind, PlaceholderTextureColor::Black)
}

/// Allocates a 1x1 gray texture and a default view for `kind`.
pub(super) fn create_gray(device: &wgpu::Device, kind: TextureBindKind) -> PlaceholderTexture {
    create_placeholder(device, kind, PlaceholderTextureColor::Gray)
}

/// Allocates a 1x1 sRGB gray texture and a default view for `kind`.
pub(super) fn create_srgb_gray(device: &wgpu::Device, kind: TextureBindKind) -> PlaceholderTexture {
    create_placeholder(device, kind, PlaceholderTextureColor::SrgbGray)
}

/// Allocates a 1x1 red texture and a default view for `kind`.
pub(super) fn create_red(device: &wgpu::Device, kind: TextureBindKind) -> PlaceholderTexture {
    create_placeholder(device, kind, PlaceholderTextureColor::Red)
}

/// Allocates a 1x1 flat tangent-space normal texture (`(0.5, 0.5, 1.0)`) in linear `Rgba8Unorm`.
///
/// Used as the fallback view for normal-map slots so missing/unloaded bump maps render as a flat
/// surface rather than the tilted `(1, 1, 1)` direction a white placeholder decodes to.
pub(super) fn create_flat_normal(
    device: &wgpu::Device,
    kind: TextureBindKind,
) -> PlaceholderTexture {
    create_placeholder(device, kind, PlaceholderTextureColor::FlatNormal)
}

/// Allocates the nonresident Texture2D checkerboard fallback.
pub(super) fn create_checkerboard_2d(device: &wgpu::Device) -> PlaceholderTexture {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("embedded_loading_checkerboard"),
        size: wgpu::Extent3d {
            width: CHECKERBOARD_TEXTURE_SIZE,
            height: CHECKERBOARD_TEXTURE_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }));
    let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("embedded_loading_checkerboard_view"),
        ..Default::default()
    }));
    crate::profiling::note_resource_churn!(
        TextureView,
        "materials::embedded_checkerboard_texture_view"
    );
    PlaceholderTexture { texture, view }
}

fn upload_placeholder(
    queue: &wgpu::Queue,
    placeholder: &PlaceholderTexture,
    kind: TextureBindKind,
    color: PlaceholderTextureColor,
) {
    let depth_or_array_layers = match kind {
        TextureBindKind::Tex2D | TextureBindKind::Tex3D => 1,
        TextureBindKind::Cube => 6,
    };
    let texel = color.rgba();
    let mut bytes: Vec<u8> = Vec::with_capacity(4 * depth_or_array_layers as usize);
    for _ in 0..depth_or_array_layers {
        bytes.extend_from_slice(&texel);
    }
    let rows_per_image = match kind {
        TextureBindKind::Tex2D => None,
        TextureBindKind::Tex3D | TextureBindKind::Cube => Some(1),
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: placeholder.texture.as_ref(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image,
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers,
        },
    );
}

/// Uploads a single white texel into every layer of `white` (1 layer for 2D / 3D, 6 for cubes).
pub(super) fn upload_white(queue: &wgpu::Queue, white: &PlaceholderTexture, kind: TextureBindKind) {
    upload_placeholder(queue, white, kind, PlaceholderTextureColor::White);
}

/// Uploads a single black texel into every layer of `black` (1 layer for 2D / 3D, 6 for cubes).
pub(super) fn upload_black(queue: &wgpu::Queue, black: &PlaceholderTexture, kind: TextureBindKind) {
    upload_placeholder(queue, black, kind, PlaceholderTextureColor::Black);
}

/// Uploads a single gray texel into every layer of `gray` (1 layer for 2D / 3D, 6 for cubes).
pub(super) fn upload_gray(queue: &wgpu::Queue, gray: &PlaceholderTexture, kind: TextureBindKind) {
    upload_placeholder(queue, gray, kind, PlaceholderTextureColor::Gray);
}

/// Uploads a single sRGB gray texel into every layer of `gray` (1 layer for 2D / 3D, 6 for cubes).
pub(super) fn upload_srgb_gray(
    queue: &wgpu::Queue,
    gray: &PlaceholderTexture,
    kind: TextureBindKind,
) {
    upload_placeholder(queue, gray, kind, PlaceholderTextureColor::SrgbGray);
}

/// Uploads a single red texel into every layer of `red` (1 layer for 2D / 3D, 6 for cubes).
pub(super) fn upload_red(queue: &wgpu::Queue, red: &PlaceholderTexture, kind: TextureBindKind) {
    upload_placeholder(queue, red, kind, PlaceholderTextureColor::Red);
}

/// Uploads a single Unity bump texel `(128, 128, 255, 128)` into `flat_normal`.
pub(super) fn upload_flat_normal(
    queue: &wgpu::Queue,
    flat_normal: &PlaceholderTexture,
    kind: TextureBindKind,
) {
    upload_placeholder(
        queue,
        flat_normal,
        kind,
        PlaceholderTextureColor::FlatNormal,
    );
}

/// Uploads the generated checkerboard fallback into `checkerboard`.
pub(super) fn upload_checkerboard_2d(queue: &wgpu::Queue, checkerboard: &PlaceholderTexture) {
    let bytes = checkerboard_rgba8();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: checkerboard.texture.as_ref(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(CHECKERBOARD_TEXTURE_SIZE * 4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: CHECKERBOARD_TEXTURE_SIZE,
            height: CHECKERBOARD_TEXTURE_SIZE,
            depth_or_array_layers: 1,
        },
    );
}

/// Builds the RGBA8 bytes for the nonresident Texture2D checkerboard fallback.
pub(super) fn checkerboard_rgba8() -> Vec<u8> {
    let len = CHECKERBOARD_TEXTURE_SIZE as usize * CHECKERBOARD_TEXTURE_SIZE as usize * 4;
    let mut bytes = Vec::with_capacity(len);
    for y in 0..CHECKERBOARD_TEXTURE_SIZE {
        for x in 0..CHECKERBOARD_TEXTURE_SIZE {
            bytes.extend_from_slice(&checkerboard_texel_rgba(x, y));
        }
    }
    bytes
}

/// Returns the checkerboard texel at `x`, `y`.
fn checkerboard_texel_rgba(x: u32, y: u32) -> [u8; 4] {
    let x_cell = x / CHECKERBOARD_CELL_SIZE;
    let y_cell = y / CHECKERBOARD_CELL_SIZE;
    if (x_cell ^ y_cell) & 1 == 0 {
        CHECKERBOARD_DARK_RGBA
    } else {
        CHECKERBOARD_LIGHT_RGBA
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CHECKERBOARD_CELL_SIZE, CHECKERBOARD_DARK_RGBA, CHECKERBOARD_LIGHT_RGBA,
        CHECKERBOARD_TEXTURE_SIZE, PlaceholderTextureColor, checkerboard_rgba8,
    };

    /// Returns one RGBA texel from the generated checkerboard byte array.
    fn texel(bytes: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * CHECKERBOARD_TEXTURE_SIZE + x) * 4) as usize;
        [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
    }

    /// Generated checkerboard bytes exactly cover the configured texture extent.
    #[test]
    fn checkerboard_generation_has_expected_size() {
        let bytes = checkerboard_rgba8();
        assert_eq!(
            bytes.len(),
            (CHECKERBOARD_TEXTURE_SIZE * CHECKERBOARD_TEXTURE_SIZE * 4) as usize
        );
    }

    /// Generated checkerboard bytes alternate between the stable dark and light texels.
    #[test]
    fn checkerboard_generation_uses_stable_dark_and_light_texels() {
        let bytes = checkerboard_rgba8();
        assert_eq!(texel(&bytes, 0, 0), CHECKERBOARD_DARK_RGBA);
        assert_eq!(
            texel(&bytes, CHECKERBOARD_CELL_SIZE, 0),
            CHECKERBOARD_LIGHT_RGBA
        );
        assert_eq!(
            texel(&bytes, 0, CHECKERBOARD_CELL_SIZE),
            CHECKERBOARD_LIGHT_RGBA
        );
        assert_eq!(
            texel(&bytes, CHECKERBOARD_CELL_SIZE, CHECKERBOARD_CELL_SIZE),
            CHECKERBOARD_DARK_RGBA
        );
    }

    /// Generated checkerboard texels are opaque.
    #[test]
    fn checkerboard_generation_is_fully_opaque() {
        let bytes = checkerboard_rgba8();
        assert!(bytes.chunks_exact(4).all(|texel| texel[3] == 255));
    }

    #[test]
    fn gray_placeholder_formats_keep_linear_and_srgb_paths_separate() {
        assert_eq!(
            PlaceholderTextureColor::Gray.format(),
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            PlaceholderTextureColor::SrgbGray.format(),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
    }
}
