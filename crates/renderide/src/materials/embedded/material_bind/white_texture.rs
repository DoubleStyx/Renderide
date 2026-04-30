//! 1×1 white placeholder textures used as the default binding for unset `@group(1)` texture slots.

use std::sync::Arc;

use super::super::bind_kind::TextureBindKind;

/// Placeholder 1×1 white texture for one [`TextureBindKind`].
pub(super) struct WhiteTexture {
    /// Underlying device texture.
    pub(super) texture: Arc<wgpu::Texture>,
    /// Default texture view used at binding time.
    pub(super) view: Arc<wgpu::TextureView>,
}

impl TextureBindKind {
    /// Texture descriptor parameters (label, dimension, layer count, view dimension).
    fn white_descriptor(self) -> WhiteDescriptor {
        match self {
            TextureBindKind::Tex2D => WhiteDescriptor {
                label: "embedded_default_white",
                view_label: None,
                dimension: wgpu::TextureDimension::D2,
                view_dimension: None,
                depth_or_array_layers: 1,
            },
            TextureBindKind::Tex3D => WhiteDescriptor {
                label: "embedded_default_white_3d",
                view_label: Some("embedded_default_white_3d_view"),
                dimension: wgpu::TextureDimension::D3,
                view_dimension: Some(wgpu::TextureViewDimension::D3),
                depth_or_array_layers: 1,
            },
            TextureBindKind::Cube => WhiteDescriptor {
                label: "embedded_default_white_cube",
                view_label: Some("embedded_default_white_cube_view"),
                dimension: wgpu::TextureDimension::D2,
                view_dimension: Some(wgpu::TextureViewDimension::Cube),
                depth_or_array_layers: 6,
            },
        }
    }
}

struct WhiteDescriptor {
    label: &'static str,
    view_label: Option<&'static str>,
    dimension: wgpu::TextureDimension,
    view_dimension: Option<wgpu::TextureViewDimension>,
    depth_or_array_layers: u32,
}

/// Allocates a 1×1 white texture and a default view for `kind`.
pub(super) fn create_white(device: &wgpu::Device, kind: TextureBindKind) -> WhiteTexture {
    let desc = kind.white_descriptor();
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
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }));
    let view_descriptor = wgpu::TextureViewDescriptor {
        label: desc.view_label,
        dimension: desc.view_dimension,
        ..Default::default()
    };
    let view = Arc::new(texture.create_view(&view_descriptor));
    WhiteTexture { texture, view }
}

/// Uploads a single white texel into every layer of `white` (1 layer for 2D / 3D, 6 for cubes).
pub(super) fn upload_white(queue: &wgpu::Queue, white: &WhiteTexture, kind: TextureBindKind) {
    let depth_or_array_layers = match kind {
        TextureBindKind::Tex2D | TextureBindKind::Tex3D => 1,
        TextureBindKind::Cube => 6,
    };
    let bytes: Vec<u8> = vec![255u8; 4 * depth_or_array_layers as usize];
    let rows_per_image = match kind {
        TextureBindKind::Tex2D => None,
        TextureBindKind::Tex3D | TextureBindKind::Cube => Some(1),
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: white.texture.as_ref(),
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
