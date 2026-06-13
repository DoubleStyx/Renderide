use std::sync::Arc;

use super::format::LightCookieAtlasFormat;

/// Edge length of each light-cookie atlas layer.
pub(super) const LIGHT_COOKIE_ATLAS_EDGE: u32 = 256;

/// Layered atlas texture and one-layer render-target views.
pub(super) struct LightCookieLayeredAtlas {
    /// Backing texture.
    texture: Arc<wgpu::Texture>,
    /// Full array view bound by frame globals.
    pub(super) view: Arc<wgpu::TextureView>,
    /// Single-layer views used as render-pass targets.
    layer_views: Vec<Arc<wgpu::TextureView>>,
    /// Array layer count.
    pub(super) layers: u32,
}

impl LightCookieLayeredAtlas {
    /// Creates a light-cookie atlas with one-layer render-target views.
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        layers: u32,
        format: LightCookieAtlasFormat,
    ) -> Self {
        let wgpu_format = format.wgpu();
        let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: LIGHT_COOKIE_ATLAS_EDGE,
                height: LIGHT_COOKIE_ATLAS_EDGE,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        write_white_layer(queue, texture.as_ref(), 0, format);
        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{label}_view")),
            format: Some(wgpu_format),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(layers),
        }));
        crate::profiling::note_resource_churn!(TextureView, "backend::light_cookie_atlas_view");
        let layer_views = (0..layers)
            .map(|layer| {
                Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("{label}_layer_{layer}")),
                    format: Some(wgpu_format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                }))
            })
            .collect::<Vec<_>>();
        crate::profiling::note_resource_churn!(TextureView, "backend::light_cookie_layer_views");
        Self {
            texture,
            view,
            layer_views,
            layers,
        }
    }

    /// Returns a single-layer render target view.
    pub(super) fn layer_view(&self, layer: u32) -> Option<&wgpu::TextureView> {
        self.layer_views.get(layer as usize).map(Arc::as_ref)
    }

    /// Retains atlas handles until driver submit.
    pub(super) fn retain_submit_resources(&self, resources: &mut crate::gpu::GpuRetainedResources) {
        resources.retain_texture(self.texture.as_ref().clone());
        resources.retain_texture_view(self.view.as_ref().clone());
        resources.retain_texture_views(self.layer_views.iter().map(|view| view.as_ref().clone()));
    }
}

/// Writes a white fallback layer.
fn write_white_layer(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layer: u32,
    format: LightCookieAtlasFormat,
) {
    let bytes = white_layer_bytes(format);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LIGHT_COOKIE_ATLAS_EDGE * format.bytes_per_texel()),
            rows_per_image: Some(LIGHT_COOKIE_ATLAS_EDGE),
        },
        wgpu::Extent3d {
            width: LIGHT_COOKIE_ATLAS_EDGE,
            height: LIGHT_COOKIE_ATLAS_EDGE,
            depth_or_array_layers: 1,
        },
    );
}

/// Builds a CPU-side full-white fallback layer for `format`.
pub(super) fn white_layer_bytes(format: LightCookieAtlasFormat) -> Vec<u8> {
    let texels = (LIGHT_COOKIE_ATLAS_EDGE * LIGHT_COOKIE_ATLAS_EDGE) as usize;
    match format {
        LightCookieAtlasFormat::R16Float => {
            let mut bytes = Vec::with_capacity(texels * 2);
            for _ in 0..texels {
                bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
            }
            bytes
        }
        LightCookieAtlasFormat::Rgba16Float => {
            let mut bytes = Vec::with_capacity(texels * 8);
            for _ in 0..(texels * 4) {
                bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
            }
            bytes
        }
        LightCookieAtlasFormat::R8Unorm => vec![255u8; texels],
    }
}
