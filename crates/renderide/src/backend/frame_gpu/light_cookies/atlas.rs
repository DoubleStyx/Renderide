use std::sync::Arc;

use super::format::LightCookieAtlasFormat;

/// Pixel extent for one packed light-cookie atlas texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LightCookieAtlasExtent {
    /// Atlas width in texels.
    pub(super) width: u32,
    /// Atlas height in texels.
    pub(super) height: u32,
}

impl LightCookieAtlasExtent {
    /// Smallest valid sampled texture extent.
    pub(super) const FALLBACK: Self = Self {
        width: 1,
        height: 1,
    };

    /// Returns an extent with both axes valid for texture allocation.
    pub(super) fn sanitized(self) -> Self {
        Self {
            width: self.width.max(1),
            height: self.height.max(1),
        }
    }
}

/// Packed light-cookie atlas texture.
pub(super) struct LightCookiePackedAtlas {
    /// Backing texture.
    texture: Arc<wgpu::Texture>,
    /// Full 2D sampled/render-target view.
    view: Arc<wgpu::TextureView>,
    /// Current allocated atlas extent.
    extent: LightCookieAtlasExtent,
    /// Atlas texture format.
    format: LightCookieAtlasFormat,
    /// Static label used for recreated atlas textures.
    label: &'static str,
}

impl LightCookiePackedAtlas {
    /// Creates a 1x1 white light-cookie atlas.
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        format: LightCookieAtlasFormat,
    ) -> Self {
        let (texture, view) =
            create_atlas_texture(device, label, LightCookieAtlasExtent::FALLBACK, format);
        write_white_texture(
            queue,
            texture.as_ref(),
            format,
            LightCookieAtlasExtent::FALLBACK,
        );
        Self {
            texture,
            view,
            extent: LightCookieAtlasExtent::FALLBACK,
            format,
            label,
        }
    }

    /// Grows the atlas so `required` fits. Returns `true` when the sampled view changed.
    pub(super) fn sync(&mut self, device: &wgpu::Device, required: LightCookieAtlasExtent) -> bool {
        let required = required.sanitized();
        if self.extent.width >= required.width && self.extent.height >= required.height {
            return false;
        }
        let next = LightCookieAtlasExtent {
            width: self.extent.width.max(required.width),
            height: self.extent.height.max(required.height),
        };
        let (texture, view) = create_atlas_texture(device, self.label, next, self.format);
        self.texture = texture;
        self.view = view;
        self.extent = next;
        true
    }

    /// Full atlas view for group-0 binding and atlas render passes.
    pub(super) fn view(&self) -> &wgpu::TextureView {
        self.view.as_ref()
    }

    /// Current atlas extent in texels.
    pub(super) fn extent(&self) -> LightCookieAtlasExtent {
        self.extent
    }

    /// Retains atlas handles until driver submit.
    pub(super) fn retain_submit_resources(&self, resources: &mut crate::gpu::GpuRetainedResources) {
        resources.retain_texture(self.texture.as_ref().clone());
        resources.retain_texture_view(self.view.as_ref().clone());
    }
}

fn create_atlas_texture(
    device: &wgpu::Device,
    label: &'static str,
    extent: LightCookieAtlasExtent,
    format: LightCookieAtlasFormat,
) -> (Arc<wgpu::Texture>, Arc<wgpu::TextureView>) {
    let extent = extent.sanitized();
    let wgpu_format = format.wgpu();
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: extent.width,
            height: extent.height,
            depth_or_array_layers: 1,
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
    let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{label}_view")),
        format: Some(wgpu_format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(1),
    }));
    crate::profiling::note_resource_churn!(TextureView, "backend::light_cookie_atlas_view");
    (texture, view)
}

/// Writes a white fallback texture.
fn write_white_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    format: LightCookieAtlasFormat,
    extent: LightCookieAtlasExtent,
) {
    let extent = extent.sanitized();
    let bytes = white_texture_bytes(format, extent.width, extent.height);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(extent.width * format.bytes_per_texel()),
            rows_per_image: Some(extent.height),
        },
        wgpu::Extent3d {
            width: extent.width,
            height: extent.height,
            depth_or_array_layers: 1,
        },
    );
}

/// Builds CPU-side full-white texels for `format` and `extent`.
pub(super) fn white_texture_bytes(
    format: LightCookieAtlasFormat,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let texels = (width.max(1) * height.max(1)) as usize;
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
