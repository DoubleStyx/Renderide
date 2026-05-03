//! Persistent headless primary color/depth pair owned by [`GpuContext`] and the
//! `&mut`/`&` accessors used by the headless render-frame substitution and PNG readback.

use super::GpuContext;

/// Persistent offscreen color + depth pair owned by [`GpuContext`] in headless mode.
///
/// The render graph treats these as a host render-texture (an `OffscreenRt` view) when
/// `render_frame` substitutes the main `Swapchain` view in headless mode. The headless
/// driver then `copy_texture_to_buffer` against [`PrimaryOffscreenTargets::color_texture`]
/// to read back the pixels and write a PNG.
pub struct PrimaryOffscreenTargets {
    /// Color attachment ([`wgpu::TextureFormat::Rgba8UnormSrgb`] + `RENDER_ATTACHMENT | COPY_SRC`).
    pub color_texture: wgpu::Texture,
    /// Default view of [`Self::color_texture`] for render passes.
    pub color_view: wgpu::TextureView,
    /// Depth-stencil texture matching the main forward pass format.
    pub depth_texture: wgpu::Texture,
    /// Default view of [`Self::depth_texture`] for render passes.
    pub depth_view: wgpu::TextureView,
    /// Pixel extent (width, height) shared by both attachments.
    pub extent_px: (u32, u32),
    /// Color format reused by the render graph when binding pipelines.
    pub color_format: wgpu::TextureFormat,
}

impl GpuContext {
    /// Returns the lazy-allocated primary offscreen color/depth pair owned by this context.
    ///
    /// Returns [`None`] when the context is windowed (it has a real swapchain instead). On the
    /// first call in headless mode, allocates the persistent textures matching `config.width x
    /// config.height` and the configured color format. Subsequent calls return the same handles
    /// until the context is dropped.
    ///
    /// `render_frame` calls this when `window.is_none()` to substitute the main `Swapchain`
    /// view with a `FrameViewTarget::OffscreenRt` backed by these textures.
    pub fn primary_offscreen_targets(&mut self) -> Option<&PrimaryOffscreenTargets> {
        if !self.is_headless() {
            return None;
        }
        if self.primary_offscreen.is_none() {
            let max_dim = self.limits.max_texture_dimension_2d();
            let req_w = self.config.width.max(1);
            let req_h = self.config.height.max(1);
            let width = req_w.min(max_dim);
            let height = req_h.min(max_dim);
            if (width, height) != (req_w, req_h) {
                logger::warn!(
                    "headless primary offscreen: {req_w}x{req_h} exceeds max_texture_dimension_2d={max_dim}; clamped to {width}x{height}",
                );
            }
            let color_format = self.config.format;
            let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("renderide-headless-primary-color"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_format =
                crate::gpu::main_forward_depth_stencil_format(self.device.features());
            let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("renderide-headless-primary-depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: depth_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.primary_offscreen = Some(PrimaryOffscreenTargets {
                color_texture,
                color_view,
                depth_texture,
                depth_view,
                extent_px: (width, height),
                color_format,
            });
        }
        self.primary_offscreen.as_ref()
    }

    /// Returns the persistent headless color texture for PNG readback.
    ///
    /// Returns [`None`] in windowed mode and also when the headless offscreen has not yet been
    /// allocated (call [`Self::primary_offscreen_targets`] first or run a render tick).
    /// Unlike [`Self::primary_offscreen_targets`], this getter takes `&self` so it does not
    /// conflict with concurrent mutable borrows on `gpu` during readback.
    pub fn headless_color_texture(&self) -> Option<&wgpu::Texture> {
        self.primary_offscreen.as_ref().map(|t| &t.color_texture)
    }
}
