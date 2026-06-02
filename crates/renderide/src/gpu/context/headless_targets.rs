//! Persistent primary final color/depth pair owned by [`GpuContext`] and the
//! `&mut`/`&` accessors used by main-view offscreen rendering and headless PNG readback.

use super::GpuContext;

/// Persistent offscreen color + depth pair owned by [`GpuContext`] for the main view.
///
/// The render graph treats these as an `OffscreenRt` view when `render_frame` substitutes the
/// main desktop/headless view. Windowed desktop presents the color attachment through the late
/// display-blit path; the headless driver can `copy_texture_to_buffer` against
/// [`PrimaryOffscreenTargets::color_texture`] to read back pixels and write a PNG.
#[derive(Clone)]
pub struct PrimaryOffscreenTargets {
    /// Color attachment used by the main-view final render target.
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
    /// Returns cloned handles for the lazy-allocated primary offscreen color/depth pair.
    ///
    /// On the first call, allocates the persistent textures matching `config.width x
    /// config.height` and the configured color format. Subsequent calls return the same handles
    /// until the context is resized, reconfigured, or dropped.
    ///
    /// `render_frame` calls this to substitute the main view with a
    /// `FrameViewTarget::OffscreenRt` backed by these textures. Desktop presentation then blits
    /// the color attachment to the swapchain as the terminal present step.
    pub fn primary_offscreen_targets(&mut self) -> PrimaryOffscreenTargets {
        let requested_extent = self.primary_offscreen_extent();
        let requested_format = self.config.format;
        let needs_recreate = self.primary_offscreen.as_ref().is_none_or(|targets| {
            targets.extent_px != requested_extent || targets.color_format != requested_format
        });
        if needs_recreate {
            let targets = self.create_primary_offscreen_targets();
            self.primary_offscreen = Some(targets);
        }
        if let Some(targets) = self.primary_offscreen.as_ref() {
            return targets.clone();
        }
        let targets = self.create_primary_offscreen_targets();
        self.primary_offscreen = Some(targets.clone());
        targets
    }

    /// Returns a cloned primary offscreen color view and extent if it has been allocated.
    pub fn primary_offscreen_color_source(&self) -> Option<(wgpu::TextureView, (u32, u32))> {
        self.primary_offscreen
            .as_ref()
            .map(|targets| (targets.color_view.clone(), targets.extent_px))
    }

    /// Drops the cached primary offscreen target so the next main-view render recreates it.
    pub(crate) fn invalidate_primary_offscreen_targets(&mut self) {
        self.primary_offscreen = None;
    }

    fn primary_offscreen_extent(&self) -> (u32, u32) {
        let max_dim = self.limits.max_texture_dimension_2d();
        (
            self.config.width.max(1).min(max_dim),
            self.config.height.max(1).min(max_dim),
        )
    }

    fn create_primary_offscreen_targets(&self) -> PrimaryOffscreenTargets {
        let max_dim = self.limits.max_texture_dimension_2d();
        let req_w = self.config.width.max(1);
        let req_h = self.config.height.max(1);
        let width = req_w.min(max_dim);
        let height = req_h.min(max_dim);
        if (width, height) != (req_w, req_h) {
            logger::warn!(
                "primary offscreen target: {req_w}x{req_h} exceeds max_texture_dimension_2d={max_dim}; clamped to {width}x{height}",
            );
        }
        let color_format = self.config.format;
        let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-primary-offscreen-color"),
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
        crate::profiling::note_resource_churn!(TextureView, "gpu::primary_offscreen_color_view");
        let depth_format = crate::gpu::main_forward_depth_stencil_format(self.device.features());
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-primary-offscreen-depth"),
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
        crate::profiling::note_resource_churn!(TextureView, "gpu::primary_offscreen_depth_view");
        PrimaryOffscreenTargets {
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            extent_px: (width, height),
            color_format,
        }
    }

    /// Returns the persistent headless color texture for PNG readback.
    ///
    /// Returns [`None`] in windowed mode and also when the primary offscreen has not yet been
    /// allocated (call [`Self::primary_offscreen_targets`] first or run a render tick).
    /// Unlike [`Self::primary_offscreen_targets`], this getter takes `&self` so it does not
    /// conflict with concurrent mutable borrows on `gpu` during readback.
    pub fn headless_color_texture(&self) -> Option<&wgpu::Texture> {
        if self.is_headless() {
            self.primary_offscreen.as_ref().map(|t| &t.color_texture)
        } else {
            None
        }
    }
}
