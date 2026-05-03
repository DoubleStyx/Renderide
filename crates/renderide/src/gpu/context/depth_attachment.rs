//! Main depth attachment ensure/recreate methods on [`GpuContext`].
//!
//! The single-sample forward depth target is owned by [`GpuContext`] and recreated on
//! resize or format change. The MSAA depth path is handled separately by
//! [`crate::gpu::msaa_depth_resolve`].

use super::GpuContext;

impl GpuContext {
    /// Ensures a stencil-capable depth attachment exists for the current surface extent.
    ///
    /// Call after [`Self::reconfigure`] or when the swapchain size may have changed.
    ///
    /// Returns an error string only if the depth attachment could not be read after allocation (defensive).
    pub fn ensure_depth_view(&mut self) -> Result<&wgpu::TextureView, &'static str> {
        self.ensure_depth_target().map(|(_, v)| v)
    }

    /// Ensures the main depth attachment exists and returns both the texture and its default view.
    ///
    /// Returns an error string only if the depth attachment could not be read after allocation (defensive).
    pub fn ensure_depth_target(
        &mut self,
    ) -> Result<(&wgpu::Texture, &wgpu::TextureView), &'static str> {
        let w = self.config.width.max(1);
        let h = self.config.height.max(1);
        let depth_stencil_format =
            crate::gpu::main_forward_depth_stencil_format(self.device.features());
        let needs_recreate = self.depth_extent_px != (w, h)
            || self
                .depth_attachment
                .as_ref()
                .is_none_or(|(tex, _)| tex.format() != depth_stencil_format);
        if needs_recreate {
            let max_dim = self.limits.wgpu.max_texture_dimension_2d;
            if w > max_dim || h > max_dim {
                logger::warn!(
                    "depth attachment extent {}x{} exceeds max_texture_dimension_2d ({max_dim}); creation may fail validation",
                    w,
                    h
                );
            }
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("renderide-depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: depth_stencil_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.depth_extent_px = (w, h);
            self.depth_attachment = Some((tex, view));
        }
        self.depth_attachment
            .as_ref()
            .map(|(t, v)| (t, v))
            .ok_or("depth attachment missing after ensure_depth_target")
    }
}
