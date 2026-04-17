//! Single-sample depth attachment matching the swapchain extent.

use crate::gpu::limits::GpuLimits;

/// Ensures a [`wgpu::TextureFormat::Depth32Float`] texture exists for `(width, height)`.
///
/// Updates `depth_attachment` and `depth_extent_px` when extent changes or the attachment is missing.
pub(super) fn ensure_depth_target<'a>(
    device: &wgpu::Device,
    limits: &GpuLimits,
    width: u32,
    height: u32,
    depth_attachment: &'a mut Option<(wgpu::Texture, wgpu::TextureView)>,
    depth_extent_px: &mut (u32, u32),
) -> Result<(&'a wgpu::Texture, &'a wgpu::TextureView), &'static str> {
    let w = width.max(1);
    let h = height.max(1);
    let needs_recreate = *depth_extent_px != (w, h) || depth_attachment.is_none();
    if needs_recreate {
        let max_dim = limits.wgpu.max_texture_dimension_2d;
        if w > max_dim || h > max_dim {
            logger::warn!(
                "depth attachment extent {}×{} exceeds max_texture_dimension_2d ({max_dim}); creation may fail validation",
                w,
                h
            );
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        *depth_extent_px = (w, h);
        *depth_attachment = Some((tex, view));
    }
    depth_attachment
        .as_ref()
        .map(|(t, v)| (t, v))
        .ok_or("depth attachment missing after ensure_depth_target")
}
