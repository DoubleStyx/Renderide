//! Shared GPU texture allocation helpers for resident texture pools.

use std::sync::Arc;

/// View configuration for a newly allocated sampled texture.
pub(super) struct TextureViewInit<'a> {
    /// Optional debug label for the view.
    pub label: Option<&'a str>,
    /// Optional explicit view dimension.
    pub dimension: Option<wgpu::TextureViewDimension>,
}

/// Allocation descriptor for resident sampled textures.
pub(super) struct SampledTextureAllocation<'a> {
    /// Debug label for the texture.
    pub label: &'a str,
    /// Texture extent.
    pub size: wgpu::Extent3d,
    /// Number of mip levels.
    pub mip_level_count: u32,
    /// Texture dimension.
    pub dimension: wgpu::TextureDimension,
    /// Storage format.
    pub format: wgpu::TextureFormat,
    /// Initial default view shape.
    pub view: TextureViewInit<'a>,
}

/// Creates resident texture storage and its default binding view.
pub(super) fn create_sampled_copy_dst_texture(
    device: &wgpu::Device,
    desc: SampledTextureAllocation<'_>,
) -> (Arc<wgpu::Texture>, Arc<wgpu::TextureView>) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(desc.label),
        size: desc.size,
        mip_level_count: desc.mip_level_count,
        sample_count: 1,
        dimension: desc.dimension,
        format: desc.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: desc.view.label,
        dimension: desc.view.dimension,
        ..Default::default()
    });
    (Arc::new(texture), Arc::new(view))
}
