//! Frame-global reflection-probe specular resources.

use std::sync::Arc;

use crate::gpu::{GpuReflectionProbeMetadata, REFLECTION_PROBE_ATLAS_FORMAT};

/// Resources bound to frame-global reflection-probe slots.
#[derive(Clone)]
pub struct ReflectionProbeSpecularResources {
    /// 2D-array atlas view sampled by PBS materials.
    pub array_view: Arc<wgpu::TextureView>,
    /// Sampler paired with [`Self::array_view`].
    pub sampler: Arc<wgpu::Sampler>,
    /// Storage buffer of [`GpuReflectionProbeMetadata`] rows.
    pub metadata_buffer: Arc<wgpu::Buffer>,
    /// Monotonic resource version; incremented when bindings need recreation.
    pub version: u64,
}

/// Borrowed view used while creating frame-global bind groups.
#[derive(Clone, Copy)]
pub(super) struct ReflectionProbeSpecularBindGroupResources<'a> {
    /// 2D-array atlas view bound at `@group(0) @binding(9)`.
    pub array_view: &'a wgpu::TextureView,
    /// Sampler bound at `@group(0) @binding(10)`.
    pub sampler: &'a wgpu::Sampler,
    /// Metadata buffer bound at `@group(0) @binding(12)`.
    pub metadata_buffer: &'a wgpu::Buffer,
}

/// Allocates black fallback reflection-probe bindings.
pub(super) fn create_reflection_probe_specular_fallback(
    device: &wgpu::Device,
) -> (
    Arc<wgpu::Texture>,
    Arc<wgpu::TextureView>,
    Arc<wgpu::Sampler>,
    Arc<wgpu::Buffer>,
) {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame_reflection_probe_black_array"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: REFLECTION_PROBE_ATLAS_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }));
    let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("frame_reflection_probe_black_array_view"),
        format: Some(REFLECTION_PROBE_ATLAS_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
    }));
    crate::profiling::note_resource_churn!(
        TextureView,
        "backend::frame_reflection_probe_black_array_view"
    );
    let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("frame_reflection_probe_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 0.0,
        ..Default::default()
    }));
    let metadata = [GpuReflectionProbeMetadata::default()];
    let metadata_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame_reflection_probe_metadata"),
        size: size_of_val(&metadata) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    }));
    crate::profiling::note_resource_churn!(Buffer, "backend::frame_reflection_probe_metadata");
    metadata_buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::cast_slice(&metadata));
    metadata_buffer.unmap();
    (texture, view, sampler, metadata_buffer)
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use crate::gpu::GpuReflectionProbeMetadata;

    #[test]
    fn reflection_probe_metadata_stride_includes_sh2_rows() {
        assert_eq!(size_of::<GpuReflectionProbeMetadata>(), 208);
    }
}
