//! Reusable `wgpu::BindGroupLayoutEntry` factories.
//!
//! Pure wgpu primitives -- no graph-framework dependency. Originally lived under
//! `render_graph/gpu_cache.rs` but are needed by `gpu/msaa_depth_resolve.rs` and other
//! GPU-tier modules that must not import the graph framework.

use std::num::NonZeroU64;

/// Creates a bind-group layout entry for a sampled texture binding.
pub fn texture_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
    multisampled: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        },
        count: None,
    }
}

/// Creates a fragment-stage filterable `texture_2d_array<f32>` layout entry.
pub fn fragment_filterable_d2_array_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    texture_layout_entry(
        binding,
        wgpu::ShaderStages::FRAGMENT,
        wgpu::TextureSampleType::Float { filterable: true },
        wgpu::TextureViewDimension::D2Array,
        false,
    )
}

/// Creates a fragment-stage filtering sampler layout entry.
pub fn fragment_filtering_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    sampler_layout_entry(
        binding,
        wgpu::ShaderStages::FRAGMENT,
        wgpu::SamplerBindingType::Filtering,
    )
}

/// Creates a bind-group layout entry for a sampler binding.
pub fn sampler_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sampler_type: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(sampler_type),
        count: None,
    }
}

/// Creates a bind-group layout entry for a uniform buffer binding.
pub fn uniform_buffer_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    min_binding_size: Option<NonZeroU64>,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size,
        },
        count: None,
    }
}

/// Creates a bind-group layout entry for a storage texture binding.
pub fn storage_texture_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    access: wgpu::StorageTextureAccess,
    format: wgpu::TextureFormat,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::StorageTexture {
            access,
            format,
            view_dimension,
        },
        count: None,
    }
}
