//! Bind-group layouts shared by color-blit subsystems.
//!
//! Both layouts use filterable 2D textures and filtering samplers. The UV variant adds a
//! 16-byte `vec4<f32>` uniform slot for scale/offset.

use std::sync::OnceLock;

/// `(0)` filterable 2D, `(1)` filtering sampler. Used by single-layer blit passes.
pub(crate) fn sampled_2d_filtered_layout(device: &wgpu::Device) -> &'static wgpu::BindGroupLayout {
    static LAYOUT: OnceLock<wgpu::BindGroupLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_kit::sampled_2d_filtered"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    })
}

/// `(0)` filterable 2D array, `(1)` filtering sampler. Used by stereo multiview blit passes.
pub(crate) fn sampled_2d_array_filtered_layout(
    device: &wgpu::Device,
) -> &'static wgpu::BindGroupLayout {
    static LAYOUT: OnceLock<wgpu::BindGroupLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_kit::sampled_2d_array_filtered"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    })
}

/// `(0)` filterable 2D, `(1)` filtering sampler, `(2)` 16-byte `vec4<f32>` UV uniform.
///
/// Used by both the desktop display-blit and the VR-mirror staging-to-surface blit.
pub(crate) fn sampled_2d_filtered_uv_layout(
    device: &wgpu::Device,
) -> &'static wgpu::BindGroupLayout {
    static LAYOUT: OnceLock<wgpu::BindGroupLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_kit::sampled_2d_filtered_uv"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(16),
                    },
                    count: None,
                },
            ],
        })
    })
}
