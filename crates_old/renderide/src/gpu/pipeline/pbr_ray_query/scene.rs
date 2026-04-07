//! Scene bind group layout and creation for PBR ray-query pipelines: lights, clusters, TLAS,
//! RT shadow uniforms, and shadow atlas (`group 1`).

use std::mem::size_of;

use super::super::rt_shadow_uniforms::{RtShadowSceneBind, RtShadowUniforms};
use super::super::uniforms::SceneUniforms;

/// Scene bind group layout for PBR + TLAS (group 1): scene uniform, storages, TLAS, RT shadow tuning + atlas.
pub(crate) fn pbr_scene_bind_group_layout_with_accel(
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, u64) {
    let scene_uniform_size = size_of::<SceneUniforms>() as u64;
    let rt_shadow_uniform_size = size_of::<RtShadowUniforms>() as u64;
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("PBR scene BGL + TLAS + RT shadow"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(scene_uniform_size),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(rt_shadow_uniform_size),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    (layout, scene_uniform_size)
}

/// Writes scene uniforms and builds the scene bind group with TLAS and RT shadow resources.
#[allow(clippy::too_many_arguments)]
pub(in crate::gpu::pipeline::pbr_ray_query) fn create_pbr_scene_bind_group_with_accel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    scene_uniform_buffer: &wgpu::Buffer,
    scene: &SceneUniforms,
    light_buffer: &wgpu::Buffer,
    cluster_light_counts: &wgpu::Buffer,
    cluster_light_indices: &wgpu::Buffer,
    tlas: &wgpu::Tlas,
    rt_shadow: &RtShadowSceneBind<'_>,
) -> wgpu::BindGroup {
    queue.write_buffer(scene_uniform_buffer, 0, bytemuck::bytes_of(scene));
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: light_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cluster_light_counts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cluster_light_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::AccelerationStructure(tlas),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: rt_shadow.uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(rt_shadow.atlas_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(rt_shadow.sampler),
            },
        ],
    })
}
