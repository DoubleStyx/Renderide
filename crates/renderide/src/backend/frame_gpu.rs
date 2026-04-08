//! Per-frame `@group(0)` resources: camera uniform + lights storage buffer.

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::backend::light_gpu::{GpuLight, MAX_LIGHTS};
use crate::gpu::frame_globals::FrameGpuUniforms;

/// GPU buffers and bind group for [`super::light_gpu::GpuLight`] storage and [`FrameGpuUniforms`].
pub struct FrameGpuResources {
    /// Uniform buffer for [`FrameGpuUniforms`] (32 bytes).
    pub frame_uniform: wgpu::Buffer,
    /// Storage buffer holding up to [`MAX_LIGHTS`] [`GpuLight`] records.
    pub lights_buffer: wgpu::Buffer,
    /// Bind group for `@group(0)` in composed mesh shaders.
    pub bind_group: Arc<wgpu::BindGroup>,
}

impl FrameGpuResources {
    /// Layout for `@group(0)`: uniform frame + read-only storage lights.
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame_globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<FrameGpuUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuLight>() as u64),
                    },
                    count: None,
                },
            ],
        })
    }

    /// Allocates frame uniform and lights storage; builds [`Self::bind_group`].
    pub fn new(device: &wgpu::Device) -> Self {
        let frame_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_globals_uniform"),
            size: std::mem::size_of::<FrameGpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_size = (MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as u64;
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_lights_storage"),
            size: lights_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = Self::bind_group_layout(device);
        let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame_globals_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_buffer.as_entire_binding(),
                },
            ],
        }));
        Self {
            frame_uniform,
            lights_buffer,
            bind_group,
        }
    }

    /// Uploads frame header and packed lights for this frame.
    pub fn write_frame(
        &self,
        queue: &wgpu::Queue,
        camera_world_pos: glam::Vec3,
        lights: &[GpuLight],
    ) {
        let n = lights.len().min(MAX_LIGHTS) as u32;
        let header = FrameGpuUniforms::new(camera_world_pos, n);
        queue.write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(&header));
        if n > 0 {
            let bytes = bytemuck::cast_slice(&lights[..n as usize]);
            queue.write_buffer(&self.lights_buffer, 0, bytes);
        } else {
            queue.write_buffer(&self.lights_buffer, 0, &[0u8; 4]);
        }
    }
}

/// Empty `@group(1)` layout for materials that declare no per-material bindings yet.
pub fn empty_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty_material_slot"),
        entries: &[],
    })
}

/// Single reusable empty bind group for [`empty_material_bind_group_layout`].
pub fn empty_material_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("empty_material_bind_group"),
        layout,
        entries: &[],
    })
}

/// Cached empty material bind group layout + instance (one per device attach).
pub struct EmptyMaterialBindGroup {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: Arc<wgpu::BindGroup>,
}

impl EmptyMaterialBindGroup {
    /// Builds layout and bind group for `@group(1)` placeholder.
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = empty_material_bind_group_layout(device);
        let bind_group = Arc::new(empty_material_bind_group(device, &layout));
        Self { layout, bind_group }
    }
}
