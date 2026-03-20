//! Skinned PBR pipelines: bone-skinning vertex stage with PBS fragment lighting.
//!
//! [`SkinnedPbrPipeline`] and [`SkinnedPbrMRTPipeline`] differ only in their fragment outputs:
//! single color target vs three-target G-buffer for RTAO. Both share the same bind group
//! layouts via [`SkinnedPbrPipeline::create_bind_group_layouts`].

use std::mem::size_of;

use nalgebra::Matrix4;

use super::super::mesh::{GpuMeshBuffers, VertexSkinned};
use super::builder;
use super::core::{RenderPipeline, UniformData};
use super::ring_buffer::SkinnedUniformRingBuffer;
use super::shaders::{SKINNED_PBR_MRT_SHADER_SRC, SKINNED_PBR_SHADER_SRC};
use super::uniforms::{SceneUniforms, SkinnedUniforms};

// ─── Shared bind group layouts ────────────────────────────────────────────────

/// Creates the two bind group layouts shared by both skinned PBR pipeline variants.
///
/// Returns `(draw_bgl, scene_bgl, scene_uniform_size_bytes)`.
/// - Group 0 (`draw_bgl`): skinned uniform buffer (dynamic offset) + blendshape storage.
/// - Group 1 (`scene_bgl`): scene uniform + lights storage + two cluster storages.
fn create_skinned_pbr_bind_group_layouts(
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout, u64) {
    let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("skinned PBR draw BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        size_of::<SkinnedUniforms>() as u64,
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let scene_uniform_size = size_of::<SceneUniforms>() as u64;
    let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("skinned PBR scene BGL"),
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
        ],
    });
    (draw_bgl, scene_bgl, scene_uniform_size)
}

// ─── Helper: per-draw and scene bind groups ───────────────────────────────────

fn create_skinned_draw_bg(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    blendshape_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(size_of::<SkinnedUniforms>() as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: blendshape_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    })
}

fn create_scene_bg(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    scene_uniform_buffer: &wgpu::Buffer,
    scene: &SceneUniforms,
    queue: &wgpu::Queue,
    light_buffer: &wgpu::Buffer,
    cluster_light_counts: &wgpu::Buffer,
    cluster_light_indices: &wgpu::Buffer,
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
        ],
    })
}

// ─── SkinnedPbrPipeline ───────────────────────────────────────────────────────

/// Skinned PBR pipeline: bone skinning vertex stage with PBS fragment lighting (single target).
pub struct SkinnedPbrPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: SkinnedUniformRingBuffer,
    bind_group_layout: wgpu::BindGroupLayout,
    dummy_blendshape_buffer: wgpu::Buffer,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_uniform_buffer: wgpu::Buffer,
}

impl SkinnedPbrPipeline {
    /// Creates a skinned PBR pipeline with single color target.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned PBR shader"),
            source: wgpu::ShaderSource::Wgsl(SKINNED_PBR_SHADER_SRC.into()),
        });
        let (draw_bgl, scene_bgl, scene_uniform_size) =
            create_skinned_pbr_bind_group_layouts(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned PBR pipeline layout"),
            bind_group_layouts: &[&draw_bgl, &scene_bgl],
            immediate_size: 0,
        });
        let uniform_ring = SkinnedUniformRingBuffer::new(device, "skinned PBR uniform ring buffer");
        let dummy_blendshape_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned PBR dummy blendshape buffer"),
            size: 1,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let scene_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned PBR scene uniform buffer"),
            size: scene_uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned PBR pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<VertexSkinned>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &builder::SKINNED_ATTRIBS,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(builder::standard_color_target(config.format))],
                compilation_options: Default::default(),
            }),
            primitive: builder::standard_primitive_state(),
            depth_stencil: Some(builder::depth_stencil_opaque()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            uniform_ring,
            bind_group_layout: draw_bgl,
            dummy_blendshape_buffer,
            scene_bind_group_layout: scene_bgl,
            scene_uniform_buffer,
        }
    }

    /// Shared bind group layouts for code that needs to reconstruct them (e.g. SkinnedPbrMRTPipeline).
    ///
    /// Returns `(draw_bgl, scene_bgl, scene_uniform_size_bytes)`.
    pub(crate) fn create_bind_group_layouts(
        device: &wgpu::Device,
    ) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout, u64) {
        create_skinned_pbr_bind_group_layouts(device)
    }
}

impl RenderPipeline for SkinnedPbrPipeline {
    fn bind_pipeline(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.pipeline);
    }

    fn bind_draw(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        let dynamic_offset = batch_index
            .map(|i| self.uniform_ring.dynamic_offset(i, frame_index))
            .unwrap_or(0);
        let bg = draw_bind_group.expect("skinned PBR pipeline requires draw_bind_group");
        pass.set_bind_group(0, bg, &[dynamic_offset]);
    }

    fn bind_scene(&self, pass: &mut wgpu::RenderPass, scene_bind_group: Option<&wgpu::BindGroup>) {
        if let Some(bg) = scene_bind_group {
            pass.set_bind_group(1, bg, &[]);
        }
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        let blendshape = buffers
            .blendshape_buffer
            .as_ref()
            .map(|b| b.as_ref())
            .unwrap_or(&self.dummy_blendshape_buffer);
        Some(create_skinned_draw_bg(
            device,
            "skinned PBR draw BG",
            &self.bind_group_layout,
            &self.uniform_ring.buffer,
            blendshape,
        ))
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        self.set_skinned_buffers(pass, buffers);
        self.draw_skinned_indexed(pass, buffers);
    }

    fn set_skinned_buffers(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        let Some((vb, ib)) = buffers.skinned_buffers() else {
            return;
        };
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), buffers.index_format);
    }

    fn draw_skinned_indexed(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        for &(index_start, index_count) in &buffers.draw_ranges() {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, items, frame_index);
    }

    fn write_scene_uniform(&self, queue: &wgpu::Queue, scene: &[u8]) {
        queue.write_buffer(&self.scene_uniform_buffer, 0, scene);
    }

    fn create_scene_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_position: [f32; 3],
        view_space_z_coeffs: [f32; 4],
        cluster_count_x: u32,
        cluster_count_y: u32,
        cluster_count_z: u32,
        near_clip: f32,
        far_clip: f32,
        light_count: u32,
        viewport_width: u32,
        viewport_height: u32,
        light_buffer: &wgpu::Buffer,
        cluster_light_counts: &wgpu::Buffer,
        cluster_light_indices: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        let scene = SceneUniforms {
            view_position,
            _pad0: 0.0,
            view_space_z_coeffs,
            cluster_count_x,
            cluster_count_y,
            cluster_count_z,
            near_clip,
            far_clip,
            light_count,
            viewport_width,
            viewport_height,
        };
        Some(create_scene_bg(
            device,
            "skinned PBR scene BG",
            &self.scene_bind_group_layout,
            &self.scene_uniform_buffer,
            &scene,
            queue,
            light_buffer,
            cluster_light_counts,
            cluster_light_indices,
        ))
    }
}

// ─── SkinnedPbrMRTPipeline ────────────────────────────────────────────────────

/// Skinned PBR MRT pipeline: same as [`SkinnedPbrPipeline`] with three-target G-buffer for RTAO.
pub struct SkinnedPbrMRTPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: SkinnedUniformRingBuffer,
    bind_group_layout: wgpu::BindGroupLayout,
    dummy_blendshape_buffer: wgpu::Buffer,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_uniform_buffer: wgpu::Buffer,
}

impl SkinnedPbrMRTPipeline {
    /// Creates a skinned PBR MRT pipeline with three color attachments.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned PBR MRT shader"),
            source: wgpu::ShaderSource::Wgsl(SKINNED_PBR_MRT_SHADER_SRC.into()),
        });
        let (draw_bgl, scene_bgl, scene_uniform_size) =
            SkinnedPbrPipeline::create_bind_group_layouts(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned PBR MRT pipeline layout"),
            bind_group_layouts: &[&draw_bgl, &scene_bgl],
            immediate_size: 0,
        });
        let uniform_ring =
            SkinnedUniformRingBuffer::new(device, "skinned PBR MRT uniform ring buffer");
        let dummy_blendshape_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned PBR MRT dummy blendshape buffer"),
            size: 1,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let scene_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned PBR MRT scene uniform buffer"),
            size: scene_uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mrt_targets = builder::mrt_color_targets(config.format);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned PBR MRT pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<VertexSkinned>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &builder::SKINNED_ATTRIBS,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &mrt_targets,
                compilation_options: Default::default(),
            }),
            primitive: builder::standard_primitive_state(),
            depth_stencil: Some(builder::depth_stencil_opaque()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            uniform_ring,
            bind_group_layout: draw_bgl,
            dummy_blendshape_buffer,
            scene_bind_group_layout: scene_bgl,
            scene_uniform_buffer,
        }
    }
}

impl RenderPipeline for SkinnedPbrMRTPipeline {
    fn bind_pipeline(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.pipeline);
    }

    fn bind_draw(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        let dynamic_offset = batch_index
            .map(|i| self.uniform_ring.dynamic_offset(i, frame_index))
            .unwrap_or(0);
        let bg = draw_bind_group.expect("skinned PBR MRT pipeline requires draw_bind_group");
        pass.set_bind_group(0, bg, &[dynamic_offset]);
    }

    fn bind_scene(&self, pass: &mut wgpu::RenderPass, scene_bind_group: Option<&wgpu::BindGroup>) {
        if let Some(bg) = scene_bind_group {
            pass.set_bind_group(1, bg, &[]);
        }
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        let blendshape = buffers
            .blendshape_buffer
            .as_ref()
            .map(|b| b.as_ref())
            .unwrap_or(&self.dummy_blendshape_buffer);
        Some(create_skinned_draw_bg(
            device,
            "skinned PBR MRT draw BG",
            &self.bind_group_layout,
            &self.uniform_ring.buffer,
            blendshape,
        ))
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        self.set_skinned_buffers(pass, buffers);
        self.draw_skinned_indexed(pass, buffers);
    }

    fn set_skinned_buffers(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        let Some((vb, ib)) = buffers.skinned_buffers() else {
            return;
        };
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), buffers.index_format);
    }

    fn draw_skinned_indexed(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        for &(index_start, index_count) in &buffers.draw_ranges() {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, items, frame_index);
    }

    fn write_scene_uniform(&self, queue: &wgpu::Queue, scene: &[u8]) {
        queue.write_buffer(&self.scene_uniform_buffer, 0, scene);
    }

    fn create_scene_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_position: [f32; 3],
        view_space_z_coeffs: [f32; 4],
        cluster_count_x: u32,
        cluster_count_y: u32,
        cluster_count_z: u32,
        near_clip: f32,
        far_clip: f32,
        light_count: u32,
        viewport_width: u32,
        viewport_height: u32,
        light_buffer: &wgpu::Buffer,
        cluster_light_counts: &wgpu::Buffer,
        cluster_light_indices: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        let scene = SceneUniforms {
            view_position,
            _pad0: 0.0,
            view_space_z_coeffs,
            cluster_count_x,
            cluster_count_y,
            cluster_count_z,
            near_clip,
            far_clip,
            light_count,
            viewport_width,
            viewport_height,
        };
        Some(create_scene_bg(
            device,
            "skinned PBR MRT scene BG",
            &self.scene_bind_group_layout,
            &self.scene_uniform_buffer,
            &scene,
            queue,
            light_buffer,
            cluster_light_counts,
            cluster_light_indices,
        ))
    }
}
