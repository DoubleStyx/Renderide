//! Native WGSL text pipelines (`ui_text_unlit.wgsl` for both UI and standalone Text/Unlit).

use wgpu::util::DeviceExt;

use crate::assets::{
    DEFAULT_OVERLAY_TINT, MaterialPropertyLookupIds, MaterialPropertyStore, NativeUiSurfaceBlend,
    UiTextUnlitMaterialUniform, UiTextUnlitPropertyIds, ui_text_unlit_material_uniform,
};

use super::super::mesh::{GpuMeshBuffers, VertexUiCanvas};
use super::builder;
use super::core::{NonSkinnedUniformUpload, RenderPipeline};
use super::ring_buffer::UniformRingBuffer;
use super::ui_unlit_native::{fallback_white, native_ui_scene_depth_bind_group_layout};

const UI_TEXT_UNLIT_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/ui_text_unlit.wgsl"));

/// Native `UI_TextUnlit` render pipeline.
pub struct UiTextUnlitNativePipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    ring_bind_group: wgpu::BindGroup,
    material_uniform: wgpu::Buffer,
    material_bind_group: wgpu::BindGroup,
    material_bgl: wgpu::BindGroupLayout,
    linear_sampler: wgpu::Sampler,
}

impl UiTextUnlitNativePipeline {
    /// Builds the pipeline for the swapchain format.
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        surface_blend: NativeUiSurfaceBlend,
    ) -> Self {
        Self::new_inner(
            device,
            config,
            surface_blend,
            UI_TEXT_UNLIT_WGSL,
            "ui_text_unlit native",
            "ui text unlit native RP",
        )
    }

    /// Builds the native `Text/Unlit` pipeline for the swapchain format.
    pub fn new_text(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        surface_blend: NativeUiSurfaceBlend,
    ) -> Self {
        Self::new_inner(
            device,
            config,
            surface_blend,
            UI_TEXT_UNLIT_WGSL,
            "text_unlit native",
            "text unlit native RP",
        )
    }

    fn new_inner(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        surface_blend: NativeUiSurfaceBlend,
        shader_source: &'static str,
        shader_label: &'static str,
        pipeline_label: &'static str,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_label),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let ring_bgl = builder::uniform_ring_bind_group_layout(device, "ui text unlit ring BGL");
        let scene_bgl = native_ui_scene_depth_bind_group_layout(device);
        let white = fallback_white(device);
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui text unlit linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui text unlit material BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            UiTextUnlitMaterialUniform,
                        >()
                            as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let initial = UiTextUnlitMaterialUniform {
            tint_color: [1.0, 1.0, 1.0, 1.0],
            overlay_tint: DEFAULT_OVERLAY_TINT,
            outline_color: [1.0, 1.0, 1.0, 0.0],
            background_color: [0.0, 0.0, 0.0, 0.0],
            range_xy: [0.001, 0.001, 0.0, 0.0],
            face_dilate: 0.0,
            face_softness: 0.0,
            outline_size: 0.0,
            pad_scalar: 0.0,
            rect: [0.0, 0.0, 1.0, 1.0],
            flags: 1,
            pad_flags: 0,
            pad_tail: [0; 2],
        };
        let material_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui text unlit material uniform"),
            contents: bytemuck::bytes_of(&initial),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui text unlit material BG"),
            layout: &material_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(white),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&linear),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui text unlit native PL"),
            bind_group_layouts: &[&ring_bgl, &scene_bgl, &material_bgl],
            immediate_size: 0,
        });
        let vb_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VertexUiCanvas>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };
        Self::finish_text_unlit_pipeline(
            device,
            config,
            shader,
            ring_bgl,
            material_bgl,
            linear,
            material_uniform,
            material_bind_group,
            pipeline_layout,
            vb_layout,
            builder::depth_stencil_no_depth(),
            pipeline_label,
            surface_blend,
        )
    }

    /// Same as [`Self::new`] with GraphicsChunk stencil masking in the overlay pass.
    pub fn new_with_stencil(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        surface_blend: NativeUiSurfaceBlend,
    ) -> Self {
        Self::new_with_stencil_inner(
            device,
            config,
            surface_blend,
            UI_TEXT_UNLIT_WGSL,
            "ui_text_unlit native stencil",
            "ui text unlit native stencil RP",
        )
    }

    /// Same as [`Self::new_text`] with GraphicsChunk stencil masking in the overlay pass.
    pub fn new_text_with_stencil(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        surface_blend: NativeUiSurfaceBlend,
    ) -> Self {
        Self::new_with_stencil_inner(
            device,
            config,
            surface_blend,
            UI_TEXT_UNLIT_WGSL,
            "text_unlit native stencil",
            "text unlit native stencil RP",
        )
    }

    fn new_with_stencil_inner(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        surface_blend: NativeUiSurfaceBlend,
        shader_source: &'static str,
        shader_label: &'static str,
        pipeline_label: &'static str,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_label),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let ring_bgl = builder::uniform_ring_bind_group_layout(device, "ui text unlit ring BGL");
        let scene_bgl = native_ui_scene_depth_bind_group_layout(device);
        let white = fallback_white(device);
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui text unlit linear stencil"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui text unlit material BGL stencil"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            UiTextUnlitMaterialUniform,
                        >()
                            as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let initial = UiTextUnlitMaterialUniform {
            tint_color: [1.0, 1.0, 1.0, 1.0],
            overlay_tint: DEFAULT_OVERLAY_TINT,
            outline_color: [1.0, 1.0, 1.0, 0.0],
            background_color: [0.0, 0.0, 0.0, 0.0],
            range_xy: [0.001, 0.001, 0.0, 0.0],
            face_dilate: 0.0,
            face_softness: 0.0,
            outline_size: 0.0,
            pad_scalar: 0.0,
            rect: [0.0, 0.0, 1.0, 1.0],
            flags: 1,
            pad_flags: 0,
            pad_tail: [0; 2],
        };
        let material_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui text unlit material uniform stencil"),
            contents: bytemuck::bytes_of(&initial),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui text unlit material BG stencil"),
            layout: &material_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(white),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&linear),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui text unlit native PL stencil"),
            bind_group_layouts: &[&ring_bgl, &scene_bgl, &material_bgl],
            immediate_size: 0,
        });
        let vb_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VertexUiCanvas>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };
        Self::finish_text_unlit_pipeline(
            device,
            config,
            shader,
            ring_bgl,
            material_bgl,
            linear,
            material_uniform,
            material_bind_group,
            pipeline_layout,
            vb_layout,
            builder::depth_stencil_native_ui_stencil_content(),
            pipeline_label,
            surface_blend,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_text_unlit_pipeline(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        shader: wgpu::ShaderModule,
        ring_bgl: wgpu::BindGroupLayout,
        material_bgl: wgpu::BindGroupLayout,
        linear: wgpu::Sampler,
        material_uniform: wgpu::Buffer,
        material_bind_group: wgpu::BindGroup,
        pipeline_layout: wgpu::PipelineLayout,
        vb_layout: wgpu::VertexBufferLayout<'_>,
        depth_stencil: wgpu::DepthStencilState,
        pipeline_label: &'static str,
        surface_blend: NativeUiSurfaceBlend,
    ) -> Self {
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(pipeline_label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vb_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(surface_blend.to_wgpu_blend_state()),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..builder::standard_primitive_state()
            },
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_ring = UniformRingBuffer::new(device, "ui text unlit native ring");
        let ring_bind_group = builder::uniform_ring_bind_group(
            device,
            "ui text unlit native ring BG",
            &ring_bgl,
            &uniform_ring.buffer,
        );
        Self {
            pipeline,
            uniform_ring,
            ring_bind_group,
            material_uniform,
            material_bind_group,
            material_bgl,
            linear_sampler: linear,
        }
    }

    /// Material bind group layout for group 2.
    pub fn material_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_bgl
    }

    /// Linear sampler for the font atlas.
    pub fn linear_sampler(&self) -> &wgpu::Sampler {
        &self.linear_sampler
    }

    /// Per-draw material uniform buffer.
    pub fn material_uniform_buffer(&self) -> &wgpu::Buffer {
        &self.material_uniform
    }

    /// Writes material uniforms for `lookup` and binds group 2.
    pub fn write_material_bind(
        &self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        ids: &UiTextUnlitPropertyIds,
    ) {
        let (u, _) = ui_text_unlit_material_uniform(store, lookup, ids);
        queue.write_buffer(&self.material_uniform, 0, bytemuck::bytes_of(&u));
        pass.set_bind_group(2, &self.material_bind_group, &[]);
    }
}

impl RenderPipeline for UiTextUnlitNativePipeline {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn bind_pipeline(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
    }

    fn bind_draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        batch_index: Option<u32>,
        frame_index: u64,
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        let dynamic_offset = batch_index
            .map(|i| self.uniform_ring.dynamic_offset(i, frame_index))
            .unwrap_or(0);
        pass.set_bind_group(0, &self.ring_bind_group, &[dynamic_offset]);
    }

    fn set_mesh_buffers(&self, pass: &mut wgpu::RenderPass<'_>, buffers: &GpuMeshBuffers) {
        let Some((vb, ib)) = buffers.ui_canvas_buffers() else {
            let (vb, ib) = buffers.uv_buffers();
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), buffers.index_format);
            return;
        };
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), buffers.index_format);
    }

    fn draw_mesh_indexed(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        buffers: &GpuMeshBuffers,
        index_range_override: Option<(u32, u32)>,
    ) {
        for &(index_start, index_count) in &buffers.effective_draw_ranges(index_range_override) {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    fn supports_instancing(&self) -> bool {
        true
    }

    fn draw_mesh_indexed_instanced(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        buffers: &GpuMeshBuffers,
        instance_count: u32,
        index_range_override: Option<(u32, u32)>,
    ) {
        for &(index_start, index_count) in &buffers.effective_draw_ranges(index_range_override) {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..instance_count);
        }
    }

    fn upload_batch(
        &self,
        queue: &wgpu::Queue,
        draws: &[NonSkinnedUniformUpload],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, draws, frame_index);
    }
}
