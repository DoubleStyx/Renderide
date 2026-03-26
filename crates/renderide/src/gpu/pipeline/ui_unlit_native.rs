//! Native WGSL `UI_Unlit` pipeline (composed `ui_unlit.wgsl` from `wgsl_modules/`).
//!
//! Binds [`super::super::mesh::VertexUiCanvas`] data, material uniforms, optional depth for the
//! `OVERLAY` keyword, and placeholder 1×1 textures until GPU texture assets are wired.

use std::sync::OnceLock;

use nalgebra::Matrix4;
use wgpu::util::DeviceExt;

use crate::assets::{
    MaterialPropertyLookupIds, MaterialPropertyStore, NativeUiSurfaceBlend, UiUnlitMaterialUniform,
    UiUnlitPropertyIds, ui_unlit_material_uniform,
};

use super::super::mesh::{GpuMeshBuffers, VertexUiCanvas};
use super::builder;
use super::core::{NonSkinnedUniformUpload, RenderPipeline};
use super::ring_buffer::UniformRingBuffer;

const UI_UNLIT_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/ui_unlit.wgsl"));

static FALLBACK_WHITE: OnceLock<(wgpu::Texture, wgpu::TextureView)> = OnceLock::new();

/// GPU uniform for [`NativeUiOverlayUnproject`] in WGSL (two column-major `mat4` inverses).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NativeUiOverlayUnprojectUniform {
    /// Inverse scene (main camera) projection, column-major rows.
    pub inv_scene_proj: [[f32; 4]; 4],
    /// Inverse UI overlay projection (orthographic screen UI or same as scene for world UI).
    pub inv_ui_proj: [[f32; 4]; 4],
}

/// Converts a column-major [`Matrix4`] to WGSL `mat4x4f` column layout (`[[f32;4];4]`).
pub(crate) fn matrix4_to_wgsl_column_major(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    std::array::from_fn(|c| {
        let col = m.column(c);
        [col.x, col.y, col.z, col.w]
    })
}

/// Bind group layout for group 1: sampled scene depth and overlay unprojection uniforms.
pub(crate) fn native_ui_scene_depth_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("native ui scene depth BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                        NativeUiOverlayUnprojectUniform,
                    >() as u64),
                },
                count: None,
            },
        ],
    })
}

/// Bind group layout for native `UI_Unlit` material uniforms + two sampled textures.
pub(crate) fn create_ui_unlit_material_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ui unlit material BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                        UiUnlitMaterialUniform,
                    >() as u64),
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub(crate) fn fallback_white(device: &wgpu::Device) -> &'static wgpu::TextureView {
    &FALLBACK_WHITE
        .get_or_init(|| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ui native 1x1 white"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        })
        .1
}

/// Native `UI_Unlit` render pipeline.
pub struct UiUnlitNativePipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    ring_bind_group: wgpu::BindGroup,
    material_uniform: wgpu::Buffer,
    material_bind_group: wgpu::BindGroup,
    material_bgl: wgpu::BindGroupLayout,
    linear_sampler: wgpu::Sampler,
}

impl UiUnlitNativePipeline {
    /// Builds the pipeline for the swapchain format and depth-stencil attachment.
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        surface_blend: NativeUiSurfaceBlend,
    ) -> Self {
        Self::new_with_depth_stencil(
            device,
            config,
            builder::depth_stencil_no_depth(),
            "ui unlit native RP",
            surface_blend,
        )
    }

    /// Same as [`Self::new`] but with GraphicsChunk stencil test for masked overlay draws.
    pub fn new_with_stencil(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        surface_blend: NativeUiSurfaceBlend,
    ) -> Self {
        Self::new_with_depth_stencil(
            device,
            config,
            builder::depth_stencil_native_ui_stencil_content(),
            "ui unlit native stencil RP",
            surface_blend,
        )
    }

    fn new_with_depth_stencil(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        depth_stencil: wgpu::DepthStencilState,
        pipeline_label: &'static str,
        surface_blend: NativeUiSurfaceBlend,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui_unlit native"),
            source: wgpu::ShaderSource::Wgsl(UI_UNLIT_WGSL.into()),
        });
        let ring_bgl = builder::uniform_ring_bind_group_layout(device, "ui unlit native ring BGL");
        let scene_bgl = native_ui_scene_depth_bind_group_layout(device);
        let white = fallback_white(device);
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui unlit linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let material_bgl = create_ui_unlit_material_bind_group_layout(device);
        let initial = UiUnlitMaterialUniform {
            tint: [1.0, 1.0, 1.0, 1.0],
            overlay_tint: [1.0, 1.0, 1.0, 0.73],
            main_tex_st: [1.0, 1.0, 0.0, 0.0],
            mask_tex_st: [1.0, 1.0, 0.0, 0.0],
            rect: [0.0, 0.0, 1.0, 1.0],
            cutoff: 0.98,
            flags: 0,
            pad_tail: [0; 2],
        };
        let material_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui unlit material uniform"),
            contents: bytemuck::bytes_of(&initial),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui unlit material BG"),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(white),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&linear),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui unlit native PL"),
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
        let uniform_ring = UniformRingBuffer::new(device, "ui unlit native ring");
        let ring_bind_group = builder::uniform_ring_bind_group(
            device,
            "ui unlit native ring BG",
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

    /// Material bind group layout (group 2); must match cached bind groups built for this pipeline.
    pub fn material_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_bgl
    }

    /// Linear sampler for `_MainTex` / `_MaskTex`.
    pub fn linear_sampler(&self) -> &wgpu::Sampler {
        &self.linear_sampler
    }

    /// Uniform buffer written each draw for material properties.
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
        ids: &UiUnlitPropertyIds,
    ) {
        let (u, _, _) = ui_unlit_material_uniform(store, lookup, ids);
        queue.write_buffer(&self.material_uniform, 0, bytemuck::bytes_of(&u));
        pass.set_bind_group(2, &self.material_bind_group, &[]);
    }
}

impl RenderPipeline for UiUnlitNativePipeline {
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

#[cfg(test)]
mod native_ui_overlay_uniform_tests {
    use super::NativeUiOverlayUnprojectUniform;

    /// Matches WGSL `NativeUiOverlayUnproject`: two `mat4x4<f32>` (column-major in the uniform).
    #[test]
    fn native_ui_overlay_unproject_uniform_is_two_mat4() {
        assert_eq!(std::mem::size_of::<NativeUiOverlayUnprojectUniform>(), 128);
    }
}
