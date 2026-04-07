//! Debug mesh material: world-space normals as RGB.

use std::num::NonZeroU64;

use crate::gpu::PER_DRAW_UNIFORM_STRIDE;
use crate::materials::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE;

/// Builtin family id for [`DebugWorldNormalsFamily`].
pub const DEBUG_WORLD_NORMALS_FAMILY_ID: MaterialFamilyId = MaterialFamilyId(2);

/// [`ShaderPermutation`] for multiview WGSL (`debug_world_normals_multiview.wgsl`).
pub const SHADER_PERM_MULTIVIEW_STEREO: ShaderPermutation = ShaderPermutation(1);

/// Minimum `min_binding_size` for the dynamic uniform binding (256-byte slots).
fn per_draw_uniform_min_binding_size() -> NonZeroU64 {
    NonZeroU64::new(PER_DRAW_UNIFORM_STRIDE as u64).expect("stride positive")
}

/// World-normal debug visualization for decomposed position/normal vertex streams.
pub struct DebugWorldNormalsFamily;

impl DebugWorldNormalsFamily {
    /// Shared layout for [`MaterialPipelineFamily::create_render_pipeline`] and bind group creation at draw time.
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("debug_world_normals_material"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(per_draw_uniform_min_binding_size()),
                },
                count: None,
            }],
        })
    }
}

impl MaterialPipelineFamily for DebugWorldNormalsFamily {
    fn family_id(&self) -> MaterialFamilyId {
        DEBUG_WORLD_NORMALS_FAMILY_ID
    }

    fn build_wgsl(&self, permutation: ShaderPermutation) -> String {
        if permutation.0 == SHADER_PERM_MULTIVIEW_STEREO.0 {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/debug_world_normals_multiview.wgsl"
            ))
            .to_string()
        } else {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/debug_world_normals.wgsl"
            ))
            .to_string()
        }
    }

    fn create_render_pipeline(
        &self,
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        desc: &MaterialPipelineDesc,
    ) -> wgpu::RenderPipeline {
        let bgl = Self::bind_group_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("debug_world_normals_material"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pos_layout = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x4,
            }],
        };
        let nrm_layout = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x4,
            }],
        };

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("debug_world_normals_material"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[pos_layout, nrm_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: desc.surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: desc
                .depth_stencil_format
                .map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(MAIN_FORWARD_DEPTH_COMPARE),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
            multisample: wgpu::MultisampleState {
                count: desc.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: desc.multiview_mask,
            cache: None,
        })
    }
}
