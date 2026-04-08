//! Debug mesh material: world-space normals as RGB (`shaders/target/debug_world_normals_*.wgsl`).

use std::num::NonZeroU64;

use crate::backend::{empty_material_bind_group_layout, FrameGpuResources};
use crate::gpu::PER_DRAW_UNIFORM_STRIDE;
use crate::materials::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE;

/// Builtin family id for [`DebugWorldNormalsFamily`].
pub const DEBUG_WORLD_NORMALS_FAMILY_ID: MaterialFamilyId = MaterialFamilyId(2);

/// [`ShaderPermutation`] for multiview WGSL (`debug_world_normals_multiview` target stem).
pub const SHADER_PERM_MULTIVIEW_STEREO: ShaderPermutation = ShaderPermutation(1);

/// Minimum `min_binding_size` for the dynamic uniform binding (256-byte slots).
fn per_draw_uniform_min_binding_size() -> NonZeroU64 {
    NonZeroU64::new(PER_DRAW_UNIFORM_STRIDE as u64).expect("stride positive")
}

/// World-normal debug visualization for decomposed position/normal vertex streams.
pub struct DebugWorldNormalsFamily;

impl DebugWorldNormalsFamily {
    /// `@group(2)` dynamic uniform layout for [`crate::backend::DebugDrawResources`].
    pub fn per_draw_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("debug_world_normals_per_draw"),
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

    fn target_stem(permutation: ShaderPermutation) -> &'static str {
        if permutation.0 == SHADER_PERM_MULTIVIEW_STEREO.0 {
            "debug_world_normals_multiview"
        } else {
            "debug_world_normals_default"
        }
    }
}

impl MaterialPipelineFamily for DebugWorldNormalsFamily {
    fn family_id(&self) -> MaterialFamilyId {
        DEBUG_WORLD_NORMALS_FAMILY_ID
    }

    fn build_wgsl(&self, permutation: ShaderPermutation) -> String {
        let stem = Self::target_stem(permutation);
        crate::embedded_shaders::embedded_target_wgsl(stem)
            .unwrap_or_else(|| {
                panic!("composed shader missing for stem {stem} (run build with shaders/source)")
            })
            .to_string()
    }

    fn create_render_pipeline(
        &self,
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        desc: &MaterialPipelineDesc,
    ) -> wgpu::RenderPipeline {
        let frame_bgl = FrameGpuResources::bind_group_layout(device);
        let empty_mat_bgl = empty_material_bind_group_layout(device);
        let per_draw_bgl = Self::per_draw_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("debug_world_normals_material"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&empty_mat_bgl), Some(&per_draw_bgl)],
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
