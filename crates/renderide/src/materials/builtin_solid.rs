//! Minimal fullscreen-triangle material: no bind groups, solid fragment color from WGSL overrides.

use crate::materials::family::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
use crate::materials::wgsl::{compose_wgsl, WgslPatch};
use crate::pipelines::ShaderPermutation;

/// Builtin family id for [`SolidColorFamily`].
pub const SOLID_COLOR_FAMILY_ID: MaterialFamilyId = MaterialFamilyId(1);

/// Reference WGSL material: vertex index draws a clip-space triangle; fragment color from patches.
pub struct SolidColorFamily;

impl MaterialPipelineFamily for SolidColorFamily {
    fn family_id(&self) -> MaterialFamilyId {
        SOLID_COLOR_FAMILY_ID
    }

    fn build_wgsl(&self, permutation: ShaderPermutation) -> String {
        let patch = if permutation.0 & 1 != 0 {
            WgslPatch::ReplaceFirst {
                needle: "// @MATERIAL_FRAG_RETURN",
                replacement: "return vec4<f32>(1.0, 0.0, 0.0, 1.0);",
            }
        } else {
            WgslPatch::ReplaceFirst {
                needle: "// @MATERIAL_FRAG_RETURN",
                replacement: "return vec4<f32>(0.02, 0.05, 0.12, 1.0);",
            }
        };
        compose_wgsl(WGSL_TEMPLATE, &[patch])
    }

    fn create_render_pipeline(
        &self,
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        desc: &MaterialPipelineDesc,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solid_color_material"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("solid_color_material"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
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
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
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

const WGSL_TEMPLATE: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    out.clip_position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // @MATERIAL_FRAG_RETURN
}
"#;
