//! Manifest-driven raster materials: one [`MaterialFamilyId`] and many composed WGSL stems.
//!
//! [`ManifestStemMaterialFamily`] is constructed per resolved stem (see [`super::MaterialRegistry`]).

use std::sync::Arc;

use crate::backend::{empty_material_bind_group_layout, FrameGpuResources};
use crate::embedded_shaders;
use crate::materials::{reflect_raster_material_wgsl, validate_per_draw_group2};
use crate::materials::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
use crate::pipelines::raster::SHADER_PERM_MULTIVIEW_STEREO;
use crate::pipelines::ShaderPermutation;
use crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE;

/// Stable id for all shaders whose Unity name maps to an entry in the embedded shader manifest.
pub const MANIFEST_RASTER_FAMILY_ID: MaterialFamilyId = MaterialFamilyId(3);

/// Composed target stem for a manifest base stem (e.g. `world_unlit_default` → `world_unlit_multiview`).
pub fn manifest_composed_stem_for_permutation(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> String {
    if permutation.0 == SHADER_PERM_MULTIVIEW_STEREO.0 {
        if base_stem.ends_with("_default") {
            return format!("{}_multiview", base_stem.trim_end_matches("_default"));
        }
        return base_stem.to_string();
    }
    if base_stem.ends_with("_multiview") {
        return format!("{}_default", base_stem.trim_end_matches("_multiview"));
    }
    base_stem.to_string()
}

/// Raster family parameterized by a manifest stem (`shaders/target/<composed_stem>.wgsl`).
#[derive(Debug)]
pub struct ManifestStemMaterialFamily {
    /// Stem from [`super::MaterialRouter::stem_for_shader_asset`] (e.g. `world_unlit_default`).
    pub stem: Arc<str>,
}

impl ManifestStemMaterialFamily {
    /// Builds a family for the given manifest stem.
    pub fn new(stem: Arc<str>) -> Self {
        Self { stem }
    }

    fn composed_stem(&self, permutation: ShaderPermutation) -> String {
        manifest_composed_stem_for_permutation(self.stem.as_ref(), permutation)
    }
}

impl MaterialPipelineFamily for ManifestStemMaterialFamily {
    fn family_id(&self) -> MaterialFamilyId {
        MANIFEST_RASTER_FAMILY_ID
    }

    fn manifest_stem(&self) -> Option<Arc<str>> {
        Some(self.stem.clone())
    }

    fn build_wgsl(&self, permutation: ShaderPermutation) -> String {
        let stem = self.composed_stem(permutation);
        embedded_shaders::embedded_target_wgsl(&stem)
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
        wgsl_source: &str,
    ) -> wgpu::RenderPipeline {
        let reflected = reflect_raster_material_wgsl(wgsl_source).expect(
            "reflect manifest stem material (must match frame globals + per-draw contract)",
        );
        validate_per_draw_group2(&reflected.per_draw_entries).expect("per_draw group2");

        let frame_bgl = FrameGpuResources::bind_group_layout(device);
        let material_bgl = if reflected.material_entries.is_empty() {
            empty_material_bind_group_layout(device)
        } else {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("manifest_stem_material"),
                entries: &reflected.material_entries,
            })
        };
        let per_draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("manifest_stem_per_draw"),
            entries: &reflected.per_draw_entries,
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("manifest_stem_material"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&material_bgl), Some(&per_draw_bgl)],
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
        let uv_layout = wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("manifest_stem_material"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[pos_layout, nrm_layout, uv_layout],
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
