//! Pipeline descriptor helper for depth-like world-mesh passes.

use std::num::NonZeroU32;

use crate::materials::{RasterFrontFace, RasterPrimitiveTopology};
use crate::render_graph::gpu_cache::create_wgsl_shader_module;

/// Pipeline descriptor inputs shared by depth-only, shadow, and normal prepasses.
pub(super) struct DepthLikePipelineSpec<'a> {
    /// Debug label used for the shader module, pipeline layout, and pipeline.
    pub(super) label: &'a str,
    /// WGSL source for this pass variant.
    pub(super) shader_source: &'a str,
    /// Bind group layouts used by this pipeline.
    pub(super) bind_group_layouts: &'a [&'a wgpu::BindGroupLayout],
    /// Vertex buffer layouts required by this pass.
    pub(super) vertex_buffers: &'a [wgpu::VertexBufferLayout<'static>],
    /// Fragment entry point, or `None` for depth-only passes.
    pub(super) fragment_entry_point: Option<&'a str>,
    /// Color targets used by the fragment entry point.
    pub(super) color_targets: &'a [Option<wgpu::ColorTargetState>],
    /// Primitive topology baked into the pipeline.
    pub(super) primitive_topology: RasterPrimitiveTopology,
    /// Front-face winding baked into the pipeline.
    pub(super) front_face: RasterFrontFace,
    /// Optional cull mode baked into the pipeline.
    pub(super) cull_mode: Option<wgpu::Face>,
    /// Depth/stencil target format.
    pub(super) depth_stencil_format: wgpu::TextureFormat,
    /// Whether the pass writes depth.
    pub(super) depth_write_enabled: bool,
    /// Depth compare operation.
    pub(super) depth_compare: wgpu::CompareFunction,
    /// Active sample count.
    pub(super) sample_count: u32,
    /// Multiview mask for stereo rendering.
    pub(super) multiview_mask: Option<NonZeroU32>,
}

/// Creates a render pipeline for world-mesh passes that share depth-like scaffolding.
pub(super) fn create_depth_like_pipeline(
    device: &wgpu::Device,
    spec: DepthLikePipelineSpec<'_>,
) -> wgpu::RenderPipeline {
    let shader = create_wgsl_shader_module(device, spec.label, spec.shader_source);
    let bind_group_layouts = spec
        .bind_group_layouts
        .iter()
        .copied()
        .map(Some)
        .collect::<Vec<_>>();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(spec.label),
        bind_group_layouts: &bind_group_layouts,
        immediate_size: 0,
    });
    let fragment = spec
        .fragment_entry_point
        .map(|entry_point| wgpu::FragmentState {
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            targets: spec.color_targets,
        });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(spec.label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: spec.vertex_buffers,
        },
        fragment,
        primitive: wgpu::PrimitiveState {
            topology: spec.primitive_topology.to_wgpu(),
            front_face: spec.front_face.to_wgpu(),
            cull_mode: spec.cull_mode,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: spec.depth_stencil_format,
            depth_write_enabled: Some(spec.depth_write_enabled),
            depth_compare: Some(spec.depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: spec.sample_count.max(1),
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: spec.multiview_mask,
        cache: None,
    })
}
