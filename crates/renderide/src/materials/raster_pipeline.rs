//! Shared [`wgpu::RenderPipeline`] construction for reflective raster materials (frame, material, per-draw groups).
//!
//! Opaque paths use no blend state and write RGB only so destination alpha stays at the clear value
//! for float render textures. Pass descriptors from `//#pass` directives can override blend, depth,
//! cull, stencil/color-write, and depth state per material.

use crate::backend::{empty_material_bind_group_layout, FrameGpuResources};
use crate::materials::material_passes::{default_pass, MaterialPassDesc, MaterialRenderState};
use crate::materials::pipeline_build_error::PipelineBuildError;
use crate::materials::{
    reflect_raster_material_wgsl, reflect_vertex_shader_needs_color_stream,
    reflect_vertex_shader_needs_uv0_stream, validate_per_draw_group2, MaterialPipelineDesc,
};

/// Vertex stream toggles, blending, depth write, and material overrides for
/// [`create_reflective_raster_mesh_forward_pipeline`].
pub(crate) struct ReflectiveRasterMeshForwardPipelineDesc {
    /// Include UV0 vertex stream when the shader references it.
    pub include_uv_vertex_buffer: bool,
    /// Include vertex color stream when the shader references it.
    pub include_color_vertex_buffer: bool,
    /// Alpha blending vs opaque RGB-only writes for the default single pass.
    pub use_alpha_blending: bool,
    /// Depth write flag for the default single pass.
    pub depth_write_enabled: bool,
    /// Runtime material overrides for color mask, stencil, and depth state.
    pub render_state: MaterialRenderState,
}

fn mesh_forward_vertex_buffer_layouts() -> [wgpu::VertexBufferLayout<'static>; 8] {
    [
        wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x4,
            }],
        },
        wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x4,
            }],
        },
        wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x2,
            }],
        },
        wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            }],
        },
        wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x4,
            }],
        },
        wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x2,
            }],
        },
        wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32x2,
            }],
        },
        wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 7,
                format: wgpu::VertexFormat::Float32x2,
            }],
        },
    ]
}

fn pipeline_layout_and_vertex_buffers(
    device: &wgpu::Device,
    wgsl_source: &str,
    label: &'static str,
    include_uv_vertex_buffer: bool,
    include_color_vertex_buffer: bool,
) -> Result<(wgpu::PipelineLayout, Vec<wgpu::VertexBufferLayout<'static>>), PipelineBuildError> {
    let reflected = reflect_raster_material_wgsl(wgsl_source)?;
    validate_per_draw_group2(&reflected.per_draw_entries)?;

    let frame_bgl = FrameGpuResources::bind_group_layout(device);
    let material_bgl = if reflected.material_entries.is_empty() {
        empty_material_bind_group_layout(device)
    } else {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_material_props")),
            entries: &reflected.material_entries,
        })
    };
    let per_draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_per_draw")),
        entries: &reflected.per_draw_entries,
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&frame_bgl), Some(&material_bgl), Some(&per_draw_bgl)],
        immediate_size: 0,
    });

    let layouts = mesh_forward_vertex_buffer_layouts();
    let use_uv = include_uv_vertex_buffer && reflect_vertex_shader_needs_uv0_stream(wgsl_source);
    let use_color =
        include_color_vertex_buffer && reflect_vertex_shader_needs_color_stream(wgsl_source);
    let use_extended = reflected.vs_max_vertex_location.is_some_and(|m| m >= 4);

    let vertex_buffers = if use_extended {
        layouts[..8].to_vec()
    } else if use_color {
        layouts[..4].to_vec()
    } else if use_uv {
        layouts[..3].to_vec()
    } else {
        layouts[..2].to_vec()
    };

    Ok((layout, vertex_buffers))
}

/// Builds one pipeline for a single [`MaterialPassDesc`] sharing the reflected layout and vertex buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pipeline_from_pass(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    desc: &MaterialPipelineDesc,
    label: &str,
    layout: &wgpu::PipelineLayout,
    vertex_buffers: &[wgpu::VertexBufferLayout<'_>],
    pass: &MaterialPassDesc,
    render_state: MaterialRenderState,
) -> wgpu::RenderPipeline {
    let pass_label = format!("{label}__{}", pass.name);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&pass_label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(pass.vertex_entry),
            compilation_options: Default::default(),
            buffers: vertex_buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(pass.fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: desc.surface_format,
                blend: pass.blend,
                write_mask: render_state.color_writes(pass.write_mask),
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: pass.cull_mode,
            ..Default::default()
        },
        depth_stencil: desc
            .depth_stencil_format
            .map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(render_state.depth_write(pass.depth_write)),
                depth_compare: Some(render_state.depth_compare(pass.depth_compare)),
                stencil: if format.has_stencil_aspect() {
                    render_state.stencil_state()
                } else {
                    wgpu::StencilState::default()
                },
                bias: render_state
                    .depth_bias(pass.depth_bias_constant, pass.depth_bias_slope_scale),
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

/// Builds a default single-pass forward mesh pipeline from reflected WGSL (`@group(0..=2)`).
pub(crate) fn create_reflective_raster_mesh_forward_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    desc: &MaterialPipelineDesc,
    wgsl_source: &str,
    label: &'static str,
    raster: ReflectiveRasterMeshForwardPipelineDesc,
) -> Result<wgpu::RenderPipeline, PipelineBuildError> {
    let pass = default_pass(raster.use_alpha_blending, raster.depth_write_enabled);
    let (layout, vertex_buffers) = pipeline_layout_and_vertex_buffers(
        device,
        wgsl_source,
        label,
        raster.include_uv_vertex_buffer,
        raster.include_color_vertex_buffer,
    )?;
    Ok(build_pipeline_from_pass(
        device,
        module,
        desc,
        label,
        &layout,
        &vertex_buffers,
        &pass,
        raster.render_state,
    ))
}

/// Builds N pipelines (one per pass descriptor) that share reflected bind-group layout and vertex streams.
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_reflective_raster_mesh_forward_pipelines(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    desc: &MaterialPipelineDesc,
    wgsl_source: &str,
    label: &'static str,
    include_uv_vertex_buffer: bool,
    include_color_vertex_buffer: bool,
    passes: &[MaterialPassDesc],
    render_state: MaterialRenderState,
) -> Result<Vec<wgpu::RenderPipeline>, PipelineBuildError> {
    assert!(
        !passes.is_empty(),
        "create_reflective_raster_mesh_forward_pipelines called with empty passes (stem {label})"
    );
    let (layout, vertex_buffers) = pipeline_layout_and_vertex_buffers(
        device,
        wgsl_source,
        label,
        include_uv_vertex_buffer,
        include_color_vertex_buffer,
    )?;

    Ok(passes
        .iter()
        .map(|pass| {
            build_pipeline_from_pass(
                device,
                module,
                desc,
                label,
                &layout,
                &vertex_buffers,
                pass,
                render_state,
            )
        })
        .collect())
}
