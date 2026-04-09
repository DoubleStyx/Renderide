//! Shared [`wgpu::RenderPipeline`] construction for reflective raster materials (frame, material, per-draw groups).
//!
//! **Transparent blending:** embedded materials that mirror Unity transparent queues (e.g. UI text) may need
//! alpha blending derived from material uniforms (`_SrcBlend` / `_DstBlend`) or host render-queue metadata.
//! The default here is opaque replacement (`blend: None`); improving that is a cross-cutting, reflection-driven
//! follow-up—not per-shader logic in the mesh pass.

use crate::backend::{empty_material_bind_group_layout, FrameGpuResources};
use crate::materials::MaterialPipelineDesc;
use crate::materials::{
    reflect_raster_material_wgsl, reflect_vertex_shader_needs_color_stream,
    reflect_vertex_shader_needs_uv0_stream, validate_per_draw_group2,
};
use crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE;

/// Builds a forward mesh render pipeline from reflected WGSL (`@group(0..=2)`), with optional UV0 and color vertex streams.
///
/// Vertex inputs are `@location(0)` position, `@location(1)` normal/extra, and optionally
/// `@location(2)` UV0 and `@location(3)` color.
///
/// Used by [`crate::pipelines::raster::DebugWorldNormalsFamily`] and embedded WGSL raster materials.
pub(crate) fn create_reflective_raster_mesh_forward_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    desc: &MaterialPipelineDesc,
    wgsl_source: &str,
    label: &'static str,
    include_uv_vertex_buffer: bool,
    include_color_vertex_buffer: bool,
    use_alpha_blending: bool,
    depth_write_enabled: bool,
) -> wgpu::RenderPipeline {
    let reflected = reflect_raster_material_wgsl(wgsl_source).unwrap_or_else(|e| {
        panic!("reflect {label} (must match frame globals + per-draw contract): {e}");
    });
    validate_per_draw_group2(&reflected.per_draw_entries).unwrap_or_else(|e| {
        panic!("{label} per_draw group2: {e}");
    });

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
    let color_layout = wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32x4,
        }],
    };

    let use_uv = include_uv_vertex_buffer && reflect_vertex_shader_needs_uv0_stream(wgsl_source);
    let use_color =
        include_color_vertex_buffer && reflect_vertex_shader_needs_color_stream(wgsl_source);

    let vertex_buffers: &[wgpu::VertexBufferLayout<'_>] = if use_color {
        &[pos_layout, nrm_layout, uv_layout, color_layout]
    } else if use_uv {
        &[pos_layout, nrm_layout, uv_layout]
    } else {
        &[pos_layout, nrm_layout]
    };
    let blend = if use_alpha_blending {
        Some(wgpu::BlendState::ALPHA_BLENDING)
    } else {
        None
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: vertex_buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: desc.surface_format,
                blend,
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
                depth_write_enabled: Some(depth_write_enabled),
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
