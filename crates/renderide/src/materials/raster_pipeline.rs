//! Shared [`wgpu::RenderPipeline`] construction for reflective raster materials (frame, material, per-draw groups).
//!
//! **Transparent blending:** embedded materials that mirror Unity transparent queues (e.g. UI text) may need
//! alpha blending derived from material uniforms (`_SrcBlend` / `_DstBlend`) or host render-queue metadata.
//! Opaque paths use `blend: None` and [`wgpu::ColorWrites::COLOR`] so only RGB is written and the clear
//! alpha (`a=1`) is preserved for `Rgba16Float` render textures. Enabling any [`wgpu::BlendState`] on opaque
//! targets would require the format to be **blendable**; `Rgba16Float` is often not, which breaks pipeline
//! creation. Alpha-blended stems use [`wgpu::BlendState::ALPHA_BLENDING`] with [`wgpu::ColorWrites::ALL`].
//! Unity-style blend from uniforms is a cross-cutting follow-up—not per-shader logic in the mesh pass.

use crate::backend::{empty_material_bind_group_layout, FrameGpuResources};
use crate::materials::pipeline_build_error::PipelineBuildError;
use crate::materials::MaterialPipelineDesc;
use crate::materials::{
    reflect_raster_material_wgsl, reflect_vertex_shader_needs_color_stream,
    reflect_vertex_shader_needs_uv0_stream, validate_per_draw_group2,
};
use crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE;

/// Vertex stream toggles, blending, and depth write for [`create_reflective_raster_mesh_forward_pipeline`].
pub(crate) struct ReflectiveRasterMeshForwardPipelineDesc {
    /// When true, include UV0 vertex stream if the shader references it.
    pub include_uv_vertex_buffer: bool,
    /// When true, include per-vertex color if the shader references it.
    pub include_color_vertex_buffer: bool,
    /// Premultiplied alpha blending vs opaque RGB-only writes.
    pub use_alpha_blending: bool,
    /// Depth buffer write enable for the depth-stencil attachment.
    pub depth_write_enabled: bool,
}

/// Fixed vertex buffer layouts for embedded forward mesh draws (`@location` 0–3).
fn mesh_forward_base_vertex_buffer_layouts() -> (
    wgpu::VertexBufferLayout<'static>,
    wgpu::VertexBufferLayout<'static>,
    wgpu::VertexBufferLayout<'static>,
    wgpu::VertexBufferLayout<'static>,
) {
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
    (pos_layout, nrm_layout, uv_layout, color_layout)
}

/// Opaque: RGB-only; transparent: premultiplied-style alpha blend and full mask.
fn mesh_forward_blend_and_color_writes(
    use_alpha_blending: bool,
) -> (Option<wgpu::BlendState>, wgpu::ColorWrites) {
    if use_alpha_blending {
        (
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::ColorWrites::ALL,
        )
    } else {
        (None, wgpu::ColorWrites::COLOR)
    }
}

/// Builds a forward mesh render pipeline from reflected WGSL (`@group(0..=2)`), with optional UV0 and color vertex streams.
///
/// Vertex inputs are `@location(0)` position, `@location(1)` normal/extra, and optionally
/// `@location(2)` UV0 and `@location(3)` color.
///
/// Used by [`crate::pipelines::raster::DebugWorldNormalsFamily`] and embedded WGSL raster materials.
///
/// `raster` groups vertex stream, blend, and depth-write options mirrored at embedded-material call sites.
pub(crate) fn create_reflective_raster_mesh_forward_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    desc: &MaterialPipelineDesc,
    wgsl_source: &str,
    label: &'static str,
    raster: ReflectiveRasterMeshForwardPipelineDesc,
) -> Result<wgpu::RenderPipeline, PipelineBuildError> {
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

    let layouts = mesh_forward_base_vertex_buffer_layouts();
    let (pos_layout, nrm_layout, uv_layout, color_layout) = layouts;

    let use_uv =
        raster.include_uv_vertex_buffer && reflect_vertex_shader_needs_uv0_stream(wgsl_source);
    let use_color =
        raster.include_color_vertex_buffer && reflect_vertex_shader_needs_color_stream(wgsl_source);

    let vertex_buffers: &[wgpu::VertexBufferLayout<'_>] = if use_color {
        &[pos_layout, nrm_layout, uv_layout, color_layout]
    } else if use_uv {
        &[pos_layout, nrm_layout, uv_layout]
    } else {
        &[pos_layout, nrm_layout]
    };
    // Opaque: no blending + write RGB only so destination alpha stays at the clear value (a=1). Do not use
    // `blend: Some(...)` here: float RT formats may not be blendable and pipeline creation can fail.
    let (blend, color_writes) = mesh_forward_blend_and_color_writes(raster.use_alpha_blending);

    Ok(
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
                    write_mask: color_writes,
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
                    depth_write_enabled: Some(raster.depth_write_enabled),
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
        }),
    )
}
