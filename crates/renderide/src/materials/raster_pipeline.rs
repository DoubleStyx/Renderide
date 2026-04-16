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
use crate::materials::material_passes::{default_pass, MaterialPassDesc, MaterialRenderState};
use crate::materials::MaterialPipelineDesc;
use crate::materials::{
    reflect_raster_material_wgsl, reflect_vertex_shader_needs_color_stream,
    reflect_vertex_shader_needs_uv0_stream, validate_per_draw_group2,
};

/// Builds a forward mesh render pipeline from reflected WGSL (`@group(0..=2)`), with optional UV0,
/// color, and extended UI vertex streams.
///
/// Vertex inputs are `@location(0)` position, `@location(1)` normal/extra, and optionally
/// `@location(2)` UV0, `@location(3)` color, `@location(4)` tangent/color, and
/// `@location(5..=7)` UV1/UV2/UV3.
///
/// Used by [`crate::pipelines::raster::DebugWorldNormalsFamily`] and embedded WGSL raster materials.
///
/// Argument list mirrors discrete wgpu pipeline options (vertex streams, blending, depth) at call sites.
#[allow(clippy::too_many_arguments)]
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
    render_state: MaterialRenderState,
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
    let tangent_layout = wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 4,
            format: wgpu::VertexFormat::Float32x4,
        }],
    };
    let uv1_layout = wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32x2,
        }],
    };
    let uv2_layout = wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 6,
            format: wgpu::VertexFormat::Float32x2,
        }],
    };
    let uv3_layout = wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 7,
            format: wgpu::VertexFormat::Float32x2,
        }],
    };

    let use_uv = include_uv_vertex_buffer && reflect_vertex_shader_needs_uv0_stream(wgsl_source);
    let use_color =
        include_color_vertex_buffer && reflect_vertex_shader_needs_color_stream(wgsl_source);
    let use_extended = reflected.vs_max_vertex_location.is_some_and(|m| m >= 4);

    let vertex_buffers: &[wgpu::VertexBufferLayout<'_>] = if use_extended {
        &[
            pos_layout,
            nrm_layout,
            uv_layout,
            color_layout,
            tangent_layout,
            uv1_layout,
            uv2_layout,
            uv3_layout,
        ]
    } else if use_color {
        &[pos_layout, nrm_layout, uv_layout, color_layout]
    } else if use_uv {
        &[pos_layout, nrm_layout, uv_layout]
    } else {
        &[pos_layout, nrm_layout]
    };
    // Opaque: no blending + write RGB only so destination alpha stays at the clear value (a=1). Do not use
    // `blend: Some(...)` on opaque passes: float RT formats may not be blendable and pipeline creation can fail.
    let pass = default_pass(use_alpha_blending, depth_write_enabled);
    build_pipeline_from_pass(
        device,
        module,
        desc,
        label,
        &layout,
        vertex_buffers,
        &pass,
        render_state,
    )
}

/// Builds one pipeline for a single [`MaterialPassDesc`] sharing the layout and vertex buffers.
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
            cull_mode: render_state.cull_mode(pass.cull_mode),
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
                bias: wgpu::DepthBiasState {
                    constant: pass.depth_bias_constant,
                    slope_scale: pass.depth_bias_slope_scale,
                    clamp: 0.0,
                },
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

/// Builds N pipelines (one per pass descriptor) that share the reflected bind-group layout and vertex streams.
///
/// Used by the embedded material path when [`crate::embedded_shaders::embedded_target_passes`] reports
/// one or more `//#pass` directives. Single-pass materials still go through
/// [`create_reflective_raster_mesh_forward_pipeline`] directly.
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
) -> Vec<wgpu::RenderPipeline> {
    assert!(
        !passes.is_empty(),
        "create_reflective_raster_mesh_forward_pipelines called with empty passes (stem {label})"
    );
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
    let tangent_layout = wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 4,
            format: wgpu::VertexFormat::Float32x4,
        }],
    };
    let uv1_layout = wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32x2,
        }],
    };
    let uv2_layout = wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 6,
            format: wgpu::VertexFormat::Float32x2,
        }],
    };
    let uv3_layout = wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 7,
            format: wgpu::VertexFormat::Float32x2,
        }],
    };

    let use_uv = include_uv_vertex_buffer && reflect_vertex_shader_needs_uv0_stream(wgsl_source);
    let use_color =
        include_color_vertex_buffer && reflect_vertex_shader_needs_color_stream(wgsl_source);
    let use_extended = reflected.vs_max_vertex_location.is_some_and(|m| m >= 4);

    let vertex_buffers: &[wgpu::VertexBufferLayout<'_>] = if use_extended {
        &[
            pos_layout,
            nrm_layout,
            uv_layout,
            color_layout,
            tangent_layout,
            uv1_layout,
            uv2_layout,
            uv3_layout,
        ]
    } else if use_color {
        &[pos_layout, nrm_layout, uv_layout, color_layout]
    } else if use_uv {
        &[pos_layout, nrm_layout, uv_layout]
    } else {
        &[pos_layout, nrm_layout]
    };

    passes
        .iter()
        .map(|p| {
            build_pipeline_from_pass(
                device,
                module,
                desc,
                label,
                &layout,
                vertex_buffers,
                p,
                render_state,
            )
        })
        .collect()
}
