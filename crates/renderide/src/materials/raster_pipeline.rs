//! Shared [`wgpu::RenderPipeline`] construction for reflective raster materials (frame, material, per-draw groups).
//!
//! Opaque paths use no blend state and write RGB only so destination alpha stays at the clear value
//! for float render textures. Pass descriptors from `//#pass` directives can override blend, depth,
//! cull, stencil/color-write, and depth state per material.

use std::num::NonZeroU32;

use crate::gpu::{
    empty_material_bind_group_layout, frame_bind_group_layout, frame_bind_group_layout_entries,
};
use crate::materials::material_passes::{DefaultPassParams, MaterialPassDesc, default_pass};
use crate::materials::pipeline_build_error::PipelineBuildError;
use crate::materials::wgsl_reflect::reflect_raster_material_wgsl_with_vertex_entries;
use crate::materials::{MaterialRenderState, RasterFrontFace, RasterPrimitiveTopology};
use crate::materials::{
    ReflectedRasterLayout, ReflectedVertexInputFormat, validate_layout_against_limits,
    validate_per_draw_group2, validate_vertex_layout_against_limits,
};

/// Swapchain-relevant state needed to build a [`wgpu::RenderPipeline`].
#[derive(Clone, Copy, Debug)]
pub struct MaterialPipelineDesc {
    /// Primary color attachment format (for example swapchain format).
    pub surface_format: wgpu::TextureFormat,
    /// Optional depth attachment (meshes / MRT later).
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    /// MSAA sample count (1 = off).
    pub sample_count: u32,
    /// When set, must match the render pass and pipeline (e.g. `0b11` for two multiview layers).
    pub multiview_mask: Option<NonZeroU32>,
}

/// Compiled shader module and [`MaterialPipelineDesc`] from the material cache before adding a pipeline label.
pub(crate) struct ShaderModuleBuildRefs<'a> {
    /// GPU device used to create pipelines.
    pub device: &'a wgpu::Device,
    /// Effective device caps used to validate reflected layouts before pipeline creation.
    pub limits: &'a crate::gpu::GpuLimits,
    /// Compiled WGSL module.
    pub module: &'a wgpu::ShaderModule,
    /// Surface and attachment formats for the material.
    pub desc: &'a MaterialPipelineDesc,
    /// Full WGSL source for reflection.
    pub wgsl_source: &'a str,
}

impl<'a> ShaderModuleBuildRefs<'a> {
    /// Fills in the raster pipeline label used for layout and pipeline naming.
    pub(crate) fn with_label(self, label: impl Into<String>) -> ReflectiveRasterShaderContext<'a> {
        ReflectiveRasterShaderContext {
            device: self.device,
            limits: self.limits,
            module: self.module,
            desc: self.desc,
            wgsl_source: self.wgsl_source,
            label: label.into(),
        }
    }
}

/// WGSL module and pipeline layout inputs shared by every pass when building multi-pass raster pipelines.
pub(crate) struct ReflectiveRasterShaderContext<'a> {
    /// GPU device used to create pipelines.
    pub device: &'a wgpu::Device,
    /// Effective device caps used to validate reflected layouts before pipeline creation.
    pub limits: &'a crate::gpu::GpuLimits,
    /// Compiled WGSL module.
    pub module: &'a wgpu::ShaderModule,
    /// Surface and attachment formats for the material.
    pub desc: &'a MaterialPipelineDesc,
    /// Full WGSL source for reflection (vertex stream layout).
    pub wgsl_source: &'a str,
    /// Label prefix for pipeline layout and pipelines.
    pub label: String,
}

/// UV / color vertex stream inclusion for [`pipeline_layout_and_vertex_streams`] and multi-pass builds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VertexStreamToggles {
    /// Request UV0 stream when the shader references it.
    pub include_uv_vertex_buffer: bool,
    /// Request vertex color stream when the shader references it.
    pub include_color_vertex_buffer: bool,
    /// Request UV1 stream when the shader references it.
    pub include_uv1_vertex_buffer: bool,
}

/// Reflected bind-group layout and vertex buffer layouts reused for each [`MaterialPassDesc`] in a batch.
pub(crate) struct MeshForwardSharedPipelineBuild<'a> {
    /// GPU device used to create the render pipeline.
    pub device: &'a wgpu::Device,
    /// Compiled WGSL module.
    pub module: &'a wgpu::ShaderModule,
    /// Surface and attachment formats for the material.
    pub desc: &'a MaterialPipelineDesc,
    /// Label prefix for pipeline naming (`{label}__{pass}`).
    pub label: &'a str,
    /// Shared pipeline layout from reflection.
    pub layout: &'a wgpu::PipelineLayout,
    /// Vertex buffer layouts selected for this shader.
    pub vertex_buffers: &'a [wgpu::VertexBufferLayout<'a>],
    /// Front-face winding for this pipeline variant.
    pub front_face: RasterFrontFace,
    /// Primitive topology baked into [`wgpu::PrimitiveState::topology`] for this variant.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// Vertex stream toggles, blending, depth write, and material overrides for
/// [`create_reflective_raster_mesh_forward_pipeline`].
pub(crate) struct ReflectiveRasterMeshForwardPipelineDesc {
    /// Include UV0 vertex stream when the shader references it.
    pub include_uv_vertex_buffer: bool,
    /// Include vertex color stream when the shader references it.
    pub include_color_vertex_buffer: bool,
    /// Include UV1 vertex stream when the shader references it.
    pub include_uv1_vertex_buffer: bool,
    /// Alpha blending vs opaque RGB-only writes for the default single pass.
    pub use_alpha_blending: bool,
    /// Depth write flag for the default single pass.
    pub depth_write_enabled: bool,
    /// Runtime material overrides for color mask, stencil, and depth state.
    pub render_state: MaterialRenderState,
    /// Front-face winding selected from the draw's model transform.
    pub front_face: RasterFrontFace,
    /// Primitive topology selected from the mesh's per-submesh topology.
    pub primitive_topology: RasterPrimitiveTopology,
}

mod vertex_layouts;

use vertex_layouts::{
    mesh_forward_vertex_buffer_layout, mesh_forward_wide_uv_vertex_buffer_layout,
};

const UV_SHADER_LOCATIONS: [u32; 8] = [2, 5, 6, 7, 8, 9, 10, 11];
const WIDE_UV_ROW_BYTES: u64 = 16;

#[derive(Default)]
struct ReflectedMeshForwardVertexStreams {
    uv_formats: [Option<ReflectedVertexInputFormat>; 8],
    color: bool,
    tangent: bool,
}

impl ReflectedMeshForwardVertexStreams {
    fn uv(&self, channel: usize) -> bool {
        self.uv_formats[channel].is_some()
    }

    fn needs_wide_uvs(&self) -> bool {
        self.uv_formats.iter().enumerate().any(|(channel, format)| {
            format.is_some_and(|format| {
                channel >= 4 || format != ReflectedVertexInputFormat::Float32x2
            })
        })
    }
}

fn reflected_mesh_forward_vertex_streams(
    reflected: &ReflectedRasterLayout,
) -> ReflectedMeshForwardVertexStreams {
    let mut streams = ReflectedMeshForwardVertexStreams::default();
    for input in &reflected.vs_vertex_inputs {
        if let Some(channel) = uv_channel_from_location(input.location) {
            if reflected_uv_format_supported(input.format) {
                streams.uv_formats[channel] = Some(input.format);
            }
            continue;
        }
        match (input.location, input.format) {
            (3, ReflectedVertexInputFormat::Float32x4) => streams.color = true,
            (4, ReflectedVertexInputFormat::Float32x4) => streams.tangent = true,
            _ => {}
        }
    }
    streams
}

fn reflected_uv_format_supported(format: ReflectedVertexInputFormat) -> bool {
    matches!(
        format,
        ReflectedVertexInputFormat::Float32x2
            | ReflectedVertexInputFormat::Float32x3
            | ReflectedVertexInputFormat::Float32x4
    )
}

fn uv_channel_from_location(location: u32) -> Option<usize> {
    UV_SHADER_LOCATIONS
        .iter()
        .position(|candidate| *candidate == location)
}

fn reflected_vertex_format_to_wgpu(
    format: ReflectedVertexInputFormat,
) -> Option<wgpu::VertexFormat> {
    match format {
        ReflectedVertexInputFormat::Float32x2 => Some(wgpu::VertexFormat::Float32x2),
        ReflectedVertexInputFormat::Float32x3 => Some(wgpu::VertexFormat::Float32x3),
        ReflectedVertexInputFormat::Float32x4 => Some(wgpu::VertexFormat::Float32x4),
        ReflectedVertexInputFormat::Unsupported => None,
    }
}

fn include_uv_channel(
    channel: usize,
    include_uv_vertex_buffer: bool,
    include_uv1_vertex_buffer: bool,
) -> bool {
    match channel {
        0 => include_uv_vertex_buffer,
        1 => include_uv1_vertex_buffer,
        _ => true,
    }
}

fn wide_uv_attributes_for_streams(
    streams: &ReflectedMeshForwardVertexStreams,
    include_uv_vertex_buffer: bool,
    include_uv1_vertex_buffer: bool,
) -> Vec<wgpu::VertexAttribute> {
    if !streams.needs_wide_uvs() {
        return Vec::new();
    }
    let mut attributes = Vec::new();
    for (channel, format) in streams.uv_formats.iter().copied().enumerate() {
        if !include_uv_channel(channel, include_uv_vertex_buffer, include_uv1_vertex_buffer) {
            continue;
        }
        let Some(format) = format.and_then(reflected_vertex_format_to_wgpu) else {
            continue;
        };
        attributes.push(wgpu::VertexAttribute {
            offset: channel as u64 * WIDE_UV_ROW_BYTES,
            shader_location: UV_SHADER_LOCATIONS[channel],
            format,
        });
    }
    attributes
}

fn mesh_forward_vertex_buffers_for_streams<'a>(
    streams: &ReflectedMeshForwardVertexStreams,
    wide_uv_attributes: &'a [wgpu::VertexAttribute],
    include_uv_vertex_buffer: bool,
    include_color_vertex_buffer: bool,
    include_uv1_vertex_buffer: bool,
) -> Vec<wgpu::VertexBufferLayout<'a>> {
    let mut vertex_buffers = vec![
        mesh_forward_vertex_buffer_layout(0),
        mesh_forward_vertex_buffer_layout(1),
    ];
    let uses_wide_uvs = !wide_uv_attributes.is_empty();
    if uses_wide_uvs {
        vertex_buffers.push(mesh_forward_wide_uv_vertex_buffer_layout(
            wide_uv_attributes,
        ));
    } else if include_uv_vertex_buffer && streams.uv(0) {
        vertex_buffers.push(mesh_forward_vertex_buffer_layout(2));
    }
    if include_color_vertex_buffer && streams.color {
        vertex_buffers.push(mesh_forward_vertex_buffer_layout(3));
    }
    if streams.tangent {
        vertex_buffers.push(mesh_forward_vertex_buffer_layout(4));
    }
    if !uses_wide_uvs {
        if include_uv1_vertex_buffer && streams.uv(1) {
            vertex_buffers.push(mesh_forward_vertex_buffer_layout(5));
        }
        if streams.uv(2) {
            vertex_buffers.push(mesh_forward_vertex_buffer_layout(6));
        }
        if streams.uv(3) {
            vertex_buffers.push(mesh_forward_vertex_buffer_layout(7));
        }
    }
    vertex_buffers
}

fn pipeline_layout_and_vertex_streams(
    device: &wgpu::Device,
    limits: &crate::gpu::GpuLimits,
    wgsl_source: &str,
    label: &str,
    vertex_entries: &[&str],
) -> Result<(wgpu::PipelineLayout, ReflectedMeshForwardVertexStreams), PipelineBuildError> {
    let reflected = reflect_raster_material_wgsl_with_vertex_entries(wgsl_source, vertex_entries)?;
    validate_per_draw_group2(&reflected.per_draw_entries)?;
    let frame_entries = frame_bind_group_layout_entries();
    validate_layout_against_limits(&reflected, &frame_entries, limits)?;

    let frame_bgl = frame_bind_group_layout(device);
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

    Ok((layout, reflected_mesh_forward_vertex_streams(&reflected)))
}

/// Builds one pipeline for a single [`MaterialPassDesc`] sharing the reflected layout and vertex buffers.
pub(crate) fn build_pipeline_from_pass(
    shared: &MeshForwardSharedPipelineBuild<'_>,
    pass: &MaterialPassDesc,
    render_state: MaterialRenderState,
) -> wgpu::RenderPipeline {
    profiling::scope!("materials::build_pipeline_from_pass");
    let pass_label = format!(
        "{}__{}__vs_{}__fs_{}",
        shared.label, pass.name, pass.vertex_entry, pass.fragment_entry
    );
    {
        profiling::scope!("materials::create_render_pipeline_wgpu");
        let pipeline = shared
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&pass_label),
                layout: Some(shared.layout),
                vertex: wgpu::VertexState {
                    module: shared.module,
                    entry_point: Some(pass.vertex_entry),
                    compilation_options: Default::default(),
                    buffers: shared.vertex_buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: shared.module,
                    entry_point: Some(pass.fragment_entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: shared.desc.surface_format,
                        blend: pass.blend,
                        write_mask: pass.resolved_color_writes(render_state),
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: shared.primitive_topology.to_wgpu(),
                    front_face: shared.front_face.to_wgpu(),
                    cull_mode: pass.resolved_cull_mode(render_state),
                    ..Default::default()
                },
                depth_stencil: shared.desc.depth_stencil_format.map(|format| {
                    wgpu::DepthStencilState {
                        format,
                        depth_write_enabled: Some(pass.resolved_depth_write(render_state)),
                        depth_compare: Some(pass.resolved_depth_compare(render_state)),
                        stencil: if format.has_stencil_aspect() {
                            pass.resolved_stencil_state(render_state)
                        } else {
                            wgpu::StencilState::default()
                        },
                        bias: pass.resolved_depth_bias(render_state),
                    }
                }),
                multisample: wgpu::MultisampleState {
                    count: shared.desc.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: pass.alpha_to_coverage
                        && shared.desc.sample_count > 1,
                },
                multiview_mask: shared.desc.multiview_mask,
                cache: None,
            });
        crate::profiling::note_resource_churn!(RenderPipeline, "materials::raster_pipeline");
        pipeline
    }
}

/// Builds a default single-pass forward mesh pipeline from reflected WGSL (`@group(0..=2)`).
pub(crate) fn create_reflective_raster_mesh_forward_pipeline(
    device: &wgpu::Device,
    limits: &crate::gpu::GpuLimits,
    module: &wgpu::ShaderModule,
    desc: &MaterialPipelineDesc,
    wgsl_source: &str,
    label: &'static str,
    raster: ReflectiveRasterMeshForwardPipelineDesc,
) -> Result<wgpu::RenderPipeline, PipelineBuildError> {
    let pass = default_pass(DefaultPassParams {
        use_alpha_blending: raster.use_alpha_blending,
        depth_write: raster.depth_write_enabled,
    });
    let vertex_entries = [pass.vertex_entry];
    let (layout, vertex_streams) =
        pipeline_layout_and_vertex_streams(device, limits, wgsl_source, label, &vertex_entries)?;
    let wide_uv_attributes = wide_uv_attributes_for_streams(
        &vertex_streams,
        raster.include_uv_vertex_buffer,
        raster.include_uv1_vertex_buffer,
    );
    let vertex_buffers = mesh_forward_vertex_buffers_for_streams(
        &vertex_streams,
        &wide_uv_attributes,
        raster.include_uv_vertex_buffer,
        raster.include_color_vertex_buffer,
        raster.include_uv1_vertex_buffer,
    );
    validate_vertex_layout_against_limits(&vertex_buffers, limits)?;
    let shared = MeshForwardSharedPipelineBuild {
        device,
        module,
        desc,
        label,
        layout: &layout,
        vertex_buffers: &vertex_buffers,
        front_face: raster.front_face,
        primitive_topology: raster.primitive_topology,
    };
    Ok(build_pipeline_from_pass(
        &shared,
        &pass,
        raster.render_state,
    ))
}

/// Builds N pipelines (one per pass descriptor) that share reflected bind-group layout and vertex streams.
pub(crate) fn create_reflective_raster_mesh_forward_pipelines(
    shader: ReflectiveRasterShaderContext<'_>,
    streams: VertexStreamToggles,
    passes: &[MaterialPassDesc],
    render_state: MaterialRenderState,
    front_face: RasterFrontFace,
    primitive_topology: RasterPrimitiveTopology,
) -> Result<Vec<wgpu::RenderPipeline>, PipelineBuildError> {
    if passes.is_empty() {
        return Err(PipelineBuildError::EmptyPasses {
            label: shader.label,
        });
    }
    let vertex_entries = passes
        .iter()
        .map(|pass| pass.vertex_entry)
        .collect::<Vec<_>>();
    let (layout, vertex_streams) = pipeline_layout_and_vertex_streams(
        shader.device,
        shader.limits,
        shader.wgsl_source,
        shader.label.as_str(),
        &vertex_entries,
    )?;
    let wide_uv_attributes = wide_uv_attributes_for_streams(
        &vertex_streams,
        streams.include_uv_vertex_buffer,
        streams.include_uv1_vertex_buffer,
    );
    let vertex_buffers = mesh_forward_vertex_buffers_for_streams(
        &vertex_streams,
        &wide_uv_attributes,
        streams.include_uv_vertex_buffer,
        streams.include_color_vertex_buffer,
        streams.include_uv1_vertex_buffer,
    );
    validate_vertex_layout_against_limits(&vertex_buffers, shader.limits)?;

    let shared = MeshForwardSharedPipelineBuild {
        device: shader.device,
        module: shader.module,
        desc: shader.desc,
        label: shader.label.as_str(),
        layout: &layout,
        vertex_buffers: &vertex_buffers,
        front_face,
        primitive_topology,
    };
    Ok(passes
        .iter()
        .map(|pass| build_pipeline_from_pass(&shared, pass, render_state))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader_locations(layouts: &[wgpu::VertexBufferLayout<'_>]) -> Vec<u32> {
        layouts
            .iter()
            .flat_map(|layout| layout.attributes.iter().map(|attr| attr.shader_location))
            .collect()
    }

    #[test]
    fn compact_uv_layouts_stay_separate_for_vec2_uv0_to_uv3() {
        let mut streams = ReflectedMeshForwardVertexStreams::default();
        streams.uv_formats[0] = Some(ReflectedVertexInputFormat::Float32x2);
        streams.uv_formats[1] = Some(ReflectedVertexInputFormat::Float32x2);
        streams.uv_formats[2] = Some(ReflectedVertexInputFormat::Float32x2);
        streams.uv_formats[3] = Some(ReflectedVertexInputFormat::Float32x2);

        let wide_uv_attributes = wide_uv_attributes_for_streams(&streams, true, true);
        let layouts = mesh_forward_vertex_buffers_for_streams(
            &streams,
            &wide_uv_attributes,
            true,
            false,
            true,
        );

        assert!(wide_uv_attributes.is_empty());
        assert_eq!(shader_locations(&layouts), vec![0, 1, 2, 5, 6, 7]);
    }

    #[test]
    fn wide_uv_layout_replaces_compact_uv_slots() {
        let mut streams = ReflectedMeshForwardVertexStreams::default();
        streams.uv_formats[0] = Some(ReflectedVertexInputFormat::Float32x4);
        streams.uv_formats[4] = Some(ReflectedVertexInputFormat::Float32x2);
        streams.uv_formats[7] = Some(ReflectedVertexInputFormat::Float32x3);
        streams.color = true;
        streams.tangent = true;

        let wide_uv_attributes = wide_uv_attributes_for_streams(&streams, true, true);
        let layouts = mesh_forward_vertex_buffers_for_streams(
            &streams,
            &wide_uv_attributes,
            true,
            true,
            true,
        );

        assert_eq!(wide_uv_attributes.len(), 3);
        assert_eq!(wide_uv_attributes[0].shader_location, 2);
        assert_eq!(wide_uv_attributes[0].offset, 0);
        assert_eq!(wide_uv_attributes[0].format, wgpu::VertexFormat::Float32x4);
        assert_eq!(wide_uv_attributes[1].shader_location, 8);
        assert_eq!(wide_uv_attributes[1].offset, 64);
        assert_eq!(wide_uv_attributes[1].format, wgpu::VertexFormat::Float32x2);
        assert_eq!(wide_uv_attributes[2].shader_location, 11);
        assert_eq!(wide_uv_attributes[2].offset, 112);
        assert_eq!(wide_uv_attributes[2].format, wgpu::VertexFormat::Float32x3);

        assert_eq!(shader_locations(&layouts), vec![0, 1, 2, 8, 11, 3, 4]);
        assert_eq!(layouts[2].array_stride, 128);
    }
}
