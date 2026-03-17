//! Composite pass: applies RTAO to mesh color and outputs to surface.
//!
//! Fullscreen pass that reads mesh color and AO texture, outputs
//! `color * (1 - ao_strength * (1 - ao))` to darken based on occlusion.

use super::{RenderPass, RenderPassContext, RenderPassError};

const COMPOSITE_SHADER_SRC: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.clip_position = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2f(x, y);
    return out;
}

struct Uniforms {
    ao_strength: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var ao_tex: texture_2d<f32>;
@group(0) @binding(3) var color_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let color = textureSample(color_tex, color_sampler, in.uv);
    let ao = textureSample(ao_tex, color_sampler, in.uv).r;
    let occlusion = 1.0 - ao;
    let factor = 1.0 - uniforms.ao_strength * occlusion;
    return vec4f(color.rgb * factor, color.a);
}
"#;

/// Composite pass: applies RTAO to mesh color, outputs to surface.
///
/// Fullscreen triangle. Reads mesh color and AO (visibility); darkens by
/// `color * (1 - ao_strength * (1 - ao))` where ao is visibility (1 = no occlusion).
pub struct CompositePass {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    sampler: Option<wgpu::Sampler>,
}

impl CompositePass {
    /// Creates a new composite pass. Pipeline is built lazily when first used.
    pub fn new() -> Self {
        Self {
            pipeline: None,
            bind_group_layout: None,
            sampler: None,
        }
    }

    fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Option<(
        &wgpu::RenderPipeline,
        &wgpu::BindGroupLayout,
        &wgpu::Sampler,
    )> {
        if self.pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("composite shader"),
                source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER_SRC.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("composite bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(16),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("composite pipeline layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
            self.pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("composite pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            }));
            self.bind_group_layout = Some(bgl);
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("composite sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });
            self.sampler = Some(sampler);
        }
        match (
            self.pipeline.as_ref(),
            self.bind_group_layout.as_ref(),
            self.sampler.as_ref(),
        ) {
            (Some(p), Some(b), Some(s)) => Some((p, b, s)),
            _ => None,
        }
    }
}

impl RenderPass for CompositePass {
    fn name(&self) -> &str {
        "composite"
    }

    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        let color_input = match ctx.render_target.mrt_color_input_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let ao_view = match ctx.render_target.mrt_ao_view {
            Some(v) => v,
            None => return Ok(()),
        };

        let (pipeline, bgl, sampler) = match self.ensure_pipeline(&ctx.gpu.device, ctx.gpu.config.format) {
            Some(x) => x,
            None => return Ok(()),
        };

        let ao_strength = ctx.session.render_config().rtao_strength;
        let uniform_data = [ao_strength, 0.0f32, 0.0f32, 0.0f32];
        let uniform_buffer = ctx.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.gpu
            .queue
            .write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform_data));
        let bind_group = ctx.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bind group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(color_input),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite pass"),
            timestamp_writes: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.render_target.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);

        Ok(())
    }
}

impl Default for CompositePass {
    fn default() -> Self {
        Self::new()
    }
}
