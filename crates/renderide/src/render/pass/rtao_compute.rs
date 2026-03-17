//! RTAO (Ray-Traced Ambient Occlusion) compute pass.
//!
//! Clustered compute pass: dispatches workgroups of 8×8; each invocation processes one pixel.
//! Reads position/normal from G-buffer, traces rays in cosine-weighted hemisphere, writes AO.

use super::{RenderPass, RenderPassError};

const TILE_SIZE: u32 = 8;

const RTAO_SHADER_SRC: &str = r#"
enable wgpu_ray_query;

struct Uniforms {
    ao_radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var position_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var ao_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var acc_struct: acceleration_structure;

fn hash11(p: f32) -> f32 {
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash21(p: vec2f) -> f32 {
    return hash11(dot(p, vec2f(127.1, 311.7)));
}

fn cosine_hemisphere_sample(u1: f32, u2: f32, n: vec3f) -> vec3f {
    let r = sqrt(u1);
    let theta = 2.0 * 3.14159265 * u2;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - u1));
    let t = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.y) < 0.999);
    let b = normalize(cross(n, t));
    let t2 = cross(b, n);
    return normalize(t2 * x + b * y + n * z);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(position_tex);
    if global_id.x >= dims.x || global_id.y >= dims.y {
        return;
    }
    let uv = (vec2f(global_id.xy) + 0.5) / vec2f(dims);
    let pos = textureLoad(position_tex, vec2i(global_id.xy), 0);
    let n = textureLoad(normal_tex, vec2i(global_id.xy), 0);
    let world_pos = pos.xyz;
    let normal = normalize(n.xyz);
    let epsilon = 0.001;
    let origin = world_pos + normal * epsilon;
    var occluded = 0u;
    let pixel_seed = f32(global_id.y * dims.x + global_id.x);
    for (var i = 0u; i < 8u; i++) {
        let u1 = hash21(vec2f(pixel_seed, f32(i) * 0.6180339887));
        let u2 = hash21(vec2f(pixel_seed + 1.0, f32(i) * 0.6180339887));
        let dir = cosine_hemisphere_sample(u1, u2, normal);
        var rq: ray_query;
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.0, uniforms.ao_radius, origin, dir));
        rayQueryProceed(&rq);
        let hit = rayQueryGetCommittedIntersection(&rq);
        if hit.kind != RAY_QUERY_INTERSECTION_NONE {
            occluded += 1u;
        }
    }
    let visibility = 1.0 - f32(occluded) / 8.0;
    textureStore(ao_output, vec2i(global_id.xy), vec4f(visibility, 0.0, 0.0, 1.0));
}
"#;

/// RTAO compute pass: traces rays per pixel, writes visibility (1 - occlusion) to AO texture.
///
/// Dispatches (width/8, height/8, 1) workgroups. Each invocation reads position/normal,
/// traces 8 rays in cosine-weighted hemisphere, accumulates occlusion, writes Rgba8Unorm.
pub struct RtaoComputePass {
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl RtaoComputePass {
    /// Creates a new RTAO compute pass. Pipeline is built lazily when first used with ray tracing.
    pub fn new() -> Self {
        Self {
            pipeline: None,
            bind_group_layout: None,
        }
    }

    fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<(&wgpu::ComputePipeline, &wgpu::BindGroupLayout)> {
        if self.pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RTAO compute shader"),
                source: wgpu::ShaderSource::Wgsl(RTAO_SHADER_SRC.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RTAO compute bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(16),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RTAO compute pipeline layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
            self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RTAO compute pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            }));
            self.bind_group_layout = Some(bgl);
        }
        self.pipeline
            .as_ref()
            .zip(self.bind_group_layout.as_ref())
    }
}

impl RenderPass for RtaoComputePass {
    fn name(&self) -> &str {
        "rtao_compute"
    }

    fn execute(&mut self, ctx: &mut super::RenderPassContext) -> Result<(), RenderPassError> {
        let pos_view = match ctx.render_target.mrt_position_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let norm_view = match ctx.render_target.mrt_normal_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let ao_view = match ctx.render_target.mrt_ao_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let tlas = match &ctx.gpu.ray_tracing_state {
            Some(rt) => match &rt.tlas {
                Some(t) => t,
                None => return Ok(()),
            },
            None => return Ok(()),
        };

        let (pipeline, bgl) = match self.ensure_pipeline(&ctx.gpu.device) {
            Some((p, b)) => (p, b),
            None => return Ok(()),
        };

        let ao_radius = ctx.session.render_config().ao_radius;
        let uniform_data = [ao_radius, 0.0f32, 0.0f32, 0.0f32];
        let uniform_buffer = ctx.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RTAO uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.gpu
            .queue
            .write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform_data));

        let bind_group = ctx.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RTAO compute bind group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(pos_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(norm_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::AccelerationStructure(tlas),
                },
            ],
        });

        let (width, height) = ctx.viewport;
        let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RTAO compute pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            (width + TILE_SIZE - 1) / TILE_SIZE,
            (height + TILE_SIZE - 1) / TILE_SIZE,
            1,
        );

        Ok(())
    }
}

impl Default for RtaoComputePass {
    fn default() -> Self {
        Self::new()
    }
}
