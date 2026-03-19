//! RTAO (Ray-Traced Ambient Occlusion) compute pass.
//!
//! Clustered compute pass: dispatches workgroups of 8×8; each invocation processes one pixel.
//! Reads position/normal from G-buffer, traces rays in cosine-weighted hemisphere, writes AO.
//! When RTAO skips (e.g. TLAS None), clears AO texture to full visibility so composite
//! does not sample uninitialized data.
//!
//! The position G-buffer stores **camera-relative** positions; this pass adds
//! [`super::mesh_pass::mrt_gbuffer_world_origin`] back to reconstruct world-space ray origins.

use super::mesh_pass::mrt_gbuffer_world_origin;
use super::{PassResources, RenderPass, RenderPassError, ResourceSlot};

const TILE_SIZE: u32 = 8;

/// Compute shader that clears AO texture to full visibility (r=1) so composite sees no occlusion.
const AO_CLEAR_SHADER_SRC: &str = r#"
@group(0) @binding(0) var ao_output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(ao_output);
    if global_id.x >= dims.x || global_id.y >= dims.y {
        return;
    }
    textureStore(ao_output, vec2i(global_id.xy), vec4f(1.0, 0.0, 0.0, 1.0));
}
"#;

const RTAO_SHADER_SRC: &str = r#"
enable wgpu_ray_query;

struct Uniforms {
    ao_radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    gbuffer_origin: vec3f,
    _pad_origin: f32,
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
    let world_pos = pos.xyz + uniforms.gbuffer_origin;
    let normal = normalize(n.xyz);
    let bias = 0.01;
    let origin = world_pos + normal * bias;
    let t_min = 0.01;
    var occluded = 0u;
    let pixel_seed = f32(global_id.y * dims.x + global_id.x);
    let seed_frac = fract(pixel_seed * 0.0001) * 10000.0;
    for (var i = 0u; i < 16u; i++) {
        let u1 = hash21(vec2f(seed_frac, f32(i) * 0.6180339887));
        let u2 = hash21(vec2f(seed_frac + 1.0, f32(i) * 0.6180339887));
        let dir = cosine_hemisphere_sample(u1, u2, normal);
        var rq: ray_query;
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, t_min, uniforms.ao_radius, origin, dir));
        rayQueryProceed(&rq);
        let hit = rayQueryGetCommittedIntersection(&rq);
        if hit.kind != RAY_QUERY_INTERSECTION_NONE {
            occluded += 1u;
        }
    }
    let visibility = 1.0 - f32(occluded) / 16.0;
    textureStore(ao_output, vec2i(global_id.xy), vec4f(visibility, 0.0, 0.0, 1.0));
}
"#;

/// Host layout for [`RtaoComputePass`] WGSL `Uniforms` (32 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtaoComputeUniforms {
    ao_radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    gbuffer_origin: [f32; 3],
    _pad_origin: f32,
}

#[cfg(test)]
mod rtao_uniform_tests {
    use super::RtaoComputeUniforms;
    use std::mem::size_of;

    #[test]
    fn rtao_compute_uniforms_is_32_bytes() {
        assert_eq!(size_of::<RtaoComputeUniforms>(), 32);
        assert_eq!(size_of::<RtaoComputeUniforms>() % 16, 0);
    }
}

/// RTAO compute pass: traces rays per pixel, writes visibility (1 - occlusion) to AO texture.
///
/// Dispatches (width/8, height/8, 1) workgroups. Each invocation reads position/normal,
/// traces 16 rays in cosine-weighted hemisphere, accumulates occlusion, writes Rgba8Unorm.
/// When skipping (TLAS None, pipeline failure), clears AO to full visibility.
pub struct RtaoComputePass {
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    clear_pipeline: Option<wgpu::ComputePipeline>,
    clear_bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl RtaoComputePass {
    /// Creates a new RTAO compute pass. Pipeline is built lazily when first used with ray tracing.
    pub fn new() -> Self {
        Self {
            pipeline: None,
            bind_group_layout: None,
            clear_pipeline: None,
            clear_bind_group_layout: None,
        }
    }

    /// Clears AO texture to full visibility (r=1) when RTAO skips so composite does not sample garbage.
    fn clear_ao_to_visibility(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        ao_view: &wgpu::TextureView,
        viewport: (u32, u32),
    ) {
        let (clear_pipeline, clear_bgl) = match self.ensure_clear_pipeline(device) {
            Some(x) => x,
            None => return,
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RTAO AO clear bind group"),
            layout: clear_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(ao_view),
            }],
        });
        let (width, height) = viewport;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RTAO AO clear pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(clear_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(TILE_SIZE), height.div_ceil(TILE_SIZE), 1);
    }

    fn ensure_clear_pipeline(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<(&wgpu::ComputePipeline, &wgpu::BindGroupLayout)> {
        if self.clear_pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RTAO AO clear shader"),
                source: wgpu::ShaderSource::Wgsl(AO_CLEAR_SHADER_SRC.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RTAO AO clear bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RTAO AO clear pipeline layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
            self.clear_pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("RTAO AO clear pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.clear_bind_group_layout = Some(bgl);
        }
        self.clear_pipeline
            .as_ref()
            .zip(self.clear_bind_group_layout.as_ref())
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
                            min_binding_size: std::num::NonZeroU64::new(32),
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
            self.pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("RTAO compute pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.bind_group_layout = Some(bgl);
        }
        self.pipeline.as_ref().zip(self.bind_group_layout.as_ref())
    }
}

impl RenderPass for RtaoComputePass {
    fn name(&self) -> &str {
        "rtao_compute"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![ResourceSlot::Position, ResourceSlot::Normal],
            writes: vec![ResourceSlot::AoRaw],
        }
    }

    fn execute(&mut self, ctx: &mut super::RenderPassContext) -> Result<(), RenderPassError> {
        let pos_view = match ctx.render_target.mrt_position_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let norm_view = match ctx.render_target.mrt_normal_view {
            Some(v) => v,
            None => {
                logger::trace!("RTAO compute skipped: mrt_normal_view is None");
                return Ok(());
            }
        };
        let ao_raw_view = match ctx.render_target.mrt_ao_raw_view {
            Some(v) => v,
            None => {
                logger::trace!("RTAO compute skipped: mrt_ao_raw_view is None");
                return Ok(());
            }
        };
        let tlas = match &ctx.gpu.ray_tracing_state {
            Some(rt) => match &rt.tlas {
                Some(t) => t,
                None => {
                    logger::trace!(
                        "RTAO compute skipped: TLAS is None (no non-overlay non-skinned geometry with BLAS)"
                    );
                    self.clear_ao_to_visibility(
                        &ctx.gpu.device,
                        ctx.encoder,
                        ao_raw_view,
                        ctx.viewport,
                    );
                    return Ok(());
                }
            },
            None => {
                logger::trace!("RTAO compute skipped: ray_tracing_state is None");
                self.clear_ao_to_visibility(
                    &ctx.gpu.device,
                    ctx.encoder,
                    ao_raw_view,
                    ctx.viewport,
                );
                return Ok(());
            }
        };

        let (pipeline, bgl) = match self.ensure_pipeline(&ctx.gpu.device) {
            Some((p, b)) => (p, b),
            None => {
                logger::trace!("RTAO compute skipped: pipeline creation failed (shader compile?)");
                self.clear_ao_to_visibility(
                    &ctx.gpu.device,
                    ctx.encoder,
                    ao_raw_view,
                    ctx.viewport,
                );
                return Ok(());
            }
        };

        const RTAO_UNIFORM_SIZE: u64 = 32;
        let device = &ctx.gpu.device;
        let rtao_uniform_buffer = match ctx.gpu.rtao_uniform_buffer.take() {
            Some(b) if b.size() >= RTAO_UNIFORM_SIZE => b,
            _ => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RTAO uniforms"),
                size: RTAO_UNIFORM_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        let ao_radius = ctx.session.render_config().ao_radius;
        let origin = mrt_gbuffer_world_origin(ctx.draw_batches, ctx.session);
        let uniform_data = RtaoComputeUniforms {
            ao_radius,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            gbuffer_origin: origin,
            _pad_origin: 0.0,
        };
        ctx.gpu
            .queue
            .write_buffer(&rtao_uniform_buffer, 0, bytemuck::bytes_of(&uniform_data));

        let bind_group = ctx
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RTAO compute bind group"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: rtao_uniform_buffer.as_entire_binding(),
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
                        resource: wgpu::BindingResource::TextureView(ao_raw_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::AccelerationStructure(tlas),
                    },
                ],
            });

        ctx.gpu.rtao_uniform_buffer = Some(rtao_uniform_buffer);

        let (width, height) = ctx.viewport;
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RTAO compute pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(TILE_SIZE), height.div_ceil(TILE_SIZE), 1);

        Ok(())
    }
}

impl Default for RtaoComputePass {
    fn default() -> Self {
        Self::new()
    }
}
