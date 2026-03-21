//! Ray-traced shadow atlas compute pass.
//!
//! When [`crate::config::RenderConfig::ray_traced_shadows_use_compute`] is enabled with RTAO MRT,
//! this pass runs **after** the mesh pass and fills a half-resolution `R16Float` texture array
//! (32 layers: cluster slot index within each tile). PBR ray-query shaders sample the atlas when
//! [`crate::gpu::pipeline::RtShadowUniforms::shadow_mode`] is [`RT_SHADOW_MODE_ATLAS`](crate::gpu::pipeline::RT_SHADOW_MODE_ATLAS);
//! that sampling uses the atlas produced **after** the previous frame’s mesh pass (one-frame latency).

use super::mesh_pass::{
    mrt_gbuffer_world_origin, pbr_primary_view_batch, pbr_view_space_z_coeffs_for_batch,
};
use super::{PassResources, RenderPass, RenderPassContext, RenderPassError, ResourceSlot};
use crate::gpu::pipeline::SceneUniforms;

const TILE: u32 = 8;

/// Clears every atlas texel to `1.0` (full light visibility).
const SHADOW_ATLAS_CLEAR_SHADER: &str = r#"
@group(0) @binding(0) var shadow_atlas_out: texture_storage_2d_array<r16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let layer = global_id.z;
    if layer >= 32u {
        return;
    }
    let dims = textureDimensions(shadow_atlas_out);
    if global_id.x >= dims.x || global_id.y >= dims.y {
        return;
    }
    textureStore(shadow_atlas_out, vec2i(global_id.xy), i32(layer), vec4f(1.0, 0.0, 0.0, 1.0));
}
"#;

/// Fills the shadow atlas using the same clustered visibility rules as [`pbr_ray_query_shadow_lib.wgsl`], without the atlas branch.
const SHADOW_ATLAS_TRACE_SHADER: &str = r#"
enable wgpu_ray_query;

struct GpuLight {
    position: vec3f,
    _pad0: f32,
    direction: vec3f,
    _pad1: f32,
    color: vec3f,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    _pad_before_shadow_params: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    _pad_trailing: array<u32, 3>,
}
struct SceneUniforms {
    view_position: vec3f,
    _pad0: f32,
    view_space_z_coeffs: vec4f,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}
struct RtShadowComputeExtra {
    gbuffer_origin: vec3f,
    _pad0: f32,
    soft_shadow_sample_count: u32,
    soft_cone_scale: f32,
    frame_counter: u32,
    /// 1 when the atlas matches the G-buffer size; 2 when the atlas is half-resolution.
    atlas_gbuffer_stride: u32,
}

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(0) @binding(3) var<storage, read> cluster_light_indices: array<u32>;
@group(0) @binding(4) var acc_struct: acceleration_structure;
@group(0) @binding(5) var<uniform> rt_extra: RtShadowComputeExtra;
@group(0) @binding(6) var position_tex: texture_2d<f32>;
@group(0) @binding(7) var normal_tex: texture_2d<f32>;
@group(0) @binding(8) var shadow_atlas_out: texture_storage_2d_array<r16float, write>;

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn cluster_xy_from_frag(frag_xy: vec2f, viewport_w: u32, viewport_h: u32) -> vec2u {
    let max_x = max(f32(viewport_w) - 0.5, 0.5);
    let max_y = max(f32(viewport_h) - 0.5, 0.5);
    let pxy = clamp(frag_xy, vec2f(0.5, 0.5), vec2f(max_x, max_y));
    let tile_f = (pxy - vec2f(0.5, 0.5)) / vec2f(f32(TILE_SIZE));
    return vec2u(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}

fn hash11(p: f32) -> f32 {
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn hash21(p: vec2f) -> f32 {
    return hash11(dot(p, vec2f(127.1, 311.7)));
}

fn trace_shadow_ray(origin: vec3f, dir: vec3f, t_min: f32, t_max: f32) -> f32 {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, t_min, t_max, origin, dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    return select(0.0, 1.0, hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

fn visibility_for_light(
    light: GpuLight,
    world_pos: vec3f,
    surf_n: vec3f,
    l: vec3f,
    attenuation: f32,
    frag_xy: vec2f,
    light_idx: u32,
) -> f32 {
    if light.shadow_type == 0u || light.shadow_strength <= 0.0 || attenuation <= 0.0 {
        return 1.0;
    }
    let light_pos = light.position.xyz;
    let ray_origin = world_pos + surf_n * light.shadow_normal_bias + l * light.shadow_bias;
    var trace_dir: vec3f;
    var t_min: f32;
    var t_max: f32;
    if light.light_type == 1u {
        t_min = max(light.shadow_near_plane, 0.001);
        trace_dir = l;
        t_max = 1.0e6;
    } else {
        let to_lp = light_pos - ray_origin;
        let dist = length(to_lp);
        trace_dir = to_lp / max(dist, 1e-8);
        let light_margin = min(0.02, dist * 0.1);
        t_max = max(dist - light_margin, 1e-5);
        var t_near = max(light.shadow_near_plane, 0.001);
        t_near = min(t_near, t_max * 0.95);
        t_min = min(t_near, max(t_max - 1e-4, 1e-6));
        if t_min >= t_max {
            t_min = max(t_max * 0.01, 1e-6);
        }
    }
    let is_hard = light.shadow_type == 1u;
    if is_hard {
        let vis = trace_shadow_ray(ray_origin, trace_dir, t_min, t_max);
        return mix(1.0, vis, light.shadow_strength);
    }
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(trace_dir.y) > 0.999);
    let uu = normalize(cross(up, trace_dir));
    let vv = cross(trace_dir, uu);
    var acc_vis = 0.0;
    let smax = min(max(rt_extra.soft_shadow_sample_count, 1u), 16u);
    let cone = 0.025 * max(rt_extra.soft_cone_scale, 0.0);
    let seed = dot(frag_xy, vec2f(12.9898, 78.233))
        + f32(light_idx) * 19.19
        + f32(rt_extra.frame_counter) * 3.174;
    for (var s = 0u; s < smax; s++) {
        let r1 = hash21(vec2f(seed + f32(s) * 0.618, f32(s) * 1.414));
        let r2 = hash21(vec2f(seed + 19.19, f32(s) * 2.718));
        let ox = (r1 * 2.0 - 1.0) * cone;
        let oy = (r2 * 2.0 - 1.0) * cone;
        let d = normalize(trace_dir + uu * ox + vv * oy);
        acc_vis += trace_shadow_ray(ray_origin, d, t_min, t_max);
    }
    let vis = acc_vis / f32(smax);
    return mix(1.0, vis, light.shadow_strength);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let layer = global_id.z;
    if layer >= 32u {
        return;
    }
    let atlas_dims = textureDimensions(shadow_atlas_out);
    if global_id.x >= atlas_dims.x || global_id.y >= atlas_dims.y {
        return;
    }

    let pos_dims = textureDimensions(position_tex);
    let stride = max(rt_extra.atlas_gbuffer_stride, 1u);
    let px = min(global_id.x * stride, pos_dims.x - 1u);
    let py = min(global_id.y * stride, pos_dims.y - 1u);
    let pos_sample = textureLoad(position_tex, vec2i(i32(px), i32(py)), 0);
    let n_sample = textureLoad(normal_tex, vec2i(i32(px), i32(py)), 0);
    let normal_len = length(n_sample.xyz);
    if normal_len < 0.1 {
        textureStore(shadow_atlas_out, vec2i(global_id.xy), i32(layer), vec4f(1.0, 0.0, 0.0, 1.0));
        return;
    }

    let world_pos = pos_sample.xyz + rt_extra.gbuffer_origin;
    let n = n_sample.xyz / normal_len;
    let frag_xy = vec2f(f32(px) + 0.5, f32(py) + 0.5);
    let view_z = dot(scene.view_space_z_coeffs.xyz, world_pos) + scene.view_space_z_coeffs.w;
    let d = clamp(-view_z, scene.near_clip, scene.far_clip);
    let cluster_z = u32(clamp(
        log(d / scene.near_clip) / log(scene.far_clip / scene.near_clip) * f32(scene.cluster_count_z),
        0.0, f32(scene.cluster_count_z - 1u)));
    let cluster_xy = cluster_xy_from_frag(frag_xy, scene.viewport_width, scene.viewport_height);
    let cluster_id = min(cluster_xy.x, scene.cluster_count_x - 1u)
        + scene.cluster_count_x * (min(cluster_xy.y, scene.cluster_count_y - 1u)
        + scene.cluster_count_y * cluster_z);
    let count = cluster_light_counts[cluster_id];
    if layer >= count {
        textureStore(shadow_atlas_out, vec2i(global_id.xy), i32(layer), vec4f(1.0, 0.0, 0.0, 1.0));
        return;
    }
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;
    let light_idx = cluster_light_indices[base_idx + layer];
    if light_idx >= scene.light_count {
        textureStore(shadow_atlas_out, vec2i(global_id.xy), i32(layer), vec4f(1.0, 0.0, 0.0, 1.0));
        return;
    }
    let light = lights[light_idx];
    let light_pos = light.position.xyz;
    let light_dir = light.direction.xyz;
    var l: vec3f;
    var attenuation: f32;
    if light.light_type == 0u {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        attenuation = select(0.0, light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)), light.range > 0.0);
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3f(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
        attenuation = select(0.0, light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001), light.range > 0.0);
    }
    let vis = visibility_for_light(light, world_pos, n, l, attenuation, frag_xy, light_idx);
    textureStore(shadow_atlas_out, vec2i(global_id.xy), i32(layer), vec4f(vis, 0.0, 0.0, 1.0));
}
"#;

/// Host layout for [`SHADOW_ATLAS_TRACE_SHADER`] `RtShadowComputeExtra` (32 bytes, WGSL-uniform aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RtShadowComputeExtra {
    gbuffer_origin: [f32; 3],
    _pad0: f32,
    soft_shadow_sample_count: u32,
    soft_cone_scale: f32,
    frame_counter: u32,
    atlas_gbuffer_stride: u32,
}

/// Compute pass that writes the RT shadow atlas (optional; requires MRT position/normal and TLAS).
pub struct RtShadowComputePass {
    trace_pipeline: Option<wgpu::ComputePipeline>,
    trace_bgl: Option<wgpu::BindGroupLayout>,
    clear_pipeline: Option<wgpu::ComputePipeline>,
    clear_bgl: Option<wgpu::BindGroupLayout>,
}

impl RtShadowComputePass {
    /// Builds a pass; GPU pipelines are created lazily on first successful use.
    pub fn new() -> Self {
        Self {
            trace_pipeline: None,
            trace_bgl: None,
            clear_pipeline: None,
            clear_bgl: None,
        }
    }

    fn ensure_clear_pipeline(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<(&wgpu::ComputePipeline, &wgpu::BindGroupLayout)> {
        if self.clear_pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RT shadow atlas clear shader"),
                source: wgpu::ShaderSource::Wgsl(SHADOW_ATLAS_CLEAR_SHADER.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RT shadow atlas clear BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                }],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RT shadow atlas clear pipeline layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
            self.clear_pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("RT shadow atlas clear pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.clear_bgl = Some(bgl);
        }
        self.clear_pipeline.as_ref().zip(self.clear_bgl.as_ref())
    }

    fn dispatch_clear(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        atlas_view: &wgpu::TextureView,
        atlas_w: u32,
        atlas_h: u32,
    ) {
        let Some((pl, bgl)) = self.ensure_clear_pipeline(device) else {
            return;
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RT shadow atlas clear bind group"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(atlas_view),
            }],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RT shadow atlas clear"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pl);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(atlas_w.div_ceil(TILE), atlas_h.div_ceil(TILE), 32);
    }

    fn ensure_trace_pipeline(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<(&wgpu::ComputePipeline, &wgpu::BindGroupLayout)> {
        if self.trace_pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RT shadow atlas trace shader"),
                source: wgpu::ShaderSource::Wgsl(SHADOW_ATLAS_TRACE_SHADER.into()),
            });
            let scene_sz = std::mem::size_of::<SceneUniforms>() as u64;
            let extra_sz = std::mem::size_of::<RtShadowComputeExtra>() as u64;
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RT shadow atlas trace BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(scene_sz),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(extra_sz),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R16Float,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                        },
                        count: None,
                    },
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RT shadow atlas trace pipeline layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
            self.trace_pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("RT shadow atlas trace pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.trace_bgl = Some(bgl);
        }
        self.trace_pipeline.as_ref().zip(self.trace_bgl.as_ref())
    }
}

impl Default for RtShadowComputePass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for RtShadowComputePass {
    fn name(&self) -> &str {
        "rt_shadow_compute"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![
                ResourceSlot::ClusterBuffers,
                ResourceSlot::LightBuffer,
                ResourceSlot::Position,
                ResourceSlot::Normal,
            ],
            writes: vec![],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        let pos_view = match ctx.render_target.mrt_position_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let norm_view = match ctx.render_target.mrt_normal_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let Some((atlas_w, atlas_h)) = ctx.gpu.rt_shadow_atlas_extent else {
            return Ok(());
        };
        let Some(ref atlas_view) = ctx.gpu.rt_shadow_atlas_main_view else {
            return Ok(());
        };

        let cluster_buffers = match ctx
            .gpu
            .cluster_buffer_cache
            .get_buffers(ctx.viewport, ctx.gpu.cluster_count_z)
        {
            Some(c) => c,
            None => {
                self.dispatch_clear(&ctx.gpu.device, ctx.encoder, atlas_view, atlas_w, atlas_h);
                return Ok(());
            }
        };
        let light_buffer = match ctx
            .gpu
            .light_buffer_cache
            .ensure_buffer(&ctx.gpu.device, ctx.gpu.light_count.max(1) as usize)
        {
            Some(b) => b,
            None => {
                self.dispatch_clear(&ctx.gpu.device, ctx.encoder, atlas_view, atlas_w, atlas_h);
                return Ok(());
            }
        };

        let tlas = match &ctx.gpu.ray_tracing_state {
            Some(rt) => match &rt.tlas {
                Some(t) => t,
                None => {
                    self.dispatch_clear(&ctx.gpu.device, ctx.encoder, atlas_view, atlas_w, atlas_h);
                    return Ok(());
                }
            },
            None => {
                self.dispatch_clear(&ctx.gpu.device, ctx.encoder, atlas_view, atlas_w, atlas_h);
                return Ok(());
            }
        };

        let (pl, bgl) = match self.ensure_trace_pipeline(&ctx.gpu.device) {
            Some(x) => x,
            None => {
                self.dispatch_clear(&ctx.gpu.device, ctx.encoder, atlas_view, atlas_w, atlas_h);
                return Ok(());
            }
        };

        const SCENE_SZ: u64 = 64;
        const EXTRA_SZ: u64 = 32;
        let device = &ctx.gpu.device;
        let scene_buf = match ctx.gpu.rt_shadow_compute_scene_buffer.take() {
            Some(b) if b.size() >= SCENE_SZ => b,
            _ => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RT shadow compute scene uniforms"),
                size: SCENE_SZ,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };
        let extra_buf = match ctx.gpu.rt_shadow_compute_extra_buffer.take() {
            Some(b) if b.size() >= EXTRA_SZ => b,
            _ => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RT shadow compute extra uniforms"),
                size: EXTRA_SZ,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        let cluster_ok = ctx.gpu.cluster_count_x > 0
            && ctx.gpu.cluster_count_y > 0
            && ctx.gpu.cluster_count_z > 0;
        if !cluster_ok {
            ctx.gpu.rt_shadow_compute_scene_buffer = Some(scene_buf);
            ctx.gpu.rt_shadow_compute_extra_buffer = Some(extra_buf);
            self.dispatch_clear(&ctx.gpu.device, ctx.encoder, atlas_view, atlas_w, atlas_h);
            return Ok(());
        }

        let batch = pbr_primary_view_batch(ctx.draw_batches, ctx.session);
        let view_space_z_coeffs = batch
            .map(pbr_view_space_z_coeffs_for_batch)
            .unwrap_or([0.0, 0.0, 0.0, 0.0]);
        let origin = mrt_gbuffer_world_origin(ctx.draw_batches, ctx.session);
        let (vw, vh) = ctx.viewport;
        let scene = SceneUniforms {
            view_position: origin,
            _pad0: 0.0,
            view_space_z_coeffs,
            cluster_count_x: ctx.gpu.cluster_count_x,
            cluster_count_y: ctx.gpu.cluster_count_y,
            cluster_count_z: ctx.gpu.cluster_count_z,
            near_clip: ctx.session.near_clip().max(0.01),
            far_clip: ctx.session.far_clip(),
            light_count: ctx.gpu.light_count,
            viewport_width: vw,
            viewport_height: vh,
        };
        let rc = ctx.session.render_config();
        let extra = RtShadowComputeExtra {
            gbuffer_origin: origin,
            _pad0: 0.0,
            soft_shadow_sample_count: rc.rt_soft_shadow_samples,
            soft_cone_scale: rc.rt_soft_shadow_cone_scale,
            frame_counter: ctx.frame_index as u32,
            atlas_gbuffer_stride: if rc.rt_shadow_atlas_half_resolution {
                2
            } else {
                1
            },
        };
        ctx.gpu
            .queue
            .write_buffer(&scene_buf, 0, bytemuck::bytes_of(&scene));
        ctx.gpu
            .queue
            .write_buffer(&extra_buf, 0, bytemuck::bytes_of(&extra));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RT shadow atlas trace bind group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cluster_buffers.cluster_light_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cluster_buffers.cluster_light_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::AccelerationStructure(tlas),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: extra_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(pos_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(norm_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(atlas_view),
                },
            ],
        });

        ctx.gpu.rt_shadow_compute_scene_buffer = Some(scene_buf);
        ctx.gpu.rt_shadow_compute_extra_buffer = Some(extra_buf);

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RT shadow atlas trace"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pl);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(atlas_w.div_ceil(TILE), atlas_h.div_ceil(TILE), 32);

        Ok(())
    }
}

#[cfg(test)]
mod rt_shadow_compute_uniform_tests {
    use super::RtShadowComputeExtra;
    use std::mem::size_of;

    #[test]
    fn rt_shadow_compute_extra_is_32_bytes() {
        assert_eq!(size_of::<RtShadowComputeExtra>(), 32);
        assert_eq!(size_of::<RtShadowComputeExtra>() % 16, 0);
    }
}
