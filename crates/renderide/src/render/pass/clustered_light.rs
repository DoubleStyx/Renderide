//! Clustered light compute pass.
//!
//! Dispatches over tiles (16x16 pixels), computes per-tile light indices by testing
//! each light against the tile frustum (point: sphere-AABB, spot: cone-AABB,
//! directional: always in). Outputs cluster_light_counts and cluster_light_indices.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use super::mesh_draw::apply_view_handedness_fix;
use super::{RenderPass, RenderPassError};
use crate::gpu::cluster_buffer::ClusterBufferRefs;
use crate::render::SpaceDrawBatch;
use crate::render::lights::MAX_LIGHTS;
use crate::scene::render_transform_to_matrix;
use crate::session::Session;

const TILE_SIZE: u32 = 16;

/// Cluster parameters uniform for the compute shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClusterParams {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    viewport_width: f32,
    viewport_height: f32,
    tile_size: u32,
    light_count: u32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    near_clip: f32,
    far_clip: f32,
}

const CLUSTER_PARAMS_SIZE: u64 = size_of::<ClusterParams>() as u64;

const CLUSTERED_LIGHT_SHADER_SRC: &str = r#"
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
    _pad2: vec4u,
}

struct ClusterParams {
    view: mat4x4f,
    proj: mat4x4f,
    inv_view_proj: mat4x4f,
    viewport_width: f32,
    viewport_height: f32,
    tile_size: u32,
    light_count: u32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    near_clip: f32,
    far_clip: f32,
}

@group(0) @binding(0) var<uniform> params: ClusterParams;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<storage, read_write> cluster_light_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cluster_light_indices: array<u32>;

const MAX_LIGHTS_PER_TILE: u32 = 32u;

struct TileAabb {
    min_v: vec3f,
    max_v: vec3f,
}

fn ndc_to_view(ndc: vec3f) -> vec3f {
    let clip = params.inv_view_proj * vec4f(ndc.x, ndc.y, ndc.z, 1.0);
    return clip.xyz / clip.w;
}

fn get_tile_aabb(cluster_x: u32, cluster_y: u32) -> TileAabb {
    let w = params.viewport_width;
    let h = params.viewport_height;
    let px_min = f32(cluster_x * params.tile_size) + 0.5;
    let px_max = f32((cluster_x + 1u) * params.tile_size) - 0.5;
    let py_min = f32(cluster_y * params.tile_size) + 0.5;
    let py_max = f32((cluster_y + 1u) * params.tile_size) - 0.5;
    let ndc_left = 2.0 * px_min / w - 1.0;
    let ndc_right = 2.0 * px_max / w - 1.0;
    let ndc_top = 1.0 - 2.0 * py_min / h;
    let ndc_bottom = 1.0 - 2.0 * py_max / h;
    let v_near_bl = ndc_to_view(vec3f(ndc_left, ndc_bottom, 1.0));
    let v_near_br = ndc_to_view(vec3f(ndc_right, ndc_bottom, 1.0));
    let v_near_tl = ndc_to_view(vec3f(ndc_left, ndc_top, 1.0));
    let v_near_tr = ndc_to_view(vec3f(ndc_right, ndc_top, 1.0));
    let v_far_bl = ndc_to_view(vec3f(ndc_left, ndc_bottom, -1.0));
    let v_far_br = ndc_to_view(vec3f(ndc_right, ndc_bottom, -1.0));
    let v_far_tl = ndc_to_view(vec3f(ndc_left, ndc_top, -1.0));
    let v_far_tr = ndc_to_view(vec3f(ndc_right, ndc_top, -1.0));
    var min_v = min(min(min(v_near_bl, v_near_br), min(v_near_tl, v_near_tr)), min(min(v_far_bl, v_far_br), min(v_far_tl, v_far_tr)));
    var max_v = max(max(max(v_near_bl, v_near_br), max(v_near_tl, v_near_tr)), max(max(v_far_bl, v_far_br), max(v_far_tl, v_far_tr)));
    return TileAabb(min_v, max_v);
}

fn sphere_aabb_intersect(center: vec3f, radius: f32, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    let closest = clamp(center, aabb_min, aabb_max);
    let d = center - closest;
    return dot(d, d) <= radius * radius;
}

fn cone_aabb_intersect(apex: vec3f, _axis: vec3f, _cos_half: f32, range: f32, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    return sphere_aabb_intersect(apex, range, aabb_min, aabb_max);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let cluster_count_x = params.cluster_count_x;
    let cluster_count_y = params.cluster_count_y;
    if global_id.x >= cluster_count_x || global_id.y >= cluster_count_y {
        return;
    }
    let cluster_id = global_id.x + global_id.y * cluster_count_x;
    let cluster_x = global_id.x;
    let cluster_y = global_id.y;

    let aabb = get_tile_aabb(cluster_x, cluster_y);
    let aabb_min = aabb.min_v;
    let aabb_max = aabb.max_v;

    var count: u32 = 0u;
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;

    for (var i = 0u; i < params.light_count; i++) {
        if count >= MAX_LIGHTS_PER_TILE {
            break;
        }
        let light = lights[i];
        let pos_view = (params.view * vec4f(light.position.x, light.position.y, light.position.z, 1.0)).xyz;
        let dir_view = (params.view * vec4f(light.direction.x, light.direction.y, light.direction.z, 0.0)).xyz;

        var intersects = false;
        if light.light_type == 0u {
            intersects = sphere_aabb_intersect(pos_view, light.range, aabb_min, aabb_max);
        } else if light.light_type == 1u {
            intersects = true;
        } else {
            let axis = -normalize(dir_view);
            intersects = cone_aabb_intersect(pos_view, axis, light.spot_cos_half_angle, light.range, aabb_min, aabb_max);
        }

        if intersects {
            cluster_light_indices[base_idx + count] = i;
            count += 1u;
        }
    }

    atomicStore(&cluster_light_counts[cluster_id], count);
}
"#;

/// Clustered light compute pass: builds per-tile light indices.
/// Uses [`ClusterBufferCache`](crate::gpu::cluster_buffer::ClusterBufferCache) from GpuState.
pub struct ClusteredLightPass {
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl ClusteredLightPass {
    /// Creates a new clustered light pass.
    pub fn new() -> Self {
        Self {
            pipeline: None,
            bind_group_layout: None,
        }
    }

    fn ensure_pipeline(&mut self, device: &wgpu::Device) -> bool {
        if self.pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("clustered light compute shader"),
                source: wgpu::ShaderSource::Wgsl(CLUSTERED_LIGHT_SHADER_SRC.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("clustered light bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(CLUSTER_PARAMS_SIZE),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("clustered light pipeline layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
            self.pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("clustered light compute pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.bind_group_layout = Some(bgl);
        }
        true
    }

    fn view_matrix_from_batches(draw_batches: &[SpaceDrawBatch]) -> Option<Mat4> {
        draw_batches.iter().find(|b| !b.is_overlay).map(|b| {
            apply_view_handedness_fix(render_transform_to_matrix(&b.view_transform).inverse())
        })
    }

    fn space_id_for_lights(session: &Session, draw_batches: &[SpaceDrawBatch]) -> Option<i32> {
        session.primary_view_space_id().or_else(|| {
            draw_batches
                .iter()
                .find(|b| !b.is_overlay)
                .map(|b| b.space_id)
        })
    }
}

impl RenderPass for ClusteredLightPass {
    fn name(&self) -> &str {
        "clustered_light"
    }

    fn execute(&mut self, ctx: &mut super::RenderPassContext) -> Result<(), RenderPassError> {
        let view_mat = match Self::view_matrix_from_batches(ctx.draw_batches) {
            Some(v) => v,
            None => {
                logger::trace!("Clustered light pass skipped: no non-overlay batch");
                return Ok(());
            }
        };

        let space_id = match Self::space_id_for_lights(ctx.session, ctx.draw_batches) {
            Some(id) => id,
            None => {
                logger::trace!("Clustered light pass skipped: no space for lights");
                return Ok(());
            }
        };

        let lights = ctx
            .session
            .resolved_lights_for_space(space_id)
            .unwrap_or_default();

        logger::trace!(
            "clustered_light pass space_id={} light_count={} lights=[{}]",
            space_id,
            lights.len(),
            lights
                .iter()
                .map(|l| format!(
                    "pos=({:.2},{:.2},{:.2}) type={:?} intensity={:.2} range={:.2}",
                    l.world_position.x,
                    l.world_position.y,
                    l.world_position.z,
                    l.light_type,
                    l.intensity,
                    l.range
                ))
                .collect::<Vec<_>>()
                .join("; ")
        );

        let light_count = lights.len().min(MAX_LIGHTS);
        let effective_light_count = light_count.max(1);
        ctx.gpu
            .light_buffer_cache
            .ensure_buffer(&ctx.gpu.device, effective_light_count);
        if !lights.is_empty() {
            ctx.gpu.light_buffer_cache.upload(&ctx.gpu.queue, lights);
        } else {
            let zero_light = crate::render::lights::GpuLight::default();
            if let Some(buf) = ctx
                .gpu
                .light_buffer_cache
                .ensure_buffer(&ctx.gpu.device, effective_light_count)
            {
                ctx.gpu
                    .queue
                    .write_buffer(buf, 0, bytemuck::bytes_of(&zero_light));
            }
        }
        let light_buffer = match ctx
            .gpu
            .light_buffer_cache
            .ensure_buffer(&ctx.gpu.device, effective_light_count)
        {
            Some(b) => b,
            None => return Ok(()),
        };

        let ClusterBufferRefs {
            cluster_light_counts: cluster_counts,
            cluster_light_indices: cluster_indices,
            params_buffer,
        } = match ctx
            .gpu
            .cluster_buffer_cache
            .ensure_buffers(&ctx.gpu.device, ctx.viewport)
        {
            Some(refs) => refs,
            None => return Ok(()),
        };

        if !self.ensure_pipeline(&ctx.gpu.device) {
            return Ok(());
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let bgl = self.bind_group_layout.as_ref().unwrap();

        let (width, height) = ctx.viewport;

        let cluster_count_x = width.div_ceil(TILE_SIZE);
        let cluster_count_y = height.div_ceil(TILE_SIZE);

        ctx.gpu.cluster_count_x = cluster_count_x;
        ctx.gpu.cluster_count_y = cluster_count_y;
        ctx.gpu.light_count = light_count as u32;

        let proj_glam = Mat4::from_cols_array(&[
            ctx.proj[(0, 0)],
            ctx.proj[(1, 0)],
            ctx.proj[(2, 0)],
            ctx.proj[(3, 0)],
            ctx.proj[(0, 1)],
            ctx.proj[(1, 1)],
            ctx.proj[(2, 1)],
            ctx.proj[(3, 1)],
            ctx.proj[(0, 2)],
            ctx.proj[(1, 2)],
            ctx.proj[(2, 2)],
            ctx.proj[(3, 2)],
            ctx.proj[(0, 3)],
            ctx.proj[(1, 3)],
            ctx.proj[(2, 3)],
            ctx.proj[(3, 3)],
        ]);

        let view_cols = view_mat.to_cols_array();
        let proj_cols = proj_glam.to_cols_array();
        let view_proj = proj_glam * view_mat;
        let inv_view_proj = view_proj.inverse();
        let inv_cols = inv_view_proj.to_cols_array();
        let params = ClusterParams {
            view: [
                [view_cols[0], view_cols[1], view_cols[2], view_cols[3]],
                [view_cols[4], view_cols[5], view_cols[6], view_cols[7]],
                [view_cols[8], view_cols[9], view_cols[10], view_cols[11]],
                [view_cols[12], view_cols[13], view_cols[14], view_cols[15]],
            ],
            proj: [
                [proj_cols[0], proj_cols[1], proj_cols[2], proj_cols[3]],
                [proj_cols[4], proj_cols[5], proj_cols[6], proj_cols[7]],
                [proj_cols[8], proj_cols[9], proj_cols[10], proj_cols[11]],
                [proj_cols[12], proj_cols[13], proj_cols[14], proj_cols[15]],
            ],
            inv_view_proj: [
                [inv_cols[0], inv_cols[1], inv_cols[2], inv_cols[3]],
                [inv_cols[4], inv_cols[5], inv_cols[6], inv_cols[7]],
                [inv_cols[8], inv_cols[9], inv_cols[10], inv_cols[11]],
                [inv_cols[12], inv_cols[13], inv_cols[14], inv_cols[15]],
            ],
            viewport_width: width as f32,
            viewport_height: height as f32,
            tile_size: TILE_SIZE,
            light_count: light_count as u32,
            cluster_count_x,
            cluster_count_y,
            near_clip: ctx.session.near_clip().max(0.01),
            far_clip: ctx.session.far_clip(),
        };

        ctx.gpu
            .queue
            .write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = ctx
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clustered light bind group"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cluster_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cluster_indices.as_entire_binding(),
                    },
                ],
            });

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clustered light pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cluster_count_x, cluster_count_y, 1);

        Ok(())
    }
}

impl Default for ClusteredLightPass {
    fn default() -> Self {
        Self::new()
    }
}
