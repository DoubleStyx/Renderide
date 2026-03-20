//! Clustered light compute pass.
//!
//! Dispatches over tiles (16x16 pixels), computes per-tile light indices by testing
//! each light against the tile frustum (point: sphere-AABB, spot: conservative bounding sphere of
//! the finite cone, directional: always in). Outputs cluster_light_counts and cluster_light_indices.
//!
//! [`crate::render::lights::order_lights_for_clustered_shading`] runs before upload so directional
//! lights occupy low buffer indices and are not dropped when local lights fill the per-cluster cap
//! ([`MAX_LIGHTS_PER_TILE`]).
//!
//! For spot lights, the cone axis in WGSL is the view-space beam forward (same as
//! [`GpuLight`](crate::render::lights::GpuLight) `direction`), matching PBR `light_dir`.
//!
//! The view matrix passed to the compute shader must match [`crate::render::visibility::view_matrix_glam_for_batch`]
//! (handedness fix + scale filter), i.e. the same eye space as mesh MVP, or light-volume tests
//! against cluster AABBs will be wrong when the camera moves.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

use super::{PassResources, RenderPass, RenderPassError, ResourceSlot};
use crate::diagnostics::LogOnChange;
use crate::gpu::cluster_buffer::ClusterBufferRefs;
use crate::render::SpaceDrawBatch;
use crate::render::lights::{MAX_LIGHTS, order_lights_for_clustered_shading};
use crate::session::Session;

const TILE_SIZE: u32 = 16;
/// Depth slice count for clustered shading (DOOM 2016 style exponential subdivision).
const CLUSTER_COUNT_Z: u32 = 24;

/// Cluster parameters uniform for the compute shader.
///
/// Padded to 240 bytes to match WGSL uniform buffer alignment (16-byte boundary).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClusterParams {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    viewport_width: f32,
    viewport_height: f32,
    tile_size: u32,
    light_count: u32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    /// Padding to 240 bytes for WGSL uniform buffer alignment.
    _pad: [u8; 16],
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
    inv_proj: mat4x4f,
    viewport_width: f32,
    viewport_height: f32,
    tile_size: u32,
    light_count: u32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
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
    let clip = params.inv_proj * vec4f(ndc.x, ndc.y, ndc.z, 1.0);
    return clip.xyz / clip.w;
}

/// Intersects the ray from origin through `ray_point` with the view-space z-plane at `z_dist`.
fn line_intersect_z_plane(ray_point: vec3f, z_dist: f32) -> vec3f {
    let t = z_dist / ray_point.z;
    return ray_point * t;
}

/// Returns the AABB of a 3D cluster in view space (DOOM 2016 exponential depth subdivision).
fn get_cluster_aabb(cluster_x: u32, cluster_y: u32, cluster_z: u32) -> TileAabb {
    let w = params.viewport_width;
    let h = params.viewport_height;
    let near = params.near_clip;
    let far = params.far_clip;
    let num_z = f32(params.cluster_count_z);
    let z = f32(cluster_z);

    let tile_near = -near * pow(far / near, z / num_z);
    let tile_far = -near * pow(far / near, (z + 1.0) / num_z);

    let px_min = f32(cluster_x * params.tile_size) + 0.5;
    let px_max = f32((cluster_x + 1u) * params.tile_size) - 0.5;
    let py_min = f32(cluster_y * params.tile_size) + 0.5;
    let py_max = f32((cluster_y + 1u) * params.tile_size) - 0.5;
    let ndc_left = 2.0 * px_min / w - 1.0;
    let ndc_right = 2.0 * px_max / w - 1.0;
    let ndc_top = 1.0 - 2.0 * py_min / h;
    let ndc_bottom = 1.0 - 2.0 * py_max / h;

    let v_bl = ndc_to_view(vec3f(ndc_left, ndc_bottom, 1.0));
    let v_br = ndc_to_view(vec3f(ndc_right, ndc_bottom, 1.0));
    let v_tl = ndc_to_view(vec3f(ndc_left, ndc_top, 1.0));
    let v_tr = ndc_to_view(vec3f(ndc_right, ndc_top, 1.0));

    let p_near_bl = line_intersect_z_plane(v_bl, tile_near);
    let p_near_br = line_intersect_z_plane(v_br, tile_near);
    let p_near_tl = line_intersect_z_plane(v_tl, tile_near);
    let p_near_tr = line_intersect_z_plane(v_tr, tile_near);
    let p_far_bl = line_intersect_z_plane(v_bl, tile_far);
    let p_far_br = line_intersect_z_plane(v_br, tile_far);
    let p_far_tl = line_intersect_z_plane(v_tl, tile_far);
    let p_far_tr = line_intersect_z_plane(v_tr, tile_far);

    var min_v = min(min(min(p_near_bl, p_near_br), min(p_near_tl, p_near_tr)), min(min(p_far_bl, p_far_br), min(p_far_tl, p_far_tr)));
    var max_v = max(max(max(p_near_bl, p_near_br), max(p_near_tl, p_near_tr)), max(max(p_far_bl, p_far_br), max(p_far_tl, p_far_tr)));
    return TileAabb(min_v, max_v);
}

fn sphere_aabb_intersect(center: vec3f, radius: f32, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    let closest = clamp(center, aabb_min, aabb_max);
    let d = center - closest;
    return dot(d, d) <= radius * radius;
}

/// Conservative sphere that fully contains the finite spotlight cone (apex, axis, half-angle, range).
/// Center is midpoint along the axis; radius reaches apex and base rim (avoids false negatives from
/// incomplete cone–AABB tests).
fn spotlight_bounds_intersect_aabb(apex: vec3f, axis: vec3f, cos_half: f32, range: f32, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    if cos_half >= 0.9999 {
        return sphere_aabb_intersect(apex, range, aabb_min, aabb_max);
    }
    let axis_n = normalize(axis);
    let sin_sq = max(0.0, 1.0 - cos_half * cos_half);
    let tan_sq = sin_sq / max(cos_half * cos_half, 1e-8);
    let radius = range * sqrt(0.25 + tan_sq);
    let center = apex + axis_n * (range * 0.5);
    return sphere_aabb_intersect(center, radius, aabb_min, aabb_max);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let cluster_count_x = params.cluster_count_x;
    let cluster_count_y = params.cluster_count_y;
    let cluster_count_z = params.cluster_count_z;
    if global_id.x >= cluster_count_x || global_id.y >= cluster_count_y || global_id.z >= cluster_count_z {
        return;
    }
    let cluster_id = global_id.x + cluster_count_x * (global_id.y + cluster_count_y * global_id.z);
    let cluster_x = global_id.x;
    let cluster_y = global_id.y;
    let cluster_z = global_id.z;

    let aabb = get_cluster_aabb(cluster_x, cluster_y, cluster_z);
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
            // Forward axis = PBR `light_dir`; avoid normalize(0) → NaN on some drivers.
            let dir_len_sq = dot(dir_view, dir_view);
            let axis = select(
                vec3f(0.0, 0.0, 1.0),
                dir_view * inverseSqrt(dir_len_sq),
                dir_len_sq > 1e-16
            );
            intersects = spotlight_bounds_intersect_aabb(pos_view, axis, light.spot_cos_half_angle, light.range, aabb_min, aabb_max);
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
    /// Set after the first successful clustered-light dispatch; used for a one-time info log.
    logged_active_once: bool,
    /// Skip-reason codes for `debug!` only when the reason changes; see [`Self::log_skip`].
    skip_kind: LogOnChange<u8>,
}

impl ClusteredLightPass {
    /// Creates a new clustered light pass.
    pub fn new() -> Self {
        Self {
            pipeline: None,
            bind_group_layout: None,
            logged_active_once: false,
            skip_kind: LogOnChange::new(),
        }
    }

    /// Emits `trace!` every call; emits `debug!` only when `kind` changes from the last skip or success.
    fn log_skip(&mut self, kind: u8, message: impl Into<String>) {
        let msg = message.into();
        logger::trace!("{}", msg);
        if self.skip_kind.changed(kind) {
            logger::debug!("{}", msg);
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

    /// Returns the view matrix for the batch whose `space_id` matches, or the first non-overlay
    /// batch if no match exists. Ensures lights and view are from the same coordinate space.
    /// The second element is `true` if a batch with matching `space_id` was found.
    fn view_matrix_for_space(
        draw_batches: &[SpaceDrawBatch],
        space_id: i32,
    ) -> Option<(Mat4, bool)> {
        let matching = draw_batches
            .iter()
            .find(|b| !b.is_overlay && b.space_id == space_id);
        let batch = matching.or_else(|| draw_batches.iter().find(|b| !b.is_overlay))?;
        let view = crate::render::visibility::view_matrix_glam_for_batch(batch);
        Some((view, matching.is_some()))
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

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![],
            writes: vec![ResourceSlot::ClusterBuffers, ResourceSlot::LightBuffer],
        }
    }

    fn execute(&mut self, ctx: &mut super::RenderPassContext) -> Result<(), RenderPassError> {
        let space_id = match Self::space_id_for_lights(ctx.session, ctx.draw_batches) {
            Some(id) => id,
            None => {
                self.log_skip(
                    1,
                    "Clustered light pass skipped: no space for lights (primary_view_space_id or non-overlay batch not found)"
                        .to_string(),
                );
                return Ok(());
            }
        };

        let (view_mat, batch_matched) = match Self::view_matrix_for_space(
            ctx.draw_batches,
            space_id,
        ) {
            Some(v) => v,
            None => {
                let overlay_count = ctx.draw_batches.iter().filter(|b| b.is_overlay).count();
                let non_overlay_count = ctx.draw_batches.len() - overlay_count;
                self.log_skip(
                    2,
                    format!(
                        "Clustered light pass skipped: no non-overlay batch (batches: {} overlay, {} non-overlay)",
                        overlay_count, non_overlay_count
                    ),
                );
                return Ok(());
            }
        };

        let lights_raw = ctx
            .session
            .resolved_lights_for_space(space_id)
            .unwrap_or(&[]);
        let lights = order_lights_for_clustered_shading(lights_raw);

        logger::trace!(
            "clustered_light pass space_id={} batch_matched={} light_count={} lights=[{}]",
            space_id,
            batch_matched,
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

        if !batch_matched && ctx.session.primary_view_space_id().is_some() && !lights.is_empty() {
            let view = view_mat;
            let pos_view_samples: Vec<_> = lights
                .iter()
                .take(3)
                .map(|l| {
                    let p = Vec4::new(
                        l.world_position.x,
                        l.world_position.y,
                        l.world_position.z,
                        1.0,
                    );
                    let v = view * p;
                    format!("({:.2},{:.2},{:.2})", v.x, v.y, v.z)
                })
                .collect();
            logger::trace!(
                "clustered_light view/space mismatch: space_id={} differs from first non-overlay batch; light pos_view (first 3)=[{}]",
                space_id,
                pos_view_samples.join("; ")
            );
        }

        let light_count = lights.len().min(MAX_LIGHTS);
        let effective_light_count = light_count.max(1);
        ctx.gpu
            .light_buffer_cache
            .ensure_buffer(&ctx.gpu.device, effective_light_count);
        if !lights.is_empty() {
            ctx.gpu.light_buffer_cache.upload(&ctx.gpu.queue, &lights);
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
            None => {
                self.log_skip(
                    3,
                    format!(
                        "Clustered light pass skipped: light buffer ensure_buffer returned None (effective_light_count={})",
                        effective_light_count
                    ),
                );
                return Ok(());
            }
        };

        let ClusterBufferRefs {
            cluster_light_counts: cluster_counts,
            cluster_light_indices: cluster_indices,
            params_buffer,
        } = match ctx.gpu.cluster_buffer_cache.ensure_buffers(
            &ctx.gpu.device,
            ctx.viewport,
            CLUSTER_COUNT_Z,
        ) {
            Some(refs) => refs,
            None => {
                self.log_skip(
                    4,
                    format!(
                        "Clustered light pass skipped: cluster ensure_buffers returned None (viewport={:?})",
                        ctx.viewport
                    ),
                );
                return Ok(());
            }
        };

        if !self.ensure_pipeline(&ctx.gpu.device) {
            self.log_skip(
                5,
                "Clustered light pass skipped: ensure_pipeline failed".to_string(),
            );
            return Ok(());
        }

        let (pipeline, bgl) = match (self.pipeline.as_ref(), self.bind_group_layout.as_ref()) {
            (Some(p), Some(b)) => (p, b),
            _ => {
                self.log_skip(
                    6,
                    "Clustered light pass skipped: pipeline or bind group layout not ready"
                        .to_string(),
                );
                return Ok(());
            }
        };

        let (width, height) = ctx.viewport;

        let cluster_count_x = width.div_ceil(TILE_SIZE);
        let cluster_count_y = height.div_ceil(TILE_SIZE);

        ctx.gpu.cluster_count_x = cluster_count_x;
        ctx.gpu.cluster_count_y = cluster_count_y;
        ctx.gpu.cluster_count_z = CLUSTER_COUNT_Z;
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
        let inv_proj = proj_glam.inverse();
        let inv_proj_cols = inv_proj.to_cols_array();
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
            inv_proj: [
                [
                    inv_proj_cols[0],
                    inv_proj_cols[1],
                    inv_proj_cols[2],
                    inv_proj_cols[3],
                ],
                [
                    inv_proj_cols[4],
                    inv_proj_cols[5],
                    inv_proj_cols[6],
                    inv_proj_cols[7],
                ],
                [
                    inv_proj_cols[8],
                    inv_proj_cols[9],
                    inv_proj_cols[10],
                    inv_proj_cols[11],
                ],
                [
                    inv_proj_cols[12],
                    inv_proj_cols[13],
                    inv_proj_cols[14],
                    inv_proj_cols[15],
                ],
            ],
            viewport_width: width as f32,
            viewport_height: height as f32,
            tile_size: TILE_SIZE,
            light_count: light_count as u32,
            cluster_count_x,
            cluster_count_y,
            cluster_count_z: CLUSTER_COUNT_Z,
            near_clip: ctx.session.near_clip().max(0.01),
            far_clip: ctx.session.far_clip(),
            _pad: [0; 16],
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
        pass.dispatch_workgroups(
            cluster_count_x.div_ceil(8),
            cluster_count_y.div_ceil(8),
            CLUSTER_COUNT_Z,
        );

        if !self.logged_active_once {
            self.logged_active_once = true;
            logger::info!(
                "Clustered light pass active (space_id={} cluster_grid={}x{}x{} lights={})",
                space_id,
                cluster_count_x,
                cluster_count_y,
                CLUSTER_COUNT_Z,
                light_count
            );
        }
        self.skip_kind.reset();

        Ok(())
    }
}

impl Default for ClusteredLightPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Quaternion, Vector3};

    fn make_batch(space_id: i32, is_overlay: bool, pos: (f32, f32, f32)) -> SpaceDrawBatch {
        SpaceDrawBatch {
            space_id,
            is_overlay,
            view_transform: crate::shared::RenderTransform {
                position: Vector3::new(pos.0, pos.1, pos.2),
                scale: Vector3::new(1.0, 1.0, 1.0),
                rotation: Quaternion::identity(),
            },
            draws: vec![],
        }
    }

    #[test]
    fn view_matrix_for_space_returns_matching_batch() {
        let batches = vec![
            make_batch(3, false, (1.0, 0.0, 0.0)),
            make_batch(5, false, (10.0, 0.0, 0.0)),
        ];
        let (view, matched) =
            ClusteredLightPass::view_matrix_for_space(&batches, 5).expect("should have view");
        assert!(matched, "space_id=5 should match batch 2");
        let inv = view.inverse();
        let cam_pos = inv.w_axis;
        assert!(
            (cam_pos.x - 10.0).abs() < 1e-5,
            "camera should be at (10,0,0)"
        );
    }

    #[test]
    fn view_matrix_for_space_fallback_when_no_match() {
        let batches = vec![
            make_batch(3, false, (1.0, 0.0, 0.0)),
            make_batch(5, false, (10.0, 0.0, 0.0)),
        ];
        let (view, matched) =
            ClusteredLightPass::view_matrix_for_space(&batches, 99).expect("should fallback");
        assert!(!matched, "space_id=99 has no match, should use fallback");
        let inv = view.inverse();
        let cam_pos = inv.w_axis;
        assert!(
            (cam_pos.x - 1.0).abs() < 1e-5,
            "fallback should use first non-overlay (space 3)"
        );
    }

    #[test]
    fn view_matrix_for_space_none_when_no_non_overlay() {
        let batches = vec![
            make_batch(3, true, (1.0, 0.0, 0.0)),
            make_batch(5, true, (10.0, 0.0, 0.0)),
        ];
        let result = ClusteredLightPass::view_matrix_for_space(&batches, 5);
        assert!(result.is_none(), "all overlay batches should return None");
    }

    /// Radius of the conservative spotlight bounding sphere (must match WGSL `spotlight_bounds_intersect_aabb`).
    fn spotlight_bounding_radius(range: f32, cos_half: f32) -> f32 {
        let sin_sq = (1.0 - cos_half * cos_half).max(0.0);
        let tan_sq = sin_sq / (cos_half * cos_half).max(1e-8);
        range * (0.25 + tan_sq).sqrt()
    }

    /// `true` when the sphere intersects the axis-aligned box (same metric as WGSL `sphere_aabb_intersect`).
    fn sphere_aabb_intersects_cpu(
        center: glam::Vec3,
        radius: f32,
        aabb_min: glam::Vec3,
        aabb_max: glam::Vec3,
    ) -> bool {
        let closest = center.clamp(aabb_min, aabb_max);
        (center - closest).length_squared() <= radius * radius
    }

    #[test]
    fn spotlight_bounding_radius_matches_base_rim_distance() {
        let range = 4.0_f32;
        let cos_half = std::f32::consts::FRAC_PI_6.cos();
        let sin_half = std::f32::consts::FRAC_PI_6.sin();
        let tan_half = sin_half / cos_half;
        let r = spotlight_bounding_radius(range, cos_half);
        let expected = range * (0.25 + tan_half * tan_half).sqrt();
        assert!((r - expected).abs() < 1e-5, "r={} expected {}", r, expected);
        let axial = range * 0.5;
        let radial = range * tan_half;
        let from_geometry = (axial * axial + radial * radial).sqrt();
        assert!(
            (r - from_geometry).abs() < 1e-4,
            "radius should reach base rim from axial midpoint"
        );
    }

    #[test]
    fn spotlight_bounding_sphere_intersects_when_axis_misses_but_cone_sweeps_box() {
        let cos_half = 0.965_925_8_f32;
        let range = 10.0_f32;
        let r = spotlight_bounding_radius(range, cos_half);
        let axis = glam::Vec3::Z;
        let apex = glam::Vec3::ZERO;
        let center = apex + axis * (range * 0.5);
        let aabb_min = glam::Vec3::new(3.0, 3.0, 2.0);
        let aabb_max = glam::Vec3::new(4.0, 4.0, 8.0);
        assert!(
            sphere_aabb_intersects_cpu(center, r, aabb_min, aabb_max),
            "conservative sphere should hit AABB that a cone-axis ray can miss"
        );
    }
}
