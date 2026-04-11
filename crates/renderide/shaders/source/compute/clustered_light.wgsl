// Clustered forward lighting: assigns light indices per view-space cluster (compute).
// `GpuLight` and `ClusterParams` layouts must match `crate::backend::GpuLight` and the
// `ClusterParams` struct in `clustered_light.rs` (including uniform padding).

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
    /// Linear cluster index offset for packed stereo (`0` = left / mono, `per_eye` = right).
    cluster_write_base: u32,
    _pad_cluster: u32,
}

@group(0) @binding(0) var<uniform> params: ClusterParams;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<storage, read_write> cluster_light_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> cluster_light_indices: array<u32>;

/// Must match `cluster_gpu::MAX_LIGHTS_PER_TILE` and `pbs_cluster.wgsl`.
const MAX_LIGHTS_PER_TILE: u32 = 64u;

/// Radians; matches `SPOT_PENUMBRA_RAD` in `pbs_brdf.wgsl` so culling bounds cover the shaded penumbra.
const SPOT_PENUMBRA_RAD: f32 = 0.1;

struct TileAabb {
    min_v: vec3f,
    max_v: vec3f,
}

fn ndc_to_view(ndc: vec3f) -> vec3f {
    let clip = params.inv_proj * vec4f(ndc.x, ndc.y, ndc.z, 1.0);
    return clip.xyz / clip.w;
}

fn line_intersect_z_plane(ray_point: vec3f, z_dist: f32) -> vec3f {
    let t = z_dist / ray_point.z;
    return ray_point * t;
}

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

// Conservative circumsphere around the finite cone; `cull_cos_half` widens the cone by `SPOT_PENUMBRA_RAD`
// so the proxy covers the PBS smoothstep penumbra in `pbs_brdf.wgsl`.
fn spotlight_bounds_intersect_aabb(apex: vec3f, axis: vec3f, cos_half: f32, range: f32, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    let cull_cos_half = max(cos_half - SPOT_PENUMBRA_RAD, -1.0);
    if cull_cos_half >= 0.9999 {
        return sphere_aabb_intersect(apex, range, aabb_min, aabb_max);
    }
    let axis_n = normalize(axis);
    let sin_sq = max(0.0, 1.0 - cull_cos_half * cull_cos_half);
    let tan_sq = sin_sq / max(cull_cos_half * cull_cos_half, 1e-8);
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
    let effective_id = cluster_id + params.cluster_write_base;
    let cluster_x = global_id.x;
    let cluster_y = global_id.y;
    let cluster_z = global_id.z;

    let aabb = get_cluster_aabb(cluster_x, cluster_y, cluster_z);
    let aabb_min = aabb.min_v;
    let aabb_max = aabb.max_v;

    var count: u32 = 0u;
    let base_idx = effective_id * MAX_LIGHTS_PER_TILE;

    for (var i = 0u; i < params.light_count; i++) {
        if count >= MAX_LIGHTS_PER_TILE {
            break;
        }
        let light = lights[i];
        // Directionals apply everywhere; evaluated in fragment from `frame.directional_light_count` prefix.
        if light.light_type == 1u {
            continue;
        }
        let pos_view = (params.view * vec4f(light.position.x, light.position.y, light.position.z, 1.0)).xyz;
        let dir_view = (params.view * vec4f(light.direction.x, light.direction.y, light.direction.z, 0.0)).xyz;

        var intersects = false;
        // Point lights: sphere radius `range` is conservative vs PBS smoothstep falloff (slightly wider cull).
        if light.light_type == 0u {
            intersects = sphere_aabb_intersect(pos_view, light.range, aabb_min, aabb_max);
        } else {
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

    cluster_light_counts[effective_id] = count;
}
