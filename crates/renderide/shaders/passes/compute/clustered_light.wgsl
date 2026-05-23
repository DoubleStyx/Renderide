// Clustered forward lighting: assigns light indices per view-space cluster (compute).
// `GpuLight` and `ClusterParams` layouts must match `crate::backend::GpuLight` and the
// `ClusterParams` struct in `clustered_light.rs` (including uniform padding).

#import renderide::lighting::cluster_math as cmath
#import renderide::frame::types as ft

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
    /// Base offset into the cluster storage buffers (0 for eye 0 / mono, N for eye 1 in stereo).
    cluster_offset: u32,
    /// Max row length of the world-to-view linear part. Lights are uploaded in world units, but
    /// `pos_view = view * light.position` is in scaled view space, so `light.range` must be
    /// multiplied by this factor to compare against the (also-scaled) cluster AABB.
    world_to_view_scale: f32,
}

@group(0) @binding(0) var<uniform> params: ClusterParams;
@group(0) @binding(1) var<storage, read> lights: array<ft::GpuLight>;
/// Per-cluster compact-index ranges: `.x` is offset in `cluster_light_indices`, `.y` is count.
@group(0) @binding(2) var<storage, read_write> cluster_light_ranges: array<vec2<u32>>;
/// Compact light indices. Each cluster is written by one compute thread, so no atomics are required.
@group(0) @binding(3) var<storage, read_write> cluster_light_indices: array<u32>;

struct TileAabb {
    min_v: vec3f,
    max_v: vec3f,
}

const SPOT_CULL_MIN_COS_HALF: f32 = 0.9999619;
const SPOT_CULL_WIDE_COS_HALF: f32 = 0.5;
const SPOT_CULL_DISTANCE_EPSILON: f32 = 0.00001;

fn ndc_to_view(ndc: vec3f) -> vec3f {
    let clip = params.inv_proj * vec4f(ndc.x, ndc.y, ndc.z, 1.0);
    if abs(clip.w) <= 1e-8 {
        return clip.xyz;
    }
    return clip.xyz / clip.w;
}

fn ndc_z_from_view_z(view_z: f32) -> f32 {
    let clip = params.proj * vec4f(0.0, 0.0, view_z, 1.0);
    return clip.z / clip.w;
}

fn view_point_at_ndc_xy_and_z(ndc_xy: vec2f, view_z: f32) -> vec3f {
    return ndc_to_view(vec3f(ndc_xy.x, ndc_xy.y, ndc_z_from_view_z(view_z)));
}

fn get_cluster_aabb(cluster_x: u32, cluster_y: u32, cluster_z: u32) -> TileAabb {
    let w = params.viewport_width;
    let h = params.viewport_height;
    let z_bounds =
        cmath::cluster_z_depth_bounds(cluster_z, params.cluster_count_z, params.near_clip, params.far_clip);
    let tile_near = -z_bounds.x;
    let tile_far = -z_bounds.y;

    // Use integer-pixel tile bounds (no 0.5 inset) so the AABB covers the exact pixel range that
    // `cluster_xy_from_frag` in `pbs_cluster.wgsl` assigns to this tile. A prior 0.5-pixel inset on
    // both edges left a 1-pixel-wide band of fragments that mapped to this tile but fell outside
    // this AABB -- producing visibly pixelated seams where lights reach the neighbor's AABB only.
    let px_min = f32(cluster_x * params.tile_size);
    let px_max = min(f32((cluster_x + 1u) * params.tile_size), w);
    let py_min = f32(cluster_y * params.tile_size);
    let py_max = min(f32((cluster_y + 1u) * params.tile_size), h);
    let ndc_left = 2.0 * px_min / w - 1.0;
    let ndc_right = 2.0 * px_max / w - 1.0;
    let ndc_top = 1.0 - 2.0 * py_min / h;
    let ndc_bottom = 1.0 - 2.0 * py_max / h;

    let ndc_bl = vec2f(ndc_left, ndc_bottom);
    let ndc_br = vec2f(ndc_right, ndc_bottom);
    let ndc_tl = vec2f(ndc_left, ndc_top);
    let ndc_tr = vec2f(ndc_right, ndc_top);

    let p_near_bl = view_point_at_ndc_xy_and_z(ndc_bl, tile_near);
    let p_near_br = view_point_at_ndc_xy_and_z(ndc_br, tile_near);
    let p_near_tl = view_point_at_ndc_xy_and_z(ndc_tl, tile_near);
    let p_near_tr = view_point_at_ndc_xy_and_z(ndc_tr, tile_near);
    let p_far_bl = view_point_at_ndc_xy_and_z(ndc_bl, tile_far);
    let p_far_br = view_point_at_ndc_xy_and_z(ndc_br, tile_far);
    let p_far_tl = view_point_at_ndc_xy_and_z(ndc_tl, tile_far);
    let p_far_tr = view_point_at_ndc_xy_and_z(ndc_tr, tile_far);

    var min_v = min(min(min(p_near_bl, p_near_br), min(p_near_tl, p_near_tr)), min(min(p_far_bl, p_far_br), min(p_far_tl, p_far_tr)));
    var max_v = max(max(max(p_near_bl, p_near_br), max(p_near_tl, p_near_tr)), max(max(p_far_bl, p_far_br), max(p_far_tl, p_far_tr)));
    if cluster_z == 0u {
        let camera_clip = params.proj * vec4f(0.0, 0.0, 0.0, 1.0);
        if abs(camera_clip.w) > 1e-8 {
            let p_zero_bl = view_point_at_ndc_xy_and_z(ndc_bl, 0.0);
            let p_zero_br = view_point_at_ndc_xy_and_z(ndc_br, 0.0);
            let p_zero_tl = view_point_at_ndc_xy_and_z(ndc_tl, 0.0);
            let p_zero_tr = view_point_at_ndc_xy_and_z(ndc_tr, 0.0);
            min_v = min(min_v, min(min(p_zero_bl, p_zero_br), min(p_zero_tl, p_zero_tr)));
            max_v = max(max_v, max(max(p_zero_bl, p_zero_br), max(p_zero_tl, p_zero_tr)));
        } else {
            min_v = min(min_v, vec3f(0.0));
            max_v = max(max_v, vec3f(0.0));
        }
    }
    let extent = max_v - min_v;
    let max_extent = max(max(max(extent.x, extent.y), extent.z), 1.0);
    let pad = max_extent * cmath::CLUSTER_BOUNDARY_EPSILON;
    return TileAabb(min_v - vec3f(pad), max_v + vec3f(pad));
}

fn sphere_aabb_intersect(center: vec3f, radius: f32, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    let closest = clamp(center, aabb_min, aabb_max);
    let d = center - closest;
    return dot(d, d) <= radius * radius;
}

fn aabb_bounding_sphere_center(aabb_min: vec3f, aabb_max: vec3f) -> vec3f {
    return (aabb_min + aabb_max) * 0.5;
}

fn aabb_bounding_sphere_radius(aabb_min: vec3f, aabb_max: vec3f, center: vec3f) -> f32 {
    return length(aabb_max - center);
}

fn spotlight_cone_intersects_sphere(
    apex: vec3f,
    axis_n: vec3f,
    cos_half: f32,
    range: f32,
    sphere_center: vec3f,
    sphere_radius: f32,
) -> bool {
    let clamped_cos_half = clamp(cos_half, 0.0, SPOT_CULL_MIN_COS_HALF);
    if clamped_cos_half <= SPOT_CULL_WIDE_COS_HALF {
        return true;
    }

    let sin_half = sqrt(max(0.0, 1.0 - clamped_cos_half * clamped_cos_half));
    let offset = sphere_center - apex;
    let axis_dist = dot(offset, axis_n);
    if axis_dist < -sphere_radius || axis_dist > range + sphere_radius {
        return false;
    }

    let offset_len_sq = dot(offset, offset);
    let lateral_len = sqrt(max(0.0, offset_len_sq - axis_dist * axis_dist));
    let closest_cone_distance = clamped_cos_half * lateral_len - axis_dist * sin_half;
    return closest_cone_distance <= sphere_radius + SPOT_CULL_DISTANCE_EPSILON;
}

fn spotlight_bounds_intersect_aabb(apex: vec3f, axis: vec3f, cos_half: f32, range: f32, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    if !sphere_aabb_intersect(apex, range, aabb_min, aabb_max) {
        return false;
    }

    let center = aabb_bounding_sphere_center(aabb_min, aabb_max);
    let radius = aabb_bounding_sphere_radius(aabb_min, aabb_max, center);
    return spotlight_cone_intersects_sphere(apex, axis, cos_half, range, center, radius);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let cluster_count_x = params.cluster_count_x;
    let cluster_count_y = params.cluster_count_y;
    let cluster_count_z = params.cluster_count_z;
    if global_id.x >= cluster_count_x || global_id.y >= cluster_count_y || global_id.z >= cluster_count_z {
        return;
    }
    let local_cluster_id = global_id.x + cluster_count_x * (global_id.y + cluster_count_y * global_id.z);
    let cluster_x = global_id.x;
    let cluster_y = global_id.y;
    let cluster_z = global_id.z;

    let aabb = get_cluster_aabb(cluster_x, cluster_y, cluster_z);
    let aabb_min = aabb.min_v;
    let aabb_max = aabb.max_v;

    let global_cluster_id = params.cluster_offset + local_cluster_id;
    let base_index = global_cluster_id * params.light_count;
    var count: u32 = 0u;

    for (var i = 0u; i < params.light_count; i++) {
        let light = lights[i];
        let pos_view = (params.view * vec4f(light.position.x, light.position.y, light.position.z, 1.0)).xyz;
        let dir_view = (params.view * vec4f(light.direction.x, light.direction.y, light.direction.z, 0.0)).xyz;

        var intersects = false;
        // `light.range` is in world units; `pos_view` and the cluster AABB are in scaled view
        // space. Multiply by `world_to_view_scale` (CPU-computed max row length of the
        // world-to-view linear part) so the sphere/spot bounds are in matching units. Without
        // this, a player avatar with non-unit scale (e.g. 0.01) culls lights with a radius that
        // is `1/scale` too small in view space, dropping lights from clusters that should
        // contain them and producing tile-shaped dark seams in the lit image.
        let cull_range = max(light.range * params.world_to_view_scale, 0.0);
        if light.light_type == 0u {
            intersects = sphere_aabb_intersect(pos_view, cull_range, aabb_min, aabb_max);
        } else if light.light_type == 1u {
            intersects = true;
        } else {
            let dir_len_sq = dot(dir_view, dir_view);
            let axis = select(
                vec3f(0.0, 0.0, 1.0),
                dir_view * inverseSqrt(dir_len_sq),
                dir_len_sq > 1e-16
            );
            intersects = spotlight_bounds_intersect_aabb(pos_view, axis, light.spot_cos_half_angle, cull_range, aabb_min, aabb_max);
        }

        if intersects {
            cluster_light_indices[base_index + count] = i;
            count += 1u;
        }
    }

    cluster_light_ranges[global_cluster_id] = vec2<u32>(base_index, count);
}
