// Clustered forward lighting: assigns light indices per view-space cluster (compute).
// `GpuLight` and `ClusterParams` layouts must match `crate::backend::GpuLight` and the
// `ClusterParams` struct in `clustered_light.rs` (including uniform padding).

#import renderide::cluster_math as cmath

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
    align_pad_vec3_tail: vec3<u32>,
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
    /// Base offset into the cluster storage buffers (0 for eye 0 / mono, N for eye 1 in stereo).
    cluster_offset: u32,
    /// Max row length of the world-to-view linear part. Lights are uploaded in world units, but
    /// `pos_view = view * light.position` is in scaled view space, so `light.range` must be
    /// multiplied by this factor to compare against the (also-scaled) cluster AABB.
    world_to_view_scale: f32,
}

@group(0) @binding(0) var<uniform> params: ClusterParams;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<storage, read_write> cluster_light_counts: array<u32>;
/// Packed light indices: 2 × `u16` per `u32` slot (low 16 bits = even slot, high 16 bits = odd
/// slot). Each cluster is written by a single compute thread, so no atomics are required.
@group(0) @binding(3) var<storage, read_write> cluster_light_indices: array<u32>;

const MAX_LIGHTS_PER_TILE: u32 = cmath::MAX_LIGHTS_PER_TILE;

struct TileAabb {
    min_v: vec3f,
    max_v: vec3f,
}

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
    // this AABB — producing visibly pixelated seams where lights reach the neighbor's AABB only.
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
    let local_cluster_id = global_id.x + cluster_count_x * (global_id.y + cluster_count_y * global_id.z);
    let cluster_x = global_id.x;
    let cluster_y = global_id.y;
    let cluster_z = global_id.z;

    let aabb = get_cluster_aabb(cluster_x, cluster_y, cluster_z);
    let aabb_min = aabb.min_v;
    let aabb_max = aabb.max_v;

    let global_cluster_id = params.cluster_offset + local_cluster_id;
    var count: u32 = 0u;
    var packed: u32 = 0u;
    let base_word = global_cluster_id * (MAX_LIGHTS_PER_TILE / 2u);

    for (var i = 0u; i < params.light_count; i++) {
        if count >= MAX_LIGHTS_PER_TILE {
            break;
        }
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
            let shift = (count & 1u) * 16u;
            packed |= (i & 0xFFFFu) << shift;
            count += 1u;
            if (count & 1u) == 0u {
                cluster_light_indices[base_word + (count >> 1u) - 1u] = packed;
                packed = 0u;
            }
        }
    }

    if (count & 1u) != 0u {
        cluster_light_indices[base_word + (count >> 1u)] = packed;
    }

    cluster_light_counts[global_cluster_id] = count;
}
