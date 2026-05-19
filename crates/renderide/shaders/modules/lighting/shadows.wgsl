//! Realtime shadow-map sampling for clustered raster lights.

#define_import_path renderide::lighting::shadows

#import renderide::frame::globals as rg
#import renderide::frame::types as ft

const SHADOW_VIEW_NONE: u32 = 0xffffffffu;

fn shadow_project(view: ft::GpuShadowView, world_pos: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let biased_world = world_pos + normal * view.params.y;
    let clip = view.view_proj * vec4<f32>(biased_world, 1.0);
    if (abs(clip.w) <= 1e-6) {
        return vec4<f32>(-1.0);
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec4<f32>(uv, ndc.z, 1.0);
}

fn shadow_coord_inside(projected: vec4<f32>) -> bool {
    return projected.w > 0.0
        && all(projected.xy >= vec2<f32>(0.0))
        && all(projected.xy <= vec2<f32>(1.0))
        && projected.z >= 0.0
        && projected.z <= 1.0;
}

fn sample_shadow_compare(view_index: u32, uv: vec2<f32>, depth: f32, texel: f32, soft: bool) -> f32 {
    let layer = i32(view_index);
    let compare_depth = clamp(depth, 0.0, 1.0);
    if (!soft) {
        return textureSampleCompare(rg::shadow_maps, rg::shadow_sampler, uv, layer, compare_depth);
    }
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum = sum + textureSampleCompare(
                rg::shadow_maps,
                rg::shadow_sampler,
                clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0)),
                layer,
                compare_depth,
            );
        }
    }
    return sum * (1.0 / 9.0);
}

fn safe_normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(value, value);
    if (len_sq <= 1e-16) {
        return fallback;
    }
    return value * inverseSqrt(len_sq);
}

fn shadow_light_dir(light: ft::GpuLight, world_pos: vec3<f32>) -> vec3<f32> {
    if (light.light_type == 1u) {
        return safe_normalize_or(-light.direction, vec3<f32>(0.0, 1.0, 0.0));
    }
    return safe_normalize_or(light.position - world_pos, vec3<f32>(0.0, 1.0, 0.0));
}

fn receiver_depth_bias(view: ft::GpuShadowView, light: ft::GpuLight, world_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    let n = safe_normalize_or(normal, vec3<f32>(0.0, 1.0, 0.0));
    let l = shadow_light_dir(light, world_pos);
    let slope = 1.0 - clamp(abs(dot(n, l)), 0.0, 1.0);
    return view.params.x + slope * view.params.w;
}

fn shadow_visibility(light_index: u32, light: ft::GpuLight, world_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    if (light.shadow_type == 0u || light.shadow_strength <= 0.0) {
        return 1.0;
    }
    let shadow_meta = rg::shadow_lights[light_index];
    if (shadow_meta.first_view == SHADOW_VIEW_NONE || shadow_meta.view_count == 0u) {
        return 1.0;
    }
    let last_view = shadow_meta.first_view + shadow_meta.view_count;
    for (var view_index = shadow_meta.first_view; view_index < last_view; view_index = view_index + 1u) {
        let view = rg::shadow_views[view_index];
        let projected = shadow_project(view, world_pos, normal);
        if (!shadow_coord_inside(projected)) {
            continue;
        }
        let soft = light.shadow_type == 2u;
        let depth_bias = receiver_depth_bias(view, light, world_pos, normal);
        let visibility = sample_shadow_compare(view_index, projected.xy, projected.z + depth_bias, view.params.z, soft);
        return mix(1.0, visibility, clamp(light.shadow_strength, 0.0, 1.0));
    }
    return 1.0;
}
