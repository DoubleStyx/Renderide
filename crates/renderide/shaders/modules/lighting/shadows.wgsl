//! Realtime shadow-map visibility for clustered direct lights.

#define_import_path renderide::lighting::shadows

#import renderide::frame::globals as rg
#import renderide::frame::types as ft

const SHADOW_UV_BORDER: f32 = 0.0005;

fn has_shadow_views(light: ft::GpuLight) -> bool {
    return light.shadow_view_count > 0u && light.shadow_strength > 0.0 && light.shadow_type != 0u;
}

fn point_face_index(light: ft::GpuLight, world_pos: vec3<f32>) -> u32 {
    let d = world_pos - light.position.xyz;
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, d.x >= 0.0);
    }
    if (a.y >= a.z) {
        return select(3u, 2u, d.y >= 0.0);
    }
    return select(5u, 4u, d.z >= 0.0);
}

fn shadow_view_kind(shadow_view: ft::GpuShadowView) -> u32 {
    return u32(max(shadow_view.light_params.x, 0.0) + 0.5);
}

fn point_shadow_compare_depth(light: ft::GpuLight, world_pos: vec3<f32>) -> f32 {
    let range = max(light.range, 0.001);
    return clamp(length(world_pos - light.position.xyz) / range, 0.0, 1.0);
}

fn projected_shadow_compare_depth(light: ft::GpuLight, shadow_view: ft::GpuShadowView, ndc: vec3<f32>) -> f32 {
    let bias = max(light.shadow_bias, shadow_view.light_params.w);
    return clamp(ndc.z - bias, 0.0, 1.0);
}

fn shadow_layer_visibility(light: ft::GpuLight, view_index: u32, world_pos: vec3<f32>) -> f32 {
    let shadow_view = rg::shadow_views[view_index];
    let clip = shadow_view.world_to_shadow * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return -1.0;
    }
    let ndc = clip.xyz / clip.w;
    let point_shadow = shadow_view_kind(shadow_view) == ft::SHADOW_VIEW_KIND_POINT;
    if (!point_shadow && (ndc.z < 0.0 || ndc.z > 1.0)) {
        return -1.0;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < SHADOW_UV_BORDER || uv.x > 1.0 - SHADOW_UV_BORDER || uv.y < SHADOW_UV_BORDER || uv.y > 1.0 - SHADOW_UV_BORDER) {
        return -1.0;
    }
    let layer = i32(shadow_view.params.x + 0.5);
    var compare_depth: f32;
    if (point_shadow) {
        compare_depth = point_shadow_compare_depth(light, world_pos);
    } else {
        compare_depth = projected_shadow_compare_depth(light, shadow_view, ndc);
    }
    return textureSampleCompare(rg::shadow_atlas, rg::shadow_sampler, uv, layer, compare_depth);
}

fn visibility(light: ft::GpuLight, world_pos: vec3<f32>) -> f32 {
    if (!has_shadow_views(light)) {
        return 1.0;
    }
    let strength = clamp(light.shadow_strength, 0.0, 1.0);
    let start = light.shadow_view_start;
    let count = light.shadow_view_count;
    if (light.light_type == 0u && count >= 6u) {
        let face = min(point_face_index(light, world_pos), count - 1u);
        let sampled = shadow_layer_visibility(light, start + face, world_pos);
        return mix(1.0, select(1.0, sampled, sampled >= 0.0), strength);
    }
    for (var i = 0u; i < count; i++) {
        let sampled = shadow_layer_visibility(light, start + i, world_pos);
        if (sampled >= 0.0) {
            return mix(1.0, sampled, strength);
        }
    }
    return 1.0;
}
