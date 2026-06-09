//! Realtime shadow-map visibility for clustered direct lights.

#define_import_path renderide::lighting::shadows

#import renderide::frame::globals as rg
#import renderide::frame::types as ft

const SHADOW_UV_BORDER: f32 = 0.0005;
const SHADOW_TYPE_SOFT: u32 = 2u;

struct PointShadowFaceUv {
    face: u32,
    local_uv: vec2<f32>,
}

fn has_shadow_views(light: ft::GpuLight) -> bool {
    return light.shadow_view_count > 0u && light.shadow_strength > 0.0 && light.shadow_type != 0u;
}

fn point_face_index(direction: vec3<f32>) -> u32 {
    let a = abs(direction);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, direction.x >= 0.0);
    }
    if (a.y >= a.z) {
        return select(3u, 2u, direction.y >= 0.0);
    }
    return select(5u, 4u, direction.z >= 0.0);
}

fn point_face_direction(face: u32) -> vec3<f32> {
    switch (face % 6u) {
        case 0u: {
            return vec3<f32>(1.0, 0.0, 0.0);
        }
        case 1u: {
            return vec3<f32>(-1.0, 0.0, 0.0);
        }
        case 2u: {
            return vec3<f32>(0.0, 1.0, 0.0);
        }
        case 3u: {
            return vec3<f32>(0.0, -1.0, 0.0);
        }
        case 4u: {
            return vec3<f32>(0.0, 0.0, 1.0);
        }
        default: {
            return vec3<f32>(0.0, 0.0, -1.0);
        }
    }
}

fn point_face_up(face: u32) -> vec3<f32> {
    switch (face % 6u) {
        case 2u: {
            return vec3<f32>(0.0, 0.0, -1.0);
        }
        case 3u: {
            return vec3<f32>(0.0, 0.0, 1.0);
        }
        default: {
            return vec3<f32>(0.0, 1.0, 0.0);
        }
    }
}

fn point_face_right(face: u32) -> vec3<f32> {
    return cross(point_face_direction(face), point_face_up(face));
}

fn point_shadow_face_uv(direction: vec3<f32>) -> PointShadowFaceUv {
    let face = point_face_index(direction);
    let face_direction = point_face_direction(face);
    let face_up = point_face_up(face);
    let face_right = point_face_right(face);
    let denom = max(dot(direction, face_direction), 1e-6);
    let ndc = vec2<f32>(dot(direction, face_right), dot(direction, face_up)) / denom;
    let local_uv = clamp(
        vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5),
        vec2<f32>(0.0),
        vec2<f32>(1.0)
    );
    return PointShadowFaceUv(face, local_uv);
}

fn point_shadow_direction_for_face_uv(face: u32, local_uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(local_uv.x * 2.0 - 1.0, 1.0 - local_uv.y * 2.0);
    return point_face_direction(face)
        + point_face_right(face) * ndc.x
        + point_face_up(face) * ndc.y;
}

fn shadow_view_kind(shadow_view: ft::GpuShadowView) -> u32 {
    return u32(max(shadow_view.light_params.x, 0.0) + 0.5);
}

fn radial_shadow_kind(kind: u32) -> bool {
    return kind == ft::SHADOW_VIEW_KIND_POINT || kind == ft::SHADOW_VIEW_KIND_SPOT;
}

fn radial_shadow_compare_depth(light: ft::GpuLight, world_pos: vec3<f32>) -> f32 {
    let range = max(light.range, 0.001);
    return clamp(length(world_pos - light.position.xyz) / range, 0.0, 1.0);
}

fn projected_shadow_compare_depth(light: ft::GpuLight, shadow_view: ft::GpuShadowView, ndc: vec3<f32>) -> f32 {
    let bias = max(light.shadow_bias, shadow_view.light_params.w);
    return clamp(ndc.z - bias, 0.0, 1.0);
}

fn receiver_position(shadow_view: ft::GpuShadowView, world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let bias = max(shadow_view.light_params.z, 0.0);
    let normal_len_sq = dot(world_normal, world_normal);
    if (bias <= 0.0 || normal_len_sq <= 1e-12) {
        return world_pos;
    }
    return world_pos + normalize(world_normal) * bias;
}

fn shadow_uv_in_bounds(uv: vec2<f32>) -> bool {
    return uv.x >= SHADOW_UV_BORDER
        && uv.x <= 1.0 - SHADOW_UV_BORDER
        && uv.y >= SHADOW_UV_BORDER
        && uv.y <= 1.0 - SHADOW_UV_BORDER;
}

fn atlas_uv(shadow_view: ft::GpuShadowView, local_uv: vec2<f32>) -> vec2<f32> {
    return shadow_view.atlas_rect.xy + local_uv * shadow_view.atlas_rect.zw;
}

fn sample_shadow_compare(shadow_view: ft::GpuShadowView, local_uv: vec2<f32>, layer: i32, compare_depth: f32) -> f32 {
    return textureSampleCompare(
        rg::shadow_atlas,
        rg::shadow_sampler,
        atlas_uv(shadow_view, local_uv),
        layer,
        compare_depth
    );
}

fn sample_hard_shadow(shadow_view: ft::GpuShadowView, local_uv: vec2<f32>, layer: i32, compare_depth: f32) -> f32 {
    return sample_shadow_compare(shadow_view, local_uv, layer, compare_depth);
}

fn sample_soft_shadow(shadow_view: ft::GpuShadowView, local_uv: vec2<f32>, layer: i32, compare_depth: f32) -> f32 {
    let texel = max(shadow_view.params.y, 1e-6);
    var sum = 0.0;
    var count = 0.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let tap_uv = local_uv + vec2<f32>(f32(x), f32(y)) * texel;
            if (shadow_uv_in_bounds(tap_uv)) {
                sum = sum + sample_shadow_compare(shadow_view, tap_uv, layer, compare_depth);
                count = count + 1.0;
            }
        }
    }
    if (count <= 0.0) {
        return 1.0;
    }
    return sum / count;
}

fn sample_shadow_visibility(light: ft::GpuLight, shadow_view: ft::GpuShadowView, local_uv: vec2<f32>, layer: i32, compare_depth: f32) -> f32 {
    if (light.shadow_type == SHADOW_TYPE_SOFT) {
        return sample_soft_shadow(shadow_view, local_uv, layer, compare_depth);
    }
    return sample_hard_shadow(shadow_view, local_uv, layer, compare_depth);
}

fn sample_point_shadow_compare(light: ft::GpuLight, start: u32, face_uv: PointShadowFaceUv, compare_depth: f32) -> f32 {
    let face = min(face_uv.face, light.shadow_view_count - 1u);
    let shadow_view = rg::shadow_views[start + face];
    let layer = i32(shadow_view.params.x + 0.5);
    return sample_shadow_compare(shadow_view, face_uv.local_uv, layer, compare_depth);
}

fn sample_point_hard_shadow(light: ft::GpuLight, start: u32, direction: vec3<f32>, compare_depth: f32) -> f32 {
    return sample_point_shadow_compare(light, start, point_shadow_face_uv(direction), compare_depth);
}

fn sample_point_soft_shadow(light: ft::GpuLight, start: u32, direction: vec3<f32>, compare_depth: f32) -> f32 {
    let base = point_shadow_face_uv(direction);
    let base_shadow_view = rg::shadow_views[start + min(base.face, light.shadow_view_count - 1u)];
    let texel = max(base_shadow_view.params.y, 1e-6);
    var sum = 0.0;
    var count = 0.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let tap_uv = base.local_uv + vec2<f32>(f32(x), f32(y)) * texel;
            let tap_direction = point_shadow_direction_for_face_uv(base.face, tap_uv);
            sum = sum + sample_point_shadow_compare(
                light,
                start,
                point_shadow_face_uv(tap_direction),
                compare_depth
            );
            count = count + 1.0;
        }
    }
    if (count <= 0.0) {
        return 1.0;
    }
    return sum / count;
}

fn sample_point_shadow_visibility(light: ft::GpuLight, start: u32, direction: vec3<f32>, compare_depth: f32) -> f32 {
    if (light.shadow_type == SHADOW_TYPE_SOFT) {
        return sample_point_soft_shadow(light, start, direction, compare_depth);
    }
    return sample_point_hard_shadow(light, start, direction, compare_depth);
}

fn shadow_layer_visibility(light: ft::GpuLight, view_index: u32, world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    let shadow_view = rg::shadow_views[view_index];
    let biased_world_pos = receiver_position(shadow_view, world_pos, world_normal);
    let clip = shadow_view.world_to_shadow * vec4<f32>(biased_world_pos, 1.0);
    if (clip.w <= 0.0) {
        return -1.0;
    }
    let ndc = clip.xyz / clip.w;
    let radial_shadow = radial_shadow_kind(shadow_view_kind(shadow_view));
    if (!radial_shadow && (ndc.z < 0.0 || ndc.z > 1.0)) {
        return -1.0;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (!shadow_uv_in_bounds(uv)) {
        return -1.0;
    }
    let layer = i32(shadow_view.params.x + 0.5);
    var compare_depth: f32;
    if (radial_shadow) {
        compare_depth = radial_shadow_compare_depth(light, biased_world_pos);
    } else {
        compare_depth = projected_shadow_compare_depth(light, shadow_view, ndc);
    }
    return sample_shadow_visibility(light, shadow_view, uv, layer, compare_depth);
}

fn point_shadow_layer_visibility(light: ft::GpuLight, start: u32, world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    let base_shadow_view = rg::shadow_views[start];
    let biased_world_pos = receiver_position(base_shadow_view, world_pos, world_normal);
    let direction = biased_world_pos - light.position.xyz;
    if (dot(direction, direction) <= 1e-12) {
        return 1.0;
    }
    let compare_depth = radial_shadow_compare_depth(light, biased_world_pos);
    return sample_point_shadow_visibility(light, start, direction, compare_depth);
}

fn visibility(light: ft::GpuLight, world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    if (!has_shadow_views(light)) {
        return 1.0;
    }
    let strength = clamp(light.shadow_strength, 0.0, 1.0);
    let start = light.shadow_view_start;
    let count = light.shadow_view_count;
    if (light.light_type == 0u && count >= 6u) {
        let sampled = point_shadow_layer_visibility(light, start, world_pos, world_normal);
        return mix(1.0, select(1.0, sampled, sampled >= 0.0), strength);
    }
    for (var i = 0u; i < count; i++) {
        let sampled = shadow_layer_visibility(light, start + i, world_pos, world_normal);
        if (sampled >= 0.0) {
            return mix(1.0, sampled, strength);
        }
    }
    return 1.0;
}
