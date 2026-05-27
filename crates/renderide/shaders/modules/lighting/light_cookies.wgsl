//! Light-cookie projection and atlas sampling helpers.

#define_import_path renderide::lighting::light_cookies

#import renderide::frame::globals as rg
#import renderide::frame::types as ft

const COOKIE_PROJECTION_EPSILON: f32 = 0.00001;

fn spot_cookie_multiplier(light: ft::GpuLight, world_pos: vec3<f32>) -> f32 {
    let tan_half = light.cookie_right_tan_half_angle.w;
    if (tan_half <= COOKIE_PROJECTION_EPSILON) {
        return 1.0;
    }
    let from_light = world_pos - light.position;
    let local = vec3<f32>(
        dot(from_light, light.cookie_right_tan_half_angle.xyz),
        dot(from_light, light.cookie_up.xyz),
        dot(from_light, light.direction),
    );
    if (local.z <= COOKIE_PROJECTION_EPSILON) {
        return 0.0;
    }
    let uv = local.xy / (local.z * tan_half) * 0.5 + vec2<f32>(0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 0.0;
    }
    return textureSample(rg::light_cookie_spot_atlas, rg::light_cookie_sampler, uv, i32(light.cookie_layer)).a;
}

fn cube_face_uv(dir: vec3<f32>) -> vec3<f32> {
    let ad = abs(dir);
    if (ad.x >= ad.y && ad.x >= ad.z) {
        if (dir.x >= 0.0) {
            return vec3<f32>(vec2<f32>(-dir.z, -dir.y) / ad.x * 0.5 + vec2<f32>(0.5), 0.0);
        }
        return vec3<f32>(vec2<f32>(dir.z, -dir.y) / ad.x * 0.5 + vec2<f32>(0.5), 1.0);
    }
    if (ad.y >= ad.z) {
        if (dir.y >= 0.0) {
            return vec3<f32>(vec2<f32>(dir.x, dir.z) / ad.y * 0.5 + vec2<f32>(0.5), 2.0);
        }
        return vec3<f32>(vec2<f32>(dir.x, -dir.z) / ad.y * 0.5 + vec2<f32>(0.5), 3.0);
    }
    if (dir.z >= 0.0) {
        return vec3<f32>(vec2<f32>(dir.x, -dir.y) / ad.z * 0.5 + vec2<f32>(0.5), 4.0);
    }
    return vec3<f32>(vec2<f32>(-dir.x, -dir.y) / ad.z * 0.5 + vec2<f32>(0.5), 5.0);
}

fn point_cookie_multiplier(light: ft::GpuLight, world_pos: vec3<f32>) -> f32 {
    let from_light = world_pos - light.position;
    let len_sq = dot(from_light, from_light);
    if (len_sq <= COOKIE_PROJECTION_EPSILON) {
        return 1.0;
    }
    let local = vec3<f32>(
        dot(from_light, light.cookie_right_tan_half_angle.xyz),
        dot(from_light, light.cookie_up.xyz),
        dot(from_light, light.direction),
    ) * inverseSqrt(len_sq);
    let face_uv = cube_face_uv(local);
    let layer = light.cookie_layer + u32(face_uv.z);
    return textureSample(rg::light_cookie_point_atlas, rg::light_cookie_sampler, face_uv.xy, i32(layer)).a;
}

fn multiplier(light: ft::GpuLight, world_pos: vec3<f32>) -> f32 {
    if (light.cookie_kind == ft::LIGHT_COOKIE_KIND_SPOT_2D) {
        return spot_cookie_multiplier(light, world_pos);
    }
    if (light.cookie_kind == ft::LIGHT_COOKIE_KIND_POINT_CUBE) {
        return point_cookie_multiplier(light, world_pos);
    }
    return 1.0;
}
