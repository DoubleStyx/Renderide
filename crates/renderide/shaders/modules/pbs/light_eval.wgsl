//! Shared PBS light direction, attenuation, cookie, and shadow evaluation.

#define_import_path renderide::pbs::light_eval

#import renderide::frame::types as ft
#import renderide::lighting::birp as bl
#import renderide::lighting::light_cookies as cookies
#import renderide::lighting::shadows as shadows

/// Unity BiRP-style distance attenuation for punctual lights.
fn distance_attenuation(dist: f32, range: f32) -> f32 {
    return bl::distance_attenuation(dist, range);
}

/// Result of evaluating one punctual light at a surface point.
struct LightSample {
    /// Direction from the surface toward the light source (unit length when `attenuation > 0`).
    l: vec3<f32>,
    /// Combined direct-light boost, distance, and spot attenuation.
    attenuation: f32,
}

/// Resolves the per-light-type direction and attenuation.
fn eval_light(light: ft::GpuLight, world_pos: vec3<f32>, world_normal: vec3<f32>) -> LightSample {
    let light_pos = light.position.xyz;
    let light_dir = light.direction.xyz;
    var out: LightSample;
    if light.light_type == 0u {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        out.l = normalize(to_light);
        out.attenuation = distance_attenuation(dist, light.range);
        out.attenuation = out.attenuation * cookies::multiplier(light, world_pos);
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        out.l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        out.attenuation = bl::direct_light_scale();
        out.attenuation = out.attenuation * cookies::multiplier(light, world_pos);
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        out.l = normalize(to_light);
        let spot_atten = bl::spot_angle_attenuation(light, out.l);
        out.attenuation = spot_atten * distance_attenuation(dist, light.range);
        out.attenuation = out.attenuation * cookies::multiplier(light, world_pos);
    }
    out.attenuation = out.attenuation * shadows::visibility(light, world_pos, world_normal);
    return out;
}

/// Signed direct radiance carried by one light sample before BRDF multiplication.
fn signed_light_radiance(light: ft::GpuLight, attenuation: f32, n_dot_l: f32) -> vec3<f32> {
    return bl::light_radiance(light) * attenuation * n_dot_l;
}
