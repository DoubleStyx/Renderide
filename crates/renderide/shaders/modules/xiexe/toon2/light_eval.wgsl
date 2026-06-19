//! Light sampling helpers for Xiexe Toon 2.0 clustered lighting.

#define_import_path renderide::xiexe::toon2::light_eval

#import renderide::xiexe::toon2::base as xb
#import renderide::frame::types as ft
#import renderide::lighting::birp as bl
#import renderide::lighting::light_cookies as cookies
#import renderide::lighting::shadows as shadows

/// Resolves a single frame light into a `LightSample` (direction toward the light,
/// linear radiance, boosted energy attenuation, unboosted style visibility, directional flag).
fn sample_light(light: ft::GpuLight, world_pos: vec3<f32>, world_normal: vec3<f32>) -> xb::LightSample {
    if (light.light_type == 1u) {
        let dir_len_sq = dot(light.direction.xyz, light.direction.xyz);
        let shadow_visibility = shadows::visibility(light, world_pos, world_normal);
        return xb::LightSample(
            select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light.direction.xyz), dir_len_sq > 1e-16),
            bl::light_radiance(light) * shadow_visibility,
            bl::direct_light_scale() * cookies::multiplier(light, world_pos),
            shadow_visibility,
            true,
        );
    }

    let to_light = light.position.xyz - world_pos;
    let dist = length(to_light);
    let l = xb::safe_normalize(to_light, vec3<f32>(0.0, 1.0, 0.0));
    var visibility = bl::distance_visibility(dist, light.range);
    if (light.light_type == 2u) {
        visibility = visibility * bl::spot_angle_attenuation(light, l);
    }
    visibility = visibility * cookies::multiplier(light, world_pos);
    visibility = visibility * shadows::visibility(light, world_pos, world_normal);
    let attenuation = visibility * bl::direct_light_scale();
    return xb::LightSample(l, bl::light_radiance(light), attenuation, visibility, false);
}
