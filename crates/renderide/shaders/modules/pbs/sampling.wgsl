//! Shared PBS sampling helpers that convert texture samples into surface-space values.

#define_import_path renderide::pbs::sampling

#import renderide::normal_decode as nd
#import renderide::pbs::normal as pnorm
#import renderide::texture_sampling as ts

fn roughness_from_smoothness(smoothness: f32) -> f32 {
    return clamp(1.0 - smoothness, 0.045, 1.0);
}

fn one_minus_reflectivity_from_specular_color(specular_color: vec3<f32>) -> f32 {
    return 1.0 - max(max(specular_color.r, specular_color.g), specular_color.b);
}

fn sample_tangent_normal(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, lod_bias: f32, scale: f32) -> vec3<f32> {
    return nd::decode_ts_normal_with_placeholder_sample(ts::sample_tex_2d(tex, samp, uv, lod_bias), scale);
}

fn sample_world_normal(
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    lod_bias: f32,
    scale: f32,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
) -> vec3<f32> {
    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    return normalize(tbn * sample_tangent_normal(tex, samp, uv, lod_bias, scale));
}

fn sample_optional_world_normal(
    enabled: bool,
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    lod_bias: f32,
    scale: f32,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
) -> vec3<f32> {
    if (!enabled) {
        return normalize(world_n);
    }
    return sample_world_normal(tex, samp, uv, lod_bias, scale, world_n, world_t);
}

fn blend_detail_tangent_normal(base: vec3<f32>, detail: vec3<f32>, detail_mask: f32) -> vec3<f32> {
    return normalize(vec3<f32>(base.xy + detail.xy * detail_mask, base.z));
}

fn unpack_packed_normal_xy(xy: vec2<f32>, scale: f32) -> vec3<f32> {
    let scaled = (xy * 2.0 - 1.0) * scale;
    let z = sqrt(max(1.0 - dot(scaled, scaled), 0.0));
    return vec3<f32>(scaled, z);
}
