//! Shared PBS sampling helpers that convert texture samples into surface-space values.

#define_import_path renderide::pbs::sampling

#import renderide::core::normal_decode as nd
#import renderide::pbs::normal as pnorm
#import renderide::core::texture_sampling as ts

fn roughness_from_smoothness(smoothness: f32) -> f32 {
    return clamp(1.0 - smoothness, 0.0, 1.0);
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

fn sample_optional_two_sided_world_normal(
    enabled: bool,
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    lod_bias: f32,
    scale: f32,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    front_facing: bool,
) -> vec3<f32> {
    if (!enabled) {
        var n = normalize(world_n);
        if (!front_facing) {
            n = -n;
        }
        return n;
    }

    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    var ts_n = sample_tangent_normal(tex, samp, uv, lod_bias, scale);
    if (!front_facing) {
        ts_n.z = -ts_n.z;
    }
    return normalize(tbn * ts_n);
}

fn two_sided_geometric_normal(world_n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    let n = normalize(world_n);
    return select(-n, n, front_facing);
}

fn unpack_packed_normal_xy(xy: vec2<f32>, scale: f32) -> vec3<f32> {
    return nd::reconstruct_ts_normal_from_scaled_xy((xy * 2.0 - 1.0) * scale);
}
