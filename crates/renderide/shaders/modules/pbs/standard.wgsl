//! Shared Unity Standard PBS sampling helpers.

#define_import_path renderide::pbs::standard

#import renderide::core::normal_decode as nd
#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu
#import renderide::pbs::normal as pnorm
#import renderide::pbs::parallax as ppar
#import renderide::pbs::sampling as psamp

fn standard_alpha(color_alpha: f32, texture_alpha: f32, smoothness_from_albedo_alpha: bool) -> f32 {
    if (smoothness_from_albedo_alpha) {
        return color_alpha;
    }
    return color_alpha * texture_alpha;
}

fn clip_standard_alpha(alpha: f32, cutoff: f32, enabled: bool) {
    if (enabled && alpha <= cutoff) {
        discard;
    }
}

fn apply_parallax(
    uv: vec2<f32>,
    enabled: bool,
    parallax: f32,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    view_layer: u32,
    parallax_map: texture_2d<f32>,
    parallax_sampler: sampler,
    parallax_lod_bias: f32,
) -> vec2<f32> {
    if (!enabled) {
        return uv;
    }
    let h = ts::sample_tex_2d(parallax_map, parallax_sampler, uv, parallax_lod_bias).g;
    return uv + ppar::unity_parallax_offset(h, parallax, world_pos, world_n, world_t, view_layer);
}

fn detail_uv(uv0: vec2<f32>, uv1: vec2<f32>, uv_sec: f32, detail_st: vec4<f32>) -> vec2<f32> {
    return uvu::apply_st(select(uv0, uv1, uv_sec >= 0.5), detail_st);
}

fn sample_detail_mask(
    enabled: bool,
    detail_mask_tex: texture_2d<f32>,
    detail_mask_sampler: sampler,
    uv_main: vec2<f32>,
    lod_bias: f32,
) -> f32 {
    if (!enabled) {
        return 0.0;
    }
    return ts::sample_tex_2d(detail_mask_tex, detail_mask_sampler, uv_main, lod_bias).a;
}

fn apply_detail_albedo(
    base_color: vec3<f32>,
    enabled: bool,
    detail_mask: f32,
    detail_tex: texture_2d<f32>,
    detail_sampler: sampler,
    uv_detail: vec2<f32>,
    lod_bias: f32,
) -> vec3<f32> {
    if (!enabled) {
        return base_color;
    }
    let detail = ts::sample_tex_2d(detail_tex, detail_sampler, uv_detail, lod_bias).rgb;
    return base_color * mix(vec3<f32>(1.0), detail * 2.0, detail_mask);
}

fn sample_world_normal(
    normal_enabled: bool,
    detail_enabled: bool,
    detail_mask: f32,
    normal_tex: texture_2d<f32>,
    normal_sampler: sampler,
    uv_main: vec2<f32>,
    normal_lod_bias: f32,
    normal_scale: f32,
    detail_normal_tex: texture_2d<f32>,
    detail_normal_sampler: sampler,
    uv_detail: vec2<f32>,
    detail_normal_lod_bias: f32,
    detail_normal_scale: f32,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    fallback_normal: vec3<f32>,
) -> vec3<f32> {
    if (!normal_enabled) {
        return fallback_normal;
    }

    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    var ts_n = nd::decode_ts_normal_with_placeholder_sample(
        ts::sample_tex_2d(normal_tex, normal_sampler, uv_main, normal_lod_bias),
        normal_scale,
    );

    if (detail_enabled && detail_mask > 0.001) {
        let ts_detail = nd::decode_ts_normal_with_placeholder_sample(
            ts::sample_tex_2d(detail_normal_tex, detail_normal_sampler, uv_detail, detail_normal_lod_bias),
            detail_normal_scale,
        );
        ts_n = normalize(vec3<f32>(ts_n.xy + ts_detail.xy * detail_mask, ts_n.z));
    }

    return normalize(tbn * ts_n);
}

fn roughness_from_smoothness(smoothness: f32) -> f32 {
    return psamp::roughness_from_smoothness(smoothness);
}

fn occlusion_from_sample(sample: f32, strength: f32) -> f32 {
    return mix(1.0, sample, clamp(strength, 0.0, 1.0));
}

fn sample_emission(
    enabled: bool,
    emission_color: vec3<f32>,
    emission_tex: texture_2d<f32>,
    emission_sampler: sampler,
    uv_main: vec2<f32>,
    lod_bias: f32,
) -> vec3<f32> {
    if (!enabled) {
        return vec3<f32>(0.0);
    }
    return ts::sample_tex_2d(emission_tex, emission_sampler, uv_main, lod_bias).rgb * emission_color;
}
