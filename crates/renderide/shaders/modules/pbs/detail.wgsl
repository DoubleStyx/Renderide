
#define_import_path renderide::pbs::detail

#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu

/// Unity's linear-space detail-albedo x2 multiplier (`unity_ColorSpaceDouble.rgb`).
const COLOR_SPACE_DOUBLE: f32 = 4.59479380;

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
    return base_color * mix(vec3<f32>(1.0), detail * COLOR_SPACE_DOUBLE, detail_mask);
}
