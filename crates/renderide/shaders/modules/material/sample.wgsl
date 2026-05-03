//! Shared material texture sampling and UV transform helpers.

#define_import_path renderide::material::sample

#import renderide::alpha_clip_sample as acs
#import renderide::texture_sampling as ts
#import renderide::uv_utils as uvu

fn sample_uv(raw_uv: vec2<f32>, st: vec4<f32>, polar_power: f32, polar_enabled: bool) -> vec2<f32> {
    let selected_uv = select(raw_uv, uvu::polar_uv(raw_uv, polar_power), polar_enabled);
    return uvu::apply_st(selected_uv, st);
}

fn sample_rgba(
    tex: texture_2d<f32>,
    samp: sampler,
    raw_uv: vec2<f32>,
    st: vec4<f32>,
    lod_bias: f32,
    polar_power: f32,
    polar_enabled: bool,
) -> vec4<f32> {
    return ts::sample_tex_2d(tex, samp, sample_uv(raw_uv, st, polar_power, polar_enabled), lod_bias);
}

fn sample_rgba_lod0(
    tex: texture_2d<f32>,
    samp: sampler,
    raw_uv: vec2<f32>,
    st: vec4<f32>,
    polar_power: f32,
    polar_enabled: bool,
) -> vec4<f32> {
    return acs::texture_rgba_base_mip(tex, samp, sample_uv(raw_uv, st, polar_power, polar_enabled));
}
