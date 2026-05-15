//! Unity `_ST` tiling/offset and polar UV helpers.
//!
//! Import with `#import renderide::core::uv as uvu` (do **not** use alias `uv` -- naga-oil rejects it).
//!
//! Sampled textures use Unity convention (V=0 at the bottom row of storage), matching mesh UVs
//! authored in the same convention. Material sampling therefore needs no V flip in the shader,
//! and `apply_st` is a plain `_ST` transform.

#define_import_path renderide::core::uv

fn apply_st(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    return uv_in * st.xy + st.zw;
}

fn polar_uv(raw_uv: vec2<f32>, radius_pow: f32) -> vec2<f32> {
    let centered = raw_uv * 2.0 - 1.0;
    let angle_len = 6.28318530718;
    let radius = pow(length(centered), radius_pow);
    let angle = atan2(centered.x, centered.y) + angle_len * 0.5;
    return vec2<f32>(angle / angle_len, radius);
}

struct PolarMapping {
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
}

fn transform_polar_mapping(polar: vec2<f32>, st: vec4<f32>) -> PolarMapping {
    let transformed = apply_st(polar, st);
    let coord0 = transformed;
    let coord1 = vec2<f32>(fract(abs(transformed.x) + 0.5), transformed.y);

    let ddx0 = dpdx(coord0);
    let ddy0 = dpdy(coord0);
    let ddx1 = dpdx(coord1);
    let ddy1 = dpdy(coord1);

    if (length(ddx1) + length(ddy1) < length(ddx0) + length(ddy0)) {
        return PolarMapping(transformed, ddx1, ddy1);
    }
    return PolarMapping(transformed, ddx0, ddy0);
}

fn polar_mapping(raw_uv: vec2<f32>, st: vec4<f32>, radius_pow: f32) -> PolarMapping {
    return transform_polar_mapping(polar_uv(raw_uv, radius_pow), st);
}
