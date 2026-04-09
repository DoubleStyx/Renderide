//! Overlay Unlit (`Shader "OverlayUnlit"`): front/behind unlit layers composed in a single pass.
//!
//! Unity implements this as two passes with different depth tests (`Greater` and `LEqual`).
//! The current renderer has a single fixed forward pass, so this WGSL path approximates the effect
//! by sampling both layers and compositing `front over behind` in one fragment shader.
//!
//! Keyword-style float fields mirror Unity `#pragma multi_compile` values:
//! `_POLARUV`, `_MUL_RGB_BY_ALPHA`, `_MUL_ALPHA_INTENSITY`.

#import renderide::globals as rg
#import renderide::per_draw as pd

struct OverlayUnlitMaterial {
    _BehindColor: vec4<f32>,
    _FrontColor: vec4<f32>,
    _BehindTex_ST: vec4<f32>,
    _FrontTex_ST: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _POLARUV: f32,
    _MUL_RGB_BY_ALPHA: f32,
    _MUL_ALPHA_INTENSITY: f32,
    _pad0: f32,
}

@group(1) @binding(0) var<uniform> mat: OverlayUnlitMaterial;
@group(1) @binding(1) var _BehindTex: texture_2d<f32>;
@group(1) @binding(2) var _BehindTex_sampler: sampler;
@group(1) @binding(3) var _FrontTex: texture_2d<f32>;
@group(1) @binding(4) var _FrontTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let world_p = pd::draw.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = pd::draw.view_proj_left;
    } else {
        vp = pd::draw.view_proj_right;
    }
#else
    let vp = pd::draw.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    return out;
}

fn apply_st(uv: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st = uv * st.xy + st.zw;
    return vec2<f32>(uv_st.x, 1.0 - uv_st.y);
}

fn polar_uv(raw_uv: vec2<f32>, radius_pow: f32) -> vec2<f32> {
    let centered = raw_uv * 2.0 - 1.0;
    let angle_len = 6.28318530718;
    let radius = pow(length(centered), radius_pow);
    let angle = atan2(centered.x, centered.y) + angle_len * 0.5;
    return vec2<f32>(angle / angle_len, radius);
}

fn sample_layer(
    tex: texture_2d<f32>,
    samp: sampler,
    tint: vec4<f32>,
    uv: vec2<f32>,
    st: vec4<f32>,
) -> vec4<f32> {
    let use_polar = mat._POLARUV > 0.99;
    let sample_uv = select(apply_st(uv, st), apply_st(polar_uv(uv, mat._PolarPow), st), use_polar);
    return textureSample(tex, samp, sample_uv) * tint;
}

fn alpha_over(front: vec4<f32>, behind: vec4<f32>) -> vec4<f32> {
    let out_a = front.a + behind.a * (1.0 - front.a);
    if (out_a <= 1e-6) {
        return vec4<f32>(0.0);
    }
    let out_rgb =
        (front.rgb * front.a + behind.rgb * behind.a * (1.0 - front.a)) / out_a;
    return vec4<f32>(out_rgb, out_a);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let behind = sample_layer(
        _BehindTex,
        _BehindTex_sampler,
        mat._BehindColor,
        in.uv,
        mat._BehindTex_ST,
    );
    let front = sample_layer(
        _FrontTex,
        _FrontTex_sampler,
        mat._FrontColor,
        in.uv,
        mat._FrontTex_ST,
    );

    var color = alpha_over(front, behind);

    if (mat._Cutoff > 0.0 && mat._Cutoff < 1.0 && color.a <= mat._Cutoff) {
        discard;
    }

    if (mat._MUL_RGB_BY_ALPHA > 0.99) {
        color = vec4<f32>(color.rgb * color.a, color.a);
    }

    if (mat._MUL_ALPHA_INTENSITY > 0.99) {
        let lum = (color.r + color.g + color.b) * 0.33333334;
        color.a = color.a * lum;
    }

    var lit: u32 = 0u;
    if (rg::frame.light_count > 0u) {
        lit = rg::lights[0].light_type;
    }
    let cluster_touch =
        f32(rg::cluster_light_counts[0u] & 255u) * 1e-10 +
        f32(rg::cluster_light_indices[0u] & 255u) * 1e-10;
    return color + vec4<f32>(vec3<f32>(f32(lit) * 1e-10 + cluster_touch), 0.0);
}
