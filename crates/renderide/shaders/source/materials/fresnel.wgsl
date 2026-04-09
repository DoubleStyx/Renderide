//! Fresnel (`Shader "Fresnel"`): blends near/far colors from view-angle Fresnel and optional normal/mask textures.
//!
//! Keyword-style float fields mirror Unity `#pragma multi_compile` values:
//! `_POLARUV`, `_NORMALMAP`, `_MASK_TEXTURE_MUL`, `_MASK_TEXTURE_CLIP`,
//! `_MUL_ALPHA_INTENSITY`.

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf

struct FresnelMaterial {
    _FarColor: vec4<f32>,
    _NearColor: vec4<f32>,
    _FarTex_ST: vec4<f32>,
    _NearTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _Exp: f32,
    _GammaCurve: f32,
    _NormalScale: f32,
    _Cutoff: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _PolarPow: f32,
    _POLARUV: f32,
    _NORMALMAP: f32,
    _MASK_TEXTURE_MUL: f32,
    _MASK_TEXTURE_CLIP: f32,
    _MUL_ALPHA_INTENSITY: f32,
}

@group(1) @binding(0) var<uniform> mat: FresnelMaterial;
@group(1) @binding(1) var _FarTex: texture_2d<f32>;
@group(1) @binding(2) var _FarTex_sampler: sampler;
@group(1) @binding(3) var _NearTex: texture_2d<f32>;
@group(1) @binding(4) var _NearTex_sampler: sampler;
@group(1) @binding(5) var _NormalMap: texture_2d<f32>;
@group(1) @binding(6) var _NormalMap_sampler: sampler;
@group(1) @binding(7) var _MaskTex: texture_2d<f32>;
@group(1) @binding(8) var _MaskTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let world_p = pd::draw.model * vec4<f32>(pos.xyz, 1.0);
    let wn = normalize((pd::draw.model * vec4<f32>(n.xyz, 0.0)).xyz);
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
    out.world_pos = world_p.xyz;
    out.world_n = wn;
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

fn decode_ts_normal(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    let nm_xy = (raw.xy * 2.0 - 1.0) * scale;
    let z = max(sqrt(max(1.0 - dot(nm_xy, nm_xy), 0.0)), 1e-6);
    return normalize(vec3<f32>(nm_xy, z));
}

fn sample_color(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, st: vec4<f32>) -> vec4<f32> {
    let use_polar = mat._POLARUV > 0.99;
    let sample_uv = select(apply_st(uv, st), apply_st(polar_uv(uv, mat._PolarPow), st), use_polar);
    return textureSample(tex, samp, sample_uv);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var n = normalize(in.world_n);
    if (mat._NORMALMAP > 0.99) {
        let uv_n = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
        let tbn = brdf::orthonormal_tbn(n);
        let ts_n = decode_ts_normal(
            textureSample(_NormalMap, _NormalMap_sampler, uv_n).xyz,
            mat._NormalScale,
        );
        n = normalize(tbn * ts_n);
    }

    let view_dir = normalize(rg::frame.camera_world_pos.xyz - in.world_pos);
    var fres = pow(1.0 - abs(dot(n, view_dir)), max(mat._Exp, 1e-4));
    fres = pow(clamp(fres, 0.0, 1.0), max(mat._GammaCurve, 1e-4));

    let far_color = mat._FarColor * sample_color(_FarTex, _FarTex_sampler, in.uv, mat._FarTex_ST);
    let near_color =
        mat._NearColor * sample_color(_NearTex, _NearTex_sampler, in.uv, mat._NearTex_ST);

    var color = mix(near_color, far_color, clamp(fres, 0.0, 1.0));

    if (mat._MASK_TEXTURE_MUL > 0.99 || mat._MASK_TEXTURE_CLIP > 0.99) {
        let uv_mask = apply_st(in.uv, mat._MaskTex_ST);
        let mask = textureSample(_MaskTex, _MaskTex_sampler, uv_mask);
        let mul = (mask.r + mask.g + mask.b) * 0.33333334 * mask.a;

        if (mat._MASK_TEXTURE_MUL > 0.99) {
            color.a = color.a * mul;
        }
        if (mat._MASK_TEXTURE_CLIP > 0.99 && mul <= mat._Cutoff) {
            discard;
        }
    }

    if (!(mat._MASK_TEXTURE_CLIP > 0.99) && mat._Cutoff > 0.0 && mat._Cutoff < 1.0 && color.a <= mat._Cutoff) {
        discard;
    }

    if (mat._MUL_ALPHA_INTENSITY > 0.99) {
        let lum = (color.r + color.g + color.b) * 0.33333334;
        color.a = color.a * lum * lum;
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
