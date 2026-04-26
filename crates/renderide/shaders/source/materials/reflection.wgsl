//! Unity `Shader "Reflection"`: samples a host-provided 2D reflection texture in screen space,
//! optionally distorted by a tangent-space normal map. **Not a grab pass** — `_ReflectionTex` is a
//! regular `sampler2D`, populated by the host with whatever reflection RT (planar, cubemap-projected,
//! etc.) is available. Multi-view eye separation is handled by the renderer's per-eye render pass,
//! not the side-by-side `eyeIndex` texture-coordinate offset Unity needs for single-pass stereo.

// unity-shader-name: Reflection

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct ReflectionMaterial {
    _Color: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Cutoff: f32,
    _Distort: f32,
    _COLOR: f32,
    _NORMALMAP: f32,
    _ALPHATEST: f32,
    _MUL_ALPHA_INTENSITY: f32,
    _MUL_RGB_BY_ALPHA: f32,
}

@group(1) @binding(0) var<uniform> mat: ReflectionMaterial;
@group(1) @binding(1) var _ReflectionTex: texture_2d<f32>;
@group(1) @binding(2) var _ReflectionTex_sampler: sampler;
@group(1) @binding(3) var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) var _NormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) screen_uv: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    let clip = vp * world_p;
    var out: VertexOutput;
    out.clip_pos = clip;
    // Equivalent of Unity's ComputeNonStereoScreenPos: ((clip.xy * vec2(1, -1) + clip.w) * 0.5, w)
    // packed into xy/z so the fragment can do uv/w to get [0..1] screen UV.
    out.screen_uv = vec3<f32>(
        (clip.x + clip.w) * 0.5,
        (clip.w - clip.y) * 0.5,
        clip.w,
    );
    out.uv = uvu::apply_st(uv, mat._NormalMap_ST);
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @location(0) screen_uv: vec3<f32>,
    @location(1) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    var screen = screen_uv.xy / max(screen_uv.z, 1e-4);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let bump = nd::decode_ts_normal_with_placeholder(
            textureSample(_NormalMap, _NormalMap_sampler, uv).xyz,
            1.0,
        );
        screen = screen + bump.xy * mat._Distort;
    }
    var col = textureSample(_ReflectionTex, _ReflectionTex_sampler, screen);
    if (uvu::kw_enabled(mat._COLOR)) {
        col = col * mat._Color;
    }
    if (uvu::kw_enabled(mat._ALPHATEST) && col.a <= mat._Cutoff) {
        discard;
    }
    if (uvu::kw_enabled(mat._MUL_RGB_BY_ALPHA)) {
        col = vec4<f32>(col.rgb * col.a, col.a);
    }
    if (uvu::kw_enabled(mat._MUL_ALPHA_INTENSITY)) {
        let mulfactor = (col.r + col.g + col.b) * 0.33333334;
        col.a = col.a * mulfactor;
    }
    return rg::retain_globals_additive(col);
}
