//! Depth-driven mesh displacement (`Shader "DepthProjection"`).
//!
//! Treats the input mesh as a UV-parameterized surface: each vertex displaces along an
//! angle-derived object-space direction by the depth sampled from `_DepthTex`. Five-tap depth
//! averaging plus a max-difference silhouette term; the fragment then
//! samples `_MainTex` and discards along silhouette edges and outside the near/far clip range.
//!
//! `DEPTH_HUE` selects hue decoding (dominant-channel hue mapped to `[0,1)`); otherwise depth
//! defaults to grayscale (`1 - r`).


//#texture_default _MainTex white
//#texture_default _DepthTex black
//#mat_default _Angle vec4 90.0 60.0 0.0 0.0
//#mat_default _DepthTo float 1.0
//#mat_default _DiscardOffset float 0.01
//#mat_default _DiscardThreshold float 0.01
//#mat_default _FarClip float 1.0

#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::core::uv as uvu
#import renderide::material::variant_bits as vb

struct DepthProjectionMaterial {
    _MainTex_ST: vec4<f32>,
    _DepthTex_ST: vec4<f32>,
    _Angle: vec4<f32>,
    _DepthFrom: f32,
    _DepthTo: f32,
    _NearClip: f32,
    _FarClip: f32,
    _DiscardThreshold: f32,
    _DiscardOffset: f32,
    _RenderideVariantBits: u32,
    _pad0: f32,
}

const DEPTHPROJECTION_KW_DEPTH_GRAYSCALE: u32 = 1u << 0u;
const DEPTHPROJECTION_KW_DEPTH_HUE: u32 = 1u << 1u;

@group(1) @binding(0) var<uniform> mat: DepthProjectionMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _DepthTex: texture_2d<f32>;
@group(1) @binding(4) var _DepthTex_sampler: sampler;

const PI: f32 = 3.1415;

fn kw_DEPTH_HUE() -> bool {
    return vb::enabled(mat._RenderideVariantBits, DEPTHPROJECTION_KW_DEPTH_HUE);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) norm_depth: f32,
    @location(2) diff: f32,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn rgb_to_hue(c: vec3<f32>) -> f32 {
    let cmax = max(max(c.r, c.g), c.b);
    let cmin = min(min(c.r, c.g), c.b);
    let delta = cmax - cmin;
    if (delta <= 0.0) {
        return 1.0;
    }
    if (cmax == c.r) {
        return (((c.g - c.b) / delta) % 6.0) / 6.0;
    }
    if (cmax == c.g) {
        return ((c.b - c.r) / delta + 2.0) / 6.0;
    }
    return ((c.r - c.g) / delta + 4.0) / 6.0;
}

fn sample_depth_at(uv: vec2<f32>) -> f32 {
    let suv = uvu::apply_st(uv, mat._DepthTex_ST);
    let c = textureSampleLevel(_DepthTex, _DepthTex_sampler, suv, 0.0);
    if (kw_DEPTH_HUE()) {
        return rgb_to_hue(c.rgb);
    }
    return 1.0 - c.r;
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
    let layer = view_idx;
#else
    let vp = mv::select_view_proj(d, 0u);
    let layer = 0u;
#endif

    let center = sample_depth_at(uv0);
    let surround = vec4<f32>(
        sample_depth_at(uv0 + vec2<f32>( mat._DiscardOffset, 0.0)),
        sample_depth_at(uv0 + vec2<f32>( 0.0,  mat._DiscardOffset)),
        sample_depth_at(uv0 - vec2<f32>( mat._DiscardOffset, 0.0)),
        sample_depth_at(uv0 - vec2<f32>( 0.0,  mat._DiscardOffset)),
    );

    var diff = abs(center - surround.x);
    diff = max(diff, abs(center - surround.y));
    diff = max(diff, abs(center - surround.z));
    diff = max(diff, abs(center - surround.w));

    let avg = (center + surround.x + surround.y + surround.z + surround.w) / 5.0;
    let log_denom = log(max(mat._DepthTo + 1.0, 1.0 + 1e-6));
    var depth = log(avg + 1.0) / log_denom * avg;
    depth = (mat._DepthTo - mat._DepthFrom) * depth + mat._DepthFrom;

    let angle_xy = (uv0 - vec2<f32>(0.5)) * tan(mat._Angle.xy * (PI / 180.0));
    let displaced = vec4<f32>(angle_xy * depth, depth, 1.0);
    let world_p = mv::world_position(d, displaced);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv0;
    out.norm_depth = avg;
    out.diff = diff;
    out.view_layer = layer;
    return out;
}

//#pass forward_two_sided
@fragment
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    if (vout.diff > mat._DiscardThreshold || vout.norm_depth < mat._NearClip || vout.norm_depth > mat._FarClip) {
        discard;
    }
    let main_uv = uvu::apply_st(vout.uv, mat._MainTex_ST);
    let col = textureSample(_MainTex, _MainTex_sampler, main_uv);
    return rg::retain_globals_additive(col);
}
