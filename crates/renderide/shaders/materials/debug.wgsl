//! Debug (`Shader "Debug"`): visualizes raw mesh vertex streams as saturated RGB.
//!
//! Froox variant bits populate `_RenderideVariantBits`; the material decodes this shader's
//! `_POSITION` / `_COLOR` / `_NORMAL` / `_UV*` selector keywords locally.

//#mat_default _Scale vec4 1.0 1.0 1.0 0.0
//#mat_default _Offset vec4 0.0 0.0 0.0 0.0

#import renderide::draw::per_draw as pd
#import renderide::frame::globals as rg
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::core::math as rmath

struct DebugMaterial {
    _Offset: vec4<f32>,
    _Scale: vec4<f32>,
    _RenderideVariantBits: u32,
    _pad0: vec3<u32>,
}

const DEBUG_KW_BITANGENT: u32 = 1u << 0u;
const DEBUG_KW_COLOR: u32 = 1u << 1u;
const DEBUG_KW_COLOR_ALPHA: u32 = 1u << 2u;
const DEBUG_KW_NORMAL: u32 = 1u << 3u;
const DEBUG_KW_NORMALIZE: u32 = 1u << 4u;
const DEBUG_KW_POSITION: u32 = 1u << 5u;
const DEBUG_KW_TANGENT: u32 = 1u << 6u;
const DEBUG_KW_TANGENT4: u32 = 1u << 7u;
const DEBUG_KW_UV0: u32 = 1u << 8u;
const DEBUG_KW_UV1: u32 = 1u << 9u;
const DEBUG_KW_UV2: u32 = 1u << 10u;
const DEBUG_KW_UV3: u32 = 1u << 11u;
const DEBUG_SELECTOR_MASK: u32 =
    DEBUG_KW_COLOR |
    DEBUG_KW_COLOR_ALPHA |
    DEBUG_KW_NORMAL |
    DEBUG_KW_POSITION |
    DEBUG_KW_TANGENT |
    DEBUG_KW_TANGENT4 |
    DEBUG_KW_UV0 |
    DEBUG_KW_UV1 |
    DEBUG_KW_UV2 |
    DEBUG_KW_UV3;

@group(1) @binding(0) var<uniform> mat: DebugMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) data: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
}

fn debug_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn selector_defaults_to_position() -> bool {
    return (mat._RenderideVariantBits & DEBUG_SELECTOR_MASK) == 0u;
}

fn selected_mesh_data(
    pos: vec4<f32>,
    n: vec4<f32>,
    color: vec4<f32>,
    t: vec4<f32>,
    uv0: vec4<f32>,
    uv1: vec4<f32>,
    uv2: vec4<f32>,
    uv3: vec4<f32>,
) -> vec3<f32> {
    if (debug_kw(DEBUG_KW_POSITION) || selector_defaults_to_position()) {
        return pos.xyz;
    }
    if (debug_kw(DEBUG_KW_COLOR)) {
        return color.rgb;
    }
    if (debug_kw(DEBUG_KW_COLOR_ALPHA)) {
        return color.aaa;
    }
    if (debug_kw(DEBUG_KW_NORMAL)) {
        return n.xyz;
    }
    if (debug_kw(DEBUG_KW_TANGENT)) {
        return t.xyz;
    }
    if (debug_kw(DEBUG_KW_TANGENT4)) {
        return vec3<f32>(t.w);
    }
    if (debug_kw(DEBUG_KW_UV0)) {
        return uv0.xyz;
    }
    if (debug_kw(DEBUG_KW_UV1)) {
        return uv1.xyz;
    }
    if (debug_kw(DEBUG_KW_UV2)) {
        return uv2.xyz;
    }
    if (debug_kw(DEBUG_KW_UV3)) {
        return uv3.xyz;
    }
    return vec3<f32>(0.0);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec4<f32>,
    @location(3) color: vec4<f32>,
    @location(4) t: vec4<f32>,
    @location(5) uv1: vec4<f32>,
    @location(6) uv2: vec4<f32>,
    @location(7) uv3: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.data = selected_mesh_data(pos, n, color, t, uv0, uv1, uv2, uv3);
    out.normal = n.xyz;
    out.tangent = t;
    return out;
}

//#pass type=forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var data = in.data;
    if (debug_kw(DEBUG_KW_BITANGENT)) {
        data = cross(in.normal, in.tangent.xyz) * in.tangent.w;
    }
    if (debug_kw(DEBUG_KW_NORMALIZE)) {
        data = rmath::safe_normalize(data, vec3<f32>(0.0));
    }
    let rgb = clamp(data * mat._Scale.xyz + mat._Offset.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    return rg::retain_globals_additive(vec4<f32>(rgb, 1.0));
}
