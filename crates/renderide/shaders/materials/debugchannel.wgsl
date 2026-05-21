//! Debug Channel (`Shader "Unlit/DebugChannel"`): visualizes one texture channel as grayscale.
//!
//! Froox variant bits populate `_RenderideVariantBits`; `_R`, `_G`, `_B`, and `_A` select the
//! sampled channel.

//#texture_default _MainTex white
//#mat_default _MainTex_LodBias float 0.0

#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu
#import renderide::frame::globals as rg
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv

struct DebugChannelMaterial {
    _MainTex_ST: vec4<f32>,
    _RenderideVariantBits: u32,
    _MainTex_LodBias: f32,
    _pad0: vec2<f32>,
}

const DEBUGCHANNEL_KW_A: u32 = 1u << 0u;
const DEBUGCHANNEL_KW_B: u32 = 1u << 1u;
const DEBUGCHANNEL_KW_G: u32 = 1u << 2u;
const DEBUGCHANNEL_KW_R: u32 = 1u << 3u;
const DEBUGCHANNEL_SELECTOR_MASK: u32 =
    DEBUGCHANNEL_KW_A |
    DEBUGCHANNEL_KW_B |
    DEBUGCHANNEL_KW_G |
    DEBUGCHANNEL_KW_R;

@group(1) @binding(0) var<uniform> mat: DebugChannelMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;

fn debugchannel_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn selector_defaults_to_r() -> bool {
    return (mat._RenderideVariantBits & DEBUGCHANNEL_SELECTOR_MASK) == 0u;
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
) -> mv::UvVertexOutput {
    let material_uv = uvu::apply_st(uv, mat._MainTex_ST);
#ifdef MULTIVIEW
    return mv::uv_vertex_main(instance_index, view_idx, pos, material_uv);
#else
    return mv::uv_vertex_main(instance_index, 0u, pos, material_uv);
#endif
}

//#pass type=forward
@fragment
fn fs_main(in: mv::UvVertexOutput) -> @location(0) vec4<f32> {
    let col = ts::sample_tex_2d(_MainTex, _MainTex_sampler, in.uv, mat._MainTex_LodBias);
    var result = vec4<f32>(0.0);
    if (debugchannel_kw(DEBUGCHANNEL_KW_R) || selector_defaults_to_r()) {
        result = vec4<f32>(vec3<f32>(col.r), 1.0);
    } else if (debugchannel_kw(DEBUGCHANNEL_KW_G)) {
        result = vec4<f32>(vec3<f32>(col.g), 1.0);
    } else if (debugchannel_kw(DEBUGCHANNEL_KW_B)) {
        result = vec4<f32>(vec3<f32>(col.b), 1.0);
    } else if (debugchannel_kw(DEBUGCHANNEL_KW_A)) {
        result = vec4<f32>(vec3<f32>(col.a), 1.0);
    }
    return rg::retain_globals_additive(result);
}
