//! Unity surface shader `Shader "Custom/TestBlend"`: metallic Standard lighting that lerps
//! between two albedo textures and clips against `_CutOff`.
//!
//! No `#pragma multi_compile` user keywords on this shader; `_RenderideVariantBits` is
//! reserved for layout consistency with the rest of the embedded materials and is never read.

//#texture_default _MainTex white
//#texture_default _MainTex2 white

#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu

struct TestBlendMaterial {
    _Color: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex2_ST: vec4<f32>,
    _Glossiness: f32,
    _Metallic: f32,
    _Lerp: f32,
    _CutOff: f32,
    _RenderideVariantBits: u32,
    _pad0: vec3<u32>,
}

@group(1) @binding(0) var<uniform> mat: TestBlendMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _MainTex2: texture_2d<f32>;
@group(1) @binding(4) var _MainTex2_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
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
    let world_p = mv::world_position(d, pos);
    let wn = mv::world_normal(d, n);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
#ifdef MULTIVIEW
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
#else
    out.view_layer = mv::packed_view_layer(instance_index, 0u);
#endif
    return out;
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    uv0: vec2<f32>,
    view_layer: u32,
) -> vec4<f32> {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);
    let uv_main2 = uvu::apply_st(uv0, mat._MainTex2_ST);
    let c1 = textureSample(_MainTex, _MainTex_sampler, uv_main);
    let c2 = textureSample(_MainTex2, _MainTex2_sampler, uv_main2);
    let c = mix(c1, c2, mat._Lerp);

    if (c.a < mat._CutOff) {
        discard;
    }

    let base_color = c.rgb;
    // Unity's Custom/TestBlend is an opaque surface shader (no `o.Alpha = ...`); emit a
    // fixed 1.0 alpha rather than propagating the texture/tint alpha through the output.
    let metallic = clamp(mat._Metallic, 0.0, 1.0);
    let smoothness = clamp(mat._Glossiness, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);
    let n = normalize(world_n);
    let surface = psurf::metallic(
        base_color,
        1.0,
        metallic,
        roughness,
        1.0,
        n,
        vec3<f32>(0.0),
    );
    // Touch the renderer-reserved uniform so naga-oil keeps the binding live across import pruning.
    let touch = f32(mat._RenderideVariantBits) * 0.0;
    return vec4<f32>(
        plight::shade_metallic_clustered(
            frag_xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ) + vec3<f32>(touch),
        1.0,
    );
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, uv0, view_layer);
}
