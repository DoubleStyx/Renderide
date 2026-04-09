//! Unity PBS lerp (`Shader "PBSLerp"`): metallic workflow blending between two material sets.
//!
//! This uses the same clustered forward lighting path as `pbsmetallic.wgsl`, but blends two
//! albedo/normal/emission/occlusion/metallic-smoothness sets by `_Lerp` or `_LerpTex`.
//! Unity surface-shader keywords are mirrored as float uniforms:
//! `_LERPTEX`, `_ALBEDOTEX`, `_EMISSIONTEX`, `_NORMALMAP`, `_METALLICMAP`,
//! `_OCCLUSION`, `_MULTI_VALUES`, `_DUALSIDED`, `_ALPHACLIP`.

// unity-shader-name: PBSLerp

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls

struct PbsLerpMaterial {
    _Color: vec4<f32>,
    _Color1: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EmissionColor1: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex1_ST: vec4<f32>,
    _LerpTex_ST: vec4<f32>,
    _Lerp: f32,
    _NormalScale: f32,
    _NormalScale1: f32,
    _Glossiness: f32,
    _Glossiness1: f32,
    _Metallic: f32,
    _Metallic1: f32,
    _AlphaClip: f32,
    _Cull: f32,
    _LERPTEX: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _METALLICMAP: f32,
    _OCCLUSION: f32,
    _MULTI_VALUES: f32,
    _DUALSIDED: f32,
    _ALPHACLIP: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsLerpMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _MainTex1: texture_2d<f32>;
@group(1) @binding(4)  var _MainTex1_sampler: sampler;
@group(1) @binding(5)  var _LerpTex: texture_2d<f32>;
@group(1) @binding(6)  var _LerpTex_sampler: sampler;
@group(1) @binding(7)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(8)  var _NormalMap_sampler: sampler;
@group(1) @binding(9)  var _NormalMap1: texture_2d<f32>;
@group(1) @binding(10) var _NormalMap1_sampler: sampler;
@group(1) @binding(11) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(12) var _EmissionMap_sampler: sampler;
@group(1) @binding(13) var _EmissionMap1: texture_2d<f32>;
@group(1) @binding(14) var _EmissionMap1_sampler: sampler;
@group(1) @binding(15) var _Occlusion: texture_2d<f32>;
@group(1) @binding(16) var _Occlusion_sampler: sampler;
@group(1) @binding(17) var _Occlusion1: texture_2d<f32>;
@group(1) @binding(18) var _Occlusion1_sampler: sampler;
@group(1) @binding(19) var _MetallicMap: texture_2d<f32>;
@group(1) @binding(20) var _MetallicMap_sampler: sampler;
@group(1) @binding(21) var _MetallicMap1: texture_2d<f32>;
@group(1) @binding(22) var _MetallicMap1_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
}

fn apply_st(uv: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st = uv * st.xy + st.zw;
    return vec2<f32>(uv_st.x, 1.0 - uv_st.y);
}

fn kw_enabled(v: f32) -> bool {
    return v > 0.5;
}

fn decode_ts_normal(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    if (all(raw > vec3<f32>(0.99, 0.99, 0.99))) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let nm_xy = (raw.xy * 2.0 - 1.0) * scale;
    let z = max(sqrt(max(1.0 - dot(nm_xy, nm_xy), 0.0)), 1e-6);
    return normalize(vec3<f32>(nm_xy, z));
}

fn sample_normal_world(
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    world_n: vec3<f32>,
    front_facing: bool,
    lerp_factor: f32,
) -> vec3<f32> {
    if (!kw_enabled(mat._NORMALMAP)) {
        var n = normalize(world_n);
        if (kw_enabled(mat._DUALSIDED) && !front_facing) {
            n = -n;
        }
        return n;
    }

    let tbn = brdf::orthonormal_tbn(normalize(world_n));
    let ts0 = decode_ts_normal(textureSample(_NormalMap, _NormalMap_sampler, uv0).xyz, mat._NormalScale);
    let ts1 =
        decode_ts_normal(textureSample(_NormalMap1, _NormalMap1_sampler, uv1).xyz, mat._NormalScale1);
    var ts = normalize(mix(ts0, ts1, vec3<f32>(lerp_factor)));
    if (kw_enabled(mat._DUALSIDED) && !front_facing) {
        ts.z = -ts.z;
    }
    return normalize(tbn * ts);
}

fn compute_lerp_factor(uv_lerp: vec2<f32>) -> f32 {
    var l = mat._Lerp;
    if (kw_enabled(mat._LERPTEX)) {
        l = textureSample(_LerpTex, _LerpTex_sampler, uv_lerp).r;
        if (kw_enabled(mat._MULTI_VALUES)) {
            l = l * mat._Lerp;
        }
    }
    return clamp(l, 0.0, 1.0);
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
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
    out.uv0 = uv0;
    return out;
}

@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0_raw: vec2<f32>,
) -> @location(0) vec4<f32> {
    let uv_main0 = apply_st(uv0_raw, mat._MainTex_ST);
    let uv_main1 = apply_st(uv0_raw, mat._MainTex1_ST);
    let uv_lerp = apply_st(uv0_raw, mat._LerpTex_ST);
    let l = compute_lerp_factor(uv_lerp);

    var c0 = mat._Color;
    var c1 = mat._Color1;
    if (kw_enabled(mat._ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main0);
        c1 = c1 * textureSample(_MainTex1, _MainTex1_sampler, uv_main1);
    }

    let c = mix(c0, c1, l);
    if (kw_enabled(mat._ALPHACLIP) && c.a <= mat._AlphaClip) {
        discard;
    }

    let base_color = c.rgb;
    let alpha = c.a;

    var occlusion0 = 1.0;
    var occlusion1 = 1.0;
    if (kw_enabled(mat._OCCLUSION)) {
        occlusion0 = textureSample(_Occlusion, _Occlusion_sampler, uv_main0).r;
        occlusion1 = textureSample(_Occlusion1, _Occlusion1_sampler, uv_main1).r;
    }
    let occlusion = mix(occlusion0, occlusion1, l);

    var emission0 = mat._EmissionColor.xyz;
    var emission1 = mat._EmissionColor1.xyz;
    if (kw_enabled(mat._EMISSIONTEX)) {
        emission0 =
            emission0 * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main0).xyz;
        emission1 =
            emission1 * textureSample(_EmissionMap1, _EmissionMap1_sampler, uv_main1).xyz;
    }
    let em = mix(emission0, emission1, l);

    var metallic0 = mat._Metallic;
    var metallic1 = mat._Metallic1;
    var smoothness0 = mat._Glossiness;
    var smoothness1 = mat._Glossiness1;
    if (kw_enabled(mat._METALLICMAP)) {
        let m0 = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main0);
        let m1 = textureSample(_MetallicMap1, _MetallicMap1_sampler, uv_main1);
        metallic0 = m0.r;
        metallic1 = m1.r;
        smoothness0 = m0.a;
        smoothness1 = m1.a;
        if (kw_enabled(mat._MULTI_VALUES)) {
            metallic0 = metallic0 * mat._Metallic;
            metallic1 = metallic1 * mat._Metallic1;
            smoothness0 = smoothness0 * mat._Glossiness;
            smoothness1 = smoothness1 * mat._Glossiness1;
        }
    }
    let metallic = clamp(mix(metallic0, metallic1, l), 0.0, 1.0);
    let smoothness = clamp(mix(smoothness0, smoothness1, l), 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);

    let n = sample_normal_world(uv_main0, uv_main1, world_n, front_facing, l);

    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );

    let count = rg::cluster_light_counts[cluster_id];
    let base_idx = cluster_id * pcls::MAX_LIGHTS_PER_TILE;
    var lo = vec3<f32>(0.0);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    for (var i = 0u; i < i_max; i++) {
        let li = rg::cluster_light_indices[base_idx + i];
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        lo = lo + brdf::direct_radiance_metallic(
            light,
            world_pos,
            n,
            v,
            roughness,
            metallic,
            base_color,
            f0,
        );
    }

    let amb = vec3<f32>(0.03);
    let color = (amb * base_color * occlusion + lo * occlusion) + em;
    return vec4<f32>(color, alpha);
}
