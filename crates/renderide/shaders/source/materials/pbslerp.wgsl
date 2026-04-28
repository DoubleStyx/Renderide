//! Unity PBS lerp (`Shader "PBSLerp"`): metallic workflow blending between two material sets.
//!
//! This uses the same clustered forward lighting path as `pbsmetallic.wgsl`, but blends two
//! albedo/normal/emission/occlusion/metallic-smoothness sets by `_Lerp` or `_LerpTex`.
//! Unity surface-shader keywords are mirrored as float uniforms:
//! `_LERPTEX`, `_ALBEDOTEX`, `_EMISSIONTEX`, `_NORMALMAP`, `_METALLICMAP`,
//! `_OCCLUSION`, `_MULTI_VALUES`, `_DUALSIDED`, `_ALPHACLIP`.

// unity-shader-name: PBSLerp

#import renderide::globals as rg
#import renderide::sh2_ambient as shamb
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsLerpMaterial {
    _Color: vec4<f32>,
    _Color1: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EmissionColor1: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _MainTex1_ST: vec4<f32>,
    _MainTex1_StorageVInverted: f32,
    _LerpTex_ST: vec4<f32>,
    _Lerp: f32,
    _NormalScale: f32,
    _NormalScale1: f32,
    _Glossiness: f32,
    _Glossiness1: f32,
    _Metallic: f32,
    _Metallic1: f32,
    _AlphaClip: f32,
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
    @location(3) @interpolate(flat) view_layer: u32,
}

fn sample_normal_world(
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    world_n: vec3<f32>,
    front_facing: bool,
    lerp_factor: f32,
) -> vec3<f32> {
    if (!uvu::kw_enabled(mat._NORMALMAP)) {
        var n = normalize(world_n);
        if (uvu::kw_enabled(mat._DUALSIDED) && !front_facing) {
            n = -n;
        }
        return n;
    }

    let tbn = pnorm::orthonormal_tbn(normalize(world_n));
    let ts0 = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap, _NormalMap_sampler, uv0),
        mat._NormalScale,
    );
    let ts1 = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap1, _NormalMap1_sampler, uv1),
        mat._NormalScale1,
    );
    var ts = normalize(mix(ts0, ts1, vec3<f32>(lerp_factor)));
    if (uvu::kw_enabled(mat._DUALSIDED) && !front_facing) {
        ts.z = -ts.z;
    }
    return normalize(tbn * ts);
}

fn compute_lerp_factor(uv_lerp: vec2<f32>) -> f32 {
    var l = mat._Lerp;
    if (uvu::kw_enabled(mat._LERPTEX)) {
        l = textureSample(_LerpTex, _LerpTex_sampler, uv_lerp).r;
        if (uvu::kw_enabled(mat._MULTI_VALUES)) {
            l = l * mat._Lerp;
        }
    }
    return clamp(l, 0.0, 1.0);
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
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let wn = normalize(d.normal_matrix * n.xyz);
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
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0_raw: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main0 = uvu::apply_st_for_storage(uv0_raw, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let uv_main1 = uvu::apply_st_for_storage(uv0_raw, mat._MainTex1_ST, mat._MainTex1_StorageVInverted);
    let uv_lerp = uvu::apply_st(uv0_raw, mat._LerpTex_ST);
    let l = compute_lerp_factor(uv_lerp);

    var c0 = mat._Color;
    var c1 = mat._Color1;
    var clip_a = mix(mat._Color.a, mat._Color1.a, l);
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main0);
        c1 = c1 * textureSample(_MainTex1, _MainTex1_sampler, uv_main1);
        clip_a = mix(
            mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main0),
            mat._Color1.a * acs::texture_alpha_base_mip(_MainTex1, _MainTex1_sampler, uv_main1),
            l,
        );
    }

    let c = mix(c0, c1, l);
    if (uvu::kw_enabled(mat._ALPHACLIP) && clip_a <= mat._AlphaClip) {
        discard;
    }

    let base_color = c.rgb;
    let alpha = c.a;

    var occlusion0 = 1.0;
    var occlusion1 = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion0 = textureSample(_Occlusion, _Occlusion_sampler, uv_main0).r;
        occlusion1 = textureSample(_Occlusion1, _Occlusion1_sampler, uv_main1).r;
    }
    let occlusion = mix(occlusion0, occlusion1, l);

    var emission0 = mat._EmissionColor.xyz;
    var emission1 = mat._EmissionColor1.xyz;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
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
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        let m0 = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main0);
        let m1 = textureSample(_MetallicMap1, _MetallicMap1_sampler, uv_main1);
        metallic0 = m0.r;
        metallic1 = m1.r;
        smoothness0 = m0.a;
        smoothness1 = m1.a;
        if (uvu::kw_enabled(mat._MULTI_VALUES)) {
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

    let aa_roughness = brdf::filter_perceptual_roughness(roughness, n);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        view_layer,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );

    let count = pcls::cluster_light_count_at(cluster_id);
    var lo = vec3<f32>(0.0);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        lo = lo + brdf::direct_radiance_metallic(
            light,
            world_pos,
            n,
            v,
            aa_roughness,
            metallic,
            base_color,
            f0,
        );
    }

    let ambient = brdf::indirect_diffuse_metallic(
        shamb::ambient_probe(n),
        base_color,
        metallic,
        occlusion,
    );
    let indirect_specular = brdf::indirect_specular(n, v, aa_roughness, f0, occlusion, true);
    let color = (ambient + indirect_specular + lo) + em;
    return vec4<f32>(color, alpha);
}
