//! Unity displaced transparent specular (`Shader "PBSDisplaceSpecularTransparent"`).
//!
//! Includes transparent specular shading, optional UV offset, and optional normal-direction
//! vertex displacement from `_VertexOffsetMap`.
// unity-shader-name: PBSDisplaceSpecularTransparent

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsDisplaceSpecularTransparentMaterial {
    _Color: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _VertexOffsetMap_ST: vec4<f32>,
    _UVOffsetMap_ST: vec4<f32>,
    _Cutoff: f32,
    _NormalScale: f32,
    _VertexOffsetMagnitude: f32,
    _VertexOffsetBias: f32,
    _UVOffsetMagnitude: f32,
    _UVOffsetBias: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Mode: f32,
    _ALPHACLIP: f32,
    _VERTEX_OFFSET: f32,
    _UV_OFFSET: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsDisplaceSpecularTransparentMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _SpecularMap: texture_2d<f32>;
@group(1) @binding(10) var _SpecularMap_sampler: sampler;
@group(1) @binding(11) var _VertexOffsetMap: texture_2d<f32>;
@group(1) @binding(12) var _VertexOffsetMap_sampler: sampler;
@group(1) @binding(13) var _UVOffsetMap: texture_2d<f32>;
@group(1) @binding(14) var _UVOffsetMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn mode_near(v: f32) -> bool {
    return abs(mat._Mode - v) < 0.5;
}

fn alpha_premultiply_enabled() -> bool {
    return mode_near(3.0);
}

fn apply_premultiply(color: vec3<f32>, alpha: f32) -> vec3<f32> {
    return select(color, color * alpha, alpha_premultiply_enabled());
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
    var local_pos = pos.xyz;
    let local_n = normalize(n.xyz);
    if (uvu::kw_enabled(mat._VERTEX_OFFSET)) {
        let uv_offset = uvu::apply_st(uv0, mat._VertexOffsetMap_ST);
        let h = textureSampleLevel(_VertexOffsetMap, _VertexOffsetMap_sampler, uv_offset, 0.0).r;
        let disp = h * mat._VertexOffsetMagnitude + mat._VertexOffsetBias;
        local_pos = local_pos + local_n * disp;
    }
    let world_p = d.model * vec4<f32>(local_pos, 1.0);
    let wn = normalize(d.normal_matrix * local_n);
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

@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    var uv = uvu::apply_st(uv0, mat._MainTex_ST);
    if (uvu::kw_enabled(mat._UV_OFFSET)) {
        let uv_off_uv = uvu::apply_st(uv0, mat._UVOffsetMap_ST);
        let uv_off_sample = textureSample(_UVOffsetMap, _UVOffsetMap_sampler, uv_off_uv).xy;
        uv = uv + uv_off_sample * mat._UVOffsetMagnitude + vec2<f32>(mat._UVOffsetBias);
    }

    let albedo_s = textureSample(_MainTex, _MainTex_sampler, uv);
    let base_color = mat._Color.rgb * albedo_s.rgb;
    let alpha = mat._Color.a * albedo_s.a;
    if (uvu::kw_enabled(mat._ALPHACLIP)) {
        let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv);
        if (clip_alpha <= mat._Cutoff) {
            discard;
        }
    }

    let tbn = brdf::orthonormal_tbn(normalize(world_n));
    let ts_n = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap, _NormalMap_sampler, uv),
        mat._NormalScale,
    );
    let n = normalize(tbn * ts_n);

    let occ = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv).r;
    let emission = textureSample(_EmissionMap, _EmissionMap_sampler, uv).rgb * mat._EmissionColor.rgb;

    let spec_sample = textureSample(_SpecularMap, _SpecularMap_sampler, uv) * mat._SpecularColor;
    let f0 = clamp(spec_sample.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec_sample.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

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
    let count = rg::cluster_light_counts[cluster_id];
    let base_idx = cluster_id * pcls::MAX_LIGHTS_PER_TILE;
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = rg::cluster_light_indices[base_idx + i];
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        lo = lo + brdf::direct_radiance_specular(
            light,
            world_pos,
            n,
            v,
            roughness,
            base_color,
            f0,
            one_minus_reflectivity,
        );
    }

    let amb = vec3<f32>(0.03);
    let color = (amb * base_color * occ + lo * occ) + emission;
    return vec4<f32>(apply_premultiply(color, alpha), alpha);
}
