//! Unity surface shader `Shader "PBSVertexColorTransparent"`: metallic Standard lighting that
//! optionally multiplies albedo, emission, or metallic/smoothness by the mesh vertex color.
//!
//! Mirrors [`pbsdualsided`](super::pbsdualsided) for the keyword + VCOLOR_* surface, dropping the
//! front-face flip (this shader is single-sided in Unity). Transparent default render state is
//! driven by the host's `_SrcBlend` / `_DstBlend` / `_ZWrite` material properties.

// unity-shader-name: PBSVertexColorTransparent

#import renderide::globals as rg
#import renderide::sh2_ambient as shamb
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsVertexColorTransparentMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _AlphaClip: f32,
    _ALPHACLIP: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _METALLICMAP: f32,
    _OCCLUSION: f32,
    VCOLOR_ALBEDO: f32,
    VCOLOR_EMIT: f32,
    VCOLOR_METALLIC: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsVertexColorTransparentMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _MetallicMap: texture_2d<f32>;
@group(1) @binding(10) var _MetallicMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>) -> vec3<f32> {
    let n = normalize(world_n);
    if (!uvu::kw_enabled(mat._NORMALMAP)) {
        return n;
    }
    let tbn = pnorm::orthonormal_tbn(n);
    let ts_n = nd::decode_ts_normal_with_placeholder(
        textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
        mat._NormalScale,
    );
    return normalize(tbn * ts_n);
}

fn sample_surface(uv0: vec2<f32>, world_n: vec3<f32>, vertex_color: vec4<f32>) -> SurfaceData {
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);

    var albedo = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        albedo = albedo * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    if (uvu::kw_enabled(mat.VCOLOR_ALBEDO)) {
        albedo = albedo * vertex_color;
    }
    let vertex_alpha = select(1.0, vertex_color.a, uvu::kw_enabled(mat.VCOLOR_ALBEDO));
    let clip_alpha = select(
        albedo.a,
        mat._Color.a
            * vertex_alpha
            * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main),
        uvu::kw_enabled(mat._ALBEDOTEX),
    );
    if (uvu::kw_enabled(mat._ALPHACLIP) && clip_alpha <= mat._AlphaClip) {
        discard;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    if (uvu::kw_enabled(mat.VCOLOR_METALLIC)) {
        metallic = metallic * dot(vertex_color.rgb, vec3<f32>(0.33333334));
        smoothness = smoothness * vertex_color.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    let emission_color = mat._EmissionColor.rgb;
    var emission = vec3<f32>(0.0);
    if (dot(emission_color, emission_color) > 1e-8) {
        emission = emission_color;
        if (uvu::kw_enabled(mat._EMISSIONTEX)) {
            emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
        }
    }
    if (uvu::kw_enabled(mat.VCOLOR_EMIT)) {
        emission = emission * vertex_color.rgb;
    }

    return SurfaceData(
        albedo.rgb,
        albedo.a,
        metallic,
        roughness,
        occlusion,
        sample_normal_world(uv_main, world_n),
        emission,
    );
}

fn clustered_direct_lighting(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: SurfaceData,
    include_directional: bool,
    include_local: bool,
) -> vec3<f32> {
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);
    let f0 = mix(vec3<f32>(0.04), s.base_color, s.metallic);

    let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
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
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        let is_directional = light.light_type == 1u;
        if ((is_directional && !include_directional) || (!is_directional && !include_local)) {
            continue;
        }
        lo = lo + brdf::direct_radiance_metallic(
            light, world_pos, s.normal, v, aa_roughness, s.metallic, s.base_color, f0,
        );
    }
    return lo;
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
    @location(3) color: vec4<f32>,
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
    out.color = color;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n, color);
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, true, true);
    let view_dir = normalize(rg::frame.camera_world_pos.xyz - world_pos);
    let f0 = brdf::metallic_f0(s.base_color, s.metallic);
    let ambient = brdf::indirect_diffuse_metallic(
        shamb::ambient_probe(s.normal),
        s.base_color,
        s.metallic,
        s.occlusion,
    );
    let indirect_specular = brdf::indirect_specular(
        s.normal,
        view_dir,
        s.roughness,
        f0,
        s.occlusion,
        true,
    );
    return vec4<f32>(ambient + indirect_specular + direct + s.emission, s.alpha);
}
