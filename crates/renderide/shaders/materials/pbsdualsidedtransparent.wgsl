//! Unity surface shader `Shader "PBSDualSidedTransparent"`: metallic Standard lighting with
//! transparent two-pass dual-sided rendering.
//!
//! Unity renders this shader as back faces first (`Cull Front`) and front faces second
//! (`Cull Back`). The two WGSL fragment entries keep that pass topology while sharing the same
//! surface evaluation.

#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::pbs::lighting as plight
#import renderide::pbs::surface as psurf
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsDualSidedTransparentMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _METALLICMAP: f32,
    _OCCLUSION: f32,
    VCOLOR_ALBEDO: f32,
    VCOLOR_EMIT: f32,
    VCOLOR_METALLIC: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsDualSidedTransparentMaterial;
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

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, front_facing: bool) -> vec3<f32> {
    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        ts_n = nd::decode_ts_normal_with_placeholder(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
            mat._NormalScale,
        );
    }
    if (!front_facing) {
        ts_n.z = -ts_n.z;
    }
    return normalize(tbn * ts_n);
}

fn sample_surface(
    uv0: vec2<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    front_facing: bool,
    vertex_color: vec4<f32>,
) -> SurfaceData {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);

    var albedo = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        albedo = albedo * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    if (uvu::kw_enabled(mat.VCOLOR_ALBEDO)) {
        albedo = albedo * vertex_color;
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
        sample_normal_world(uv_main, world_n, world_t, front_facing),
        emission,
    );
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
    @location(4) t: vec4<f32>,
) -> mv::WorldColorVertexOutput {
#ifdef MULTIVIEW
    return mv::world_color_vertex_main(instance_index, view_idx, pos, n, t, uv0, color);
#else
    return mv::world_color_vertex_main(instance_index, 0u, pos, n, t, uv0, color);
#endif
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    uv0: vec2<f32>,
    vertex_color: vec4<f32>,
    view_layer: u32,
    front_facing: bool,
) -> vec4<f32> {
    let s = sample_surface(uv0, world_n, world_t, front_facing, vertex_color);
    let surface = psurf::metallic(
        s.base_color,
        s.alpha,
        s.metallic,
        s.roughness,
        s.occlusion,
        s.normal,
        s.emission,
    );
    return vec4<f32>(
        plight::shade_metallic_clustered(
            frag_xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        s.alpha,
    );
}

//#pass forward_transparent_cull_front
@fragment
fn fs_back_faces(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, world_t, uv0, color, view_layer, front_facing);
}

//#pass forward_transparent_cull_back
@fragment
fn fs_front_faces(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, world_t, uv0, color, view_layer, front_facing);
}
