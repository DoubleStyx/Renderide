//! Unity PBS rim transparent specular (`Shader "PBSRimTransparentSpecular"`): same surface logic
//! as [`pbsrimspecular`](super::pbsrimspecular).
//!
//! Transparent default render state is driven by the host's `_SrcBlend` / `_DstBlend` / `_ZWrite`
//! material properties; the WGSL is identical to the opaque sibling.


#import renderide::globals as rg
#import renderide::material::fresnel as mf
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::uv_utils as uvu

struct PbsRimTransparentSpecularMaterial {
    _Color: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _RimColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _NormalScale: f32,
    _RimPower: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsRimTransparentSpecularMaterial;
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

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> vec3<f32> {
    return psamp::sample_world_normal(_NormalMap, _NormalMap_sampler, uv_main, 0.0, mat._NormalScale, world_n, world_t);
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
    @location(4) t: vec4<f32>,
) -> mv::WorldVertexOutput {
#ifdef MULTIVIEW
    return mv::world_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return mv::world_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);

    let albedo_s = textureSample(_MainTex, _MainTex_sampler, uv_main);
    let base_color = mat._Color.xyz * albedo_s.xyz;
    let alpha = mat._Color.a * albedo_s.a;

    let spec_s = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    let spec = mat._SpecularColor * spec_s;
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    let occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).x;

    let n = sample_normal_world(uv_main, world_n, world_t);

    let emission = textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).xyz * mat._EmissionColor.xyz;

    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let rim = mf::rim_factor(n, view_dir, mat._RimPower);
    let rim_emission = mat._RimColor.rgb * rim;
    let surface = psurf::specular(base_color, alpha, f0, roughness, occlusion, n, emission + rim_emission);
    let color = plight::shade_specular_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
    return vec4<f32>(color, alpha);
}
