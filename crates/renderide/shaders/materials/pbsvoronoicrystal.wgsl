//! Unity surface shader `Shader "PBSVoronoiCrystal"`: metallic Standard lighting layered over a
//! procedural Voronoi pattern.
//!
//! Each fragment scans a 3×3 cell neighborhood of the scaled UV; the nearest cell drives albedo
//! / smoothness / emission via gradient-texture lookups; the second-nearest distance drives an
//! `_EdgeThickness`-wide border that blends to `_EdgeColor` / `_EdgeMetallic` / `_EdgeGloss` etc.
//! Cell centers animate by `_AnimationOffset` (host-driven; this renderer doesn't expose seconds-
//! since-startup so the host must drive the animation directly).


#import renderide::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd
#import renderide::voronoi as vor

struct PbsVoronoiCrystalMaterial {
    _ColorTint: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EdgeColor: vec4<f32>,
    _EdgeEmission: vec4<f32>,
    _Scale: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _NormalStrength: f32,
    _EdgeThickness: f32,
    _EdgeGloss: f32,
    _EdgeMetallic: f32,
    _EdgeNormalStrength: f32,
    _AnimationOffset: f32,
    _Glossiness: f32,
    _Metallic: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsVoronoiCrystalMaterial;
@group(1) @binding(1)  var _ColorGradient: texture_2d<f32>;
@group(1) @binding(2)  var _ColorGradient_sampler: sampler;
@group(1) @binding(3)  var _GlossGradient: texture_2d<f32>;
@group(1) @binding(4)  var _GlossGradient_sampler: sampler;
@group(1) @binding(5)  var _EmissionGradient: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionGradient_sampler: sampler;
@group(1) @binding(7)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(8)  var _NormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) uv_normal: vec2<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
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
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
    let wn = mv::world_normal(d, n);
    let wt = mv::world_tangent(d, t);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.world_t = wt;
    out.uv = uv0;
    out.uv_normal = uvu::apply_st(uv0, mat._NormalMap_ST);
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    uv: vec2<f32>,
    uv_normal: vec2<f32>,
    view_layer: u32,
) -> vec4<f32> {
    let scale = mat._Scale.xy;
    let v_result = vor::voronoi_full(uv * scale, scale, mat._AnimationOffset);
    let cell_offset = vec2<f32>(0.5) + vec2<f32>(0.5) * sin(mat._AnimationOffset + 6.2831 * v_result.min_point);
    let border_dist = v_result.second_min_dist - v_result.min_dist;
    let aaf = fwidth(border_dist);
    let border_lerp = smoothstep(mat._EdgeThickness - aaf, mat._EdgeThickness, border_dist);

    let edge_dir = normalize(vec2<f32>(dpdx(border_dist), dpdy(border_dist))) * mat._EdgeNormalStrength;
    let edge_normal_ts = normalize(vec3<f32>(edge_dir, 1.0));
    let cell_normal_ts = nd::decode_ts_normal_with_placeholder(
        textureSample(_NormalMap, _NormalMap_sampler, uv_normal + v_result.min_point).xyz,
        mat._NormalStrength,
    );
    let n_blend_ts = mix(edge_normal_ts, cell_normal_ts, border_lerp);
    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    let n = normalize(tbn * n_blend_ts);

    let cell_color = textureSample(_ColorGradient, _ColorGradient_sampler, cell_offset).rgb * mat._ColorTint.rgb;
    let base_color = mix(mat._EdgeColor.rgb, cell_color, border_lerp);
    let metallic = clamp(mix(mat._EdgeMetallic, mat._Metallic, border_lerp), 0.0, 1.0);
    let gloss_sample = textureSample(_GlossGradient, _GlossGradient_sampler, v_result.min_point).x;
    let smoothness = clamp(mix(mat._EdgeGloss, mat._Glossiness * gloss_sample, border_lerp), 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);
    let cell_emission = textureSample(_EmissionGradient, _EmissionGradient_sampler, cell_offset).rgb * mat._EmissionColor.rgb;
    let emission = mix(mat._EdgeEmission.rgb, cell_emission, border_lerp);

    let surface = psurf::metallic(
        base_color,
        1.0,
        metallic,
        roughness,
        1.0,
        n,
        emission,
    );
    return vec4<f32>(
        plight::shade_metallic_clustered(
            frag_xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        1.0,
    );
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) uv_normal: vec2<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, world_t, uv, uv_normal, view_layer);
}
