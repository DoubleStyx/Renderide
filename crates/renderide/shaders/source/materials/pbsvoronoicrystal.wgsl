//! Unity surface shader `Shader "PBSVoronoiCrystal"`: metallic Standard lighting layered over a
//! procedural Voronoi pattern.
//!
//! Each fragment scans a 3×3 cell neighborhood of the scaled UV; the nearest cell drives albedo
//! / smoothness / emission via gradient-texture lookups; the second-nearest distance drives an
//! `_EdgeThickness`-wide border that blends to `_EdgeColor` / `_EdgeMetallic` / `_EdgeGloss` etc.
//! Cell centers animate by `_AnimationOffset` (host-driven; this renderer doesn't expose seconds-
//! since-startup so the host must drive the animation directly).

// unity-shader-name: PBSVoronoiCrystal

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

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
    @location(2) uv: vec2<f32>,
    @location(3) uv_normal: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
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
    out.uv = uv0;
    out.uv_normal = uvu::apply_st(uv0, mat._NormalMap_ST);
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

fn random2(p: vec2<f32>) -> vec2<f32> {
    let s = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(s) * 43758.5453);
}

fn wrap_tile(tile_in: vec2<f32>, scale: vec2<f32>) -> vec2<f32> {
    var tile = tile_in - floor(tile_in / scale) * scale;
    if (tile.x < 0.0) { tile.x = tile.x + scale.x; }
    if (tile.y < 0.0) { tile.y = tile.y + scale.y; }
    return tile;
}

struct VoronoiResult {
    min_dist: f32,
    second_min_dist: f32,
    min_point: vec2<f32>,
}

fn voronoi(uv_scaled: vec2<f32>, scale: vec2<f32>) -> VoronoiResult {
    let i_uv = floor(uv_scaled);
    let f_uv = fract(uv_scaled);
    var min_dist: f32 = 2.0;
    var second_min: f32 = 2.0;
    var min_point: vec2<f32> = vec2<f32>(0.0);
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let tile = wrap_tile(i_uv + neighbor, scale);
            let p_orig = random2(tile);
            let p = vec2<f32>(0.5) + vec2<f32>(0.5) * sin(mat._AnimationOffset + 6.2831 * p_orig);
            let diff = neighbor + p - f_uv;
            let dist = length(diff);
            if (dist < min_dist) {
                second_min = min_dist;
                min_dist = dist;
                min_point = p_orig;
            } else if (dist < second_min) {
                second_min = dist;
            }
        }
    }
    return VoronoiResult(min_dist, second_min, min_point);
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    uv: vec2<f32>,
    uv_normal: vec2<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
) -> vec4<f32> {
    let scale = mat._Scale.xy;
    let v_result = voronoi(uv * scale, scale);
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
    let tbn = pnorm::orthonormal_tbn(normalize(world_n));
    let n = normalize(tbn * n_blend_ts);

    let cell_color = textureSample(_ColorGradient, _ColorGradient_sampler, cell_offset).rgb * mat._ColorTint.rgb;
    let base_color = mix(mat._EdgeColor.rgb, cell_color, border_lerp);
    let metallic = clamp(mix(mat._EdgeMetallic, mat._Metallic, border_lerp), 0.0, 1.0);
    let gloss_sample = textureSample(_GlossGradient, _GlossGradient_sampler, v_result.min_point).x;
    let smoothness = clamp(mix(mat._EdgeGloss, mat._Glossiness * gloss_sample, border_lerp), 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let cell_emission = textureSample(_EmissionGradient, _EmissionGradient_sampler, cell_offset).rgb * mat._EmissionColor.rgb;
    let emission = mix(mat._EdgeEmission.rgb, cell_emission, border_lerp);

    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let aa_roughness = brdf::filter_perceptual_roughness(roughness, n);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy, world_pos, rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right,
        view_layer, rg::frame.viewport_width, rg::frame.viewport_height,
        rg::frame.cluster_count_x, rg::frame.cluster_count_y, rg::frame.cluster_count_z,
        rg::frame.near_clip, rg::frame.far_clip,
    );
    let count = rg::cluster_light_counts[cluster_id];
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
            light, world_pos, n, v, aa_roughness, metallic, base_color, f0,
        );
    }
    let ambient = select(vec3<f32>(0.0), vec3<f32>(0.03) * base_color, include_directional);
    let extra = select(vec3<f32>(0.0), emission, include_directional);
    return vec4<f32>(ambient + lo + extra, 1.0);
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) uv_normal: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, uv, uv_normal, view_layer, true, true);
}
