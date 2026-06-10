//! Shared view-space refraction helpers for grab-pass material filters.

#define_import_path renderide::post::filter_refraction

#import renderide::core::normal_decode as nd
#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu
#import renderide::draw::per_draw as pd
#import renderide::frame::scene_depth_sample as sds
#import renderide::frame::view_basis as vbasis
#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::post::filter_math as fm
#import renderide::post::filter_vertex as fv

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
    @location(4) view_n: vec3<f32>,
    @location(5) obj_xy: vec2<f32>,
    @location(6) view_t: vec4<f32>,
    @location(7) clip_w: f32,
}

fn view_tangent_for_draw(instance_index: u32, view_idx: u32, world_t: vec4<f32>) -> vec4<f32> {
    let d = pd::get_draw(instance_index);
    let vp = mv::select_view_proj(d, view_idx);
    return vec4<f32>(vbasis::world_to_view_normal(world_t.xyz, vp), world_t.w);
}

fn vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    uv0: vec2<f32>,
) -> VertexOutput {
    let inner = fv::vertex_main(instance_index, view_idx, pos, n, t, uv0);
    var out: VertexOutput;
    out.clip_pos = inner.clip_pos;
    out.primary_uv = inner.primary_uv;
    out.world_pos = inner.world_pos;
    out.world_n = inner.world_n;
    out.view_layer = inner.view_layer;
    out.view_n = inner.view_n;
    out.obj_xy = pos.xy;
    out.view_t = view_tangent_for_draw(instance_index, view_idx, inner.world_t);
    out.clip_w = inner.clip_pos.w;
    return out;
}

fn billboard_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    uv0: vec2<f32>,
    vertex_index: u32,
    uv1: vec2<f32>
) -> VertexOutput {
    let inner = fv::billboard_vertex_main(instance_index, view_idx, pos, n, t, uv0, vertex_index, uv1);
    var out: VertexOutput;
    out.clip_pos = inner.clip_pos;
    out.primary_uv = inner.primary_uv;
    out.world_pos = inner.world_pos;
    out.world_n = inner.world_n;
    out.view_layer = inner.view_layer;
    out.view_n = inner.view_n;
    out.obj_xy = pos.xy;
    out.view_t = view_tangent_for_draw(instance_index, view_idx, inner.world_t);
    out.clip_w = inner.clip_pos.w;
    return out;
}

fn normal_offset(
    refraction_enabled: bool,
    normal_map_enabled: bool,
    uv0: vec2<f32>,
    view_n: vec3<f32>,
    view_t: vec4<f32>,
    clip_w: f32,
    strength: f32,
    normal_map_st: vec4<f32>,
    normal_lod_bias: f32,
    normal_map: texture_2d<f32>,
    normal_sampler: sampler,
) -> vec2<f32> {
    if (!refraction_enabled) {
        return vec2<f32>(0.0);
    }

    var n = normalize(view_n);
    if (normal_map_enabled) {
        let tangent_normal = nd::decode_ts_normal_with_placeholder_sample(
            ts::sample_tex_2d(
                normal_map,
                normal_sampler,
                uvu::apply_st(uv0, normal_map_st),
                normal_lod_bias,
            ),
            1.0,
        );
        let tbn = pnorm::orthonormal_tbn(n, view_t);
        n = normalize(tbn * tangent_normal);
    }

    return n.xy / max(abs(clip_w), 0.000001) * strength;
}

fn guarded_refracted_screen_uv(
    screen_uv: vec2<f32>,
    uv0: vec2<f32>,
    view_n: vec3<f32>,
    view_t: vec4<f32>,
    frag_pos: vec4<f32>,
    clip_w: f32,
    world_pos: vec3<f32>,
    view_layer: u32,
    normal_map_enabled: bool,
    normal_map_st: vec4<f32>,
    normal_lod_bias: f32,
    normal_map: texture_2d<f32>,
    normal_sampler: sampler,
    strength: f32,
    depth_bias: f32,
    depth_divisor: f32,
) -> vec2<f32> {
    let fade = sds::depth_fade(frag_pos, world_pos, view_layer, depth_divisor);
    let offset = normal_offset(
        true,
        normal_map_enabled,
        uv0,
        view_n,
        view_t,
        clip_w,
        strength,
        normal_map_st,
        normal_lod_bias,
        normal_map,
        normal_sampler,
    ) * fade * fm::screen_vignette(screen_uv);
    let grab_uv = screen_uv - offset;

    // Linearizing the fragment depth twice, and the scene depth not at all.
    // This is what Unity does, and somehow it works!?
    let sampled_xy = sds::scene_depth_xy_from_uv(grab_uv);
    let sampled_depth = sds::raw_depth_at_xy(sampled_xy, view_layer);
    let fragment_depth = sds::fragment_linear_depth(world_pos, view_layer);
    if (sampled_depth > sds::linear_depth_from_raw(fragment_depth, view_layer) + depth_bias) {
        return screen_uv;
    }

    return grab_uv;
}
