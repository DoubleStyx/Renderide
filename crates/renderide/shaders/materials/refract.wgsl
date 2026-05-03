//! Grab-pass refraction filter (`Shader "Filters/Refract"`).


#import renderide::filter_math as fm
#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::normal_decode as nd
#import renderide::scene_depth_sample as sds
#import renderide::ui::rect_clip as uirc
#import renderide::uv_utils as uvu

struct FiltersRefractMaterial {
    _NormalMap_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _RefractionStrength: f32,
    _DepthBias: f32,
    _DepthDivisor: f32,
    _NORMALMAP: f32,
    _RectClip: f32,
    _pad0: vec3<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersRefractMaterial;
@group(1) @binding(1) var _NormalMap: texture_2d<f32>;
@group(1) @binding(2) var _NormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
    @location(4) view_n: vec3<f32>,
    @location(5) obj_xy: vec2<f32>,
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
#ifdef MULTIVIEW
    let base = fv::vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    let base = fv::vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
    var out: VertexOutput;
    out.clip_pos = base.clip_pos;
    out.primary_uv = base.primary_uv;
    out.world_pos = base.world_pos;
    out.world_n = base.world_n;
    out.view_layer = base.view_layer;
    out.view_n = base.view_n;
    out.obj_xy = pos.xy;
    return out;
}

fn refract_offset(uv0: vec2<f32>, view_n: vec3<f32>, clip_recip_w: f32) -> vec2<f32> {
    var n = normalize(view_n);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uvu::apply_st(uv0, mat._NormalMap_ST)),
            1.0,
        );
        n = normalize(vec3<f32>(n.xy + ts.xy, n.z));
    }
    return n.xy * clip_recip_w * mat._RefractionStrength;
}

fn refracted_screen_uv(
    screen_uv: vec2<f32>,
    uv0: vec2<f32>,
    view_n: vec3<f32>,
    frag_pos: vec4<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
) -> vec2<f32> {
    let fade = sds::depth_fade(frag_pos, world_pos, view_layer, mat._DepthDivisor);
    let offset = refract_offset(uv0, view_n, frag_pos.w) * fade * fm::screen_vignette(screen_uv);
    let grab_uv = screen_uv - offset;
    let sampled_depth = sds::scene_linear_depth_at_uv(grab_uv, view_layer);
    let fragment_depth = sds::fragment_linear_depth(world_pos, view_layer);
    if (sampled_depth > fragment_depth + mat._DepthBias) {
        return screen_uv;
    }
    return grab_uv;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(in.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }
    let screen_uv = gp::frag_screen_uv(in.clip_pos);
    let color = gp::sample_scene_color(
        refracted_screen_uv(screen_uv, in.primary_uv, in.view_n, in.clip_pos, in.world_pos, in.view_layer),
        in.view_layer,
    );
    return rg::retain_globals_additive(color);
}
