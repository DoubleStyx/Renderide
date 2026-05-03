//! Grab-pass posterize filter (`Shader "Filters/Posterize"`).


#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::ui::rect_clip as uirc

struct FiltersPosterizeMaterial {
    _Rect: vec4<f32>,
    _Levels: f32,
    _RectClip: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersPosterizeMaterial;

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

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(in.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }
    let c = gp::sample_scene_color(gp::frag_screen_uv(in.clip_pos), in.view_layer);
    let levels = max(mat._Levels, 1.0);
    let filtered = round(c.rgb * levels) / levels;
    return rg::retain_globals_additive(vec4<f32>(filtered, c.a));
}
