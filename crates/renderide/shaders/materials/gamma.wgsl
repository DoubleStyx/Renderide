//! Grab-pass gamma filter (`Shader "Filters/Gamma"`).


#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::ui::rect_clip as uirc

struct FiltersGammaMaterial {
    _Rect: vec4<f32>,
    _Gamma: f32,
    _RectClip: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersGammaMaterial;

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
) -> fv::RectVertexOutput {
#ifdef MULTIVIEW
    return fv::rect_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fv::rect_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward
@fragment
fn fs_main(in: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(in.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }

    let c = gp::sample_scene_color(gp::frag_screen_uv(in.clip_pos), in.view_layer);
    let filtered = pow(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(max(mat._Gamma, 1e-4)));
    return rg::retain_globals_additive(vec4<f32>(filtered, c.a));
}