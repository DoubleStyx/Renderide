//! Grab-pass 3D LUT filter (`Shader "Filters/LUT"`).


#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::texture_sampling as ts
#import renderide::ui::rect_clip as uirc

struct FiltersLutMaterial {
    _Rect: vec4<f32>,
    _Lerp: f32,
    _LUT_LodBias: f32,
    _SecondaryLUT_LodBias: f32,
    _RectClip: f32,
    LERP: f32,
    SRGB: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersLutMaterial;
@group(1) @binding(1) var _LUT: texture_3d<f32>;
@group(1) @binding(2) var _LUT_sampler: sampler;
@group(1) @binding(3) var _SecondaryLUT: texture_3d<f32>;
@group(1) @binding(4) var _SecondaryLUT_sampler: sampler;

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
    let inner = fv::vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    let inner = fv::vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
    var out: VertexOutput;
    out.clip_pos = inner.clip_pos;
    out.primary_uv = inner.primary_uv;
    out.world_pos = inner.world_pos;
    out.world_n = inner.world_n;
    out.view_layer = inner.view_layer;
    out.view_n = inner.view_n;
    out.obj_xy = pos.xy;
    return out;
}

//#pass forward
@fragment
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(vout.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }

    let c = gp::sample_scene_color(gp::frag_screen_uv(vout.clip_pos), vout.view_layer);
    let gain = max(max(c.r, c.g), max(c.b, 1.0));
    var normalized = c.rgb / gain;
    if (mat.SRGB > 0.5) {
        normalized = pow(normalized, vec3<f32>(1.0 / 2.2));
    }
    let coords = clamp(normalized, vec3<f32>(0.0), vec3<f32>(1.0));
    var filtered = ts::sample_tex_3d(_LUT, _LUT_sampler, coords, mat._LUT_LodBias).rgb;
    if (mat.LERP > 0.5) {
        let secondary = ts::sample_tex_3d(_SecondaryLUT, _SecondaryLUT_sampler, coords, mat._SecondaryLUT_LodBias).rgb;
        filtered = mix(filtered, secondary, mat._Lerp);
    }
    filtered = filtered * gain;
    return rg::retain_globals_additive(vec4<f32>(filtered, c.a));
}
