//! Grab-pass 3D LUT filter (`Shader "Filters/LUT"`).


//#render_queue Transparent+500
//#texture_default _LUT empty
//#texture_default _SecondaryLUT empty

#import renderide::post::filter_vertex as fv
#import renderide::post::filter_common as fc
#import renderide::core::texture_sampling as ts
#import renderide::material::variant_bits as vb

struct FiltersLutMaterial {
    _Rect: vec4<f32>,
    _Lerp: f32,
    _RenderideVariantBits: u32,
}

const LUT_KW_LERP: u32 = 1u << 0u;
const LUT_KW_RECTCLIP: u32 = 1u << 1u;
const LUT_KW_SRGB: u32 = 1u << 2u;

@group(1) @binding(0) var<uniform> mat: FiltersLutMaterial;
@group(1) @binding(1) var _LUT: texture_3d<f32>;
@group(1) @binding(2) var _LUT_sampler: sampler;
@group(1) @binding(3) var _SecondaryLUT: texture_3d<f32>;
@group(1) @binding(4) var _SecondaryLUT_sampler: sampler;

fn lut_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_LERP() -> bool {
    return lut_kw(LUT_KW_LERP);
}

fn kw_RECTCLIP() -> bool {
    return lut_kw(LUT_KW_RECTCLIP);
}

fn kw_SRGB() -> bool {
    return lut_kw(LUT_KW_SRGB);
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
) -> fv::RectVertexOutput {
#ifdef MULTIVIEW
    return fv::rect_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fv::rect_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass type=forward name=forward_filter blend=material_filter
@fragment
fn fs_main(in: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    let c = fc::sample_clipped_scene_color_at_clip(in.obj_xy, mat._Rect, kw_RECTCLIP(), in.clip_pos, in.view_layer);
    let gain = max(max(c.r, c.g), max(c.b, 1.0));
    var normalized = c.rgb / gain;
    if (kw_SRGB()) {
        normalized = pow(normalized, vec3<f32>(1.0 / 2.2));
    }
    let coords = normalized;
    var filtered = ts::sample_tex_3d_level(_LUT, _LUT_sampler, coords, 0.0).rgb;
    if (kw_LERP()) {
        let secondary = ts::sample_tex_3d_level(_SecondaryLUT, _SecondaryLUT_sampler, coords, 0.0).rgb;
        filtered = mix(filtered, secondary, mat._Lerp);
    }
    filtered = filtered * gain;
    return fc::retain_scene_alpha(c, filtered);
}
