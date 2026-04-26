//! Unity unlit `Shader "Unlit/TextureDebug"`: visualizes a single texture channel as grayscale.
//! `_TextureChannel` selects R (0) / G (1) / B (2) / A (3); other values pass the texture through.

// unity-shader-name: Unlit/TextureDebug
// unity-shader-name: TextureDebug

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu

struct TextureDebugMaterial {
    _MainTex_ST: vec4<f32>,
    _TextureChannel: f32,
}

@group(1) @binding(0) var<uniform> mat: TextureDebugMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
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
    out.uv = uvu::apply_st(uv, mat._MainTex_ST);
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    let col = textureSample(_MainTex, _MainTex_sampler, uv);
    let ch = i32(round(mat._TextureChannel));
    var result = col;
    if (ch == 0) {
        result = vec4<f32>(vec3<f32>(col.r), 1.0);
    } else if (ch == 1) {
        result = vec4<f32>(vec3<f32>(col.g), 1.0);
    } else if (ch == 2) {
        result = vec4<f32>(vec3<f32>(col.b), 1.0);
    } else if (ch == 3) {
        result = vec4<f32>(vec3<f32>(col.a), 1.0);
    }
    return rg::retain_globals_additive(result);
}
