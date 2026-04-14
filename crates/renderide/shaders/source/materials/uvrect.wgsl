//! UV Rect (route key `uvrect`, Unity shader `Unlit/UVRect`): colors inside/outside a UV-space rect.
//!
//! `_RectClip` is a synthetic helper field so the CPU packer can reuse existing `RECTCLIP` keyword handling.

#import renderide::globals as rg
#import renderide::per_draw as pd

struct UvRectMaterial {
    _Rect: vec4<f32>,
    _ClipRect: vec4<f32>,
    _OuterColor: vec4<f32>,
    _InnerColor: vec4<f32>,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _RectClip: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: UvRectMaterial;

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
    out.uv = uv;
    return out;
}

fn inside_rect01(p: vec2<f32>, r: vec4<f32>) -> f32 {
    let inside_x = step(r.x, p.x) * step(p.x, r.z);
    let inside_y = step(r.y, p.y) * step(p.y, r.w);
    return inside_x * inside_y;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (mat._RectClip > 0.5 && inside_rect01(in.uv, mat._ClipRect) < 0.5) {
        discard;
    }

    let inner = inside_rect01(in.uv, mat._Rect);
    let color = mix(mat._OuterColor, mat._InnerColor, inner);

    return rg::retain_globals_additive(color);
}
