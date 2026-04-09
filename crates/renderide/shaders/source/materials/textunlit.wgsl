//! World text unlit (route key `textunlit`): alias of `text_unlit` for plain-label routing.
//!
//! Keep this source in sync with `text_unlit.wgsl`.

#import renderide::globals as rg
#import renderide::per_draw as pd

struct TextUnlitMaterial {
    _TintColor: vec4<f32>,
    _OutlineColor: vec4<f32>,
    _BackgroundColor: vec4<f32>,
    _Range: vec4<f32>,
    _FaceDilate: f32,
    _FaceSoftness: f32,
    _OutlineSize: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _TextMode: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(1) @binding(0) var<uniform> mat: TextUnlitMaterial;
@group(1) @binding(1) var _FontAtlas: texture_2d<f32>;
@group(1) @binding(2) var _FontAtlas_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) extra_data: vec4<f32>,
    @location(2) vtx_color: vec4<f32>,
}

fn median3(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

fn text_mode_clamped(tm: f32) -> i32 {
    return clamp(i32(round(tm)), 0, 2);
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) extra_n: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
) -> VertexOutput {
    let world_p = pd::draw.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = pd::draw.view_proj_left;
    } else {
        vp = pd::draw.view_proj_right;
    }
#else
    let vp = pd::draw.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    out.extra_data = extra_n;
    out.vtx_color = color;
    return out;
}

fn shade_distance_field(
    sig_dist_in: f32,
    vo: VertexOutput,
    vtx_color: vec4<f32>,
    range_xy: vec2<f32>,
) -> vec4<f32> {
    var sig_dist = sig_dist_in;
    sig_dist = sig_dist + mat._FaceDilate + vo.extra_data.x;

    let fw = vec2<f32>(fwidth(vo.uv.x), fwidth(vo.uv.y));
    let anti_aliasing = dot(range_xy, vec2<f32>(0.5) / max(fw, vec2<f32>(1e-6)));
    let aa = max(anti_aliasing, 1.0);

    var glyph_lerp = mix(sig_dist * aa, sig_dist, mat._FaceSoftness);
    glyph_lerp = clamp(glyph_lerp + 0.5, 0.0, 1.0);

    if (max(glyph_lerp, mat._BackgroundColor.a) < 0.001) {
        discard;
    }

    var fill_color = mat._TintColor * vtx_color;
    let outline_w = mat._OutlineSize + vo.extra_data.y;
    if (outline_w > 1e-6 || mat._OutlineSize > 1e-6) {
        let outline_dist = sig_dist - outline_w;
        var outline_lerp = mix(outline_dist * aa, outline_dist, mat._FaceSoftness);
        outline_lerp = clamp(outline_lerp + 0.5, 0.0, 1.0);
        fill_color = mix(
            mat._OutlineColor * vec4<f32>(1.0, 1.0, 1.0, vtx_color.a),
            fill_color,
            outline_lerp,
        );
    }

    return mix(mat._BackgroundColor * vtx_color, fill_color, glyph_lerp);
}

@fragment
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    let vtx_color = vout.vtx_color;
    let atlas_color = textureSample(_FontAtlas, _FontAtlas_sampler, vout.uv);
    let range_xy = mat._Range.xy;
    let mode = text_mode_clamped(mat._TextMode);

    var c: vec4<f32>;

    if (mode == 1) {
        c = atlas_color * mat._TintColor * vtx_color;
        if (c.a < 0.001) {
            discard;
        }
    } else if (mode == 2) {
        let sig_dist = atlas_color.a - 0.5;
        c = shade_distance_field(sig_dist, vout, vtx_color, range_xy);
    } else {
        let m = median3(atlas_color.r, atlas_color.g, atlas_color.b);
        let sig_dist = m - 0.5;
        c = shade_distance_field(sig_dist, vout, vtx_color, range_xy);
    }

    var lit: u32 = 0u;
    if (rg::frame.light_count > 0u) {
        lit = rg::lights[0].light_type;
    }
    let cluster_touch =
        f32(rg::cluster_light_counts[0u] & 255u) * 1e-10 +
        f32(rg::cluster_light_indices[0u] & 255u) * 1e-10;
    return c + vec4<f32>(vec3<f32>(f32(lit) * 1e-10 + cluster_touch), 0.0);
}
