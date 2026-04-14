//! Canvas UI text unlit (Unity shader asset `UI_TextUnlit`, normalized key `ui_textunlit`): MSDF/SDF/Raster font atlas, tint, outline, rect clip.
//!
//! Build emits `ui_textunlit_default` / `ui_textunlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` global names match Unity `UI_TextUnlit.shader` material property names for host reflection.
//!
//! **Vertex color:** Unity multiplies `_TintColor * vertexColor`. The mesh pass provides a float4
//! color stream at `@location(3)` with opaque-white fallback when absent on the host mesh.
//!
//! **Glyph mode (Unity `RASTER` / `SDF` / `MSDF` keywords):** FrooxEngine always sends `_Range` from
//! `PixelRange / atlasSize` for both raster and distance-field fonts, so **mode cannot be inferred from `_Range` alone**.
//! Use **`_TextMode`**: `0` = MSDF (median RGB), `1` = RASTER (`atlas * tint`, alpha clip), `2` = SDF (single-channel alpha distance).
//! The host may set `_TextMode` explicitly or rely on keyword floats (`MSDF`, `SDF`, `RASTER`) mapped in
//! [`crate::backend::embedded_material_bind`].
//!
//! **Rect clip (Unity `RECTCLIP` keyword):** When **`_RectClip` > 0.5** and `_Rect` has non-zero area, fragments outside
//! the rect in object XY are discarded. If `_RectClip` is unset, **`RECTCLIP` / `rectclip` keyword floats** are used.
//!
//! **OVERLAY** depth compositing is not implemented; when `_OverlayTint.a` is high, a simple tint approximation may be applied.
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
#import renderide::text_sdf as tsdf

struct UiTextUnlitMaterial {
    _TintColor: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _OutlineColor: vec4<f32>,
    _BackgroundColor: vec4<f32>,
    _Range: vec4<f32>,
    _Rect: vec4<f32>,
    _FaceDilate: f32,
    _FaceSoftness: f32,
    _OutlineSize: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _StencilComp: f32,
    _Stencil: f32,
    _StencilOp: f32,
    _StencilWriteMask: f32,
    _StencilReadMask: f32,
    _ColorMask: f32,
    /// `0` = MSDF, `1` = RASTER, `2` = SDF (Unity shader keyword modes).
    _TextMode: f32,
    /// `1` when rect clipping is enabled (Unity `RECTCLIP`); gates use of `_Rect`.
    _RectClip: f32,
    _pad: f32,
}

@group(1) @binding(0) var<uniform> mat: UiTextUnlitMaterial;
@group(1) @binding(1) var _FontAtlas: texture_2d<f32>;
@group(1) @binding(2) var _FontAtlas_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) extra_data: vec4<f32>,
    @location(2) vtx_color: vec4<f32>,
    @location(3) obj_xy: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) extra_n: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
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
    out.extra_data = extra_n;
    out.vtx_color = color;
    out.obj_xy = pos.xy;
    return out;
}

/// Returns true when `p` is outside the axis-aligned rect (min = xy, size = zw) in object XY.
fn outside_rect_clip(p: vec2<f32>, r: vec4<f32>) -> bool {
    let min_v = r.xy;
    let max_v = r.xy + r.zw;
    return p.x < min_v.x || p.x > max_v.x || p.y < min_v.y || p.y > max_v.y;
}

/// Signed-distance shading shared by MSDF and SDF paths (Unity `MSDF` / `SDF` blocks).
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
        fill_color = mix(mat._OutlineColor * vec4<f32>(1.0, 1.0, 1.0, vtx_color.a), fill_color, outline_lerp);
    }

    return mix(mat._BackgroundColor * vtx_color, fill_color, glyph_lerp);
}

@fragment
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    let vtx_color = vout.vtx_color;

    let rect = mat._Rect;
    let use_rect_clip = mat._RectClip > 0.5 && rect.z * rect.w > 1e-6;
    if (use_rect_clip && outside_rect_clip(vout.obj_xy, rect)) {
        discard;
    }

    let atlas_color = textureSample(_FontAtlas, _FontAtlas_sampler, vout.uv);
    let atlas_clip = acs::texture_rgba_base_mip(_FontAtlas, _FontAtlas_sampler, vout.uv);
    let range_xy = mat._Range.xy;
    let mode = tsdf::text_mode_clamped(mat._TextMode);

    var c: vec4<f32>;

    if (mode == 1) {
        c = atlas_color * vtx_color;
        if (atlas_clip.a * vtx_color.a < 0.001) {
            discard;
        }
    } else if (mode == 2) {
        let sig_dist = atlas_clip.a - 0.5;
        c = shade_distance_field(sig_dist, vout, vtx_color, range_xy);
    } else {
        let m = tsdf::median3(atlas_clip.r, atlas_clip.g, atlas_clip.b);
        let sig_dist = m - 0.5;
        c = shade_distance_field(sig_dist, vout, vtx_color, range_xy);
    }

    let o = mat._OverlayTint;
    if (o.a > 0.01) {
        c = vec4<f32>(c.rgb * mix(vec3<f32>(1.0), o.rgb, o.a), c.a);
    }

    return rg::retain_globals_additive(c);
}
