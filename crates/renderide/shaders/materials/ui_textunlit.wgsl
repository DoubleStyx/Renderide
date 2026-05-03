//! Canvas UI text unlit (Unity shader asset `UI_TextUnlit`, normalized key `ui_textunlit`): MSDF/SDF/Raster font atlas, tint, outline, rect clip.
//!
//! Build emits `ui_textunlit_default` / `ui_textunlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` global names match Unity `UI_TextUnlit.shader` material property names for host reflection.
//!
//! **Vertex color:** Unity multiplies `_TintColor * vertexColor`. The mesh pass provides a float4
//! color stream at `@location(3)` with opaque-white fallback when absent on the host mesh.
//!
//! **Glyph mode (Unity `RASTER` / `SDF` / `MSDF` keywords):** FrooxEngine sends `_Range` from
//! `PixelRange / atlasSize` for both raster and distance-field fonts, so mode is not derived from `_Range`.
//! Use **`_TextMode`**: `0` = MSDF (median RGB), `1` = RASTER (`atlas * tint`, alpha clip), `2` = SDF (single-channel alpha distance).
//! Missing `_TextMode` is inferred from `_FontAtlas` color profile by CPU uniform packing.
//!
//! **Rect clip (Unity `RECTCLIP` keyword):** When **`_RectClip` > 0.5** and `_Rect` has non-zero area, fragments outside
//! the rect in object XY are discarded. Missing `_RectClip` defaults to off.
//!
//! **OVERLAY** depth compositing uses `_OVERLAY` to gate the scene-depth comparison before applying `_OverlayTint`.
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::text_sdf as tsdf
#import renderide::scene_depth_sample as sds
#import renderide::ui::rect_clip as uirc
#import renderide::uv_utils as uvu

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
    /// `0` = MSDF, `1` = RASTER, `2` = SDF (Unity shader keyword modes).
    _TextMode: f32,
    /// `1` when rect clipping is enabled (Unity `RECTCLIP`); gates use of `_Rect`.
    _RectClip: f32,
    /// `1` when overlay depth compositing is enabled (Unity `OVERLAY`).
    _OVERLAY: f32,
    /// Padding for 16-byte uniform alignment.
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
    @location(4) world_pos: vec3<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
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
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    out.extra_data = extra_n;
    out.vtx_color = color;
    out.obj_xy = pos.xy;
    out.world_pos = world_p.xyz;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

//#pass forward
@fragment
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    let vtx_color = vout.vtx_color;

    if (uirc::should_clip_rect(vout.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }

    let atlas_color = textureSample(_FontAtlas, _FontAtlas_sampler, vout.uv);
    let style = tsdf::distance_field_style(
        mat._TintColor,
        mat._OutlineColor,
        mat._BackgroundColor,
        mat._Range,
        mat._FaceDilate,
        mat._FaceSoftness,
        mat._OutlineSize,
    );
    let text_input = tsdf::DistanceFieldInput(0.0, vout.uv, vout.extra_data, vtx_color);
    let mode = tsdf::text_mode_clamped(mat._TextMode);
    var c = tsdf::shade_text_sample(atlas_color, style, text_input, vtx_color, mode);

    if (mat._OVERLAY > 0.5) {
        let scene_z = sds::scene_linear_depth(vout.clip_pos, vout.view_layer);
        let part_z = sds::fragment_linear_depth(vout.world_pos, vout.view_layer);
        if (part_z > scene_z) {
            c = c * mat._OverlayTint;
        }
    }

    return rg::retain_globals_additive(c);
}
