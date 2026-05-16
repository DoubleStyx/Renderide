//! Canvas UI text unlit (Unity shader asset `UI_TextUnlit`, normalized key `ui_textunlit`):
//! MSDF/SDF/Raster font atlas, tint, outline, rect clip, overlay tint.
//!
//! Build emits `ui_textunlit_default` / `ui_textunlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` field names match Unity `UI_TextUnlit.shader` material property names for host reflection.
//!
//! Vertex color: Unity multiplies `_TintColor * vertexColor`. The mesh pass provides a float4
//! color stream at `@location(3)` with opaque-white fallback when absent on the host mesh.
//!
//! Froox `#pragma multi_compile` keywords (`RASTER`/`SDF`/`MSDF`, `OUTLINE`, `RECTCLIP`, `OVERLAY`)
//! are decoded from the renderer-reserved `_RenderideVariantBits` uniform; bit positions match
//! Froox's sorted `UniqueKeywords` list.
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::draw::per_draw`].


//#texture_default _FontAtlas white
//#mat_default _OutlineColor vec4 1.0 1.0 1.0 0.0
//#mat_default _OverlayTint vec4 1.0 1.0 1.0 0.5
//#mat_default _Range vec4 0.001 0.001 0.0 0.0
//#mat_default _Rect vec4 0.0 0.0 1.0 1.0
//#mat_default _TintColor vec4 1.0 1.0 1.0 1.0

#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::material::text_sdf as tsdf
#import renderide::material::variant_bits as vb
#import renderide::core::texture_sampling as ts
#import renderide::ui::overlay_tint as uiot
#import renderide::ui::rect_clip as uirc

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
    _RenderideVariantBits: u32,
    _FontAtlas_LodBias: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

const UITEXTUNLIT_KW_MSDF: u32 = 1u << 0u;
const UITEXTUNLIT_KW_OUTLINE: u32 = 1u << 1u;
const UITEXTUNLIT_KW_OVERLAY: u32 = 1u << 2u;
const UITEXTUNLIT_KW_RASTER: u32 = 1u << 3u;
const UITEXTUNLIT_KW_RECTCLIP: u32 = 1u << 4u;
const UITEXTUNLIT_KW_SDF: u32 = 1u << 5u;

@group(1) @binding(0) var<uniform> mat: UiTextUnlitMaterial;
@group(1) @binding(1) var _FontAtlas: texture_2d<f32>;
@group(1) @binding(2) var _FontAtlas_sampler: sampler;

fn ui_textunlit_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

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
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
#else
    out.view_layer = mv::packed_view_layer(instance_index, 0u);
#endif
    return out;
}

//#pass forward
@fragment
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    let vtx_color = vout.vtx_color;

    if (uirc::should_clip_rect_kw(vout.obj_xy, mat._Rect, ui_textunlit_kw(UITEXTUNLIT_KW_RECTCLIP))) {
        discard;
    }

    let atlas_color = ts::sample_tex_2d(
        _FontAtlas,
        _FontAtlas_sampler,
        vout.uv,
        mat._FontAtlas_LodBias,
    );
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
    let mode = tsdf::text_mode_from_keywords(
        ui_textunlit_kw(UITEXTUNLIT_KW_RASTER),
        ui_textunlit_kw(UITEXTUNLIT_KW_SDF),
    );
    var c = tsdf::shade_text_sample(
        atlas_color,
        style,
        text_input,
        vtx_color,
        mode,
        ui_textunlit_kw(UITEXTUNLIT_KW_OUTLINE),
    );

    c = uiot::apply_overlay_tint(
        c,
        mat._OverlayTint,
        vout.clip_pos,
        vout.world_pos,
        vout.view_layer,
        ui_textunlit_kw(UITEXTUNLIT_KW_OVERLAY),
    );

    return rg::retain_globals_additive(c);
}
