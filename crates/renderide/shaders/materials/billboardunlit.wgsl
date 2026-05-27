//! Billboard/Unlit (`Shader "Billboard/Unlit"`).
//!
//! The material expects one point expanded into four quad vertices. WGSL has no geometry stage, so
//! this shader billboards already-quad geometry in the vertex stage. Meshes with duplicated center
//! positions and quad UVs match the expected point expansion closely.
//!
//! Variant bits cover the material keyword set.
//! `_POINT_UV` is a pipeline-affecting decision (the host pre-bakes per-point
//! `texcoord + texscale * (corner - 0.5)` into the expanded quad's UVs), so the WGSL only needs
//! the bit reserved at its sorted index even though no fragment branch reads it.

//#texture_default _Tex white
//#texture_default _OffsetTex black
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _OffsetMagnitude vec4 0.1 0.1 0.0 0.0
//#mat_default _PointSize vec4 0.1 0.1 0.0 0.0
//#mat_default _PolarPow float 1.0
//#mat_default _Cutoff float 0.5
//#mat_default _Tex_LodBias float 0.0
//#mat_default _OffsetTex_LodBias float 0.0

#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu
#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::material::variant_bits as vb
#import renderide::material::vertex_color as vc
#import renderide::mesh::billboard as mb
#import renderide::mesh::vertex as mv

struct BillboardUnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _RightEye_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _PointSize: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _RenderideVariantBits: u32,
    _Tex_LodBias: f32,
    _OffsetTex_LodBias: f32,
    _pad0: vec2<f32>,
}

const BILLBOARDUNLIT_KW_ALPHATEST: u32 = 1u << 0u;
const BILLBOARDUNLIT_KW_COLOR: u32 = 1u << 1u;
const BILLBOARDUNLIT_KW_MUL_ALPHA_INTENSITY: u32 = 1u << 2u;
const BILLBOARDUNLIT_KW_MUL_RGB_BY_ALPHA: u32 = 1u << 3u;
const BILLBOARDUNLIT_KW_OFFSET_TEXTURE: u32 = 1u << 4u;
const BILLBOARDUNLIT_KW_POINT_ROTATION: u32 = 1u << 5u;
const BILLBOARDUNLIT_KW_POINT_SIZE: u32 = 1u << 6u;
const BILLBOARDUNLIT_KW_POINT_UV: u32 = 1u << 7u;
const BILLBOARDUNLIT_KW_POLARUV: u32 = 1u << 8u;
const BILLBOARDUNLIT_KW_RIGHT_EYE_ST: u32 = 1u << 9u;
const BILLBOARDUNLIT_KW_TEXTURE: u32 = 1u << 10u;
const BILLBOARDUNLIT_KW_VERTEX_HDRSRGB_COLOR: u32 = 1u << 11u;
const BILLBOARDUNLIT_KW_VERTEX_HDRSRGBALPHA_COLOR: u32 = 1u << 12u;
const BILLBOARDUNLIT_KW_VERTEX_LINEAR_COLOR: u32 = 1u << 13u;
const BILLBOARDUNLIT_KW_VERTEX_SRGB_COLOR: u32 = 1u << 14u;
const BILLBOARDUNLIT_KW_VERTEXCOLORS: u32 = 1u << 15u;
const BILLBOARDUNLIT_KW_RENDER_BUFFER: u32 = 1u << 16u;

@group(1) @binding(0) var<uniform> mat: BillboardUnlitMaterial;
@group(1) @binding(1) var _Tex: texture_2d<f32>;
@group(1) @binding(2) var _Tex_sampler: sampler;
@group(1) @binding(3) var _OffsetTex: texture_2d<f32>;
@group(1) @binding(4) var _OffsetTex_sampler: sampler;

fn bb_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALPHATEST() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_ALPHATEST);
}

fn kw_COLOR() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_COLOR);
}

fn kw_MUL_ALPHA_INTENSITY() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_MUL_ALPHA_INTENSITY);
}

fn kw_MUL_RGB_BY_ALPHA() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_MUL_RGB_BY_ALPHA);
}

fn kw_OFFSET_TEXTURE() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_OFFSET_TEXTURE);
}

fn kw_POINT_ROTATION() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_POINT_ROTATION);
}

fn kw_POINT_SIZE() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_POINT_SIZE);
}

fn kw_RENDER_BUFFER() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_RENDER_BUFFER);
}

fn kw_POLARUV() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_POLARUV);
}

fn kw_RIGHT_EYE_ST() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_RIGHT_EYE_ST);
}

fn kw_TEXTURE() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_TEXTURE);
}

fn kw_VERTEX_SRGB_COLOR() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_VERTEX_SRGB_COLOR);
}

fn kw_VERTEX_HDRSRGB_COLOR() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_VERTEX_HDRSRGB_COLOR);
}

fn kw_VERTEX_HDRSRGBALPHA_COLOR() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_VERTEX_HDRSRGBALPHA_COLOR);
}

fn kw_VERTEXCOLORS() -> bool {
    return bb_kw(BILLBOARDUNLIT_KW_VERTEXCOLORS);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) view_layer: u32,
}

fn billboard_size(pointdata: vec3<f32>, model: mat4x4<f32>) -> vec2<f32> {
    if (kw_RENDER_BUFFER()) {
        return max(abs(pointdata.xy), vec2<f32>(1e-6, 1e-6)) * mb::model_uniform_scale(model);
    }
    return mb::billboard_size(pointdata, mat._PointSize.xy, model, kw_POINT_SIZE());
}

fn render_buffer_billboard_unit_corner(vertex_index: u32) -> vec2<f32> {
    let corner = vertex_index % 4u;
    return vec2<f32>(
        select(0.0, 1.0, (corner & 1u) != 0u),
        select(0.0, 1.0, (corner & 2u) != 0u),
    );
}

fn billboard_corner_for_vertex(pos: vec3<f32>, uv: vec2<f32>, vertex_index: u32) -> vec2<f32> {
    if (kw_RENDER_BUFFER()) {
        return render_buffer_billboard_unit_corner(vertex_index) * 2.0 - vec2<f32>(1.0, 1.0);
    }
    return mb::billboard_corner(pos, uv);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) pointdata_in: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let pointdata = pointdata_in.xyz;
#ifdef MULTIVIEW
    let layer = view_idx;
#else
    let layer = 0u;
#endif

    let center_world = mv::world_position(d, pos).xyz;
    let use_rotation = kw_POINT_ROTATION() && abs(pointdata.z) > 1e-4;
    let axes = mb::billboard_axes(center_world, pointdata, layer, use_rotation);
    let corner = billboard_corner_for_vertex(pos.xyz, uv, vertex_index);
    let size = billboard_size(pointdata, d.model);
    let world_p = center_world + axes.right * (corner.x * size.x) + axes.up * (corner.y * size.y);

#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * vec4<f32>(world_p, 1.0);
    out.uv = uv;
    out.color = color;
    out.view_layer = layer;
    return out;
}

fn main_st(view_layer: u32) -> vec4<f32> {
    if (kw_RIGHT_EYE_ST() && view_layer != 0u) {
        return mat._RightEye_ST;
    }
    return mat._Tex_ST;
}

fn texture_uv(base_uv: vec2<f32>, view_layer: u32) -> vec2<f32> {
    let st = main_st(view_layer);
    var uv: vec2<f32>;
    if (kw_POLARUV()) {
        uv = uvu::apply_st(uvu::polar_uv(base_uv, mat._PolarPow), st);
    } else {
        uv = uvu::apply_st(base_uv, st);
    }
    if (kw_OFFSET_TEXTURE()) {
        let uv_off = uvu::apply_st(base_uv, mat._OffsetTex_ST);
        let offset_s = ts::sample_tex_2d(_OffsetTex, _OffsetTex_sampler, uv_off, mat._OffsetTex_LodBias);
        uv = uv + offset_s.xy * mat._OffsetMagnitude.xy;
    }
    return uv;
}

fn offset_texture_uv(uv: vec2<f32>, base_uv: vec2<f32>) -> vec2<f32> {
    if (kw_OFFSET_TEXTURE()) {
        let uv_off = uvu::apply_st(base_uv, mat._OffsetTex_ST);
        let offset_s = ts::sample_tex_2d(_OffsetTex, _OffsetTex_sampler, uv_off, mat._OffsetTex_LodBias);
        return uv + offset_s.xy * mat._OffsetMagnitude.xy;
    }
    return uv;
}

fn sample_main_texture(base_uv: vec2<f32>, view_layer: u32) -> vec4<f32> {
    if (kw_POLARUV()) {
        let mapped = uvu::polar_mapping(base_uv, main_st(view_layer), mat._PolarPow);
        let uv_main = offset_texture_uv(mapped.uv, base_uv);
        return textureSampleGrad(_Tex, _Tex_sampler, uv_main, mapped.ddx_uv, mapped.ddy_uv);
    }
    return ts::sample_tex_2d(_Tex, _Tex_sampler, texture_uv(base_uv, view_layer), mat._Tex_LodBias);
}

fn vertex_color(color: vec4<f32>) -> vec4<f32> {
    if (kw_VERTEX_HDRSRGBALPHA_COLOR()) {
        let rgb_linear = vc::srgb_to_linear_hdr(color);
        return vec4<f32>(rgb_linear.rgb, pow(color.a, 1.0 / 2.2));
    }
    if (kw_VERTEX_HDRSRGB_COLOR()) {
        return vc::srgb_to_linear_hdr(color);
    }
    if (kw_VERTEX_SRGB_COLOR()) {
        return vc::srgb_to_linear_ldr(color);
    }
    return color;
}

//#pass type=forward name=forward_billboard blend=material_filter offset=0,0
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let use_texture = kw_TEXTURE();
    let use_color = kw_COLOR();

    var col: vec4<f32>;
    var clip_alpha: f32;
    if (use_texture) {
        let tex = sample_main_texture(in.uv, in.view_layer);
        clip_alpha = tex.a;
        if (use_color) {
            col = tex * mat._Color;
            clip_alpha = clip_alpha * mat._Color.a;
        } else {
            col = tex;
        }
    } else if (use_color) {
        col = mat._Color;
        clip_alpha = mat._Color.a;
    } else {
        col = vec4<f32>(1.0);
        clip_alpha = 1.0;
    }

    if (kw_ALPHATEST() && clip_alpha <= mat._Cutoff) {
        discard;
    }

    if (kw_VERTEXCOLORS()) {
        col = col * vertex_color(in.color);
    }

    if (kw_MUL_RGB_BY_ALPHA()) {
        col = vec4<f32>(col.rgb * col.a, col.a);
    }

    if (kw_MUL_ALPHA_INTENSITY()) {
        col = vec4<f32>(col.rgb, col.a * dot(col.rgb, vec3<f32>(0.3333333)));
    }

    return rg::retain_globals_additive(col);
}
