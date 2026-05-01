//! World text unlit (Unity shader asset `TextUnlit`): MSDF / SDF / raster font atlas in world space.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
#import renderide::mesh::vertex as mv
#import renderide::text_sdf as tsdf
#import renderide::uv_utils as uvu

struct TextUnlitMaterial {
    _TintColor: vec4<f32>,
    _OutlineColor: vec4<f32>,
    _BackgroundColor: vec4<f32>,
    _Range: vec4<f32>,
    _FaceDilate: f32,
    _FaceSoftness: f32,
    _OutlineSize: f32,
    _TextMode: f32,
    _FontAtlas_StorageVInverted: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
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
    out.uv = uvu::flip_v_for_storage(uv, mat._FontAtlas_StorageVInverted);
    out.extra_data = extra_n;
    out.vtx_color = color;
    return out;
}

//#pass forward
@fragment
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    let vtx_color = vout.vtx_color;
    let atlas_color = textureSample(_FontAtlas, _FontAtlas_sampler, vout.uv);
    let atlas_clip = acs::texture_rgba_base_mip(_FontAtlas, _FontAtlas_sampler, vout.uv);
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
    let c = tsdf::shade_text_sample(atlas_color, atlas_clip, style, text_input, mat._TintColor * vtx_color, mode);

    return rg::retain_globals_additive(c);
}
