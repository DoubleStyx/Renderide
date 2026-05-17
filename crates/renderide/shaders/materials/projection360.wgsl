//! Unity Projection360 (`Shader "Projection360"`): equirectangular/cubemap projection with
//! optional second texture, tint texture, offset map, and rectangular clipping.
//!
//! Froox variant bits populate `_RenderideVariantBits`; the shared Projection360 module
//! decodes shader-specific keyword bits and owns the repeated projection sampling.

//#texture_default _MainTex black
//#texture_default _SecondTex black
//#texture_default _TintTex white
//#texture_default _OffsetTex black
//#texture_default _OffsetMask white
//#texture_default _MainCube black
//#texture_default _SecondCube black
//#mat_default _Exposure float 1.0
//#mat_default _FOV vec4 6.283185 3.141593 0.0 0.0
//#mat_default _Gamma float 1.0
//#mat_default _MaxIntensity float 4.0
//#mat_default _OffsetMagnitude vec4 0.1 0.1 0.0 0.0
//#mat_default _PerspectiveFOV vec4 0.785398 0.785398 0.0 0.0
//#mat_default _Tint vec4 1.0 1.0 1.0 1.0
//#mat_default _Tint0 vec4 1.0 0.0 0.0 1.0
//#mat_default _Tint1 vec4 0.0 1.0 0.0 1.0

#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::core::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::skybox::projection360 as p360
#import renderide::skybox::projection360_material as p360m

struct Projection360Material {
    _Tint: vec4<f32>,
    _OutsideColor: vec4<f32>,
    _Tint0: vec4<f32>,
    _Tint1: vec4<f32>,
    _FOV: vec4<f32>,
    _SecondTexOffset: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _PerspectiveFOV: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _RightEye_ST: vec4<f32>,
    _TintTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _TextureLerp: f32,
    _CubeLOD: f32,
    _MainCube_StorageVInverted: f32,
    _SecondCube_StorageVInverted: f32,
    _Exposure: f32,
    _Gamma: f32,
    _MaxIntensity: f32,
    _RenderideVariantBits: u32,
}

@group(1) @binding(0) var<uniform> mat: Projection360Material;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _SecondTex: texture_2d<f32>;
@group(1) @binding(4) var _SecondTex_sampler: sampler;
@group(1) @binding(5) var _TintTex: texture_2d<f32>;
@group(1) @binding(6) var _TintTex_sampler: sampler;
@group(1) @binding(7) var _OffsetTex: texture_2d<f32>;
@group(1) @binding(8) var _OffsetTex_sampler: sampler;
@group(1) @binding(9) var _OffsetMask: texture_2d<f32>;
@group(1) @binding(10) var _OffsetMask_sampler: sampler;
@group(1) @binding(11) var _MainCube: texture_cube<f32>;
@group(1) @binding(12) var _MainCube_sampler: sampler;
@group(1) @binding(13) var _SecondCube: texture_cube<f32>;
@group(1) @binding(14) var _SecondCube_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_os: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal_os: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) dist: f32,
    @location(5) local_xy: vec2<f32>,
    @location(6) @interpolate(flat) view_layer: u32,
    @location(7) object_view_dir: vec3<f32>,
}

fn projection360_params() -> p360m::Projection360Params {
    return p360m::Projection360Params(
        mat._Tint,
        mat._OutsideColor,
        mat._Tint0,
        mat._Tint1,
        mat._FOV,
        mat._SecondTexOffset,
        mat._OffsetMagnitude,
        mat._PerspectiveFOV,
        mat._MainTex_ST,
        mat._RightEye_ST,
        mat._TintTex_ST,
        mat._OffsetTex_ST,
        mat._TextureLerp,
        mat._CubeLOD,
        mat._MainCube_StorageVInverted,
        mat._SecondCube_StorageVInverted,
        mat._Exposure,
        mat._Gamma,
        mat._MaxIntensity,
        mat._RenderideVariantBits,
    );
}

/// Object-space view direction at the vertex, intentionally un-normalized.
///
/// The result is linear in `world_pos`, so perspective-correct interpolation across the
/// triangle yields the per-fragment direction from the surface point to the camera.
/// Normalizing per vertex would skew the interpolated direction (the angular error scales
/// with the triangle's angular extent and breaks narrow-FOV projections); the fragment
/// shader normalizes the interpolated value, matching the per-fragment recompute used by
/// the original Unity shader.
fn object_space_view_dir(model: mat4x4<f32>, world_pos: vec3<f32>, view_layer: u32) -> vec3<f32> {
    let model3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    return transpose(model3) * (rg::camera_world_pos_for_view(view_layer) - world_pos);
}

fn perspective_view_dir(uv: vec2<f32>) -> vec3<f32> {
    return p360::perspective_view_dir_from_ndc((uv - vec2<f32>(0.5)) * 2.0, mat._PerspectiveFOV);
}

fn base_view_dir(in: VertexOutput) -> vec3<f32> {
    if (p360m::kw_PERSPECTIVE(mat._RenderideVariantBits)) {
        return perspective_view_dir(in.uv);
    }
    if (p360m::kw_NORMAL(mat._RenderideVariantBits)) {
        return normalize(in.normal_os);
    }
    if (p360m::kw_WORLD_VIEW(mat._RenderideVariantBits)) {
        return rg::view_dir_for_world_pos(in.world_pos, in.view_layer);
    }
    return normalize(in.object_view_dir);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
    let layer = view_idx;
#else
    let vp = mv::select_view_proj(d, 0u);
    let layer = 0u;
#endif
    let clip = vp * world_p;

    var out: VertexOutput;
    out.clip_pos = clip;
    out.pos_os = pos.xyz;
    out.world_pos = world_p.xyz;
    out.normal_os = normalize(-n.xyz);
    out.uv = uv;
    out.dist = clip.w;
    out.local_xy = pos.xy;
    out.object_view_dir = object_space_view_dir(d.model, world_p.xyz, layer);
    out.view_layer = layer;
    return out;
}

//#pass type=forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let params = projection360_params();
    if (p360m::kw_RECTCLIP(params.variant_bits) && rmath::outside_rect(in.local_xy, mat._Rect)) {
        discard;
    }

    let view_dir = p360m::apply_offset(
        base_view_dir(in),
        params,
        _OffsetTex,
        _OffsetTex_sampler,
        _OffsetMask,
        _OffsetMask_sampler,
    );
    let c = p360m::sample_projection(
        view_dir,
        in.view_layer,
        params,
        _MainTex,
        _MainTex_sampler,
        _SecondTex,
        _SecondTex_sampler,
        _TintTex,
        _TintTex_sampler,
        _MainCube,
        _MainCube_sampler,
        _SecondCube,
        _SecondCube_sampler,
    );
    return rg::retain_globals_additive(p360m::finish_material_color(c, in.dist, params));
}
