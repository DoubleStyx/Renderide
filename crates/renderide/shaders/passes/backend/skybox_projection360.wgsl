//! Fullscreen Projection360 sky draw.
//!
//! The shared Projection360 module owns shader-specific keyword decoding and projection
//! sampling for both this pass-side sky draw and the material root.

#import renderide::frame::globals as rg
#import renderide::skybox::common as skybox
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
@group(2) @binding(0) var<uniform> view: skybox::SkyboxView;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
    @location(1) @interpolate(flat) view_layer: u32,
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

fn base_view_dir(ndc: vec2<f32>, view_layer: u32) -> vec3<f32> {
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, view_layer != 0u);
    let camera_ray_view = skybox::view_ray_from_ndc(
        ndc,
        proj_params,
        skybox::view_is_orthographic(view, view_layer),
    );
    let camera_ray_world = skybox::world_ray_from_view_ray(camera_ray_view, view, view_layer);

    if (p360m::kw_PERSPECTIVE(mat._RenderideVariantBits)) {
        return p360::perspective_view_dir_from_ndc(ndc, mat._PerspectiveFOV);
    }
    return normalize(-camera_ray_world);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
) -> VertexOutput {
    let clip = skybox::fullscreen_clip_pos(vertex_index);
    var out: VertexOutput;
    out.clip_pos = clip;
    out.ndc = vec2<f32>(clip.x, clip.y * view.ndc_y_sign_pad.x);
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let params = projection360_params();
    let view_dir = p360m::apply_offset(
        base_view_dir(in.ndc, in.view_layer),
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
    return rg::retain_globals_additive(p360m::finish_skybox_color(c, params));
}
