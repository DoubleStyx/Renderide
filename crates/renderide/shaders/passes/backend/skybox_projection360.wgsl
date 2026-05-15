//! Fullscreen Projection360 sky draw.
//!
//! Material struct matches `materials/projection360.wgsl` so the reflected
//! `@group(1) @binding(0)` layout (which is taken from the material-side shader) matches
//! this pass-side shader's bind requirement. Froox variant bits populate
//! `_RenderideVariantBits`; this shader decodes Projection360's shader-specific keyword
//! bits locally. Unity `multi_compile` groups without a `_` placeholder still have an
//! implicit first keyword when the material sets no bit; this shader reconstructs that
//! default for the affected groups.

#import renderide::skybox::cubemap_storage as cubemap_storage
#import renderide::frame::globals as rg
#import renderide::skybox::common as skybox
#import renderide::core::uv as uvu
#import renderide::material::variant_bits as vb
#import renderide::skybox::projection360 as p360

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

const P360_KW_CLAMP_INTENSITY: u32 = 1u << 0u;
const P360_KW_NORMAL: u32 = 1u << 1u;
const P360_KW_OFFSET: u32 = 1u << 2u;
const P360_KW_PERSPECTIVE: u32 = 1u << 3u;
const P360_KW_RIGHT_EYE_ST: u32 = 1u << 4u;
const P360_KW_VIEW: u32 = 1u << 5u;
const P360_KW_WORLD_VIEW: u32 = 1u << 6u;
const P360_KW_CUBEMAP: u32 = 1u << 7u;
const P360_KW_CUBEMAP_LOD: u32 = 1u << 8u;
const P360_KW_EQUIRECTANGULAR: u32 = 1u << 9u;
const P360_KW_OUTSIDE_CLAMP: u32 = 1u << 10u;
const P360_KW_OUTSIDE_CLIP: u32 = 1u << 11u;
const P360_KW_OUTSIDE_COLOR: u32 = 1u << 12u;
const P360_KW_RECTCLIP: u32 = 1u << 13u;
const P360_KW_SECOND_TEXTURE: u32 = 1u << 14u;
const P360_KW_TINT_TEX_DIRECT: u32 = 1u << 15u;
const P360_KW_TINT_TEX_LERP: u32 = 1u << 16u;
const P360_KW_TINT_TEX_NONE: u32 = 1u << 17u;

const P360_GROUP_VIEW: u32 =
    P360_KW_VIEW | P360_KW_WORLD_VIEW | P360_KW_NORMAL | P360_KW_PERSPECTIVE;
const P360_GROUP_OUTSIDE: u32 =
    P360_KW_OUTSIDE_CLIP | P360_KW_OUTSIDE_COLOR | P360_KW_OUTSIDE_CLAMP;
const P360_GROUP_TINT_TEX: u32 =
    P360_KW_TINT_TEX_NONE | P360_KW_TINT_TEX_DIRECT | P360_KW_TINT_TEX_LERP;
const P360_GROUP_TEXTURE_MODE: u32 =
    P360_KW_EQUIRECTANGULAR | P360_KW_CUBEMAP | P360_KW_CUBEMAP_LOD;

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

fn proj360_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn proj360_group_default(group_mask: u32, this_bit: u32) -> bool {
    return (mat._RenderideVariantBits & group_mask) == 0u
        || vb::enabled(mat._RenderideVariantBits, this_bit);
}

fn kw_CUBEMAP() -> bool { return proj360_kw(P360_KW_CUBEMAP); }
fn kw_CUBEMAP_LOD() -> bool { return proj360_kw(P360_KW_CUBEMAP_LOD); }
fn kw_EQUIRECTANGULAR() -> bool {
    return proj360_group_default(P360_GROUP_TEXTURE_MODE, P360_KW_EQUIRECTANGULAR);
}
fn kw_OUTSIDE_CLAMP() -> bool { return proj360_kw(P360_KW_OUTSIDE_CLAMP); }
fn kw_OUTSIDE_CLIP() -> bool {
    return proj360_group_default(P360_GROUP_OUTSIDE, P360_KW_OUTSIDE_CLIP);
}
fn kw_OUTSIDE_COLOR() -> bool { return proj360_kw(P360_KW_OUTSIDE_COLOR); }
fn kw_SECOND_TEXTURE() -> bool { return proj360_kw(P360_KW_SECOND_TEXTURE); }
fn kw_TINT_TEX_DIRECT() -> bool { return proj360_kw(P360_KW_TINT_TEX_DIRECT); }
fn kw_TINT_TEX_LERP() -> bool { return proj360_kw(P360_KW_TINT_TEX_LERP); }
fn kw_TINT_TEX_NONE() -> bool {
    return proj360_group_default(P360_GROUP_TINT_TEX, P360_KW_TINT_TEX_NONE);
}
fn kw_CLAMP_INTENSITY() -> bool { return proj360_kw(P360_KW_CLAMP_INTENSITY); }
fn kw_OFFSET() -> bool { return proj360_kw(P360_KW_OFFSET); }
fn kw_PERSPECTIVE() -> bool { return proj360_kw(P360_KW_PERSPECTIVE); }
fn kw_RIGHT_EYE_ST() -> bool { return proj360_kw(P360_KW_RIGHT_EYE_ST); }
fn kw_VIEW() -> bool {
    return proj360_group_default(P360_GROUP_VIEW, P360_KW_VIEW);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
    @location(1) @interpolate(flat) view_layer: u32,
}

fn apply_offset(view_dir: vec3<f32>) -> vec3<f32> {
    if (!kw_OFFSET()) {
        return view_dir;
    }

    let offset_uv = p360::dir_to_uv(view_dir, mat._FOV);
    let offset_sample =
        textureSampleLevel(_OffsetTex, _OffsetTex_sampler, uvu::apply_st(offset_uv, mat._OffsetTex_ST), 0.0).rg;
    let offset_mask = textureSampleLevel(_OffsetMask, _OffsetMask_sampler, offset_uv, 0.0).rg;
    let offset = (offset_sample * 2.0 - vec2<f32>(1.0)) * offset_mask * mat._OffsetMagnitude.xy;
    return p360::rotate_dir(view_dir, offset);
}

fn sample_equirect(view_dir: vec3<f32>, view_layer: u32) -> vec4<f32> {
    var uv = p360::dir_to_uv(view_dir, mat._FOV);
    if (p360::is_outside_uv(uv)) {
        if (kw_OUTSIDE_COLOR()) {
            return mat._OutsideColor;
        }
        if (kw_OUTSIDE_CLIP()) {
            discard;
        }
    }
    uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

    var st = mat._MainTex_ST;
    if (kw_RIGHT_EYE_ST() && view_layer != 0u) {
        st = mat._RightEye_ST;
    }
    let sample_uv = uvu::apply_st(uv, st);
    var c = textureSampleLevel(_MainTex, _MainTex_sampler, sample_uv, 0.0);
    if (kw_SECOND_TEXTURE()) {
        let secondary_offset = vec2<f32>(mat._SecondTexOffset.x, -mat._SecondTexOffset.y);
        let sc = textureSampleLevel(_SecondTex, _SecondTex_sampler, sample_uv + secondary_offset, 0.0);
        c = mix(c, sc, mat._TextureLerp);
    }

    if (kw_TINT_TEX_DIRECT()) {
        c = c * textureSampleLevel(_TintTex, _TintTex_sampler, sample_uv, 0.0);
    } else if (kw_TINT_TEX_LERP()) {
        let tint_uv = uvu::apply_st(uv, vec4<f32>(mat._TintTex_ST.xy, mat._TintTex_ST.w, mat._TintTex_ST.z));
        let l = textureSampleLevel(_TintTex, _TintTex_sampler, tint_uv, 0.0).r;
        c = c * mix(mat._Tint0, mat._Tint1, l);
    }
    return c;
}

fn sample_cubemap(view_dir: vec3<f32>) -> vec4<f32> {
    let dir = normalize(-view_dir);
    let main_dir = cubemap_storage::sample_dir(dir, mat._MainCube_StorageVInverted);
    var c: vec4<f32>;
    if (kw_CUBEMAP_LOD()) {
        c = textureSampleLevel(_MainCube, _MainCube_sampler, main_dir, mat._CubeLOD);
    } else {
        c = textureSample(_MainCube, _MainCube_sampler, main_dir);
    }
    if (kw_SECOND_TEXTURE()) {
        let second_dir = cubemap_storage::sample_dir(dir, mat._SecondCube_StorageVInverted);
        var sc: vec4<f32>;
        if (kw_CUBEMAP_LOD()) {
            sc = textureSampleLevel(_SecondCube, _SecondCube_sampler, second_dir, mat._CubeLOD);
        } else {
            sc = textureSample(_SecondCube, _SecondCube_sampler, second_dir);
        }
        c = mix(c, sc, mat._TextureLerp);
    }
    return c;
}

fn finish_color(c_in: vec4<f32>) -> vec4<f32> {
    let c = p360::apply_tint_exposure_and_clamp(
        c_in,
        mat._Tint,
        mat._Gamma,
        mat._Exposure,
        kw_CLAMP_INTENSITY(),
        mat._MaxIntensity,
    );
    return rg::retain_globals_additive(c);
}

fn base_view_dir(ndc: vec2<f32>, view_layer: u32) -> vec3<f32> {
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, view_layer != 0u);
    let camera_ray_view = skybox::view_ray_from_ndc(
        ndc,
        proj_params,
        skybox::view_is_orthographic(view, view_layer),
    );
    let camera_ray_world = skybox::world_ray_from_view_ray(camera_ray_view, view, view_layer);

    if (kw_PERSPECTIVE()) {
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
    let view_dir = apply_offset(base_view_dir(in.ndc, in.view_layer));
    var c: vec4<f32>;
    if (kw_CUBEMAP() || kw_CUBEMAP_LOD()) {
        c = sample_cubemap(view_dir);
    } else {
        c = sample_equirect(view_dir, in.view_layer);
    }
    return finish_color(c);
}
