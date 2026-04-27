//! Fullscreen Projection360 sky draw.

#import renderide::globals as rg
#import renderide::skybox_common as skybox
#import renderide::uv_utils as uvu

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
    _MainTex_StorageVInverted: f32,
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
    _RectClip: f32,
    _VIEW: f32,
    _WORLD_VIEW: f32,
    _NORMAL: f32,
    _PERSPECTIVE: f32,
    _RIGHT_EYE_ST: f32,
    OUTSIDE_CLIP: f32,
    OUTSIDE_COLOR: f32,
    OUTSIDE_CLAMP: f32,
    TINT_TEX_DIRECT: f32,
    TINT_TEX_LERP: f32,
    _CLAMP_INTENSITY: f32,
    SECOND_TEXTURE: f32,
    EQUIRECTANGULAR: f32,
    CUBEMAP: f32,
    CUBEMAP_LOD: f32,
    _OFFSET: f32,
    RECTCLIP: f32,
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

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

fn positive_fmod(v: vec2<f32>, wrap: vec2<f32>) -> vec2<f32> {
    var r = v - trunc(v / wrap) * wrap;
    r = r + wrap;
    return r - trunc(r / wrap) * wrap;
}

fn dir_to_uv(view_dir: vec3<f32>) -> vec2<f32> {
    var angle = vec2<f32>(
        atan2(view_dir.x, view_dir.z),
        acos(clamp(dot(view_dir, vec3<f32>(0.0, 1.0, 0.0)), -1.0, 1.0)) - PI * 0.5,
    );
    angle = angle + mat._FOV.xy * 0.5 + mat._FOV.zw;
    angle = positive_fmod(angle, vec2<f32>(TAU, PI));
    return angle / max(abs(mat._FOV.xy), vec2<f32>(0.000001));
}

fn rotate_dir(view_dir: vec3<f32>, rotate: vec2<f32>) -> vec3<f32> {
    let sy = sin(rotate.y);
    let cy = cos(rotate.y);
    let x_rot = vec3<f32>(
        view_dir.x,
        view_dir.y * cy - view_dir.z * sy,
        view_dir.y * sy + view_dir.z * cy,
    );

    let sx = sin(rotate.x);
    let cx = cos(rotate.x);
    return vec3<f32>(
        x_rot.x * cx + x_rot.z * sx,
        x_rot.y,
        -x_rot.x * sx + x_rot.z * cx,
    );
}

fn perspective_view_dir(ndc: vec2<f32>) -> vec3<f32> {
    var plane_pos = ndc;
    plane_pos.y = -plane_pos.y;
    let plane_dir = tan(mat._PerspectiveFOV.xy * 0.5) * plane_pos;
    return rotate_dir(normalize(vec3<f32>(plane_dir, 1.0)), mat._PerspectiveFOV.zw);
}

fn apply_offset(view_dir: vec3<f32>) -> vec3<f32> {
    if (!uvu::kw_enabled(mat._OFFSET)) {
        return view_dir;
    }

    let offset_uv = dir_to_uv(view_dir);
    let offset_sample =
        textureSampleLevel(_OffsetTex, _OffsetTex_sampler, uvu::apply_st(offset_uv, mat._OffsetTex_ST), 0.0).rg;
    let offset_mask = textureSampleLevel(_OffsetMask, _OffsetMask_sampler, uvu::flip_v(offset_uv), 0.0).rg;
    let offset = (offset_sample * 2.0 - vec2<f32>(1.0)) * offset_mask * mat._OffsetMagnitude.xy;
    return rotate_dir(view_dir, offset);
}

fn is_outside_uv(uv: vec2<f32>) -> bool {
    return uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
}

fn sample_equirect(view_dir: vec3<f32>, view_layer: u32) -> vec4<f32> {
    var uv = dir_to_uv(view_dir);
    if (is_outside_uv(uv)) {
        if (uvu::kw_enabled(mat.OUTSIDE_COLOR)) {
            return mat._OutsideColor;
        }
        if (!uvu::kw_enabled(mat.OUTSIDE_CLAMP)) {
            discard;
        }
    }
    uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

    var st = mat._MainTex_ST;
    if (uvu::kw_enabled(mat._RIGHT_EYE_ST) && view_layer != 0u) {
        st = mat._RightEye_ST;
    }
    let sample_uv = uvu::apply_st_for_storage(uv, st, mat._MainTex_StorageVInverted);
    var c = textureSampleLevel(_MainTex, _MainTex_sampler, sample_uv, 0.0);
    if (uvu::kw_enabled(mat.SECOND_TEXTURE)) {
        let secondary_offset = vec2<f32>(mat._SecondTexOffset.x, -mat._SecondTexOffset.y);
        let sc = textureSampleLevel(_SecondTex, _SecondTex_sampler, sample_uv + secondary_offset, 0.0);
        c = mix(c, sc, clamp(mat._TextureLerp, 0.0, 1.0));
    }

    if (uvu::kw_enabled(mat.TINT_TEX_DIRECT)) {
        c = c * textureSampleLevel(_TintTex, _TintTex_sampler, sample_uv, 0.0);
    } else if (uvu::kw_enabled(mat.TINT_TEX_LERP)) {
        let tint_uv = uvu::apply_st(uv, vec4<f32>(mat._TintTex_ST.xy, mat._TintTex_ST.w, mat._TintTex_ST.z));
        let l = textureSampleLevel(_TintTex, _TintTex_sampler, tint_uv, 0.0).r;
        c = c * mix(mat._Tint0, mat._Tint1, l);
    }
    return c;
}

fn cubemap_storage_dir(dir: vec3<f32>, storage_v_inverted: f32) -> vec3<f32> {
    if (!uvu::kw_enabled(storage_v_inverted)) {
        return dir;
    }
    let a = abs(dir);
    if (a.y >= a.x && a.y >= a.z) {
        return vec3<f32>(dir.x, dir.y, -dir.z);
    }
    return vec3<f32>(dir.x, -dir.y, dir.z);
}

fn sample_cubemap(view_dir: vec3<f32>) -> vec4<f32> {
    let dir = normalize(-view_dir);
    var lod = 0.0;
    if (uvu::kw_enabled(mat.CUBEMAP_LOD)) {
        lod = mat._CubeLOD;
    }
    let main_dir = cubemap_storage_dir(dir, mat._MainCube_StorageVInverted);
    var c = textureSampleLevel(_MainCube, _MainCube_sampler, main_dir, lod);
    if (uvu::kw_enabled(mat.SECOND_TEXTURE)) {
        let second_dir = cubemap_storage_dir(dir, mat._SecondCube_StorageVInverted);
        let sc = textureSampleLevel(_SecondCube, _SecondCube_sampler, second_dir, lod);
        c = mix(c, sc, clamp(mat._TextureLerp, 0.0, 1.0));
    }
    return c;
}

fn finish_color(c_in: vec4<f32>) -> vec4<f32> {
    var c = c_in;
    c = vec4<f32>(
        pow(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(max(mat._Gamma, 0.000001))) * mat._Exposure,
        c.a,
    ) * mat._Tint;

    if (uvu::kw_enabled(mat._CLAMP_INTENSITY)) {
        let m = max(c.r, max(c.g, c.b));
        if (m > mat._MaxIntensity && m > 0.0) {
            c = vec4<f32>(c.rgb * (mat._MaxIntensity / m), c.a);
        }
    }
    return rg::retain_globals_additive(c);
}

fn base_view_dir(ndc: vec2<f32>, view_layer: u32) -> vec3<f32> {
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, view_layer != 0u);
    let camera_ray_view = skybox::view_ray_from_ndc(ndc, proj_params);
    let camera_ray_world = skybox::world_ray_from_view_ray(camera_ray_view, view, view_layer);

    if (uvu::kw_enabled(mat._PERSPECTIVE)) {
        return perspective_view_dir(ndc);
    }
    if (uvu::kw_enabled(mat._NORMAL) || uvu::kw_enabled(mat._WORLD_VIEW)) {
        return normalize(-camera_ray_world);
    }
    return normalize(-camera_ray_view);
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
    out.ndc = clip.xy;
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
    if (uvu::kw_enabled(mat.CUBEMAP) || uvu::kw_enabled(mat.CUBEMAP_LOD)) {
        c = sample_cubemap(view_dir);
    } else {
        c = sample_equirect(view_dir, in.view_layer);
    }
    return finish_color(c);
}
