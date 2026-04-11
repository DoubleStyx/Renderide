struct PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

struct FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    camera_world_pos: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    view_space_z_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    stereo_cluster_layers: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}

struct GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    position: vec3<f32>,
    align_pad_vec3_pos: f32,
    direction: vec3<f32>,
    align_pad_vec3_dir: f32,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    align_pad_before_shadow: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    align_pad_vec3_tail: vec3<u32>,
}

struct OverlayUnlitMaterial {
    _BehindColor: vec4<f32>,
    _FrontColor: vec4<f32>,
    _BehindTex_ST: vec4<f32>,
    _FrontTex_ST: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _POLARUV: f32,
    _MUL_RGB_BY_ALPHA: f32,
    _MUL_ALPHA_INTENSITY: f32,
    _pad0_: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX: f32 = 0f;

@group(2) @binding(0) 
var<uniform> drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX;
@group(0) @binding(0) 
var<uniform> frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX;
@group(0) @binding(1) 
var<storage> lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX>;
@group(0) @binding(2) 
var<storage> cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(3) 
var<storage> cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(1) @binding(0) 
var<uniform> mat: OverlayUnlitMaterial;
@group(1) @binding(1) 
var _BehindTex: texture_2d<f32>;
@group(1) @binding(2) 
var _BehindTex_sampler: sampler;
@group(1) @binding(3) 
var _FrontTex: texture_2d<f32>;
@group(1) @binding(4) 
var _FrontTex_sampler: sampler;

fn view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx_1: u32) -> mat4x4<f32> {
    if (view_idx_1 == 0u) {
        let _e5: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
        return _e5;
    }
    let _e8: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_right;
    return _e8;
}

fn apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv_1: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_1 * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn polar_uvX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYG63DBOJPXK5QX(raw_uv: vec2<f32>, radius_pow: f32) -> vec2<f32> {
    let centered: vec2<f32> = ((raw_uv * 2f) - vec2(1f));
    let radius: f32 = pow(length(centered), radius_pow);
    let angle: f32 = (atan2(centered.x, centered.y) + (6.2831855f * 0.5f));
    return vec2<f32>((angle / 6.2831855f), radius);
}

fn texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_2: vec2<f32>) -> vec4<f32> {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_2, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4;
}

fn fragment_watermark_rgbX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX3SMV2GK3TUNFXW4X() -> vec3<f32> {
    var lit: u32 = 0u;

    let _e3: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e3 > 0u) {
        let _e9: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e9;
    }
    let _e13: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e21: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e30: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e41: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let cluster_touch: f32 = (((f32((_e13 & 255u)) * 0.0000000001f) + (f32((_e21 & 255u)) * 0.0000000001f)) + ((dot(_e30, vec4<f32>(1f, 1f, 1f, 1f)) * 0.0000000001f) + (f32(_e41) * 0.0000000001f)));
    let _e47: u32 = lit;
    return vec3(((f32(_e47) * 0.0000000001f) + cluster_touch));
}

fn sample_layer(tex_1: texture_2d<f32>, samp_1: sampler, tint: vec4<f32>, uv_3: vec2<f32>, st_1: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv_3, st_1);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uvX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYG63DBOJPXK5QX(uv_3, _e10);
    let _e12: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(_e11, st_1);
    let sample_uv: vec2<f32> = select(_e7, _e12, use_polar);
    let _e17: vec4<f32> = textureSample(tex_1, samp_1, sample_uv);
    return (_e17 * tint);
}

fn sample_layer_lod0_(tex_2: texture_2d<f32>, samp_2: sampler, tint_1: vec4<f32>, uv_4: vec2<f32>, st_2: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar_1: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv_4, st_2);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uvX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYG63DBOJPXK5QX(uv_4, _e10);
    let _e12: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(_e11, st_2);
    let sample_uv_1: vec2<f32> = select(_e7, _e12, use_polar_1);
    let _e16: vec4<f32> = texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_2, samp_2, sample_uv_1);
    return (_e16 * tint_1);
}

fn alpha_over(front: vec4<f32>, behind: vec4<f32>) -> vec4<f32> {
    let out_a: f32 = (front.w + (behind.w * (1f - front.w)));
    if (out_a <= 0.000001f) {
        return vec4(0f);
    }
    let out_rgb: vec3<f32> = (((front.xyz * front.w) + ((behind.xyz * behind.w) * (1f - front.w))) / vec3(out_a));
    return vec4<f32>(out_rgb, out_a);
}

@vertex 
fn vs_main(@builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) _n: vec4<f32>, @location(2) uv: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let _e9: mat4x4<f32> = view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx);
    out.clip_pos = (_e9 * world_p);
    out.uv = uv;
    let _e15: VertexOutput = out;
    return _e15;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec4<f32>;
    var local: bool;
    var local_1: bool;

    let _e3: vec4<f32> = mat._BehindColor;
    let _e7: vec4<f32> = mat._BehindTex_ST;
    let _e10: vec4<f32> = sample_layer(_BehindTex, _BehindTex_sampler, _e3, in.uv, _e7);
    let _e13: vec4<f32> = mat._FrontColor;
    let _e17: vec4<f32> = mat._FrontTex_ST;
    let _e20: vec4<f32> = sample_layer(_FrontTex, _FrontTex_sampler, _e13, in.uv, _e17);
    let _e21: vec4<f32> = alpha_over(_e20, _e10);
    color = _e21;
    let _e25: vec4<f32> = mat._BehindColor;
    let _e29: vec4<f32> = mat._BehindTex_ST;
    let _e32: vec4<f32> = sample_layer_lod0_(_BehindTex, _BehindTex_sampler, _e25, in.uv, _e29);
    let _e35: vec4<f32> = mat._FrontColor;
    let _e39: vec4<f32> = mat._FrontTex_ST;
    let _e42: vec4<f32> = sample_layer_lod0_(_FrontTex, _FrontTex_sampler, _e35, in.uv, _e39);
    let _e43: vec4<f32> = alpha_over(_e42, _e32);
    let _e46: f32 = mat._Cutoff;
    if (_e46 > 0f) {
        let _e51: f32 = mat._Cutoff;
        local = (_e51 < 1f);
    } else {
        local = false;
    }
    let _e57: bool = local;
    if _e57 {
        let _e61: f32 = mat._Cutoff;
        local_1 = (_e43.w <= _e61);
    } else {
        local_1 = false;
    }
    let _e66: bool = local_1;
    if _e66 {
        discard;
    }
    let _e69: f32 = mat._MUL_RGB_BY_ALPHA;
    if (_e69 > 0.99f) {
        let _e72: vec4<f32> = color;
        let _e75: f32 = color.w;
        let _e78: f32 = color.w;
        color = vec4<f32>((_e72.xyz * _e75), _e78);
    }
    let _e82: f32 = mat._MUL_ALPHA_INTENSITY;
    if (_e82 > 0.99f) {
        let _e86: f32 = color.x;
        let _e88: f32 = color.y;
        let _e91: f32 = color.z;
        let lum: f32 = (((_e86 + _e88) + _e91) * 0.33333334f);
        let _e97: f32 = color.w;
        color.w = (_e97 * lum);
    }
    let _e99: vec4<f32> = color;
    let _e100: vec3<f32> = fragment_watermark_rgbX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX3SMV2GK3TUNFXW4X();
    return (_e99 + vec4<f32>(_e100, 0f));
}
