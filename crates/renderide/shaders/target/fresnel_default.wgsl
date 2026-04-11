struct PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
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

struct FresnelMaterial {
    _FarColor: vec4<f32>,
    _NearColor: vec4<f32>,
    _FarTex_ST: vec4<f32>,
    _NearTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _Exp: f32,
    _GammaCurve: f32,
    _NormalScale: f32,
    _Cutoff: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _PolarPow: f32,
    _POLARUV: f32,
    _NORMALMAP: f32,
    _MASK_TEXTURE_MUL: f32,
    _MASK_TEXTURE_CLIP: f32,
    _MUL_ALPHA_INTENSITY: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
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
var<uniform> mat: FresnelMaterial;
@group(1) @binding(1) 
var _FarTex: texture_2d<f32>;
@group(1) @binding(2) 
var _FarTex_sampler: sampler;
@group(1) @binding(3) 
var _NearTex: texture_2d<f32>;
@group(1) @binding(4) 
var _NearTex_sampler: sampler;
@group(1) @binding(5) 
var _NormalMap: texture_2d<f32>;
@group(1) @binding(6) 
var _NormalMap_sampler: sampler;
@group(1) @binding(7) 
var _MaskTex: texture_2d<f32>;
@group(1) @binding(8) 
var _MaskTex_sampler: sampler;

fn view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx: u32) -> mat4x4<f32> {
    let _e2: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
    return _e2;
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

fn mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_1: texture_2d<f32>, samp_1: sampler, uv_3: vec2<f32>) -> f32 {
    let mask: vec4<f32> = textureSampleLevel(tex_1, samp_1, uv_3, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return ((((mask.x + mask.y) + mask.z) * 0.33333334f) * mask.w);
}

fn orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_2: vec3<f32>) -> mat3x3<f32> {
    let up: vec3<f32> = select(vec3<f32>(0f, 1f, 0f), vec3<f32>(1f, 0f, 0f), (abs(n_2.y) > 0.99f));
    let t: vec3<f32> = normalize(cross(up, n_2));
    let b: vec3<f32> = cross(n_2, t);
    return mat3x3<f32>(t, b, n_2);
}

fn decode_ts_normalX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    let nm_xy: vec2<f32> = (((raw.xy * 2f) - vec2(1f)) * scale);
    let z: f32 = max(sqrt(max((1f - dot(nm_xy, nm_xy)), 0f)), 0.000001f);
    return normalize(vec3<f32>(nm_xy, z));
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

fn sample_color(tex_2: texture_2d<f32>, samp_2: sampler, uv_4: vec2<f32>, st_1: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv_4, st_1);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uvX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYG63DBOJPXK5QX(uv_4, _e10);
    let _e12: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(_e11, st_1);
    let sample_uv: vec2<f32> = select(_e7, _e12, use_polar);
    let _e16: vec4<f32> = textureSample(tex_2, samp_2, sample_uv);
    return _e16;
}

fn sample_color_lod0_(tex_3: texture_2d<f32>, samp_3: sampler, uv_5: vec2<f32>, st_2: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar_1: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv_5, st_2);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uvX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYG63DBOJPXK5QX(uv_5, _e10);
    let _e12: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(_e11, st_2);
    let sample_uv_1: vec2<f32> = select(_e7, _e12, use_polar_1);
    let _e16: vec4<f32> = texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_3, samp_3, sample_uv_1);
    return _e16;
}

@vertex 
fn vs_main(@location(0) pos: vec4<f32>, @location(1) n: vec4<f32>, @location(2) uv: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let _e11: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let wn: vec3<f32> = normalize((_e11 * vec4<f32>(n.xyz, 0f)).xyz);
    let _e19: mat4x4<f32> = view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(0u);
    out.clip_pos = (_e19 * world_p);
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv = uv;
    let _e28: VertexOutput = out;
    return _e28;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var n_1: vec3<f32>;
    var fres: f32;
    var color: vec4<f32>;
    var clip_a: f32;
    var local: bool;
    var local_1: bool;
    var local_2: bool;
    var local_3: bool;
    var local_4: bool;

    n_1 = normalize(in.world_n);
    let _e6: f32 = mat._NORMALMAP;
    if (_e6 > 0.99f) {
        let uv_n: vec2<f32> = vec2<f32>(in.uv.x, (1f - in.uv.y));
        let _e16: vec3<f32> = n_1;
        let _e17: mat3x3<f32> = orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(_e16);
        let _e20: vec4<f32> = textureSample(_NormalMap, _NormalMap_sampler, uv_n);
        let _e24: f32 = mat._NormalScale;
        let _e25: vec3<f32> = decode_ts_normalX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(_e20.xyz, _e24);
        n_1 = normalize((_e17 * _e25));
    }
    let _e30: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let view_dir: vec3<f32> = normalize((_e30.xyz - in.world_pos));
    let _e35: vec3<f32> = n_1;
    let _e42: f32 = mat._Exp;
    fres = pow((1f - abs(dot(_e35, view_dir))), max(_e42, 0.0001f));
    let _e47: f32 = fres;
    let _e53: f32 = mat._GammaCurve;
    fres = pow(clamp(_e47, 0f, 1f), max(_e53, 0.0001f));
    let _e59: vec4<f32> = mat._FarColor;
    let _e63: vec4<f32> = mat._FarTex_ST;
    let _e66: vec4<f32> = sample_color(_FarTex, _FarTex_sampler, in.uv, _e63);
    let far_color: vec4<f32> = (_e59 * _e66);
    let _e70: vec4<f32> = mat._NearColor;
    let _e74: vec4<f32> = mat._NearTex_ST;
    let _e77: vec4<f32> = sample_color(_NearTex, _NearTex_sampler, in.uv, _e74);
    let near_color: vec4<f32> = (_e70 * _e77);
    let _e79: f32 = fres;
    color = mix(near_color, far_color, clamp(_e79, 0f, 1f));
    let _e87: vec4<f32> = mat._FarColor;
    let _e91: vec4<f32> = mat._FarTex_ST;
    let _e94: vec4<f32> = sample_color_lod0_(_FarTex, _FarTex_sampler, in.uv, _e91);
    let far_clip: vec4<f32> = (_e87 * _e94);
    let _e98: vec4<f32> = mat._NearColor;
    let _e102: vec4<f32> = mat._NearTex_ST;
    let _e105: vec4<f32> = sample_color_lod0_(_NearTex, _NearTex_sampler, in.uv, _e102);
    let near_clip: vec4<f32> = (_e98 * _e105);
    let _e109: f32 = fres;
    clip_a = mix(near_clip.w, far_clip.w, clamp(_e109, 0f, 1f));
    let _e117: f32 = mat._MASK_TEXTURE_MUL;
    if !((_e117 > 0.99f)) {
        let _e123: f32 = mat._MASK_TEXTURE_CLIP;
        local = (_e123 > 0.99f);
    } else {
        local = true;
    }
    let _e129: bool = local;
    if _e129 {
        let _e133: vec4<f32> = mat._MaskTex_ST;
        let _e134: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(in.uv, _e133);
        let mask_1: vec4<f32> = textureSample(_MaskTex, _MaskTex_sampler, _e134);
        let mul: f32 = ((((mask_1.x + mask_1.y) + mask_1.z) * 0.33333334f) * mask_1.w);
        let _e149: f32 = mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MaskTex, _MaskTex_sampler, _e134);
        let _e152: f32 = mat._MASK_TEXTURE_MUL;
        if (_e152 > 0.99f) {
            let _e157: f32 = color.w;
            color.w = (_e157 * mul);
            let _e159: f32 = clip_a;
            clip_a = (_e159 * _e149);
        }
        let _e163: f32 = mat._MASK_TEXTURE_CLIP;
        if (_e163 > 0.99f) {
            let _e168: f32 = mat._Cutoff;
            local_1 = (_e149 <= _e168);
        } else {
            local_1 = false;
        }
        let _e173: bool = local_1;
        if _e173 {
            discard;
        }
    }
    let _e176: f32 = mat._MASK_TEXTURE_CLIP;
    if !((_e176 > 0.99f)) {
        let _e182: f32 = mat._Cutoff;
        local_2 = (_e182 > 0f);
    } else {
        local_2 = false;
    }
    let _e188: bool = local_2;
    if _e188 {
        let _e191: f32 = mat._Cutoff;
        local_3 = (_e191 < 1f);
    } else {
        local_3 = false;
    }
    let _e197: bool = local_3;
    if _e197 {
        let _e198: f32 = clip_a;
        let _e201: f32 = mat._Cutoff;
        local_4 = (_e198 <= _e201);
    } else {
        local_4 = false;
    }
    let _e206: bool = local_4;
    if _e206 {
        discard;
    }
    let _e209: f32 = mat._MUL_ALPHA_INTENSITY;
    if (_e209 > 0.99f) {
        let _e213: f32 = color.x;
        let _e215: f32 = color.y;
        let _e218: f32 = color.z;
        let lum: f32 = (((_e213 + _e215) + _e218) * 0.33333334f);
        let _e224: f32 = color.w;
        color.w = ((_e224 * lum) * lum);
    }
    let _e227: vec4<f32> = color;
    let _e228: vec3<f32> = fragment_watermark_rgbX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX3SMV2GK3TUNFXW4X();
    return (_e227 + vec4<f32>(_e228, 0f));
}
