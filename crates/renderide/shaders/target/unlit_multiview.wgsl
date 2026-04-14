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

struct UnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    flags: u32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX: f32 = 0f;

@group(2) @binding(0) 
var<storage> instancesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX: array<PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX>;
@group(0) @binding(0) 
var<uniform> frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX;
@group(0) @binding(1) 
var<storage> lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX>;
@group(0) @binding(2) 
var<storage> cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(3) 
var<storage> cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(1) @binding(0) 
var<uniform> mat: UnlitMaterial;
@group(1) @binding(1) 
var _Tex: texture_2d<f32>;
@group(1) @binding(2) 
var _Tex_sampler: sampler;
@group(1) @binding(3) 
var _OffsetTex: texture_2d<f32>;
@group(1) @binding(4) 
var _OffsetTex_sampler: sampler;
@group(1) @binding(5) 
var _MaskTex: texture_2d<f32>;
@group(1) @binding(6) 
var _MaskTex_sampler: sampler;

fn get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_idx: u32) -> PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    let _e3: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = instancesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX[instance_idx];
    return _e3;
}

fn apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_in * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_1: vec2<f32>) -> f32 {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_1, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4.w;
}

fn mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_1: texture_2d<f32>, samp_1: sampler, uv_2: vec2<f32>) -> f32 {
    let mask: vec4<f32> = textureSampleLevel(tex_1, samp_1, uv_2, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return ((((mask.x + mask.y) + mask.z) * 0.33333334f) * mask.w);
}

fn retain_globals_additiveX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX(color: vec4<f32>) -> vec4<f32> {
    var lit: u32 = 0u;

    let _e3: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e3 > 0u) {
        let _e9: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e9;
    }
    let _e13: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e21: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let cluster_touch: f32 = ((f32((_e13 & 255u)) * 0.0000000001f) + (f32((_e21 & 255u)) * 0.0000000001f));
    let _e28: u32 = lit;
    return (color + vec4<f32>(vec3(((f32(_e28) * 0.0000000001f) + cluster_touch)), 0f));
}

@vertex 
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) _n: vec4<f32>, @location(2) uv: vec2<f32>) -> VertexOutput {
    var vp: mat4x4<f32>;
    var out: VertexOutput;

    let _e1: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_index);
    let world_p: vec4<f32> = (_e1.model * vec4<f32>(pos.xyz, 1f));
    if (view_idx == 0u) {
        vp = _e1.view_proj_left;
    } else {
        vp = _e1.view_proj_right;
    }
    let _e16: mat4x4<f32> = vp;
    out.clip_pos = (_e16 * world_p);
    out.uv = uv;
    let _e20: VertexOutput = out;
    return _e20;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var albedo: vec4<f32>;
    var clip_a: f32;
    var uv_main: vec2<f32>;
    var local: bool;

    let _e2: vec4<f32> = mat._Color;
    albedo = _e2;
    let _e7: f32 = mat._Color.w;
    clip_a = _e7;
    let _e11: u32 = mat.flags;
    if ((_e11 & 1u) != 0u) {
        let _e20: vec4<f32> = mat._Tex_ST;
        let _e21: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(in.uv, _e20);
        uv_main = _e21;
        let _e25: u32 = mat.flags;
        if ((_e25 & 4u) != 0u) {
            let _e33: vec4<f32> = mat._OffsetTex_ST;
            let _e34: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(in.uv, _e33);
            let offset_s: vec4<f32> = textureSample(_OffsetTex, _OffsetTex_sampler, _e34);
            let _e38: vec2<f32> = uv_main;
            let _e42: vec4<f32> = mat._OffsetMagnitude;
            uv_main = (_e38 + (offset_s.xy * _e42.xy));
        }
        let _e48: vec2<f32> = uv_main;
        let t: vec4<f32> = textureSample(_Tex, _Tex_sampler, _e48);
        let _e53: f32 = mat._Color.w;
        let _e54: vec2<f32> = uv_main;
        let _e57: f32 = texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_Tex, _Tex_sampler, _e54);
        clip_a = (_e53 * _e57);
        let _e59: vec4<f32> = albedo;
        albedo = (_e59 * t);
    }
    let _e63: u32 = mat.flags;
    if ((_e63 & 24u) != 0u) {
        let _e71: vec4<f32> = mat._MaskTex_ST;
        let _e72: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(in.uv, _e71);
        let mask_1: vec4<f32> = textureSample(_MaskTex, _MaskTex_sampler, _e72);
        let mul: f32 = ((((mask_1.x + mask_1.y) + mask_1.z) * 0.33333334f) * mask_1.w);
        let _e87: f32 = mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MaskTex, _MaskTex_sampler, _e72);
        let _e90: u32 = mat.flags;
        if ((_e90 & 8u) != 0u) {
            let _e97: f32 = albedo.w;
            albedo.w = (_e97 * mul);
            let _e99: f32 = clip_a;
            clip_a = (_e99 * _e87);
        }
        let _e103: u32 = mat.flags;
        if ((_e103 & 16u) != 0u) {
            let _e110: f32 = mat._Cutoff;
            if (_e87 <= _e110) {
                discard;
            }
        }
    }
    let _e114: u32 = mat.flags;
    if ((_e114 & 2u) != 0u) {
        let _e121: u32 = mat.flags;
        local = ((_e121 & 16u) == 0u);
    } else {
        local = false;
    }
    let _e129: bool = local;
    if _e129 {
        let _e130: f32 = clip_a;
        let _e133: f32 = mat._Cutoff;
        if (_e130 <= _e133) {
            discard;
        }
    }
    let _e137: u32 = mat.flags;
    if ((_e137 & 32u) != 0u) {
        let _e142: vec4<f32> = albedo;
        let _e145: f32 = albedo.w;
        let _e148: f32 = albedo.w;
        albedo = vec4<f32>((_e142.xyz * _e145), _e148);
    }
    let _e152: u32 = mat.flags;
    if ((_e152 & 64u) != 0u) {
        let _e158: f32 = albedo.x;
        let _e160: f32 = albedo.y;
        let _e163: f32 = albedo.z;
        let lum: f32 = (((_e158 + _e160) + _e163) * 0.33333334f);
        let _e169: f32 = albedo.w;
        albedo.w = (_e169 * lum);
    }
    let _e171: vec4<f32> = albedo;
    let _e172: vec4<f32> = retain_globals_additiveX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX(_e171);
    return _e172;
}
