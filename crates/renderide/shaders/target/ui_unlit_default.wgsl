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
    directional_light_count: u32,
    _pad_frame: vec2<u32>,
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

struct UiUnlitMaterial {
    _MainTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _Tint: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _Rect: vec4<f32>,
    _Cutoff: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _StencilComp: f32,
    _Stencil: f32,
    _StencilOp: f32,
    _StencilWriteMask: f32,
    _StencilReadMask: f32,
    _ColorMask: f32,
    flags: u32,
    _pad_end: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) obj_xy: vec2<f32>,
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
var<uniform> mat: UiUnlitMaterial;
@group(1) @binding(1) 
var _MainTex: texture_2d<f32>;
@group(1) @binding(2) 
var _MainTex_sampler: sampler;
@group(1) @binding(3) 
var _MaskTex: texture_2d<f32>;
@group(1) @binding(4) 
var _MaskTex_sampler: sampler;

fn view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx: u32) -> mat4x4<f32> {
    let _e2: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
    return _e2;
}

fn apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv_1: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_1 * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_2: vec2<f32>) -> f32 {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_2, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4.w;
}

fn mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_1: texture_2d<f32>, samp_1: sampler, uv_3: vec2<f32>) -> f32 {
    let mask: vec4<f32> = textureSampleLevel(tex_1, samp_1, uv_3, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return ((((mask.x + mask.y) + mask.z) * 0.33333334f) * mask.w);
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

@vertex 
fn vs_main(@location(0) pos: vec4<f32>, @location(1) _n: vec4<f32>, @location(2) uv: vec2<f32>, @location(3) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let _e9: mat4x4<f32> = view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(0u);
    out.clip_pos = (_e9 * world_p);
    out.uv = uv;
    let _e19: vec4<f32> = mat._Tint;
    out.color = (color * _e19);
    out.obj_xy = pos.xy;
    let _e23: VertexOutput = out;
    return _e23;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color_1: vec4<f32>;
    var clip_a: f32;
    var local: bool;
    var local_1: bool;
    var local_2: bool;
    var local_3: bool;

    color_1 = in.color;
    clip_a = in.color.w;
    let _e8: u32 = mat.flags;
    if ((_e8 & 1u) != 0u) {
        let _e16: vec4<f32> = mat._MainTex_ST;
        let _e17: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(in.uv, _e16);
        let t: vec4<f32> = textureSample(_MainTex, _MainTex_sampler, _e17);
        let _e25: f32 = texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MainTex, _MainTex_sampler, _e17);
        clip_a = (in.color.w * _e25);
        let _e27: vec4<f32> = color_1;
        color_1 = (_e27 * t);
    }
    let _e31: u32 = mat.flags;
    if ((_e31 & 4u) != 0u) {
        let r: vec4<f32> = mat._Rect;
        let min_v: vec2<f32> = r.xy;
        let max_v: vec2<f32> = (r.xy + r.zw);
        if !((in.obj_xy.x < min_v.x)) {
            local = (in.obj_xy.x > max_v.x);
        } else {
            local = true;
        }
        let _e55: bool = local;
        if !(_e55) {
            local_1 = (in.obj_xy.y < min_v.y);
        } else {
            local_1 = true;
        }
        let _e64: bool = local_1;
        if !(_e64) {
            local_2 = (in.obj_xy.y > max_v.y);
        } else {
            local_2 = true;
        }
        let _e73: bool = local_2;
        if _e73 {
            discard;
        }
    }
    let _e76: u32 = mat.flags;
    if ((_e76 & 48u) != 0u) {
        let _e84: vec4<f32> = mat._MaskTex_ST;
        let _e85: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(in.uv, _e84);
        let mask_1: vec4<f32> = textureSample(_MaskTex, _MaskTex_sampler, _e85);
        let mul: f32 = ((((mask_1.x + mask_1.y) + mask_1.z) * 0.33333334f) * mask_1.w);
        let _e100: f32 = mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MaskTex, _MaskTex_sampler, _e85);
        let _e103: u32 = mat.flags;
        if ((_e103 & 16u) != 0u) {
            let _e110: f32 = color_1.w;
            color_1.w = (_e110 * mul);
            let _e112: f32 = clip_a;
            clip_a = (_e112 * _e100);
        }
        let _e116: u32 = mat.flags;
        if ((_e116 & 32u) != 0u) {
            let _e123: f32 = mat._Cutoff;
            if (_e100 <= _e123) {
                discard;
            }
        }
    }
    let _e127: u32 = mat.flags;
    if ((_e127 & 2u) != 0u) {
        let _e134: u32 = mat.flags;
        local_3 = ((_e134 & 32u) == 0u);
    } else {
        local_3 = false;
    }
    let _e142: bool = local_3;
    if _e142 {
        let _e143: f32 = clip_a;
        let _e146: f32 = mat._Cutoff;
        if (_e143 <= _e146) {
            discard;
        }
    }
    let _e150: u32 = mat.flags;
    if ((_e150 & 8u) != 0u) {
        let o: vec4<f32> = mat._OverlayTint;
        let _e158: vec4<f32> = color_1;
        let _e167: f32 = color_1.w;
        color_1 = vec4<f32>((_e158.xyz * mix(vec3(1f), o.xyz, o.w)), _e167);
    }
    let _e169: vec4<f32> = color_1;
    let _e170: vec3<f32> = fragment_watermark_rgbX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX3SMV2GK3TUNFXW4X();
    return (_e169 + vec4<f32>(_e170, 0f));
}
