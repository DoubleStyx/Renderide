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

struct UnlitUvRectMaterial {
    _Rect: vec4<f32>,
    _ClipRect: vec4<f32>,
    _OuterColor: vec4<f32>,
    _InnerColor: vec4<f32>,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _RectClip: f32,
    _pad0_: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

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
var<uniform> mat: UnlitUvRectMaterial;

fn view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx_1: u32) -> mat4x4<f32> {
    if (view_idx_1 == 0u) {
        let _e5: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
        return _e5;
    }
    let _e8: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_right;
    return _e8;
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

fn inside_rect01_(p: vec2<f32>, r: vec4<f32>) -> f32 {
    let inside_x: f32 = (step(r.x, p.x) * step(p.x, r.z));
    let inside_y: f32 = (step(r.y, p.y) * step(p.y, r.w));
    return (inside_x * inside_y);
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
    var local: bool;

    let _e2: f32 = mat._RectClip;
    if (_e2 > 0.5f) {
        let _e9: vec4<f32> = mat._ClipRect;
        let _e10: f32 = inside_rect01_(in.uv, _e9);
        local = (_e10 < 0.5f);
    } else {
        local = false;
    }
    let _e16: bool = local;
    if _e16 {
        discard;
    }
    let _e20: vec4<f32> = mat._Rect;
    let _e21: f32 = inside_rect01_(in.uv, _e20);
    let _e24: vec4<f32> = mat._OuterColor;
    let _e27: vec4<f32> = mat._InnerColor;
    let color: vec4<f32> = mix(_e24, _e27, _e21);
    let _e29: vec3<f32> = fragment_watermark_rgbX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX3SMV2GK3TUNFXW4X();
    return (color + vec4<f32>(_e29, 0f));
}
