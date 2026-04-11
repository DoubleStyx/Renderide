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

struct UiTextUnlitMaterial {
    _TintColor: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _OutlineColor: vec4<f32>,
    _BackgroundColor: vec4<f32>,
    _Range: vec4<f32>,
    _Rect: vec4<f32>,
    _FaceDilate: f32,
    _FaceSoftness: f32,
    _OutlineSize: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _StencilComp: f32,
    _Stencil: f32,
    _StencilOp: f32,
    _StencilWriteMask: f32,
    _StencilReadMask: f32,
    _ColorMask: f32,
    _TextMode: f32,
    _RectClip: f32,
    _pad: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) extra_data: vec4<f32>,
    @location(2) vtx_color: vec4<f32>,
    @location(3) obj_xy: vec2<f32>,
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
var<uniform> mat: UiTextUnlitMaterial;
@group(1) @binding(1) 
var _FontAtlas: texture_2d<f32>;
@group(1) @binding(2) 
var _FontAtlas_sampler: sampler;

fn view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx_1: u32) -> mat4x4<f32> {
    if (view_idx_1 == 0u) {
        let _e5: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
        return _e5;
    }
    let _e8: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_right;
    return _e8;
}

fn texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_1: vec2<f32>) -> vec4<f32> {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_1, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
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

fn median3_(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

fn text_mode_clamped(tm: f32) -> i32 {
    return clamp(i32(round(tm)), 0i, 2i);
}

fn outside_rect_clip(p: vec2<f32>, r_1: vec4<f32>) -> bool {
    var local_2: bool;
    var local_3: bool;
    var local_4: bool;

    let min_v: vec2<f32> = r_1.xy;
    let max_v: vec2<f32> = (r_1.xy + r_1.zw);
    if !((p.x < min_v.x)) {
        local_2 = (p.x > max_v.x);
    } else {
        local_2 = true;
    }
    let _e16: bool = local_2;
    if !(_e16) {
        local_3 = (p.y < min_v.y);
    } else {
        local_3 = true;
    }
    let _e24: bool = local_3;
    if !(_e24) {
        local_4 = (p.y > max_v.y);
    } else {
        local_4 = true;
    }
    let _e32: bool = local_4;
    return _e32;
}

fn shade_distance_field(sig_dist_in: f32, vo: VertexOutput, vtx_color: vec4<f32>, range_xy: vec2<f32>) -> vec4<f32> {
    var sig_dist: f32;
    var glyph_lerp: f32;
    var fill_color: vec4<f32>;
    var local_5: bool;
    var outline_lerp: f32;

    sig_dist = sig_dist_in;
    let _e2: f32 = sig_dist;
    let _e6: f32 = mat._FaceDilate;
    sig_dist = ((_e2 + _e6) + vo.extra_data.x);
    let _e13: f32 = fwidth(vo.uv.x);
    let _e16: f32 = fwidth(vo.uv.y);
    let fw: vec2<f32> = vec2<f32>(_e13, _e16);
    let anti_aliasing: f32 = dot(range_xy, (vec2(0.5f) / max(fw, vec2(0.000001f))));
    let aa: f32 = max(anti_aliasing, 1f);
    let _e28: f32 = sig_dist;
    let _e30: f32 = sig_dist;
    let _e33: f32 = mat._FaceSoftness;
    glyph_lerp = mix((_e28 * aa), _e30, _e33);
    let _e36: f32 = glyph_lerp;
    glyph_lerp = clamp((_e36 + 0.5f), 0f, 1f);
    let _e42: f32 = glyph_lerp;
    let _e46: f32 = mat._BackgroundColor.w;
    if (max(_e42, _e46) < 0.001f) {
        discard;
    }
    let _e53: vec4<f32> = mat._TintColor;
    fill_color = (_e53 * vtx_color);
    let _e58: f32 = mat._OutlineSize;
    let outline_w: f32 = (_e58 + vo.extra_data.y);
    if !((outline_w > 0.000001f)) {
        let _e67: f32 = mat._OutlineSize;
        local_5 = (_e67 > 0.000001f);
    } else {
        local_5 = true;
    }
    let _e73: bool = local_5;
    if _e73 {
        let _e74: f32 = sig_dist;
        let outline_dist: f32 = (_e74 - outline_w);
        let _e79: f32 = mat._FaceSoftness;
        outline_lerp = mix((outline_dist * aa), outline_dist, _e79);
        let _e82: f32 = outline_lerp;
        outline_lerp = clamp((_e82 + 0.5f), 0f, 1f);
        let _e90: vec4<f32> = mat._OutlineColor;
        let _e97: vec4<f32> = fill_color;
        let _e98: f32 = outline_lerp;
        fill_color = mix((_e90 * vec4<f32>(1f, 1f, 1f, vtx_color.w)), _e97, _e98);
    }
    let _e102: vec4<f32> = mat._BackgroundColor;
    let _e104: vec4<f32> = fill_color;
    let _e105: f32 = glyph_lerp;
    return mix((_e102 * vtx_color), _e104, _e105);
}

@vertex 
fn vs_main(@builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) extra_n: vec4<f32>, @location(2) uv: vec2<f32>, @location(3) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let _e9: mat4x4<f32> = view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx);
    out.clip_pos = (_e9 * world_p);
    out.uv = uv;
    out.extra_data = extra_n;
    out.vtx_color = color;
    out.obj_xy = pos.xy;
    let _e21: VertexOutput = out;
    return _e21;
}

@fragment 
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    var local: bool;
    var local_1: bool;
    var c: vec4<f32>;

    let vtx_color_1: vec4<f32> = vout.vtx_color;
    let rect: vec4<f32> = mat._Rect;
    let _e7: f32 = mat._RectClip;
    if (_e7 > 0.5f) {
        local = ((rect.z * rect.w) > 0.000001f);
    } else {
        local = false;
    }
    let use_rect_clip: bool = local;
    if use_rect_clip {
        let _e20: bool = outside_rect_clip(vout.obj_xy, rect);
        local_1 = _e20;
    } else {
        local_1 = false;
    }
    let _e24: bool = local_1;
    if _e24 {
        discard;
    }
    let atlas_color: vec4<f32> = textureSample(_FontAtlas, _FontAtlas_sampler, vout.uv);
    let _e32: vec4<f32> = texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_FontAtlas, _FontAtlas_sampler, vout.uv);
    let _e35: vec4<f32> = mat._Range;
    let range_xy_1: vec2<f32> = _e35.xy;
    let _e39: f32 = mat._TextMode;
    let _e40: i32 = text_mode_clamped(_e39);
    if (_e40 == 1i) {
        c = (atlas_color * vtx_color_1);
        if ((_e32.w * vtx_color_1.w) < 0.001f) {
            discard;
        }
    } else {
        if (_e40 == 2i) {
            let sig_dist_1: f32 = (_e32.w - 0.5f);
            let _e55: vec4<f32> = shade_distance_field(sig_dist_1, vout, vtx_color_1, range_xy_1);
            c = _e55;
        } else {
            let _e59: f32 = median3_(_e32.x, _e32.y, _e32.z);
            let sig_dist_2: f32 = (_e59 - 0.5f);
            let _e62: vec4<f32> = shade_distance_field(sig_dist_2, vout, vtx_color_1, range_xy_1);
            c = _e62;
        }
    }
    let o: vec4<f32> = mat._OverlayTint;
    if (o.w > 0.01f) {
        let _e69: vec4<f32> = c;
        let _e78: f32 = c.w;
        c = vec4<f32>((_e69.xyz * mix(vec3(1f), o.xyz, o.w)), _e78);
    }
    let _e80: vec4<f32> = c;
    let _e81: vec3<f32> = fragment_watermark_rgbX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX3SMV2GK3TUNFXW4X();
    return (_e80 + vec4<f32>(_e81, 0f));
}
