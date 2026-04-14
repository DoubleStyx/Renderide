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

struct TextUnitMaterial {
    _TintColor: vec4<f32>,
    _OutlineColor: vec4<f32>,
    _BackgroundColor: vec4<f32>,
    _Range: vec4<f32>,
    _FaceDilate: f32,
    _FaceSoftness: f32,
    _OutlineSize: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _TextMode: f32,
    _pad0_: f32,
    _pad1_: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) extra_data: vec4<f32>,
    @location(2) vtx_color: vec4<f32>,
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
var<uniform> mat: TextUnitMaterial;
@group(1) @binding(1) 
var _FontAtlas: texture_2d<f32>;
@group(1) @binding(2) 
var _FontAtlas_sampler: sampler;

fn get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_idx: u32) -> PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    let _e3: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = instancesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX[instance_idx];
    return _e3;
}

fn texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_1: vec2<f32>) -> vec4<f32> {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_1, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4;
}

fn median3X_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2GK6DUL5ZWIZQX(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

fn text_mode_clampedX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2GK6DUL5ZWIZQX(tm: f32) -> i32 {
    return clamp(i32(round(tm)), 0i, 2i);
}

fn retain_globals_additiveX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX(color_1: vec4<f32>) -> vec4<f32> {
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
    return (color_1 + vec4<f32>(vec3(((f32(_e28) * 0.0000000001f) + cluster_touch)), 0f));
}

fn shade_distance_field(sig_dist_in: f32, vo: VertexOutput, vtx_color: vec4<f32>, range_xy: vec2<f32>) -> vec4<f32> {
    var sig_dist: f32;
    var glyph_lerp: f32;
    var fill_color: vec4<f32>;
    var local: bool;
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
        local = (_e67 > 0.000001f);
    } else {
        local = true;
    }
    let _e73: bool = local;
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
fn vs_main(@builtin(instance_index) instance_index: u32, @location(0) pos: vec4<f32>, @location(1) extra_n: vec4<f32>, @location(2) uv: vec2<f32>, @location(3) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e1: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_index);
    let world_p: vec4<f32> = (_e1.model * vec4<f32>(pos.xyz, 1f));
    let vp: mat4x4<f32> = _e1.view_proj_left;
    out.clip_pos = (vp * world_p);
    out.uv = uv;
    out.extra_data = extra_n;
    out.vtx_color = color;
    let _e18: VertexOutput = out;
    return _e18;
}

@fragment 
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    var c: vec4<f32>;

    let vtx_color_1: vec4<f32> = vout.vtx_color;
    let atlas_color: vec4<f32> = textureSample(_FontAtlas, _FontAtlas_sampler, vout.uv);
    let _e9: vec4<f32> = texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_FontAtlas, _FontAtlas_sampler, vout.uv);
    let _e12: vec4<f32> = mat._Range;
    let range_xy_1: vec2<f32> = _e12.xy;
    let _e16: f32 = mat._TextMode;
    let _e17: i32 = text_mode_clampedX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2GK6DUL5ZWIZQX(_e16);
    if (_e17 == 1i) {
        let _e22: vec4<f32> = mat._TintColor;
        c = ((atlas_color * _e22) * vtx_color_1);
        let _e30: f32 = mat._TintColor.w;
        if (((_e9.w * _e30) * vtx_color_1.w) < 0.001f) {
            discard;
        }
    } else {
        if (_e17 == 2i) {
            let sig_dist_1: f32 = (_e9.w - 0.5f);
            let _e41: vec4<f32> = shade_distance_field(sig_dist_1, vout, vtx_color_1, range_xy_1);
            c = _e41;
        } else {
            let _e45: f32 = median3X_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2GK6DUL5ZWIZQX(_e9.x, _e9.y, _e9.z);
            let sig_dist_2: f32 = (_e45 - 0.5f);
            let _e48: vec4<f32> = shade_distance_field(sig_dist_2, vout, vtx_color_1, range_xy_1);
            c = _e48;
        }
    }
    let _e49: vec4<f32> = c;
    let _e50: vec4<f32> = retain_globals_additiveX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX(_e49);
    return _e50;
}
