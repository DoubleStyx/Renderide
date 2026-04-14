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

struct PbsLerpSpecularMaterial {
    _Color: vec4<f32>,
    _Color1_: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _SpecularColor1_: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EmissionColor1_: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex1_ST: vec4<f32>,
    _LerpTex_ST: vec4<f32>,
    _Lerp: f32,
    _NormalScale: f32,
    _NormalScale1_: f32,
    _AlphaClip: f32,
    _Cull: f32,
    _LERPTEX: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _SPECULARMAP: f32,
    _OCCLUSION: f32,
    _MULTI_VALUES: f32,
    _DUALSIDED: f32,
    _ALPHACLIP: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0_: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

const CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX: f32 = 0f;
const MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX: u32 = 32u;

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
var<uniform> mat: PbsLerpSpecularMaterial;
@group(1) @binding(1) 
var _MainTex: texture_2d<f32>;
@group(1) @binding(2) 
var _MainTex_sampler: sampler;
@group(1) @binding(3) 
var _MainTex1_: texture_2d<f32>;
@group(1) @binding(4) 
var _MainTex1_sampler: sampler;
@group(1) @binding(5) 
var _LerpTex: texture_2d<f32>;
@group(1) @binding(6) 
var _LerpTex_sampler: sampler;
@group(1) @binding(7) 
var _NormalMap: texture_2d<f32>;
@group(1) @binding(8) 
var _NormalMap_sampler: sampler;
@group(1) @binding(9) 
var _NormalMap1_: texture_2d<f32>;
@group(1) @binding(10) 
var _NormalMap1_sampler: sampler;
@group(1) @binding(11) 
var _EmissionMap: texture_2d<f32>;
@group(1) @binding(12) 
var _EmissionMap_sampler: sampler;
@group(1) @binding(13) 
var _EmissionMap1_: texture_2d<f32>;
@group(1) @binding(14) 
var _EmissionMap1_sampler: sampler;
@group(1) @binding(15) 
var _Occlusion: texture_2d<f32>;
@group(1) @binding(16) 
var _Occlusion_sampler: sampler;
@group(1) @binding(17) 
var _Occlusion1_: texture_2d<f32>;
@group(1) @binding(18) 
var _Occlusion1_sampler: sampler;
@group(1) @binding(19) 
var _SpecularMap: texture_2d<f32>;
@group(1) @binding(20) 
var _SpecularMap_sampler: sampler;
@group(1) @binding(21) 
var _SpecularMap1_: texture_2d<f32>;
@group(1) @binding(22) 
var _SpecularMap1_sampler: sampler;

fn apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_in * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(v: f32) -> bool {
    return (v > 0.5f);
}

fn orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_1: vec3<f32>) -> mat3x3<f32> {
    let up: vec3<f32> = select(vec3<f32>(0f, 1f, 0f), vec3<f32>(1f, 0f, 0f), (abs(n_1.y) > 0.99f));
    let t: vec3<f32> = normalize(cross(up, n_1));
    let b: vec3<f32> = cross(n_1, t);
    return mat3x3<f32>(t, b, n_1);
}

fn pow5X_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(x: f32) -> f32 {
    let x2_: f32 = (x * x);
    return ((x2_ * x2_) * x);
}

fn fresnel_schlickX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(cos_theta: f32, f0_: vec3<f32>) -> vec3<f32> {
    let _e7: f32 = pow5X_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX((1f - cos_theta));
    return (f0_ + ((vec3(1f) - f0_) * _e7));
}

fn distribution_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_h: f32, roughness: f32) -> f32 {
    let a: f32 = (roughness * roughness);
    let a2_: f32 = (a * a);
    let denom: f32 = (((n_dot_h * n_dot_h) * (a2_ - 1f)) + 1f);
    return (a2_ / max(((denom * denom) * 3.1415927f), 0.0001f));
}

fn geometry_schlick_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v: f32, roughness_1: f32) -> f32 {
    let r: f32 = (roughness_1 + 1f);
    let k: f32 = ((r * r) / 8f);
    return (n_dot_v / max(((n_dot_v * (1f - k)) + k), 0.0001f));
}

fn geometry_smithX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v_1: f32, n_dot_l: f32, roughness_2: f32) -> f32 {
    let _e2: f32 = geometry_schlick_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v_1, roughness_2);
    let _e4: f32 = geometry_schlick_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_l, roughness_2);
    return (_e2 * _e4);
}

fn direct_radiance_specularX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, world_pos_1: vec3<f32>, n_2: vec3<f32>, v_1: vec3<f32>, roughness_3: f32, base_color: vec3<f32>, f0_1: vec3<f32>, one_minus_reflectivity: f32) -> vec3<f32> {
    var l: vec3<f32>;
    var attenuation: f32;

    let light_pos: vec3<f32> = light.position.xyz;
    let light_dir: vec3<f32> = light.direction.xyz;
    let light_color: vec3<f32> = light.color.xyz;
    if (light.light_type == 0u) {
        let to_light: vec3<f32> = (light_pos - world_pos_1);
        let dist: f32 = length(to_light);
        l = normalize(to_light);
        attenuation = select(0f, ((light.intensity / max((dist * dist), 0.0001f)) * (1f - smoothstep((light.range * 0.9f), light.range, dist))), (light.range > 0f));
    } else {
        if (light.light_type == 1u) {
            let dir_len_sq: f32 = dot(light_dir, light_dir);
            l = select(vec3<f32>(0f, 0f, 1f), normalize(-(light_dir)), (dir_len_sq > 0.0000000000000001f));
            attenuation = light.intensity;
        } else {
            let to_light_1: vec3<f32> = (light_pos - world_pos_1);
            let dist_1: f32 = length(to_light_1);
            l = normalize(to_light_1);
            let _e51: vec3<f32> = l;
            let spot_cos: f32 = dot(-(_e51), normalize(light_dir));
            let inner_cos: f32 = min((light.spot_cos_half_angle + 0.1f), 1f);
            let spot_atten: f32 = smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
            attenuation = select(0f, (((light.intensity * spot_atten) * (1f - smoothstep((light.range * 0.9f), light.range, dist_1))) / max((dist_1 * dist_1), 0.0001f)), (light.range > 0f));
        }
    }
    let _e82: vec3<f32> = l;
    let h: vec3<f32> = normalize((v_1 + _e82));
    let _e86: vec3<f32> = l;
    let n_dot_l_1: f32 = max(dot(n_2, _e86), 0f);
    let n_dot_v_2: f32 = max(dot(n_2, v_1), 0.0001f);
    let n_dot_h_1: f32 = max(dot(n_2, h), 0f);
    let _e96: f32 = attenuation;
    let radiance: vec3<f32> = ((light_color * _e96) * n_dot_l_1);
    if (n_dot_l_1 <= 0f) {
        return vec3(0f);
    }
    let _e107: vec3<f32> = fresnel_schlickX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(max(dot(h, v_1), 0f), f0_1);
    let _e109: f32 = distribution_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_h_1, roughness_3);
    let _e110: f32 = geometry_smithX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v_2, n_dot_l_1, roughness_3);
    let spec: vec3<f32> = (((_e109 * _e110) * _e107) / vec3(max(((4f * n_dot_v_2) * n_dot_l_1), 0.0001f)));
    let kd: vec3<f32> = ((vec3(1f) - _e107) * one_minus_reflectivity);
    let diffuse: vec3<f32> = ((kd * base_color) / vec3(3.1415927f));
    return ((diffuse + spec) * radiance);
}

fn decode_ts_normal_with_placeholderX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    if all((raw > vec3<f32>(0.99f, 0.99f, 0.99f))) {
        return vec3<f32>(0f, 0f, 1f);
    }
    let nm_xy: vec2<f32> = (((raw.xy * 2f) - vec2(1f)) * scale);
    let z: f32 = max(sqrt(max((1f - dot(nm_xy, nm_xy)), 0f)), 0.000001f);
    return normalize(vec3<f32>(nm_xy, z));
}

fn get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_idx: u32) -> PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    let _e3: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = instancesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX[instance_idx];
    return _e3;
}

fn texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>) -> f32 {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4.w;
}

fn cluster_z_from_view_zX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_z: f32, near_clip: f32, far_clip: f32, cluster_count_z: u32) -> u32 {
    let d: f32 = clamp(-(view_z), near_clip, far_clip);
    let z_1: f32 = ((log((d / near_clip)) / log((far_clip / near_clip))) * f32(cluster_count_z));
    return u32(clamp(z_1, 0f, f32((cluster_count_z - 1u))));
}

fn cluster_xy_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(frag_xy: vec2<f32>, viewport_w: u32, viewport_h: u32) -> vec2<u32> {
    let max_x: f32 = max((f32(viewport_w) - 0.5f), 0.5f);
    let max_y: f32 = max((f32(viewport_h) - 0.5f), 0.5f);
    let pxy: vec2<f32> = clamp(frag_xy, vec2<f32>(0.5f, 0.5f), vec2<f32>(max_x, max_y));
    let tile_f: vec2<f32> = ((pxy - vec2<f32>(0.5f, 0.5f)) / vec2(16f));
    return vec2<u32>(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}

fn cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy: vec2<f32>, world_pos_2: vec3<f32>, view_space_z_coeffs: vec4<f32>, view_space_z_coeffs_right: vec4<f32>, view_index: u32, viewport_w_1: u32, viewport_h_1: u32, cluster_count_x: u32, cluster_count_y: u32, cluster_count_z_1: u32, near_clip_1: f32, far_clip_1: f32) -> u32 {
    let z_coeffs: vec4<f32> = select(view_space_z_coeffs, view_space_z_coeffs_right, (view_index != 0u));
    let view_z_1: f32 = (dot(z_coeffs.xyz, world_pos_2) + z_coeffs.w);
    let _e14: u32 = cluster_z_from_view_zX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_z_1, near_clip_1, far_clip_1, cluster_count_z_1);
    let _e18: vec2<u32> = cluster_xy_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy, viewport_w_1, viewport_h_1);
    let cx: u32 = min(_e18.x, (cluster_count_x - 1u));
    let cy: u32 = min(_e18.y, (cluster_count_y - 1u));
    let local_id: u32 = (cx + (cluster_count_x * (cy + (cluster_count_y * _e14))));
    let cluster_offset: u32 = (((view_index * cluster_count_x) * cluster_count_y) * cluster_count_z_1);
    return (cluster_offset + local_id);
}

fn sample_normal_world(uv0_1: vec2<f32>, uv1_: vec2<f32>, world_n_1: vec3<f32>, front_facing_1: bool, lerp_factor: f32) -> vec3<f32> {
    var n_3: vec3<f32>;
    var local_1: bool;
    var ts: vec3<f32>;
    var local_2: bool;

    let _e2: f32 = mat._NORMALMAP;
    let _e3: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e2);
    if !(_e3) {
        n_3 = normalize(world_n_1);
        let _e10: f32 = mat._DUALSIDED;
        let _e11: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e10);
        if _e11 {
            local_1 = !(front_facing_1);
        } else {
            local_1 = false;
        }
        let _e17: bool = local_1;
        if _e17 {
            let _e18: vec3<f32> = n_3;
            n_3 = -(_e18);
        }
        let _e20: vec3<f32> = n_3;
        return _e20;
    }
    let _e22: mat3x3<f32> = orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(normalize(world_n_1));
    let _e26: vec4<f32> = textureSample(_NormalMap, _NormalMap_sampler, uv0_1);
    let _e30: f32 = mat._NormalScale;
    let _e31: vec3<f32> = decode_ts_normal_with_placeholderX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(_e26.xyz, _e30);
    let _e35: vec4<f32> = textureSample(_NormalMap1_, _NormalMap1_sampler, uv1_);
    let _e39: f32 = mat._NormalScale1_;
    let _e40: vec3<f32> = decode_ts_normal_with_placeholderX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(_e35.xyz, _e39);
    ts = normalize(mix(_e31, _e40, vec3(lerp_factor)));
    let _e48: f32 = mat._DUALSIDED;
    let _e49: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e48);
    if _e49 {
        local_2 = !(front_facing_1);
    } else {
        local_2 = false;
    }
    let _e54: bool = local_2;
    if _e54 {
        let _e57: f32 = ts.z;
        ts.z = -(_e57);
    }
    let _e59: vec3<f32> = ts;
    return normalize((_e22 * _e59));
}

fn compute_lerp_factor(uv_lerp: vec2<f32>) -> f32 {
    var l_1: f32;

    let _e2: f32 = mat._Lerp;
    l_1 = _e2;
    let _e6: f32 = mat._LERPTEX;
    let _e7: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e6);
    if _e7 {
        let _e11: vec4<f32> = textureSample(_LerpTex, _LerpTex_sampler, uv_lerp);
        l_1 = _e11.x;
        let _e15: f32 = mat._MULTI_VALUES;
        let _e16: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e15);
        if _e16 {
            let _e17: f32 = l_1;
            let _e20: f32 = mat._Lerp;
            l_1 = (_e17 * _e20);
        }
    }
    let _e22: f32 = l_1;
    return clamp(_e22, 0f, 1f);
}

@vertex 
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) n: vec4<f32>, @location(2) uv0_: vec2<f32>) -> VertexOutput {
    var vp: mat4x4<f32>;
    var out: VertexOutput;

    let _e1: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_index);
    let world_p: vec4<f32> = (_e1.model * vec4<f32>(pos.xyz, 1f));
    let wn: vec3<f32> = normalize((_e1.model * vec4<f32>(n.xyz, 0f)).xyz);
    if (view_idx == 0u) {
        vp = _e1.view_proj_left;
    } else {
        vp = _e1.view_proj_right;
    }
    let _e24: mat4x4<f32> = vp;
    out.clip_pos = (_e24 * world_p);
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0_ = uv0_;
    out.view_layer = view_idx;
    let _e32: VertexOutput = out;
    return _e32;
}

@fragment 
fn fs_main(@builtin(position) frag_pos: vec4<f32>, @builtin(front_facing) front_facing: bool, @location(0) world_pos: vec3<f32>, @location(1) world_n: vec3<f32>, @location(2) uv0_raw: vec2<f32>, @location(3) @interpolate(flat) view_layer: u32) -> @location(0) vec4<f32> {
    var c0_: vec4<f32>;
    var c1_: vec4<f32>;
    var clip_a: f32;
    var local: bool;
    var occlusion0_: f32 = 1f;
    var occlusion1_: f32 = 1f;
    var emission0_: vec3<f32>;
    var emission1_: vec3<f32>;
    var spec0_: vec4<f32>;
    var spec1_: vec4<f32>;
    var lo: vec3<f32> = vec3(0f);
    var i: u32 = 0u;

    let _e6: vec4<f32> = mat._MainTex_ST;
    let _e8: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv0_raw, _e6);
    let _e11: vec4<f32> = mat._MainTex1_ST;
    let _e12: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv0_raw, _e11);
    let _e15: vec4<f32> = mat._LerpTex_ST;
    let _e16: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv0_raw, _e15);
    let _e17: f32 = compute_lerp_factor(_e16);
    let _e20: vec4<f32> = mat._Color;
    c0_ = _e20;
    let _e24: vec4<f32> = mat._Color1_;
    c1_ = _e24;
    let _e29: f32 = mat._Color.w;
    let _e33: f32 = mat._Color1_.w;
    clip_a = mix(_e29, _e33, _e17);
    let _e38: f32 = mat._ALBEDOTEX;
    let _e39: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e38);
    if _e39 {
        let _e40: vec4<f32> = c0_;
        let _e43: vec4<f32> = textureSample(_MainTex, _MainTex_sampler, _e8);
        c0_ = (_e40 * _e43);
        let _e45: vec4<f32> = c1_;
        let _e48: vec4<f32> = textureSample(_MainTex1_, _MainTex1_sampler, _e12);
        c1_ = (_e45 * _e48);
        let _e53: f32 = mat._Color.w;
        let _e56: f32 = texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MainTex, _MainTex_sampler, _e8);
        let _e61: f32 = mat._Color1_.w;
        let _e64: f32 = texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MainTex1_, _MainTex1_sampler, _e12);
        clip_a = mix((_e53 * _e56), (_e61 * _e64), _e17);
    }
    let _e67: vec4<f32> = c0_;
    let _e68: vec4<f32> = c1_;
    let c: vec4<f32> = mix(_e67, _e68, _e17);
    let _e72: f32 = mat._ALPHACLIP;
    let _e73: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e72);
    if _e73 {
        let _e74: f32 = clip_a;
        let _e77: f32 = mat._AlphaClip;
        local = (_e74 <= _e77);
    } else {
        local = false;
    }
    let _e82: bool = local;
    if _e82 {
        discard;
    }
    let base_color_1: vec3<f32> = c.xyz;
    let alpha: f32 = c.w;
    let _e87: f32 = mat._OCCLUSION;
    let _e88: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e87);
    if _e88 {
        let _e91: vec4<f32> = textureSample(_Occlusion, _Occlusion_sampler, _e8);
        occlusion0_ = _e91.x;
        let _e96: vec4<f32> = textureSample(_Occlusion1_, _Occlusion1_sampler, _e12);
        occlusion1_ = _e96.x;
    }
    let _e99: f32 = occlusion0_;
    let _e100: f32 = occlusion1_;
    let occlusion: f32 = mix(_e99, _e100, _e17);
    let _e104: vec4<f32> = mat._EmissionColor;
    emission0_ = _e104.xyz;
    let _e109: vec4<f32> = mat._EmissionColor1_;
    emission1_ = _e109.xyz;
    let _e114: f32 = mat._EMISSIONTEX;
    let _e115: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e114);
    if _e115 {
        let _e116: vec3<f32> = emission0_;
        let _e119: vec4<f32> = textureSample(_EmissionMap, _EmissionMap_sampler, _e8);
        emission0_ = (_e116 * _e119.xyz);
        let _e122: vec3<f32> = emission1_;
        let _e125: vec4<f32> = textureSample(_EmissionMap1_, _EmissionMap1_sampler, _e12);
        emission1_ = (_e122 * _e125.xyz);
    }
    let _e128: vec3<f32> = emission0_;
    let _e129: vec3<f32> = emission1_;
    let emission: vec3<f32> = mix(_e128, _e129, _e17);
    let _e133: vec4<f32> = mat._SpecularColor;
    spec0_ = _e133;
    let _e137: vec4<f32> = mat._SpecularColor1_;
    spec1_ = _e137;
    let _e141: f32 = mat._SPECULARMAP;
    let _e142: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e141);
    if _e142 {
        let _e145: vec4<f32> = textureSample(_SpecularMap, _SpecularMap_sampler, _e8);
        spec0_ = _e145;
        let _e148: vec4<f32> = textureSample(_SpecularMap1_, _SpecularMap1_sampler, _e12);
        spec1_ = _e148;
        let _e151: f32 = mat._MULTI_VALUES;
        let _e152: bool = kw_enabledX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e151);
        if _e152 {
            let _e153: vec4<f32> = spec0_;
            let _e156: vec4<f32> = mat._SpecularColor;
            spec0_ = (_e153 * _e156);
            let _e158: vec4<f32> = spec1_;
            let _e161: vec4<f32> = mat._SpecularColor1_;
            spec1_ = (_e158 * _e161);
        }
    }
    let _e163: vec4<f32> = spec0_;
    let _e164: vec4<f32> = spec1_;
    let spec_1: vec4<f32> = mix(_e163, _e164, _e17);
    let f0_2: vec3<f32> = spec_1.xyz;
    let smoothness: f32 = clamp(spec_1.w, 0f, 1f);
    let roughness_4: f32 = clamp((1f - smoothness), 0.045f, 1f);
    let one_minus_reflectivity_1: f32 = (1f - max(max(f0_2.x, f0_2.y), f0_2.z));
    let _e185: vec3<f32> = sample_normal_world(_e8, _e12, world_n, front_facing, _e17);
    let _e188: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let cam: vec3<f32> = _e188.xyz;
    let v_2: vec3<f32> = normalize((cam - world_pos));
    let _e197: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs;
    let _e200: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e203: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_width;
    let _e206: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_height;
    let _e209: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_x;
    let _e212: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_y;
    let _e215: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_z;
    let _e218: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.near_clip;
    let _e221: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.far_clip;
    let _e223: u32 = cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(frag_pos.xy, world_pos, _e197, _e200, view_layer, _e203, _e206, _e209, _e212, _e215, _e218, _e221);
    let count: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[_e223];
    let base_idx: u32 = (_e223 * MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX);
    let i_max: u32 = min(count, MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX);
    loop {
        let _e232: u32 = i;
        if (_e232 < i_max) {
        } else {
            break;
        }
        {
            let _e235: u32 = i;
            let li: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[(base_idx + _e235)];
            let _e241: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
            if (li >= _e241) {
                continue;
            }
            let light_1: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[li];
            let _e247: vec3<f32> = lo;
            let _e248: vec3<f32> = direct_radiance_specularX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_1, world_pos, _e185, v_2, roughness_4, base_color_1, f0_2, one_minus_reflectivity_1);
            lo = (_e247 + _e248);
        }
        continuing {
            let _e251: u32 = i;
            i = (_e251 + 1u);
        }
    }
    let amb: vec3<f32> = vec3(0.03f);
    let _e257: vec3<f32> = lo;
    let color: vec3<f32> = ((((amb * base_color_1) * occlusion) + (_e257 * occlusion)) + emission);
    return vec4<f32>(color, alpha);
}
