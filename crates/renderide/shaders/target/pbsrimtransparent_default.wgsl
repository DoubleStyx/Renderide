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
    normal_matrix: mat3x3<f32>,
    _pad: vec4<f32>,
}

struct FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    camera_world_pos: vec4<f32>,
    view_space_x_coeffs: vec4<f32>,
    view_space_y_coeffs: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    view_space_x_coeffs_right: vec4<f32>,
    view_space_y_coeffs_right: vec4<f32>,
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

struct PbsRimTransparentMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _RimColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _Glossiness: f32,
    _Metallic: f32,
    _NormalScale: f32,
    _RimPower: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0_: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

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
var<uniform> mat: PbsRimTransparentMaterial;
@group(1) @binding(1) 
var _MainTex: texture_2d<f32>;
@group(1) @binding(2) 
var _MainTex_sampler: sampler;
@group(1) @binding(3) 
var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) 
var _NormalMap_sampler: sampler;
@group(1) @binding(5) 
var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6) 
var _EmissionMap_sampler: sampler;
@group(1) @binding(7) 
var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8) 
var _OcclusionMap_sampler: sampler;
@group(1) @binding(9) 
var _MetallicMap: texture_2d<f32>;
@group(1) @binding(10) 
var _MetallicMap_sampler: sampler;

fn orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_2: vec3<f32>) -> mat3x3<f32> {
    let sign_: f32 = select(-1f, 1f, (n_2.z >= 0f));
    let a: f32 = (-1f / (sign_ + n_2.z));
    let b: f32 = ((n_2.x * n_2.y) * a);
    let t: vec3<f32> = vec3<f32>((1f + (((sign_ * n_2.x) * n_2.x) * a)), (sign_ * b), (-(sign_) * n_2.x));
    let bitan: vec3<f32> = vec3<f32>(b, (sign_ + ((n_2.y * n_2.y) * a)), -(n_2.y));
    return mat3x3<f32>(normalize(t), normalize(bitan), n_2);
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
    let a_1: f32 = (roughness * roughness);
    let a2_: f32 = (a_1 * a_1);
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

fn direct_radiance_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, world_pos_1: vec3<f32>, n_3: vec3<f32>, v: vec3<f32>, roughness_3: f32, metallic: f32, base_color: vec3<f32>, f0_1: vec3<f32>) -> vec3<f32> {
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
    let h: vec3<f32> = normalize((v + _e82));
    let _e86: vec3<f32> = l;
    let n_dot_l_1: f32 = max(dot(n_3, _e86), 0f);
    let n_dot_v_2: f32 = max(dot(n_3, v), 0.0001f);
    let n_dot_h_1: f32 = max(dot(n_3, h), 0f);
    let _e96: f32 = attenuation;
    let radiance: vec3<f32> = ((light_color * _e96) * n_dot_l_1);
    if (n_dot_l_1 <= 0f) {
        return vec3(0f);
    }
    let _e107: vec3<f32> = fresnel_schlickX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(max(dot(h, v), 0f), f0_1);
    let _e109: f32 = distribution_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_h_1, roughness_3);
    let _e110: f32 = geometry_smithX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v_2, n_dot_l_1, roughness_3);
    let spec: vec3<f32> = (((_e109 * _e110) * _e107) / vec3(max(((4f * n_dot_v_2) * n_dot_l_1), 0.0001f)));
    let kd: vec3<f32> = ((vec3(1f) - _e107) * (1f - metallic));
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

fn apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_in * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
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

fn sample_normal_world(uv_main: vec2<f32>, world_n_1: vec3<f32>) -> vec3<f32> {
    let _e1: mat3x3<f32> = orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(world_n_1);
    let _e5: vec4<f32> = textureSample(_NormalMap, _NormalMap_sampler, uv_main);
    let _e9: f32 = mat._NormalScale;
    let _e10: vec3<f32> = decode_ts_normal_with_placeholderX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(_e5.xyz, _e9);
    return normalize((_e1 * _e10));
}

fn metallic_roughness(uv: vec2<f32>) -> vec2<f32> {
    let mg: vec4<f32> = textureSample(_MetallicMap, _MetallicMap_sampler, uv);
    let _e6: f32 = mat._Metallic;
    let metallic_1: f32 = clamp((_e6 * mg.x), 0f, 1f);
    let _e14: f32 = mat._Glossiness;
    let smoothness: f32 = clamp((_e14 * mg.w), 0f, 1f);
    let roughness_4: f32 = clamp((1f - smoothness), 0.045f, 1f);
    return vec2<f32>(metallic_1, roughness_4);
}

@vertex 
fn vs_main(@builtin(instance_index) instance_index: u32, @location(0) pos: vec4<f32>, @location(1) n: vec4<f32>, @location(2) uv0_: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e1: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_index);
    let world_p: vec4<f32> = (_e1.model * vec4<f32>(pos.xyz, 1f));
    let wn: vec3<f32> = normalize((_e1.normal_matrix * n.xyz));
    let vp: mat4x4<f32> = _e1.view_proj_left;
    out.clip_pos = (vp * world_p);
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0_ = uv0_;
    out.view_layer = 0u;
    let _e24: VertexOutput = out;
    return _e24;
}

@fragment 
fn fs_main(@builtin(position) frag_pos: vec4<f32>, @location(0) world_pos: vec3<f32>, @location(1) world_n: vec3<f32>, @location(2) uv0_1: vec2<f32>, @location(3) @interpolate(flat) view_layer: u32) -> @location(0) vec4<f32> {
    var n_1: vec3<f32>;
    var lo: vec3<f32> = vec3(0f);
    var i: u32 = 0u;

    let _e5: vec4<f32> = mat._MainTex_ST;
    let _e7: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv0_1, _e5);
    let albedo_s: vec4<f32> = textureSample(_MainTex, _MainTex_sampler, _e7);
    let _e13: vec4<f32> = mat._Color;
    let base_color_1: vec3<f32> = (_e13.xyz * albedo_s.xyz);
    let _e20: f32 = mat._Color.w;
    let alpha: f32 = (_e20 * albedo_s.w);
    let _e23: vec2<f32> = metallic_roughness(_e7);
    let metallic_2: f32 = _e23.x;
    let roughness_5: f32 = _e23.y;
    let _e28: vec4<f32> = textureSample(_OcclusionMap, _OcclusionMap_sampler, _e7);
    let occlusion: f32 = _e28.x;
    n_1 = normalize(world_n);
    let _e33: vec3<f32> = n_1;
    let _e34: vec3<f32> = sample_normal_world(_e7, _e33);
    n_1 = _e34;
    let _e37: vec4<f32> = textureSample(_EmissionMap, _EmissionMap_sampler, _e7);
    let _e41: vec4<f32> = mat._EmissionColor;
    let emission: vec3<f32> = (_e37.xyz * _e41.xyz);
    let _e46: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let cam: vec3<f32> = _e46.xyz;
    let v_1: vec3<f32> = normalize((cam - world_pos));
    let f0_2: vec3<f32> = mix(vec3(0.04f), base_color_1, metallic_2);
    let _e54: vec3<f32> = n_1;
    let _e65: f32 = mat._RimPower;
    let rim: f32 = pow(max((1f - clamp(dot(v_1, _e54), 0f, 1f)), 0f), max(_e65, 0.0001f));
    let _e71: vec4<f32> = mat._RimColor;
    let rim_emission: vec3<f32> = (_e71.xyz * rim);
    let _e78: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs;
    let _e81: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e84: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_width;
    let _e87: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_height;
    let _e90: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_x;
    let _e93: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_y;
    let _e96: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_z;
    let _e99: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.near_clip;
    let _e102: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.far_clip;
    let _e104: u32 = cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(frag_pos.xy, world_pos, _e78, _e81, view_layer, _e84, _e87, _e90, _e93, _e96, _e99, _e102);
    let count: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[_e104];
    let base_idx: u32 = (_e104 * MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX);
    let i_max: u32 = min(count, MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX);
    loop {
        let _e113: u32 = i;
        if (_e113 < i_max) {
        } else {
            break;
        }
        {
            let _e116: u32 = i;
            let li: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[(base_idx + _e116)];
            let _e122: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
            if (li >= _e122) {
                continue;
            }
            let light_1: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[li];
            let _e128: vec3<f32> = lo;
            let _e129: vec3<f32> = n_1;
            let _e130: vec3<f32> = direct_radiance_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_1, world_pos, _e129, v_1, roughness_5, metallic_2, base_color_1, f0_2);
            lo = (_e128 + _e130);
        }
        continuing {
            let _e133: u32 = i;
            i = (_e133 + 1u);
        }
    }
    let amb: vec3<f32> = vec3(0.03f);
    let _e139: vec3<f32> = lo;
    let color: vec3<f32> = (((((amb * base_color_1) * occlusion) + (_e139 * occlusion)) + emission) + rim_emission);
    return vec4<f32>(color, alpha);
}
