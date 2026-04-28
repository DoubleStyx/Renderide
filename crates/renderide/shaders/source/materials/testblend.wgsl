//! Unity surface shader `Shader "Custom/TestBlend"`: metallic Standard lighting that lerps
//! between two albedo textures and clips against `_CutOff`.

// unity-shader-name: Custom/TestBlend
// unity-shader-name: TestBlend

#import renderide::globals as rg
#import renderide::sh2_ambient as shamb
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu

struct TestBlendMaterial {
    _Color: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _MainTex2_ST: vec4<f32>,
    _MainTex2_StorageVInverted: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _Lerp: f32,
    _CutOff: f32,
}

@group(1) @binding(0) var<uniform> mat: TestBlendMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _MainTex2: texture_2d<f32>;
@group(1) @binding(4) var _MainTex2_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let wn = normalize(d.normal_matrix * n.xyz);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    uv0: vec2<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
) -> vec4<f32> {
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let uv_main2 = uvu::apply_st_for_storage(uv0, mat._MainTex2_ST, mat._MainTex2_StorageVInverted);
    let c1 = textureSample(_MainTex, _MainTex_sampler, uv_main);
    let c2 = textureSample(_MainTex2, _MainTex2_sampler, uv_main2);
    let lerp_factor = clamp(mat._Lerp, 0.0, 1.0);
    let c = mix(c1, c2, lerp_factor);

    let alpha_a = acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main);
    let alpha_b = acs::texture_alpha_base_mip(_MainTex2, _MainTex2_sampler, uv_main2);
    let clip_alpha = mix(alpha_a, alpha_b, lerp_factor);
    if (clip_alpha <= mat._CutOff) {
        discard;
    }

    let base_color = c.rgb * mat._Color.rgb;
    let metallic = clamp(mat._Metallic, 0.0, 1.0);
    let smoothness = clamp(mat._Glossiness, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let n = normalize(world_n);
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let aa_roughness = brdf::filter_perceptual_roughness(roughness, n);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy, world_pos, rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right,
        view_layer, rg::frame.viewport_width, rg::frame.viewport_height,
        rg::frame.cluster_count_x, rg::frame.cluster_count_y, rg::frame.cluster_count_z,
        rg::frame.near_clip, rg::frame.far_clip,
    );
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        let is_directional = light.light_type == 1u;
        if ((is_directional && !include_directional) || (!is_directional && !include_local)) {
            continue;
        }
        lo = lo + brdf::direct_radiance_metallic(
            light, world_pos, n, v, aa_roughness, metallic, base_color, f0,
        );
    }
    let ambient_probe = select(vec3<f32>(0.0), shamb::ambient_probe(n), include_directional);
    let ambient = brdf::indirect_diffuse_metallic(
        ambient_probe,
        base_color,
        metallic,
        1.0,
    );
    let indirect_specular = select(
        vec3<f32>(0.0),
        brdf::indirect_specular(n, v, aa_roughness, f0, 1.0, true),
        include_directional,
    );
    return vec4<f32>(ambient + indirect_specular + lo, mat._Color.a * c.a);
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, uv0, view_layer, true, true);
}
