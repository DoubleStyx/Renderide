//! Unity surface shader `Shader "Art/PaintPBS"`: metallic Standard lighting with a paint-pattern
//! overlay sampled from four horizontal strips of `_PaintTex`. Faded at horizontal edges and
//! gated through `pow(paint, _Pow)` × `_PaintGain` × `_OutputScale` for the final alpha mask.
//! Default render state is transparent (host-driven via `_SrcBlend` / `_DstBlend` / `_ZWrite`).

// unity-shader-name: Art/PaintPBS
// unity-shader-name: PaintPBS

#import renderide::globals as rg
#import renderide::sh2_ambient as shamb
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu

struct PaintPBSMaterial {
    _Color: vec4<f32>,
    _PaintTexOffsets: vec4<f32>,
    _PaintTexShifts: vec4<f32>,
    _PaintTexScales: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _PaintTex_ST: vec4<f32>,
    _SideFadeSize: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _Pow: f32,
    _PaintBias: f32,
    _PaintGain: f32,
    _OutputScale: f32,
}

@group(1) @binding(0) var<uniform> mat: PaintPBSMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _PaintTex: texture_2d<f32>;
@group(1) @binding(4) var _PaintTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv_main: vec2<f32>,
    @location(3) uv_paint: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
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
    out.uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    out.uv_paint = uvu::apply_st(uv0, mat._PaintTex_ST);
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
    uv_main: vec2<f32>,
    uv_paint: vec2<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
) -> vec4<f32> {
    var c = textureSample(_MainTex, _MainTex_sampler, uv_main) * mat._Color;
    let side_fade = clamp(min(uv_main.x / mat._SideFadeSize, (1.0 - uv_main.x) / mat._SideFadeSize), 0.0, 1.0);
    c.a = c.a * side_fade;

    let offsets = uv_paint.y * mat._PaintTexScales + mat._PaintTexOffsets + uv_paint.x * mat._PaintTexShifts;
    let p = vec4<f32>(
        textureSample(_PaintTex, _PaintTex_sampler, vec2<f32>(uv_paint.x, offsets.x)).r,
        textureSample(_PaintTex, _PaintTex_sampler, vec2<f32>(uv_paint.x, offsets.y)).g,
        textureSample(_PaintTex, _PaintTex_sampler, vec2<f32>(uv_paint.x, offsets.z)).b,
        textureSample(_PaintTex, _PaintTex_sampler, vec2<f32>(uv_paint.x, offsets.w)).a,
    );
    let paint = (p.x + p.y + p.z + p.w) * 0.25 * mat._PaintGain + mat._PaintBias;
    let strength = clamp((c.a + pow(max(paint, 0.0), max(mat._Pow, 1e-4)) - 1.0) * mat._OutputScale, 0.0, 1.0);

    let base_color = c.rgb;
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
    let count = rg::cluster_light_counts[cluster_id];
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
    let ambient = select(vec3<f32>(0.0), shamb::ambient_probe(n) * base_color, include_directional);
    return vec4<f32>(ambient + lo, strength);
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv_main: vec2<f32>,
    @location(3) uv_paint: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, uv_main, uv_paint, view_layer, true, true);
}
