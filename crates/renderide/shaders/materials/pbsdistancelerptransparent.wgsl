//! Unity surface shader `Shader "PBSDistanceLerpTransparent"`: transparent metallic Standard
//! lighting with vertex displacement and emission driven by distance to up to 16 reference points.
//!
//! Reference space is selected by `WORLD_SPACE` / `OBJECT_SPACE`; if neither host keyword is
//! present, the shader defaults to world space.

#import renderide::mesh::vertex as mv
#import renderide::per_draw as pd
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::uv_utils as uvu

struct PbsDistanceLerpTransparentMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _DistanceGridSize: vec4<f32>,
    _DistanceGridOffset: vec4<f32>,
    _EmissionColorFrom: vec4<f32>,
    _EmissionColorTo: vec4<f32>,
    _DisplacementDirection: vec4<f32>,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _DisplaceDistanceFrom: f32,
    _DisplaceDistanceTo: f32,
    _DisplaceMagnitudeFrom: f32,
    _DisplaceMagnitudeTo: f32,
    _EmissionDistanceFrom: f32,
    _EmissionDistanceTo: f32,
    _PointCount: f32,
    WORLD_SPACE: f32,
    OBJECT_SPACE: f32,
    OVERRIDE_DISPLACE_DIRECTION: f32,
    _METALLICMAP: f32,
    _NORMALMAP: f32,
    _Points: array<vec4<f32>, 16>,
    _TintColors: array<vec4<f32>, 16>,
}

@group(1) @binding(0)  var<uniform> mat: PbsDistanceLerpTransparentMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _MetallicMap: texture_2d<f32>;
@group(1) @binding(10) var _MetallicMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) point_emission: vec3<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

fn safe_inverse_range(band_start: f32, band_end: f32) -> f32 {
    let denom = band_end - band_start;
    return select(1.0 / denom, 0.0, abs(denom) < 1e-6);
}

fn band_lerp(d: f32, band_start: f32, inv_range: f32) -> f32 {
    return clamp((d - band_start) * inv_range, 0.0, 1.0);
}

fn snap_reference(p: vec3<f32>) -> vec3<f32> {
    let size = mat._DistanceGridSize.xyz;
    let offset = mat._DistanceGridOffset.xyz;
    let safe_size = vec3<f32>(
        select(size.x, 1.0, size.x == 0.0),
        select(size.y, 1.0, size.y == 0.0),
        select(size.z, 1.0, size.z == 0.0),
    );
    let snapped = round((p + offset) / safe_size) * safe_size;
    return select(snapped, p, size == vec3<f32>(0.0));
}

struct DisplaceResult {
    displace: f32,
    emission: vec3<f32>,
}

fn accumulate_points(reference: vec3<f32>) -> DisplaceResult {
    let dist_inv = safe_inverse_range(mat._DisplaceDistanceFrom, mat._DisplaceDistanceTo);
    let em_inv = safe_inverse_range(mat._EmissionDistanceFrom, mat._EmissionDistanceTo);
    let count = u32(clamp(mat._PointCount, 0.0, 16.0));
    var displace = 0.0;
    var emission = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let pt = mat._Points[i].xyz;
        let d = distance(reference, pt);
        let displace_lerp = band_lerp(d, mat._DisplaceDistanceFrom, dist_inv);
        let emission_lerp = band_lerp(d, mat._EmissionDistanceFrom, em_inv);
        displace = displace + mix(mat._DisplaceMagnitudeFrom, mat._DisplaceMagnitudeTo, displace_lerp);
        let tint = mat._TintColors[i];
        let em_color = mix(mat._EmissionColorFrom, mat._EmissionColorTo, emission_lerp);
        emission = emission + (tint * em_color).rgb;
    }
    return DisplaceResult(displace, emission);
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        ts_n = psamp::sample_tangent_normal(_NormalMap, _NormalMap_sampler, uv_main, 0.0, mat._NormalScale);
    }
    if (!front_facing) {
        ts_n.z = -ts_n.z;
    }
    return psamp::tangent_to_world(world_n, ts_n);
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
    let world_p_pre = d.model * vec4<f32>(pos.xyz, 1.0);
    let use_world = uvu::kw_enabled(mat.WORLD_SPACE) || (!uvu::kw_enabled(mat.OBJECT_SPACE));
    let reference_raw = select(pos.xyz, world_p_pre.xyz, use_world);
    let reference = snap_reference(reference_raw);
    let acc = accumulate_points(reference);

    let direction = select(
        normalize(n.xyz),
        normalize(mat._DisplacementDirection.xyz),
        uvu::kw_enabled(mat.OVERRIDE_DISPLACE_DIRECTION),
    );
    let displaced_obj = pos.xyz + direction * acc.displace;
    let world_p = d.model * vec4<f32>(displaced_obj, 1.0);
    let wn = normalize(d.normal_matrix * n.xyz);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
    out.point_emission = acc.emission;
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
    point_emission: vec3<f32>,
    view_layer: u32,
    front_facing: bool,
    include_directional: bool,
    include_local: bool,
) -> vec4<f32> {
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let albedo_s = textureSample(_MainTex, _MainTex_sampler, uv_main);
    let base_color = (mat._Color * albedo_s).rgb;
    let alpha = mat._Color.a * albedo_s.a;

    let occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    let n = sample_normal_world(uv_main, world_n, front_facing);

    let emission_tex = textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    let emission = mat._EmissionColor.rgb * emission_tex + point_emission;
    let surface = psurf::metallic(base_color, alpha, metallic, roughness, occlusion, n, emission);
    let options = plight::ClusterLightingOptions(include_directional, include_local, true, true);
    let color = plight::shade_metallic_clustered(frag_xy, world_pos, view_layer, surface, options);
    return vec4<f32>(color, alpha);
}

//#pass forward_transparent
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) point_emission: vec3<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, uv0, point_emission, view_layer, front_facing, true, true);
}
