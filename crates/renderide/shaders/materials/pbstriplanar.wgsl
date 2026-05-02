//! Unity surface shader `Shader "PBSTriplanar"`: metallic Standard lighting with triplanar
//! projection sampled from world or object space.
//!
//! Each texture (`_MainTex`, `_MetallicMap`, `_EmissionMap`, `_NormalMap`, `_OcclusionMap`) is
//! sampled three times — once per axis-aligned plane (ZY for X, XZ for Y, XY for Z) — and blended
//! by `pow(abs(world_normal), _TriBlendPower)`. Normal maps use Reoriented Normal Mapping (RNM)
//! per plane, after Ben Golus's 2017 example. World-space vs object-space is selected by
//! `_WORLDSPACE` (1.0) / `_OBJECTSPACE` (1.0); this matches Unity's `_WORLDSPACE`/`_OBJECTSPACE`
//! `multi_compile`.


#import renderide::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

/// Material uniforms for `PBSTriplanar`.
struct PbsTriplanarMaterial {
    /// Tint color (`Color`).
    _Color: vec4<f32>,
    /// Emission color (`EmissionColor`).
    _EmissionColor: vec4<f32>,
    /// Albedo `_ST` applied to all three projected planes.
    _MainTex_ST: vec4<f32>,
    /// Tangent-space normal scale (`Normal Scale`).
    _NormalScale: f32,
    /// Smoothness fallback when `_METALLICMAP` is disabled.
    _Glossiness: f32,
    /// Metallic fallback when `_METALLICMAP` is disabled.
    _Metallic: f32,
    /// Triplanar blend exponent — higher values produce sharper transitions between planes.
    _TriBlendPower: f32,
    /// Keyword: project from world space (mutually exclusive with `_OBJECTSPACE`).
    _WORLDSPACE: f32,
    /// Keyword: project from object space (mutually exclusive with `_WORLDSPACE`).
    _OBJECTSPACE: f32,
    /// Keyword: enable albedo texture sampling (otherwise tint-only).
    _ALBEDOTEX: f32,
    /// Keyword: enable emission texture sampling.
    _EMISSIONTEX: f32,
    /// Keyword: enable normal map sampling with RNM blending across planes.
    _NORMALMAP: f32,
    /// Keyword: read metallic + smoothness from `_MetallicMap` (R=metallic, A=smoothness).
    _METALLICMAP: f32,
    /// Keyword: read occlusion from `_OcclusionMap.g` (matches Unity's reference).
    _OCCLUSION: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsTriplanarMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _MetallicMap: texture_2d<f32>;
@group(1) @binding(6)  var _MetallicMap_sampler: sampler;
@group(1) @binding(7)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(8)  var _EmissionMap_sampler: sampler;
@group(1) @binding(9)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(10) var _OcclusionMap_sampler: sampler;

/// Interpolated vertex output forwarded to both forward-base and forward-add fragments.
///
/// `proj_pos` is the projection source — world or object space, selected at vertex time so that
/// the fragment shader can stay branchless.
struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) proj_pos: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

/// Resolved per-fragment shading inputs for the metallic Cook–Torrance path.
struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

/// Reoriented Normal Mapping blend — combines a base world-space normal with a per-plane
/// tangent-space normal so the projected plane's surface variation rotates onto the underlying
/// world normal without flattening near pole transitions. From Barré-Brisebois & Hill, 2012.
fn blend_rnm(n1_in: vec3<f32>, n2_in: vec3<f32>) -> vec3<f32> {
    let n1 = vec3<f32>(n1_in.x, n1_in.y, n1_in.z + 1.0);
    let n2 = vec3<f32>(-n2_in.x, -n2_in.y, n2_in.z);
    return n1 * dot(n1, n2) / max(n1.z, 1e-4) - n2;
}

/// Apply Unity's `TRANSFORM_TEX` against `_MainTex_ST` to a planar projection coordinate.
fn apply_main_st(uv: vec2<f32>) -> vec2<f32> {
    return uvu::apply_st(uv, mat._MainTex_ST);
}

/// Triplanar weights from a world-space normal: `pow(|n|, _TriBlendPower)` normalized so that
/// the three weights sum to 1, with a small epsilon to avoid division by zero on grazing normals.
fn triplanar_weights(world_n: vec3<f32>) -> vec3<f32> {
    let blend_power = max(mat._TriBlendPower, 0.0001);
    let raw = pow(abs(world_n), vec3<f32>(blend_power));
    let sum = max(raw.x + raw.y + raw.z, 1e-4);
    return raw / sum;
}

/// Sample-with-projection helper: builds the three planar UVs (with the
/// `TRIPLANAR_CORRECT_PROJECTED_U` x-flip) and per-axis sign for downstream normal correction.
struct PlanarUvs {
    uv_x: vec2<f32>,
    uv_y: vec2<f32>,
    uv_z: vec2<f32>,
    axis_sign: vec3<f32>,
}

/// Construct the three planar UVs from a projection-space position and the world normal.
fn build_planar_uvs(proj_pos: vec3<f32>, world_n: vec3<f32>) -> PlanarUvs {
    var uvs: PlanarUvs;
    uvs.uv_x = apply_main_st(proj_pos.zy);
    uvs.uv_y = apply_main_st(proj_pos.xz);
    uvs.uv_z = apply_main_st(proj_pos.xy);
    let axis_sign = vec3<f32>(
        select(-1.0, 1.0, world_n.x >= 0.0),
        select(-1.0, 1.0, world_n.y >= 0.0),
        select(-1.0, 1.0, world_n.z >= 0.0),
    );
    // TRIPLANAR_CORRECT_PROJECTED_U: flip horizontal axis for backside projections so
    // mirrored projections don't read backward; the `_z` axis is double-negated to keep
    // its winding consistent with the others.
    uvs.uv_x.x = uvs.uv_x.x * axis_sign.x;
    uvs.uv_y.x = uvs.uv_y.x * axis_sign.y;
    uvs.uv_z.x = uvs.uv_z.x * -axis_sign.z;
    uvs.axis_sign = axis_sign;
    return uvs;
}

/// Sample a 2D texture three times along the planar UVs and blend by triplanar weights.
fn triplanar_rgba(
    tex: texture_2d<f32>,
    samp: sampler,
    uvs: PlanarUvs,
    weights: vec3<f32>,
) -> vec4<f32> {
    let cx = textureSample(tex, samp, uvs.uv_x);
    let cy = textureSample(tex, samp, uvs.uv_y);
    let cz = textureSample(tex, samp, uvs.uv_z);
    return cx * weights.x + cy * weights.y + cz * weights.z;
}

/// Build a triplanar world-space normal via Reoriented Normal Mapping when `_NORMALMAP` is on,
/// otherwise return the interpolated geometric normal (renormalized). Triplanar shading does not
/// rely on the mesh tangent — RNM rotates each per-plane tangent-space normal directly onto the
/// interpolated world normal — so no tangent is plumbed through this material.
fn sample_normal_world(uvs: PlanarUvs, world_n: vec3<f32>, weights: vec3<f32>) -> vec3<f32> {
    let n_geo = normalize(world_n);
    if (!uvu::kw_enabled(mat._NORMALMAP)) {
        return n_geo;
    }

    var t_x = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap, _NormalMap_sampler, uvs.uv_x),
        mat._NormalScale,
    );
    var t_y = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap, _NormalMap_sampler, uvs.uv_y),
        mat._NormalScale,
    );
    var t_z = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap, _NormalMap_sampler, uvs.uv_z),
        mat._NormalScale,
    );

    // Match the projected-U flip applied at sample time so RNM aligns the maps.
    t_x.x = t_x.x * uvs.axis_sign.x;
    t_y.x = t_y.x * uvs.axis_sign.y;
    t_z.x = t_z.x * -uvs.axis_sign.z;

    let abs_n = abs(n_geo);
    let n_x_base = vec3<f32>(n_geo.z, n_geo.y, abs_n.x);
    let n_y_base = vec3<f32>(n_geo.x, n_geo.z, abs_n.y);
    let n_z_base = vec3<f32>(n_geo.x, n_geo.y, abs_n.z);

    var blended_x = blend_rnm(n_x_base, t_x);
    var blended_y = blend_rnm(n_y_base, t_y);
    var blended_z = blend_rnm(n_z_base, t_z);

    blended_x.z = blended_x.z * uvs.axis_sign.x;
    blended_y.z = blended_y.z * uvs.axis_sign.y;
    blended_z.z = blended_z.z * uvs.axis_sign.z;

    // Swizzle each per-plane tangent normal back into world space, then weight-blend.
    let world_x = vec3<f32>(blended_x.z, blended_x.y, blended_x.x);
    let world_y = vec3<f32>(blended_y.x, blended_y.z, blended_y.y);
    let world_z = vec3<f32>(blended_z.x, blended_z.y, blended_z.z);

    return normalize(world_x * weights.x + world_y * weights.y + world_z * weights.z);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's triplanar `surf` for `PBSTriplanar`.
fn sample_surface(world_pos: vec3<f32>, world_n: vec3<f32>, proj_pos: vec3<f32>) -> SurfaceData {
    let uvs = build_planar_uvs(proj_pos, world_n);
    let weights = triplanar_weights(world_n);

    var c = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c = c * triplanar_rgba(_MainTex, _MainTex_sampler, uvs, weights);
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        let m = triplanar_rgba(_MetallicMap, _MetallicMap_sampler, uvs, weights);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        // Unity's reference reads occlusion from the green channel here.
        let occ = triplanar_rgba(_OcclusionMap, _OcclusionMap_sampler, uvs, weights);
        occlusion = occ.g;
    }

    var emission = mat._EmissionColor;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * triplanar_rgba(_EmissionMap, _EmissionMap_sampler, uvs, weights);
    }

    return SurfaceData(
        c.rgb,
        c.a,
        metallic,
        roughness,
        occlusion,
        sample_normal_world(uvs, world_n, weights),
        emission.rgb,
    );
}

/// Vertex stage: forward world position, world-space normal, and the projection-space position
/// (world or object) selected by the `_WORLDSPACE`/`_OBJECTSPACE` keywords.
@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
    let wn = mv::world_normal(d, n);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    // Default to world-space projection; switch to object-space when `_OBJECTSPACE` keyword wins.
    out.proj_pos = select(world_p.xyz, pos.xyz, uvu::kw_enabled(mat._OBJECTSPACE));
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

/// Forward-base pass: ambient + directional lighting + emission.
//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) proj_pos: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(world_pos, world_n, proj_pos);
    let surface = psurf::metallic(
        s.base_color,
        s.alpha,
        s.metallic,
        s.roughness,
        s.occlusion,
        s.normal,
        s.emission,
    );
    return vec4<f32>(
        plight::shade_metallic_clustered(
            frag_pos.xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        s.alpha,
    );
}
