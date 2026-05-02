//! Surface decoding for Xiexe Toon 2.0: vertex transform, tangent-space normal handling,
//! and the `SurfaceData` aggregator consumed by the lighting and outline modules.
//!
//! The forward and outline paths take separate normal-decoding routes. The forward path
//! flips world-space `(N, T, B)` for back-facing fragments so two-sided meshes light
//! correctly. The outline path skips that flip — outline visible fragments are the
//! back-faces of an extruded shell whose geometric normals already point outward (matching
//! the Unity reference where outlines use the unflipped `i.ntb` of the front-face vertex).

#define_import_path renderide::xiexe::toon2::surface

#import renderide::xiexe::toon2::base as xb
#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::normal as pnorm
#import renderide::normal_decode as nd
#import renderide::uv_utils as uvu
#import renderide::alpha_clip_sample as acs

/// Forward-pass vertex transform. Builds the world-space TBN, applies the per-eye VP,
/// and forwards UVs / vertex color unchanged. Outline extrusion lives in
/// `xiexe_toon2_outline::vertex_outline`, which delegates back through here once the
/// position has been displaced.
fn vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    uv_primary: vec2<f32>,
    color: vec4<f32>,
    tangent: vec4<f32>,
    uv_secondary: vec2<f32>,
) -> xb::VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = xb::safe_normalize(d.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let world_tangent = vec4<f32>((d.model * vec4<f32>(tangent.xyz, 0.0)).xyz, tangent.w);
    let tbn = pnorm::orthonormal_tbn(world_n, world_tangent);
    let vp = xb::view_projection_for_draw(d, view_idx);

    var out: xb::VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_n;
    out.world_t = tbn[0];
    out.world_b = tbn[1];
    out.uv_primary = uv_primary;
    out.uv_secondary = uv_secondary;
    out.color = color;
    out.obj_pos = xb::safe_normalize(pos.xyz, vec3<f32>(0.0, 0.0, 1.0));
    out.view_layer = view_idx;
    return out;
}

/// Builds a perturbed TBN from the interpolated geometry frame. The base normal map and
/// (when `_DetailMask.r > 0`) the detail normal map are blended in tangent space using
/// Unity's `BlendNormals` formula — `xy` adds, `z` multiplies — so content authored
/// against Unity's `BlendNormals` (`#include UnityCG.cginc`) reproduces.
///
/// `flip_back_face` toggles the dual-sided correction. The forward path passes `true` so
/// back-facing fragments of two-sided meshes light from the visible side; the outline
/// path passes `false` because the visible outline fragments are back-faces of an
/// extruded shell whose geometric normals already face outward.
fn decode_normal_world(
    uv_normal: vec2<f32>,
    uv_detail_normal: vec2<f32>,
    uv_detail_mask: vec2<f32>,
    world_n: vec3<f32>,
    world_t: vec3<f32>,
    world_b: vec3<f32>,
    front_facing: bool,
    flip_back_face: bool,
) -> mat3x3<f32> {
    var n = xb::safe_normalize(world_n, vec3<f32>(0.0, 0.0, 1.0));
    var t = xb::safe_normalize(world_t, vec3<f32>(1.0, 0.0, 0.0));
    var b = xb::safe_normalize(world_b, vec3<f32>(0.0, 1.0, 0.0));

    if (flip_back_face && !front_facing) {
        n = -n;
        t = -t;
        b = -b;
    }

    if (xb::normal_map_enabled()) {
        let base_ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(xb::_BumpMap, xb::_BumpMap_sampler, uv_normal),
            xb::mat._BumpScale,
        );
        let detail_mask = textureSample(xb::_DetailMask, xb::_DetailMask_sampler, uv_detail_mask).r;
        let detail_ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(xb::_DetailNormalMap, xb::_DetailNormalMap_sampler, uv_detail_normal),
            xb::mat._DetailNormalMapScale,
        );
        let blended_ts = xb::safe_normalize(
            vec3<f32>(
                base_ts.xy + detail_ts.xy * detail_mask,
                base_ts.z * mix(1.0, detail_ts.z, detail_mask),
            ),
            vec3<f32>(0.0, 0.0, 1.0),
        );
        let tbn = mat3x3<f32>(t, b, n);
        n = xb::safe_normalize(tbn * blended_ts, n);
        t = xb::safe_normalize(cross(b, n), t);
        b = xb::safe_normalize(cross(n, t), b);
    }

    return mat3x3<f32>(t, b, n);
}

/// Decodes albedo, metallic-gloss, emission, AO, thickness, ramp-mask and the perturbed
/// TBN into a `SurfaceData` blob. `flip_back_face` is forwarded to `decode_normal_world`
/// so the outline path can opt out of the dual-sided flip.
fn sample_surface(
    flip_back_face: bool,
    front_facing: bool,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec3<f32>,
    world_b: vec3<f32>,
    uv_primary: vec2<f32>,
    uv_secondary: vec2<f32>,
    color: vec4<f32>,
) -> xb::SurfaceData {
    let uv_albedo = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetAlbedo), xb::mat._MainTex_ST);
    let uv_normal = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetNormal), xb::mat._BumpMap_ST);
    let uv_detail_normal = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetDetNormal), xb::mat._DetailNormalMap_ST);
    let uv_detail_mask = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetDetMask), xb::mat._DetailMask_ST);
    let uv_metallic = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetMetallic), xb::mat._MetallicGlossMap_ST);
    let uv_emission = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetEmission), xb::mat._EmissionMap_ST);
    let uv_occlusion = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetOcclusion), xb::mat._OcclusionMap_ST);
    let uv_thickness = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetThickness), xb::mat._ThicknessMap_ST);
    let uv_reflectivity = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetReflectivity), xb::mat._ReflectivityMask_ST);
    let uv_specular = uvu::apply_st(xb::uv_select(uv_primary, uv_secondary, xb::mat._UVSetSpecular), xb::mat._SpecularMap_ST);

    var albedo = textureSample(xb::_MainTex, xb::_MainTex_sampler, uv_albedo) * xb::mat._Color;
    let clip_alpha = xb::mat._Color.a * acs::texture_alpha_base_mip(xb::_MainTex, xb::_MainTex_sampler, uv_albedo);
    if (xb::vertex_color_albedo_enabled()) {
        albedo = vec4<f32>(albedo.rgb * color.rgb, albedo.a);
    }
    // `diffuse_color` keeps the original (saturated) base color for tinting paths
    // (specular albedo tint, rim/shadow-rim tints, outline tint) — see
    // `XSFrag.cginc:81` (`o.diffuseColor = o.albedo.rgb` *before* the metallic discount)
    // followed by `BRDF_XSLighting:35` (saturation pass).
    let diffuse_color = xb::maybe_saturate_color(albedo.rgb);
    let raw_normal = select(
        xb::safe_normalize(world_n, vec3<f32>(0.0, 1.0, 0.0)),
        -xb::safe_normalize(world_n, vec3<f32>(0.0, 1.0, 0.0)),
        flip_back_face && !front_facing,
    );

    let tbn = decode_normal_world(
        uv_normal,
        uv_detail_normal,
        uv_detail_mask,
        world_n,
        world_t,
        world_b,
        front_facing,
        flip_back_face,
    );

    var metallic = clamp(xb::mat._Metallic, 0.0, 1.0);
    var smoothness = clamp(xb::mat._Glossiness, 0.0, 1.0);
    let mg = textureSample(xb::_MetallicGlossMap, xb::_MetallicGlossMap_sampler, uv_metallic);
    if (xb::metallic_map_enabled()) {
        metallic = clamp(xb::mat._Metallic * mg.r, 0.0, 1.0);
        smoothness = clamp(xb::mat._Glossiness * mg.a, 0.0, 1.0);
    }
    var roughness = 1.0 - smoothness;
    roughness = clamp(roughness * (1.7 - 0.7 * roughness), 0.045, 1.0);
    let clearcoat_strength = clamp(xb::mat._ClearcoatStrength * mg.b, 0.0, 1.0);
    let clearcoat_smoothness = clamp(xb::mat._ClearcoatSmoothness * mg.g, 0.0, 1.0);

    // Direct-lighting albedo is the metallic-discounted tinted base — `BRDF_XSLighting:33`
    // does `i.albedo.rgb *= (1 - metallic)` before the lighting walk so a perfect metal
    // contributes no diffuse term. Multiplication is linear w.r.t. the saturation lerp,
    // so applying `(1 - metallic)` after the desaturation is equivalent to before.
    albedo = vec4<f32>(diffuse_color * (1.0 - metallic), albedo.a);

    let reflectivity = clamp(xb::mat._Reflectivity, 0.0, 1.0);
    let reflectivity_mask = textureSample(xb::_ReflectivityMask, xb::_ReflectivityMask_sampler, uv_reflectivity).r;

    var occlusion = vec3<f32>(1.0);
    if (xb::occlusion_enabled()) {
        let occ = textureSample(xb::_OcclusionMap, xb::_OcclusionMap_sampler, uv_occlusion).r;
        occlusion = mix(xb::mat._OcclusionColor.rgb, vec3<f32>(1.0), occ);
    }

    var emission = vec3<f32>(0.0);
    if (xb::emission_map_enabled()) {
        emission = textureSample(xb::_EmissionMap, xb::_EmissionMap_sampler, uv_emission).rgb;
    }

    var ramp_mask = 0.0;
    if (xb::ramp_mask_enabled()) {
        ramp_mask = textureSample(xb::_RampSelectionMask, xb::_RampSelectionMask_sampler, uv_primary).r;
    }

    var thickness = 1.0;
    if (xb::thickness_enabled()) {
        thickness = textureSample(xb::_ThicknessMap, xb::_ThicknessMap_sampler, uv_thickness).r;
    }

    let specular_mask = textureSample(xb::_SpecularMap, xb::_SpecularMap_sampler, uv_specular);

    return xb::SurfaceData(
        albedo,
        clip_alpha,
        diffuse_color,
        raw_normal,
        tbn[2],
        tbn[0],
        tbn[1],
        metallic,
        roughness,
        smoothness,
        clearcoat_strength,
        clearcoat_smoothness,
        reflectivity,
        reflectivity_mask,
        occlusion,
        emission,
        ramp_mask,
        thickness,
        specular_mask,
    );
}
