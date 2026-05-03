//! Unity `Shader "Volume/FogBox"` port.
//!
//! Uses the original property names and keyword-style flags exposed by FrooxEngine
//! (`FogBoxVolumeMaterial`) so host-driven material updates map directly.

#import renderide::globals as rg
#import renderide::mesh::vertex as mv
#import renderide::scene_depth_sample as sds
#import renderide::uv_utils as uvu

struct FogBoxVolumeMaterial {
    _BaseColor: vec4<f32>,
    _AccumulationColor: vec4<f32>,
    _AccumulationColorBottom: vec4<f32>,
    _AccumulationColorTop: vec4<f32>,
    _AccumulationRate: f32,
    _GammaCurve: f32,
    _FogStart: f32,
    _FogEnd: f32,
    _FogDensity: f32,
    // Keyword mirrors from FrooxEngine/FogBoxVolumeMaterial.
    SATURATE_ALPHA: f32,
    SATURATE_COLOR: f32,
    COLOR_CONSTANT: f32,
    COLOR_VERT_GRADIENT: f32,
    OBJECT_SPACE: f32,
    WORLD_SPACE: f32,
    FOG_LINEAR: f32,
    FOG_EXP: f32,
    FOG_EXP2: f32,
    _pad0: vec3<f32>,
}

@group(1) @binding(0) var<uniform> mat: FogBoxVolumeMaterial;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(4) t: vec4<f32>,
) -> mv::WorldVertexOutput {
#ifdef MULTIVIEW
    return mv::world_vertex_main(instance_index, view_idx, pos, n, t, uv);
#else
    return mv::world_vertex_main(instance_index, 0u, pos, n, t, uv);
#endif
}

fn accumulation_color(world_pos: vec3<f32>) -> vec4<f32> {
    if (uvu::kw_enabled(mat.COLOR_VERT_GRADIENT)) {
        // Parity note: Unity gradient is evaluated on the sampled segment in local space.
        // Until full object-space segment reconstruction lands, use world Y as a stable proxy.
        let avg_y = world_pos.y * 0.5 + 0.5;
        return mix(
            mat._AccumulationColorBottom,
            mat._AccumulationColorTop,
            clamp(avg_y, 0.0, 1.0),
        );
    }
    return mat._AccumulationColor;
}

//#pass forward_transparent_volume
@fragment
fn fs_main(
    in: mv::WorldVertexOutput,
) -> @location(0) vec4<f32> {
    // Unity `Volume/FogBox`: samples scene depth and integrates fog along the view segment.
    //
    // Render path uses `forward_transparent_volume` (depth test Always), matching Unity `ZTest Always`:
    // a giant proxy sphere around the world must still run this shader over terrain/sky pixels when
    // the shell is farther than opaque geometry — otherwise fog only appears on the horizon “rim”.
    //
    // WORLD_SPACE uses `scene_z` only (linear depth to the nearest opaque surface at this pixel).
    // Capping with `min(scene_z, part_z)` made fog thickness track the inner proxy shell’s eye-space
    // depth `part_z`, which swings with view angle (short toward zenith, long toward the horizon) and
    // reads as a circular sky rim / thin-center “bubble”. Scene depth gives uniform distance fog on
    // sky and matches Unity-style world atmosphere. Use OBJECT_SPACE for fog tied to a box proxy.
    //
    // Fog modes are mutually exclusive (`#pragma multi_compile FOG_LINEAR FOG_EXP FOG_EXP2`).
    // For `FOG_LINEAR`, `_FogDensity` is not compiled in — UI often leaves density at 0; exp
    // math would zero out. Treat ~zero density like linear when EXP isn’t intended.
    let scene_z = sds::scene_linear_depth(in.clip_pos, in.view_layer);
    let part_z = sds::fragment_linear_depth(in.world_pos, in.view_layer);
    let cam_world = rg::camera_world_pos_for_view(in.view_layer);

    var dist: f32;
    if (uvu::kw_enabled(mat.OBJECT_SPACE)) {
        // Unity OBJECT_SPACE: cube-clamped segment (see `IntersectUnitCube`); still simplified here.
        let end_ratio = min(scene_z / max(part_z, 1e-6), 1.0);
        let max_dist = distance(cam_world, in.world_pos);
        dist = max_dist * end_ratio;
    } else {
        dist = scene_z;
    }

    let use_linear_fog =
        uvu::kw_enabled(mat.FOG_LINEAR) || mat._FogDensity <= 1e-8;
    if (use_linear_fog) {
        dist = min(mat._FogEnd, dist);
        dist = max(0.0, dist - mat._FogStart);
    } else if (uvu::kw_enabled(mat.FOG_EXP2)) {
        let d = dist * mat._FogDensity;
        dist = 1.0 - (1.0 / exp(d * d));
    } else {
        // Default to FOG_EXP when neither linear nor exp2 is enabled.
        dist = 1.0 - (1.0 / exp(dist * mat._FogDensity));
    }

    // Linear fog: `dist` is view-linear length inside [_FogStart, _FogEnd] (world units). Feeding
    // that directly into `pow(dist * rate, γ)` overshoots Unity when the span is modest, but
    // dividing by the full `_FogEnd - _FogStart` undershoots badly when Froox uses a huge span
    // (near far-clip): `dist/span` stays ~0 and fog disappears. Normalize with a **capped** reference
    // span so tight ranges behave correctly and large spans still get a usable 0–1 factor.
    let fog_span = max(mat._FogEnd - mat._FogStart, 1e-5);
    let linear_accum_span = max(min(fog_span, 450.0), 1e-5);
    let accumulation_t = select(
        dist,
        clamp(dist / linear_accum_span, 0.0, 1.0),
        use_linear_fog,
    );

    let acc_color = accumulation_color(in.world_pos);
    let gamma = max(mat._GammaCurve, 1e-5);
    // Unity `Volume/FogBox` shader Properties default `_AccumulationColor` to (0.1,0.1,0.1,0.1).
    // FrooxEngine defaults that slot to white `(1,1,1,1)`, which makes `pow(...) * accColor` blow out
    // and hides `_BaseColor`. Scale so full-white matches Unity's default tint strength (same factor
    // on RGBA as Unity's property defaults).
    let unity_accumulation_property_scale = 0.1;
    let acc = pow(max(accumulation_t * mat._AccumulationRate, 0.0), gamma)
        * acc_color
        * unity_accumulation_property_scale;
    var result_color = mat._BaseColor + acc;

    if (uvu::kw_enabled(mat.SATURATE_ALPHA)) {
        result_color = vec4<f32>(result_color.rgb, clamp(result_color.a, 0.0, 1.0));
    } else if (uvu::kw_enabled(mat.SATURATE_COLOR)) {
        result_color = clamp(result_color, vec4<f32>(0.0), vec4<f32>(1.0));
    }

    return rg::retain_globals_additive(result_color);
}
