//! Unity `Shader "Volume/FogBox"` port.
//!
//! Uses the original property names and keyword-style flags exposed by FrooxEngine
//! (`FogBoxVolumeMaterial`) so host-driven material updates map directly.

#import renderide::globals as rg
#import renderide::mesh::vertex as mv
#import renderide::per_draw as pd
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

/// Carries object-space vertex position plus the per-draw instance index so OBJECT_SPACE fog
/// math (`cam_obj = inverse(model) * cam_world`) can be evaluated per-fragment, avoiding any
/// per-vertex / flat-interpolation drift across the proxy surface.
struct FogBoxVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) object_pos: vec3<f32>,
    @location(2) @interpolate(flat) instance_idx: u32,
    @location(3) @interpolate(flat) view_layer: u32,
}

/// Inverse of a 3x3 matrix via cofactors. Used to map world-space points into the proxy's
/// object space when the model matrix is general affine (rotation + non-uniform scale).
fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let det = dot(a, cross(b, c));
    // Singular / degenerate model: return identity so the shader produces stable output rather than NaN.
    if (abs(det) < 1e-20) {
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    let inv_det = 1.0 / det;
    return transpose(mat3x3<f32>(
        cross(b, c) * inv_det,
        cross(c, a) * inv_det,
        cross(a, b) * inv_det,
    ));
}

/// Maps a world-space point into the proxy's object space.
fn world_to_object_pos(model: mat4x4<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let linear = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let translation = model[3].xyz;
    return mat3_inverse(linear) * (world_pos - translation);
}

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
) -> FogBoxVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = mv::world_position(draw, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(draw, view_idx);
    let layer = view_idx;
#else
    let vp = mv::select_view_proj(draw, 0u);
    let layer = 0u;
#endif

    var out: FogBoxVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.object_pos = pos.xyz;
    out.instance_idx = instance_index;
    out.view_layer = layer;
    return out;
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

/// Intersection of an oriented line with a single axis-aligned plane.
fn line_plane_intersection(
    line_point: vec3<f32>,
    line_direction: vec3<f32>,
    plane_point: vec3<f32>,
    plane_normal: vec3<f32>,
) -> vec3<f32> {
    let diff = line_point - plane_point;
    let prod1 = dot(diff, plane_normal);
    let prod2 = dot(line_direction, plane_normal);
    return line_point - line_direction * (prod1 / prod2);
}

/// Port of Unity `IntersectUnitCube`: closest point where a ray enters/exits the unit cube
/// `[-0.5, 0.5]^3`. Tests all six face planes and keeps the in-bounds candidate nearest the line origin.
fn intersect_unit_cube(line_point: vec3<f32>, line_direction: vec3<f32>) -> vec3<f32> {
    let half_size = vec2<f32>(0.5);
    let i0 = line_plane_intersection(line_point, line_direction, vec3<f32>(-0.5, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0));
    let i1 = line_plane_intersection(line_point, line_direction, vec3<f32>(0.5, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0));
    let i2 = line_plane_intersection(line_point, line_direction, vec3<f32>(0.0, -0.5, 0.0), vec3<f32>(0.0, -1.0, 0.0));
    let i3 = line_plane_intersection(line_point, line_direction, vec3<f32>(0.0, 0.5, 0.0), vec3<f32>(0.0, 1.0, 0.0));
    let i4 = line_plane_intersection(line_point, line_direction, vec3<f32>(0.0, 0.0, -0.5), vec3<f32>(0.0, 0.0, -1.0));
    let i5 = line_plane_intersection(line_point, line_direction, vec3<f32>(0.0, 0.0, 0.5), vec3<f32>(0.0, 0.0, 1.0));

    var best_point = line_point;
    var best_dist = 65000.0;
    if (all(abs(i0.yz) <= half_size)) {
        let d = distance(line_point, i0);
        if (d < best_dist) { best_dist = d; best_point = i0; }
    }
    if (all(abs(i1.yz) <= half_size)) {
        let d = distance(line_point, i1);
        if (d < best_dist) { best_dist = d; best_point = i1; }
    }
    if (all(abs(i2.xz) <= half_size)) {
        let d = distance(line_point, i2);
        if (d < best_dist) { best_dist = d; best_point = i2; }
    }
    if (all(abs(i3.xz) <= half_size)) {
        let d = distance(line_point, i3);
        if (d < best_dist) { best_dist = d; best_point = i3; }
    }
    if (all(abs(i4.xy) <= half_size)) {
        let d = distance(line_point, i4);
        if (d < best_dist) { best_dist = d; best_point = i4; }
    }
    if (all(abs(i5.xy) <= half_size)) {
        let d = distance(line_point, i5);
        if (d < best_dist) { best_dist = d; best_point = i5; }
    }
    return best_point;
}

fn clamp_inside_unit_cube(pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    if (all(abs(pos) <= vec3<f32>(0.5))) {
        return pos;
    }
    return intersect_unit_cube(pos, dir);
}

/// Ray origin for OBJECT_SPACE fog. Unity's shader assumes a **unit cube** proxy; Resonite often
/// uses a **unit sphere** (r≈1 in object space). `clamp_inside_unit_cube` is wrong when the camera
/// sits between the inscribed cube (|x|,|y|,|z|≤0.5) and the sphere shell: it snaps `start` to
/// cube face intersections, which swings with view and reads as heat-haze / swimming. If the
/// camera is inside the unit sphere, use it directly as the segment start (same as “inside cube”
/// for Unity's common case); otherwise fall back to cube entry for cube-only / exterior cameras.
fn object_space_ray_start(cam_pos: vec3<f32>, ndir: vec3<f32>) -> vec3<f32> {
    if (dot(cam_pos, cam_pos) <= 1.0) {
        return cam_pos;
    }
    return clamp_inside_unit_cube(cam_pos, ndir);
}

//#pass forward_transparent_volume
@fragment
fn fs_main(
    in: FogBoxVertexOutput,
) -> @location(0) vec4<f32> {
    // Unity `Volume/FogBox`: integrates fog along the camera ray clipped to the proxy volume,
    // then either linearly clamps or exponentially attenuates by `_FogDensity`.
    //
    // The render path uses `forward_transparent_volume` (depth test Always) -- matches Unity
    // `ZTest Always` so a world-enclosing proxy still shades pixels where opaque geometry
    // sits in front of (or behind) the proxy shell.
    //
    // WORLD_SPACE evaluates fog in world meters using `scene_z` directly. OBJECT_SPACE
    // evaluates fog in unit-cube units (proxy-local), which is what `_FogStart` / `_FogEnd`
    // are calibrated against in Unity.
    let scene_z = sds::scene_linear_depth(in.clip_pos, in.view_layer);
    let part_z = sds::fragment_linear_depth(in.world_pos, in.view_layer);

    // `#pragma multi_compile OBJECT_SPACE WORLD_SPACE` -- Unity defaults to the first listed
    // keyword (`OBJECT_SPACE`) when neither variant is selected. FrooxEngine often does not
    // serialize the default keyword over the wire, so when both arrive as 0 we treat the
    // material as object-space (matching Unity's compile-time fallback).
    let use_world_space =
        uvu::kw_enabled(mat.WORLD_SPACE) && !uvu::kw_enabled(mat.OBJECT_SPACE);

    var dist: f32;
    if (use_world_space) {
        dist = scene_z;
    } else {
        // Object-space ray: from camera (in proxy object space) toward this fragment's
        // object-space position, clipped to the unit cube and shortened by scene depth.
        // Computed here in the fragment shader so `cam_obj` cannot drift across the proxy
        // surface from per-vertex flat-interpolation noise.
        let draw = pd::get_draw(in.instance_idx);
        let cam_pos = world_to_object_pos(
            draw.model,
            rg::camera_world_pos_for_view(in.view_layer),
        );
        let end_pos = in.object_pos;
        let ndir = normalize(end_pos - cam_pos);
        let start = object_space_ray_start(cam_pos, ndir);
        let max_dist = distance(cam_pos, end_pos);
        let end_ratio = min(scene_z / max(part_z, 1e-6), 1.0);
        let end = cam_pos + ndir * max_dist * end_ratio;
        // Mirrors Unity's `if (distance_sqr(camPos, end) < distance_sqr(camPos, start)) discard;`
        // -- if scene depth is closer than the proxy entry, no fog accumulates along this ray.
        if (dot(end - cam_pos, end - cam_pos) < dot(start - cam_pos, start - cam_pos)) {
            discard;
        }
        dist = distance(start, end);
    }

    // Fog modes are mutually exclusive (Unity `#pragma multi_compile FOG_LINEAR FOG_EXP FOG_EXP2`).
    // For `FOG_LINEAR`, `_FogDensity` is not compiled in -- UI often leaves density at 0; exp
    // math would zero out. Treat ~zero density like linear when EXP isn't intended.
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

    let acc_color = accumulation_color(in.world_pos);
    let gamma = max(mat._GammaCurve, 1e-5);
    // Unity `Volume/FogBox` exactly: `pow(dist * _AccumulationRate, _GammaCurve) * accColor`.
    // Previous `0.1 * 0.1` parity scales forced ~100x higher `_AccumulationRate` / `_AccumulationColor`
    // than the Unity shader for the same authored values (the product sat inside `pow`).
    let acc = pow(max(dist * mat._AccumulationRate, 0.0), gamma) * acc_color;
    var result_color = mat._BaseColor + acc;

    if (uvu::kw_enabled(mat.SATURATE_ALPHA)) {
        result_color = vec4<f32>(result_color.rgb, clamp(result_color.a, 0.0, 1.0));
    } else if (uvu::kw_enabled(mat.SATURATE_COLOR)) {
        result_color = clamp(result_color, vec4<f32>(0.0), vec4<f32>(1.0));
    }

    return rg::retain_globals_additive(result_color);
}
