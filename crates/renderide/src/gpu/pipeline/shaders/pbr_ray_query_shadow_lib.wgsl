struct RtShadowUniforms {
    soft_shadow_sample_count: u32,
    soft_cone_scale: f32,
    frame_counter: u32,
    shadow_mode: u32,
    full_viewport_width: u32,
    full_viewport_height: u32,
    shadow_atlas_width: u32,
    shadow_atlas_height: u32,
    gbuffer_origin: vec3f,
    _pad0: f32,
}
@group(1) @binding(5) var<uniform> rt_shadow: RtShadowUniforms;
@group(1) @binding(6) var shadow_atlas: texture_2d_array<f32>;
@group(1) @binding(7) var shadow_sampler: sampler;

fn hash11(p: f32) -> f32 {
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn hash21(p: vec2f) -> f32 {
    return hash11(dot(p, vec2f(127.1, 311.7)));
}

fn trace_shadow_ray(origin: vec3f, dir: vec3f, t_min: f32, t_max: f32) -> f32 {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, t_min, t_max, origin, dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    return select(0.0, 1.0, hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

fn rt_shadow_atlas_uv(frag_xy: vec2f) -> vec2f {
    let fw = max(f32(rt_shadow.full_viewport_width), 1.0);
    let fh = max(f32(rt_shadow.full_viewport_height), 1.0);
    return vec2f((frag_xy.x + 0.5) / fw, (frag_xy.y + 0.5) / fh);
}

fn light_shadow_visibility(
    light: GpuLight,
    world_pos: vec3f,
    surf_n: vec3f,
    l: vec3f,
    attenuation: f32,
    frag_xy: vec2f,
    light_idx: u32,
    cluster_slot: u32,
) -> f32 {
    if light.shadow_type == 0u || light.shadow_strength <= 0.0 || attenuation <= 0.0 {
        return 1.0;
    }
    if rt_shadow.shadow_mode == 1u {
        let vis = textureSampleLevel(
            shadow_atlas,
            shadow_sampler,
            rt_shadow_atlas_uv(frag_xy),
            i32(cluster_slot),
            0.0
        ).r;
        return mix(1.0, vis, light.shadow_strength);
    }
    let light_pos = light.position.xyz;
    let ray_origin = world_pos + surf_n * light.shadow_normal_bias + l * light.shadow_bias;
    var trace_dir: vec3f;
    var t_min: f32;
    var t_max: f32;
    if light.light_type == 1u {
        // Directional: shadow_near_plane is a world-space-style offset along the ray; unbounded range.
        t_min = max(light.shadow_near_plane, 0.001);
        trace_dir = l;
        t_max = 1.0e6;
    } else {
        // Point / spot: trace from biased receiver toward the light. Host near-plane is often tuned
        // for directional lights; if it exceeds the distance to the light, t_min lands past the
        // source and shadows break for nearby receivers.
        let to_lp = light_pos - ray_origin;
        let dist = length(to_lp);
        trace_dir = to_lp / max(dist, 1e-8);
        let light_margin = min(0.02, dist * 0.1);
        t_max = max(dist - light_margin, 1e-5);
        var t_near = max(light.shadow_near_plane, 0.001);
        t_near = min(t_near, t_max * 0.95);
        t_min = min(t_near, max(t_max - 1e-4, 1e-6));
        if t_min >= t_max {
            t_min = max(t_max * 0.01, 1e-6);
        }
    }
    let is_hard = light.shadow_type == 1u;
    if is_hard {
        let vis = trace_shadow_ray(ray_origin, trace_dir, t_min, t_max);
        return mix(1.0, vis, light.shadow_strength);
    }
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(trace_dir.y) > 0.999);
    let uu = normalize(cross(up, trace_dir));
    let vv = cross(trace_dir, uu);
    var acc_vis = 0.0;
    let smax = min(max(rt_shadow.soft_shadow_sample_count, 1u), 16u);
    let cone = 0.025 * max(rt_shadow.soft_cone_scale, 0.0);
    let seed = dot(frag_xy, vec2f(12.9898, 78.233))
        + f32(light_idx) * 19.19
        + f32(rt_shadow.frame_counter) * 3.174;
    for (var s = 0u; s < smax; s++) {
        let r1 = hash21(vec2f(seed + f32(s) * 0.618, f32(s) * 1.414));
        let r2 = hash21(vec2f(seed + 19.19, f32(s) * 2.718));
        let ox = (r1 * 2.0 - 1.0) * cone;
        let oy = (r2 * 2.0 - 1.0) * cone;
        let d = normalize(trace_dir + uu * ox + vv * oy);
        acc_vis += trace_shadow_ray(ray_origin, d, t_min, t_max);
    }
    let vis = acc_vis / f32(smax);
    return mix(1.0, vis, light.shadow_strength);
}
