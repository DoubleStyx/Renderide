//! Cook–Torrance GGX BRDF, tangent space helpers, and clustered direct-light terms for PBS materials
//! (metallic / specular workflows).
//!
//! Import with `#import renderide::pbs::brdf`. Depends on [`renderide::globals`] for [`GpuLight`].

#import renderide::globals as rg

#define_import_path renderide::pbs::brdf

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(denom * denom * 3.14159265, 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow5(1.0 - cos_theta);
}

/// Range-coupled windowed inverse-square distance attenuation for punctual lights.
/// `(saturate(1 - (d/r)^4))^2 / d² * 4π*r²` — Karis/Lagarde window (exactly zero at `range` with a
/// wide smooth transition zone that hides the per-cluster cull boundary) plus a `4π*range²`
/// range-coupling term so that larger `range` reads as a brighter light, matching Resonite's
/// BiRP-style authoring convention where increasing a light's range is expected to increase
/// perceived brightness as well as extend the falloff. Intensity is applied by the call site.
fn distance_attenuation(dist: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 0.0;
    }
    let inv_d2 = 1.0 / max(dist * dist, 0.01 * 0.01);
    let t = dist / range;
    let window_inner = clamp(1.0 - t * t * t * t, 0.0, 1.0);
    let window = window_inner * window_inner;
    let range_boost = range * range;
    return inv_d2 * window * range_boost;
}

/// Builds an orthonormal TBN from a world-space normal (fallback when mesh tangents are unavailable).
///
/// Uses the branchless construction from *Building an Orthonormal Basis, Revisited* (Duff et al., JCGT 2017)
/// so there is no discontinuity near the poles (unlike a fixed world-up cross).
fn orthonormal_tbn(n: vec3<f32>) -> mat3x3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitan = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    return mat3x3<f32>(normalize(t), normalize(bitan), n);
}

/// Metallic workflow: Cook–Torrance direct light with GGX + Schlick; diffuse scaled by `(1 - metallic)`.
fn direct_radiance_metallic(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
    metallic: f32,
    base_color: vec3<f32>,
    f0: vec3<f32>,
) -> vec3<f32> {
    let light_pos = light.position.xyz;
    let light_dir = light.direction.xyz;
    let light_color = light.color.xyz;
    var l: vec3<f32>;
    var attenuation: f32;
    if light.light_type == 0u {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        attenuation = light.intensity * distance_attenuation(dist, light.range);
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let inner_cos = min(light.spot_cos_half_angle + 0.1, 1.0);
        let spot_atten = smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
        attenuation = light.intensity * spot_atten * distance_attenuation(dist, light.range);
    }
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0001);
    let n_dot_h = max(dot(n, h), 0.0);
    let radiance = light_color * attenuation * n_dot_l;
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let spec = (distribution_ggx(n_dot_h, roughness) * geometry_smith(n_dot_v, n_dot_l, roughness) * f)
        / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / 3.14159265;
    return (diffuse + spec) * radiance;
}

/// Specular workflow (Unity Standard SpecularSetup): diffuse albedo scaled by `one_minus_reflectivity`
/// (energy taken by colored specular); `f0` is the tinted specular color.
fn direct_radiance_specular(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
    base_color: vec3<f32>,
    f0: vec3<f32>,
    one_minus_reflectivity: f32,
) -> vec3<f32> {
    let light_pos = light.position.xyz;
    let light_dir = light.direction.xyz;
    let light_color = light.color.xyz;
    var l: vec3<f32>;
    var attenuation: f32;
    if light.light_type == 0u {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        attenuation = light.intensity * distance_attenuation(dist, light.range);
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let inner_cos = min(light.spot_cos_half_angle + 0.1, 1.0);
        let spot_atten = smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
        attenuation = light.intensity * spot_atten * distance_attenuation(dist, light.range);
    }
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0001);
    let n_dot_h = max(dot(n, h), 0.0);
    let radiance = light_color * attenuation * n_dot_l;
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let spec = (distribution_ggx(n_dot_h, roughness) * geometry_smith(n_dot_v, n_dot_l, roughness) * f)
        / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let kd = (vec3<f32>(1.0) - f) * one_minus_reflectivity;
    let diffuse = kd * base_color / 3.14159265;
    return (diffuse + spec) * radiance;
}

/// Lambertian only (specular highlights disabled), metallic path.
fn diffuse_only_metallic(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    base_color: vec3<f32>,
) -> vec3<f32> {
    let light_pos = light.position.xyz;
    let light_dir = light.direction.xyz;
    let light_color = light.color.xyz;
    var l: vec3<f32>;
    var attenuation: f32;
    if light.light_type == 0u {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        attenuation = light.intensity * distance_attenuation(dist, light.range);
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let inner_cos = min(light.spot_cos_half_angle + 0.1, 1.0);
        let spot_atten = smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
        attenuation = light.intensity * spot_atten * distance_attenuation(dist, light.range);
    }
    let n_dot_l = max(dot(n, l), 0.0);
    return base_color / 3.14159265 * light_color * attenuation * n_dot_l;
}

/// Lambertian only with diffuse energy scaled for specular workflow.
fn diffuse_only_specular(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    base_color: vec3<f32>,
    one_minus_reflectivity: f32,
) -> vec3<f32> {
    let light_pos = light.position.xyz;
    let light_dir = light.direction.xyz;
    let light_color = light.color.xyz;
    var l: vec3<f32>;
    var attenuation: f32;
    if light.light_type == 0u {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        attenuation = light.intensity * distance_attenuation(dist, light.range);
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let inner_cos = min(light.spot_cos_half_angle + 0.1, 1.0);
        let spot_atten = smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
        attenuation = light.intensity * spot_atten * distance_attenuation(dist, light.range);
    }
    let n_dot_l = max(dot(n, l), 0.0);
    return base_color * one_minus_reflectivity / 3.14159265 * light_color * attenuation * n_dot_l;
}
