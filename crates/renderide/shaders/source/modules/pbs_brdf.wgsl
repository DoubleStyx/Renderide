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

/// Builds an orthonormal TBN from a world-space normal (Mikkelsen-style fallback when no tangent).
fn orthonormal_tbn(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

/// Decodes a tangent-space normal from an RGB normal map sample (DXT5nm / Unity-style packed Z).
fn decode_ts_normal(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    let nm_xy = (raw.xy * 2.0 - 1.0) * scale;
    let z = max(sqrt(max(1.0 - dot(nm_xy, nm_xy), 0.0)), 1e-6);
    return normalize(vec3<f32>(nm_xy, z));
}

/// Like [`decode_ts_normal`], but treats near-white RGB `(1,1,1)` as flat Z-up (placeholder texels).
fn decode_ts_normal_placeholder_flat(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    if (all(raw > vec3<f32>(0.99, 0.99, 0.99))) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return decode_ts_normal(raw, scale);
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
        attenuation = select(
            0.0,
            light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)),
            light.range > 0.0
        );
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
        attenuation = select(
            0.0,
            light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001),
            light.range > 0.0
        );
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
        attenuation = select(
            0.0,
            light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)),
            light.range > 0.0
        );
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
        attenuation = select(
            0.0,
            light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001),
            light.range > 0.0
        );
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
        attenuation = select(
            0.0,
            light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)),
            light.range > 0.0
        );
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
        attenuation = select(
            0.0,
            light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001),
            light.range > 0.0
        );
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
        attenuation = select(
            0.0,
            light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)),
            light.range > 0.0
        );
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
        attenuation = select(
            0.0,
            light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001),
            light.range > 0.0
        );
    }
    let n_dot_l = max(dot(n, l), 0.0);
    return base_color * one_minus_reflectivity / 3.14159265 * light_color * attenuation * n_dot_l;
}
