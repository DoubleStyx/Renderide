//! High-level PBS channel contracts consumed by shared lighting.

#define_import_path renderide::pbs::surface

struct MetallicSurface {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

struct SpecularSurface {
    base_color: vec3<f32>,
    alpha: f32,
    specular_color: vec3<f32>,
    roughness: f32,
    one_minus_reflectivity: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn metallic(
    base_color: vec3<f32>,
    alpha: f32,
    metallic_value: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
) -> MetallicSurface {
    return MetallicSurface(
        base_color,
        alpha,
        clamp(metallic_value, 0.0, 1.0),
        roughness,
        occlusion,
        normalize(normal),
        emission,
    );
}

fn specular(
    base_color: vec3<f32>,
    alpha: f32,
    specular_color: vec3<f32>,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
) -> SpecularSurface {
    let clamped_specular_color = clamp(specular_color, vec3<f32>(0.0), vec3<f32>(1.0));
    return SpecularSurface(
        base_color,
        alpha,
        clamped_specular_color,
        roughness,
        1.0 - max(max(clamped_specular_color.r, clamped_specular_color.g), clamped_specular_color.b),
        occlusion,
        normalize(normal),
        emission,
    );
}
