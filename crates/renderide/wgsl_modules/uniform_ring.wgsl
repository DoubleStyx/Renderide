#define_import_path uniform_ring

/// Matches [`crate::gpu::pipeline::uniforms::Uniforms`] (256 bytes): MVP, model, optional PBR host factors, padding.
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    /// When `.w >= 0.5`, `.xyz` overrides the stock PBR base color; otherwise defaults to gray.
    host_base_color: vec4f,
    /// When `.z >= 0.5`, `.x`/`.y` are metallic and perceptual roughness; else shader uses 0.5 / 0.5.
    host_metallic_roughness: vec4f,
    _pad: array<vec4f, 6>,
}
