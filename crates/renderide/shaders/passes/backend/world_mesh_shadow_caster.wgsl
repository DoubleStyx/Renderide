//! Projected shadow caster for world meshes.
//!
//! Per-draw rows carry only model data. The active shadow-view projection is bound once per atlas
//! layer so cascades and cubemap faces can reuse the same caster slab.

struct ShadowCasterDraw {
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    _pad: array<vec4<f32>, 25>,
}

struct ShadowLayerUniforms {
    view_proj: mat4x4<f32>,
    light_position_range: vec4<f32>,
    shadow_params: vec4<f32>,
    _pad: array<vec4<f32>, 26>,
}

@group(0) @binding(0) var<storage, read> instances: array<ShadowCasterDraw>;
@group(1) @binding(0) var<uniform> shadow_layer: ShadowLayerUniforms;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @location(0) pos: vec4<f32>,
) -> VertexOutput {
    let draw = instances[instance_index];
    let world_p = draw.model * vec4<f32>(pos.xyz, 1.0);

    var out: VertexOutput;
    out.clip_pos = shadow_layer.view_proj * world_p;
    return out;
}

@fragment
fn fs_main() {
}
