//! Point-light shadow caster for world meshes.
//!
//! The raster projection selects the cubemap face and atlas UVs. Fragment depth stores normalized
//! radial distance so point-light shadows compare consistently across all six faces.

struct ShadowCasterUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    light_position_range: vec4<f32>,
    shadow_params: vec4<f32>,
    _pad: array<vec4<f32>, 15>,
}

@group(0) @binding(0) var<storage, read> instances: array<ShadowCasterUniforms>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @location(0) pos: vec4<f32>,
) -> VertexOutput {
    let draw = instances[instance_index];
    let world_p = draw.model * vec4<f32>(pos.xyz, 1.0);

    var out: VertexOutput;
    out.clip_pos = draw.view_proj_left * world_p;
    out.world_pos = world_p.xyz;
    out.instance_index = instance_index;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @builtin(frag_depth) f32 {
    let draw = instances[in.instance_index];
    let range = max(draw.light_position_range.w, 0.001);
    let bias = max(draw.shadow_params.x, 0.0);
    let radial_depth = (length(in.world_pos - draw.light_position_range.xyz) + bias) / range;
    return clamp(radial_depth, 0.0, 1.0);
}
