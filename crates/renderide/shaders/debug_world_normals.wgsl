// Debug raster: visualize world-space normals (RGB = n * 0.5 + 0.5).
// One 256-byte uniform slot per draw (WebGPU dynamic uniform alignment).
// Stereo: `view_proj_left` / `view_proj_right`; desktop duplicates the same matrix in both.

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> draw: PerDrawUniforms;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_n: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,
) -> VertexOutput {
    let world_p = draw.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = normalize((draw.model * vec4<f32>(normal.xyz, 0.0)).xyz);
    var out: VertexOutput;
    out.clip_pos = draw.view_proj_left * world_p;
    out.world_n = world_n;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.world_n * 0.5 + 0.5, 1.0);
}
