// Multiview variant: `@builtin(view_index)` selects per-eye view–projection (single draw, two layers).

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
    @builtin(view_index) view_idx: u32,
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,
) -> VertexOutput {
    let world_p = draw.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = normalize((draw.model * vec4<f32>(normal.xyz, 0.0)).xyz);
    // WGSL `select` only allows scalars or vecN, not matrices — branch per eye.
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = draw.view_proj_left;
    } else {
        vp = draw.view_proj_right;
    }
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_n = world_n;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.world_n * 0.5 + 0.5, 1.0);
}
