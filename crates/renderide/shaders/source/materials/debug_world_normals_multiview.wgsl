//! Debug raster: world-space normals (RGB). Multiview stereo (`@builtin(view_index)`).
//! Composed to `shaders/target/debug_world_normals_multiview.wgsl`.
//! `build.rs` prepends `source/modules/globals.wgsl`.

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

@group(2) @binding(0) var<uniform> draw: PerDrawUniforms;

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
    let n = in.world_n * 0.5 + 0.5;
    var lit: u32 = 0u;
    if (frame.light_count > 0u) {
        lit = lights[0].light_type;
    }
    let c = vec3<f32>(n) + frame.camera_world_pos.xyz * 0.0001 + vec3<f32>(f32(lit) * 1e-10);
    return vec4<f32>(c, 1.0);
}
