struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    host_base_color: vec4f,
    host_metallic_roughness: vec4f,
    _pad: array<vec4f, 6>,
}
struct MrtGbufferFrame {
    view_position: vec3f,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> mrt_frame: MrtGbufferFrame;
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    let world_pos = u.model * vec4f(in.position, 1.0);
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.world_normal = (u.model * vec4f(in.normal, 0.0)).xyz;
    out.world_position = world_pos.xyz;
    return out;
}
struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let n = normalize(in.world_normal);
    let rel = in.world_position - mrt_frame.view_position;
    return FragmentOutput(
        vec4f(n * 0.5 + 0.5, 1.0),
        vec4f(rel, 1.0),
        vec4f(n, 0.0),
    );
}

