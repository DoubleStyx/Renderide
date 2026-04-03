struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    host_base_color: vec4f,
    host_metallic_roughness: vec4f,
    _pad: array<vec4f, 6>,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    return vec4f(rgb, 1.0);
}

