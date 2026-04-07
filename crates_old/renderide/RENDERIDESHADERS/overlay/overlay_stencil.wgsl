
#import renderide_color_util

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
struct Uniforms {
    mvp: mat4x4f,
    model: mat4x4f,
    clip_rect: vec4f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let rect = uniforms.clip_rect;
    if rect.z > 0.0 {
        let ndc = in.clip_position.xy / in.clip_position.w;
        let nx = (ndc.x + 1.0) * 0.5;
        let ny = 1.0 - (ndc.y + 1.0) * 0.5;
        if nx < rect.x || nx > rect.x + rect.z || ny < rect.y || ny > rect.y + rect.w {
            discard;
        }
    }
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = renderide_color_util::hsv_to_rgb(hue, sat, 1.0);
    return vec4f(rgb, 1.0);
}
