//! Fullscreen triangle helpers shared by post, present, and backend blit passes.

#define_import_path renderide::fullscreen

struct FullscreenVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct FullscreenClipOutput {
    @builtin(position) clip_pos: vec4<f32>,
}

fn fullscreen_clip_pos(vertex_index: u32) -> vec4<f32> {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

fn fullscreen_uv(vertex_index: u32) -> vec2<f32> {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec2<f32>(x, y);
}

fn fullscreen_uv_flipped_y(vertex_index: u32) -> vec2<f32> {
    let uv = fullscreen_uv(vertex_index);
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

fn vertex_clip_main(vertex_index: u32) -> FullscreenClipOutput {
    var out: FullscreenClipOutput;
    out.clip_pos = fullscreen_clip_pos(vertex_index);
    return out;
}

fn vertex_main(vertex_index: u32) -> FullscreenVertexOutput {
    var out: FullscreenVertexOutput;
    out.clip_pos = fullscreen_clip_pos(vertex_index);
    out.uv = fullscreen_uv(vertex_index);
    return out;
}

fn vertex_flipped_y_main(vertex_index: u32) -> FullscreenVertexOutput {
    var out: FullscreenVertexOutput;
    out.clip_pos = fullscreen_clip_pos(vertex_index);
    out.uv = fullscreen_uv_flipped_y(vertex_index);
    return out;
}
