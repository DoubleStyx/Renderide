//! Fullscreen solid background for host `CameraClearMode::Color`.

#import renderide::skybox_common as skybox

@group(0) @binding(0) var<uniform> view: skybox::SkyboxView;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = skybox::fullscreen_clip_pos(vertex_index);
    return out;
}

@fragment
fn fs_main(_in: VertexOutput) -> @location(0) vec4<f32> {
    return view.clear_color;
}
