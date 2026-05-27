struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var source_cookie: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let p = pos[vertex_index];
    var out: VsOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + vec2<f32>(0.5);
    return out;
}

fn source_sample(in: VsOut) -> vec4<f32> {
    return textureSample(source_cookie, source_sampler, clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0)));
}

fn source_alpha(in: VsOut) -> f32 {
    return source_sample(in).a;
}

fn source_red(in: VsOut) -> f32 {
    return source_sample(in).r;
}

@fragment
fn fs_alpha_scalar(in: VsOut) -> @location(0) f32 {
    return source_alpha(in);
}

@fragment
fn fs_red_scalar(in: VsOut) -> @location(0) f32 {
    return source_red(in);
}

@fragment
fn fs_alpha_rgba(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(source_alpha(in), 1.0, 1.0, 1.0);
}

@fragment
fn fs_red_rgba(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(source_red(in), 1.0, 1.0, 1.0);
}
