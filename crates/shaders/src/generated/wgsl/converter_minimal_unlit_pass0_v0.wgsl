// Reference WGSL for `Converter/MinimalUnlit` (SampleShaders/MinimalUnlit.shader).
// Regenerate via `slangc` when available; this file matches the sample HLSL entry points `vert` / `frag`.

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vert(@location(0) vertex: vec4<f32>) -> VertexOutput {
    var o: VertexOutput;
    o.pos = vertex;
    return o;
}

@fragment
fn frag() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
