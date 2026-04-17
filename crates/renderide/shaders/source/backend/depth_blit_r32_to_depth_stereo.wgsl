// Multiview fullscreen pass: writes resolved linear depth from an R32Float 2D array into a 2-layer
// `Depth32Float` attachment. `@builtin(view_index)` selects the per-eye layer.

struct VsOut {
    @builtin(position) pos: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var o: VsOut;
    o.pos = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}

@group(0) @binding(0) var src_r32: texture_2d_array<f32>;

@fragment
fn fs_main(@builtin(position) pos: vec4f, @builtin(view_index) view: u32) -> @builtin(frag_depth) f32 {
    let dims = textureDimensions(src_r32);
    let xy = vec2i(i32(pos.x), i32(pos.y));
    let cx = min(u32(max(xy.x, 0)), dims.x - 1u);
    let cy = min(u32(max(xy.y, 0)), dims.y - 1u);
    let d = textureLoad(src_r32, vec2i(i32(cx), i32(cy)), i32(view), 0).r;
    return d;
}
