// Fullscreen pass: restores a layer's cached static-caster depth into the live shadow atlas layer.
//
// The source is a single array layer of the static store exposed as a plain 2D depth view, so no
// layer index needs passing in. Depth compare is Always with depth writes on, which reproduces the
// stored depth texel for texel. Dynamic casters are then drawn over the result with the normal
// LessEqual test, which is exactly what a combined static+dynamic pass would have produced.

#import renderide::core::fullscreen as fs

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> fs::FullscreenClipOutput {
    return fs::vertex_clip_main(vi);
}

@group(0) @binding(0) var static_depth: texture_depth_2d;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @builtin(frag_depth) f32 {
    let dims = textureDimensions(static_depth);
    let xy = vec2i(i32(pos.x), i32(pos.y));
    let cx = min(u32(max(xy.x, 0)), dims.x - 1u);
    let cy = min(u32(max(xy.y, 0)), dims.y - 1u);
    return textureLoad(static_depth, vec2i(i32(cx), i32(cy)), 0);
}
