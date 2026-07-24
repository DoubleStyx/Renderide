// Fullscreen pass: writes a single-sample depth attachment's texels into the R32Float
// scene-depth snapshot. Stereo runs one pass per layer, so no multiview variant exists.

#import renderide::core::fullscreen as fs

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> fs::FullscreenClipOutput {
    return fs::vertex_clip_main(vi);
}

@group(0) @binding(0) var src_depth: texture_depth_2d;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) f32 {
    let dims = textureDimensions(src_depth);
    let xy = vec2i(i32(pos.x), i32(pos.y));
    let cx = min(u32(max(xy.x, 0)), dims.x - 1u);
    let cy = min(u32(max(xy.y, 0)), dims.y - 1u);
    return textureLoad(src_depth, vec2i(i32(cx), i32(cy)), 0);
}
