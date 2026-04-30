// Fullscreen pass: writes resolved linear depth from an R32Float texture into a depth attachment.
// `#ifdef MULTIVIEW` selects the per-eye array layer via `@builtin(view_index)` and a
// `texture_2d_array<f32>` source; the non-multiview path uses a plain `texture_2d<f32>`.

#import renderide::fullscreen as fs

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> fs::FullscreenClipOutput {
    return fs::vertex_clip_main(vi);
}

#ifdef MULTIVIEW
@group(0) @binding(0) var src_r32: texture_2d_array<f32>;
#else
@group(0) @binding(0) var src_r32: texture_2d<f32>;
#endif

@fragment
fn fs_main(
    @builtin(position) pos: vec4f,
#ifdef MULTIVIEW
    @builtin(view_index) view: u32,
#endif
) -> @builtin(frag_depth) f32 {
    let dims = textureDimensions(src_r32);
    let xy = vec2i(i32(pos.x), i32(pos.y));
    let cx = min(u32(max(xy.x, 0)), dims.x - 1u);
    let cy = min(u32(max(xy.y, 0)), dims.y - 1u);
#ifdef MULTIVIEW
    let d = textureLoad(src_r32, vec2i(i32(cx), i32(cy)), i32(view), 0).r;
#else
    let d = textureLoad(src_r32, vec2i(i32(cx), i32(cy)), 0).r;
#endif
    return d;
}
