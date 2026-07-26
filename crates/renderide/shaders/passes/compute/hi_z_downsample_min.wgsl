// Min-reduction downsample for reverse-Z Hi-Z. Each texel stores the farthest depth in its source
// footprint, so rectangle occlusion tests remain conservative when a coarse texel contains a hole.

struct DownsampleParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

@group(0) @binding(0) var src: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> sizes: DownsampleParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dx = gid.x;
    let dy = gid.y;
    if (dx >= sizes.dst_w || dy >= sizes.dst_h) {
        return;
    }
    let sx0 = (dx * sizes.src_w) / sizes.dst_w;
    let sy0 = (dy * sizes.src_h) / sizes.dst_h;
    let sx1 = min(((dx + 1u) * sizes.src_w + sizes.dst_w - 1u) / sizes.dst_w, sizes.src_w);
    let sy1 = min(((dy + 1u) * sizes.src_h + sizes.dst_h - 1u) / sizes.dst_h, sizes.src_h);
    var m = 1.0;
    for (var sy = sy0; sy < max(sy1, sy0 + 1u); sy++) {
        for (var sx = sx0; sx < max(sx1, sx0 + 1u); sx++) {
            let v = textureLoad(src, vec2i(i32(sx), i32(sy))).x;
            m = min(m, v);
        }
    }
    textureStore(dst, vec2i(i32(dx), i32(dy)), vec4f(m, 0.0, 0.0, 1.0));
}
