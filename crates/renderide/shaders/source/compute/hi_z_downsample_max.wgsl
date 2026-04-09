// Max-reduction downsample for reverse-Z Hi-Z (closest surface has the largest stored depth).

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
    let sx0 = i32(dx * 2u);
    let sy0 = i32(dy * 2u);
    let sw = i32(sizes.src_w);
    let sh = i32(sizes.src_h);
    var m = 0.0;
    for (var j = 0u; j < 2u; j++) {
        for (var i = 0u; i < 2u; i++) {
            let xx = sx0 + i32(i);
            let yy = sy0 + i32(j);
            if (xx >= 0 && xx < sw && yy >= 0 && yy < sh) {
                let v = textureLoad(src, vec2i(xx, yy)).x;
                m = max(m, v);
            }
        }
    }
    textureStore(dst, vec2i(i32(dx), i32(dy)), vec4f(m, 0.0, 0.0, 1.0));
}
