// Copies reverse-Z depth from the main attachment into mip0 of an R32Float pyramid (possibly downscaled).

@group(0) @binding(0) var src_depth: texture_depth_2d;
@group(0) @binding(1) var dst_mip0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let src_dims = textureDimensions(src_depth);
    let dst_dims = textureDimensions(dst_mip0);
    let x = gid.x;
    let y = gid.y;
    if (x >= dst_dims.x || y >= dst_dims.y) {
        return;
    }
    let sx = min((x * src_dims.x + dst_dims.x / 2u) / dst_dims.x, src_dims.x - 1u);
    let sy = min((y * src_dims.y + dst_dims.y / 2u) / dst_dims.y, src_dims.y - 1u);
    let d = textureLoad(src_depth, vec2i(i32(sx), i32(sy)), 0);
    textureStore(dst_mip0, vec2i(i32(x), i32(y)), vec4f(d, 0.0, 0.0, 1.0));
}
