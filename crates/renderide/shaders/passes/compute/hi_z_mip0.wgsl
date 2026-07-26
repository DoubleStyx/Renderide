// Conservatively reduces reverse-Z depth from the main attachment into mip0 of an R32Float
// farthest-depth pyramid. `#ifdef MULTIVIEW` selects a `texture_depth_2d_array` source indexed by
// a per-dispatch layer uniform (WGSL forbids `@builtin(view_index)` in compute); the non-multiview
// path samples a plain `texture_depth_2d`.

#ifdef MULTIVIEW
struct LayerParams {
    layer: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var src_depth: texture_depth_2d_array;
@group(0) @binding(1) var<uniform> layer_params: LayerParams;
@group(0) @binding(2) var dst_mip0: texture_storage_2d<r32float, write>;
#else
@group(0) @binding(0) var src_depth: texture_depth_2d;
@group(0) @binding(1) var dst_mip0: texture_storage_2d<r32float, write>;
#endif

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let src_dims = textureDimensions(src_depth);
    let dst_dims = textureDimensions(dst_mip0);
    let x = gid.x;
    let y = gid.y;
    if (x >= dst_dims.x || y >= dst_dims.y) {
        return;
    }
    let sx0 = (x * src_dims.x) / dst_dims.x;
    let sy0 = (y * src_dims.y) / dst_dims.y;
    let sx1 = min(((x + 1u) * src_dims.x + dst_dims.x - 1u) / dst_dims.x, src_dims.x);
    let sy1 = min(((y + 1u) * src_dims.y + dst_dims.y - 1u) / dst_dims.y, src_dims.y);
    var d = 1.0;
    for (var sy = sy0; sy < max(sy1, sy0 + 1u); sy++) {
        for (var sx = sx0; sx < max(sx1, sx0 + 1u); sx++) {
#ifdef MULTIVIEW
            let sample = textureLoad(src_depth, vec2i(i32(sx), i32(sy)), i32(layer_params.layer), 0);
#else
            let sample = textureLoad(src_depth, vec2i(i32(sx), i32(sy)), 0);
#endif
            d = min(d, sample);
        }
    }
    textureStore(dst_mip0, vec2i(i32(x), i32(y)), vec4f(d, 0.0, 0.0, 1.0));
}
