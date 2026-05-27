//! Compute pass: raw reverse-Z depth -> GTAO view-space depth levels 0..4.

#import renderide::frame::types as ft
#import renderide::post::gtao_params as gparams

#ifdef MULTIVIEW
@group(0) @binding(0) var raw_depth: texture_depth_2d_array;
#else
@group(0) @binding(0) var raw_depth: texture_depth_2d;
#endif
@group(0) @binding(1) var<uniform> frame: ft::FrameGlobals;
@group(0) @binding(2) var<uniform> gtao: gparams::GtaoParams;
#ifdef MULTIVIEW
@group(0) @binding(3) var dst_mip0: texture_storage_2d_array<r32float, write>;
@group(0) @binding(4) var dst_mip1: texture_storage_2d_array<r32float, write>;
@group(0) @binding(5) var dst_mip2: texture_storage_2d_array<r32float, write>;
@group(0) @binding(6) var dst_mip3: texture_storage_2d_array<r32float, write>;
@group(0) @binding(7) var dst_mip4: texture_storage_2d_array<r32float, write>;
#else
@group(0) @binding(3) var dst_mip0: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var dst_mip1: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var dst_mip2: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var dst_mip3: texture_storage_2d<r32float, write>;
@group(0) @binding(7) var dst_mip4: texture_storage_2d<r32float, write>;
#endif

var<workgroup> previous_mip_depth: array<array<f32, 8>, 8>;

fn projection_flags_for_layer(layer: u32) -> u32 {
#ifdef MULTIVIEW
    if ((layer & 1u) != 0u) {
        return frame.frame_tail.z;
    }
#endif
    return frame.frame_tail.y;
}

fn view_is_orthographic(layer: u32) -> bool {
    let flags = projection_flags_for_layer(layer);
    return (flags & ft::FRAME_PROJECTION_FLAG_ORTHOGRAPHIC) != 0u;
}

fn linearize_depth(d: f32, layer: u32) -> f32 {
    let near = frame.near_clip;
    let far = frame.far_clip;
    if (view_is_orthographic(layer)) {
        return far - d * (far - near);
    }
    let denom = d * (far - near) + near;
    return (near * far) / max(denom, 1e-6);
}

fn raw_dimensions() -> vec2<u32> {
#ifdef MULTIVIEW
    let dim = textureDimensions(raw_depth);
#else
    let dim = textureDimensions(raw_depth);
#endif
    return vec2<u32>(dim.xy);
}

fn load_raw_view_z(pix: vec2<i32>, layer: u32) -> f32 {
#ifdef MULTIVIEW
    let raw = textureLoad(raw_depth, pix, i32(layer), 0);
#else
    let raw = textureLoad(raw_depth, pix, 0);
#endif
    return select(0.0, linearize_depth(raw, layer), raw > 0.0);
}

fn load_mip0_depth(mip0_pix: vec2<i32>, mip0_max: vec2<i32>, raw_dim: vec2<u32>, layer: u32) -> f32 {
    let clamped_mip0_pix = clamp(mip0_pix, vec2<i32>(0), mip0_max);
    let divisor = clamp(gtao.resolution_divisor, 1u, 4u);
    let raw_base = clamped_mip0_pix * i32(divisor);
    var sum = 0.0;
    var count = 0.0;
    for (var y = 0u; y < 4u; y = y + 1u) {
        if (y >= divisor) {
            continue;
        }
        for (var x = 0u; x < 4u; x = x + 1u) {
            if (x >= divisor) {
                continue;
            }
            let raw_pix = raw_base + vec2<i32>(i32(x), i32(y));
            if (raw_pix.x >= i32(raw_dim.x) || raw_pix.y >= i32(raw_dim.y)) {
                continue;
            }
            sum = sum + load_raw_view_z(raw_pix, layer);
            count = count + 1.0;
        }
    }
    return select(0.0, sum / max(count, 1.0), count > 0.0);
}

fn depth_mip_filter(d0: f32, d1: f32, d2: f32, d3: f32) -> f32 {
    let max_depth = max(max(d0, d1), max(d2, d3));
    if (max_depth <= 0.0) {
        return 0.0;
    }

    let effect_radius = max(gtao.radius_world * gtao.radius_multiplier, 1e-4) * 0.75;
    let falloff_fraction = clamp(gtao.falloff_range, 0.01, 2.0);
    let falloff_range = max(falloff_fraction * effect_radius, 1e-4);
    let falloff_from = effect_radius * (1.0 - falloff_fraction);
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = falloff_from / falloff_range + 1.0;

    let w0 = clamp((max_depth - d0) * falloff_mul + falloff_add, 0.0, 1.0);
    let w1 = clamp((max_depth - d1) * falloff_mul + falloff_add, 0.0, 1.0);
    let w2 = clamp((max_depth - d2) * falloff_mul + falloff_add, 0.0, 1.0);
    let w3 = clamp((max_depth - d3) * falloff_mul + falloff_add, 0.0, 1.0);
    let weight_sum = max(w0 + w1 + w2 + w3, 1e-5);
    return (w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3) / weight_sum;
}

fn store_mip0(pix: vec2<i32>, layer: u32, value: f32) {
    let dim = textureDimensions(dst_mip0);
    if (pix.x < 0 || pix.y < 0 || pix.x >= i32(dim.x) || pix.y >= i32(dim.y)) {
        return;
    }
#ifdef MULTIVIEW
    textureStore(dst_mip0, pix, i32(layer), vec4<f32>(value, 0.0, 0.0, 1.0));
#else
    textureStore(dst_mip0, pix, vec4<f32>(value, 0.0, 0.0, 1.0));
#endif
}

fn store_mip1(pix: vec2<i32>, layer: u32, value: f32) {
    let dim = textureDimensions(dst_mip1);
    if (pix.x < 0 || pix.y < 0 || pix.x >= i32(dim.x) || pix.y >= i32(dim.y)) {
        return;
    }
#ifdef MULTIVIEW
    textureStore(dst_mip1, pix, i32(layer), vec4<f32>(value, 0.0, 0.0, 1.0));
#else
    textureStore(dst_mip1, pix, vec4<f32>(value, 0.0, 0.0, 1.0));
#endif
}

fn store_mip2(pix: vec2<i32>, layer: u32, value: f32) {
    let dim = textureDimensions(dst_mip2);
    if (pix.x < 0 || pix.y < 0 || pix.x >= i32(dim.x) || pix.y >= i32(dim.y)) {
        return;
    }
#ifdef MULTIVIEW
    textureStore(dst_mip2, pix, i32(layer), vec4<f32>(value, 0.0, 0.0, 1.0));
#else
    textureStore(dst_mip2, pix, vec4<f32>(value, 0.0, 0.0, 1.0));
#endif
}

fn store_mip3(pix: vec2<i32>, layer: u32, value: f32) {
    let dim = textureDimensions(dst_mip3);
    if (pix.x < 0 || pix.y < 0 || pix.x >= i32(dim.x) || pix.y >= i32(dim.y)) {
        return;
    }
#ifdef MULTIVIEW
    textureStore(dst_mip3, pix, i32(layer), vec4<f32>(value, 0.0, 0.0, 1.0));
#else
    textureStore(dst_mip3, pix, vec4<f32>(value, 0.0, 0.0, 1.0));
#endif
}

fn store_mip4(pix: vec2<i32>, layer: u32, value: f32) {
    let dim = textureDimensions(dst_mip4);
    if (pix.x < 0 || pix.y < 0 || pix.x >= i32(dim.x) || pix.y >= i32(dim.y)) {
        return;
    }
#ifdef MULTIVIEW
    textureStore(dst_mip4, pix, i32(layer), vec4<f32>(value, 0.0, 0.0, 1.0));
#else
    textureStore(dst_mip4, pix, vec4<f32>(value, 0.0, 0.0, 1.0));
#endif
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let mip0_dim = textureDimensions(dst_mip0);
    let mip0_max = vec2<i32>(i32(mip0_dim.x) - 1, i32(mip0_dim.y) - 1);
    let raw_dim = raw_dimensions();
    let layer = gid.z;
    let base_mip1 = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_mip0 = base_mip1 * 2;

    let d0 = load_mip0_depth(base_mip0 + vec2<i32>(0, 0), mip0_max, raw_dim, layer);
    let d1 = load_mip0_depth(base_mip0 + vec2<i32>(1, 0), mip0_max, raw_dim, layer);
    let d2 = load_mip0_depth(base_mip0 + vec2<i32>(0, 1), mip0_max, raw_dim, layer);
    let d3 = load_mip0_depth(base_mip0 + vec2<i32>(1, 1), mip0_max, raw_dim, layer);
    store_mip0(base_mip0 + vec2<i32>(0, 0), layer, d0);
    store_mip0(base_mip0 + vec2<i32>(1, 0), layer, d1);
    store_mip0(base_mip0 + vec2<i32>(0, 1), layer, d2);
    store_mip0(base_mip0 + vec2<i32>(1, 1), layer, d3);

    let depth_mip1 = depth_mip_filter(d0, d1, d2, d3);
    store_mip1(base_mip1, layer, depth_mip1);
    previous_mip_depth[lid.x][lid.y] = depth_mip1;

    workgroupBarrier();

    if all(lid.xy % vec2<u32>(2u) == vec2<u32>(0u)) {
        let depth0 = previous_mip_depth[lid.x + 0u][lid.y + 0u];
        let depth1 = previous_mip_depth[lid.x + 1u][lid.y + 0u];
        let depth2 = previous_mip_depth[lid.x + 0u][lid.y + 1u];
        let depth3 = previous_mip_depth[lid.x + 1u][lid.y + 1u];
        let depth_mip2 = depth_mip_filter(depth0, depth1, depth2, depth3);
        store_mip2(base_mip1 / 2, layer, depth_mip2);
        previous_mip_depth[lid.x][lid.y] = depth_mip2;
    }

    workgroupBarrier();

    if all(lid.xy % vec2<u32>(4u) == vec2<u32>(0u)) {
        let depth0 = previous_mip_depth[lid.x + 0u][lid.y + 0u];
        let depth1 = previous_mip_depth[lid.x + 2u][lid.y + 0u];
        let depth2 = previous_mip_depth[lid.x + 0u][lid.y + 2u];
        let depth3 = previous_mip_depth[lid.x + 2u][lid.y + 2u];
        let depth_mip3 = depth_mip_filter(depth0, depth1, depth2, depth3);
        store_mip3(base_mip1 / 4, layer, depth_mip3);
        previous_mip_depth[lid.x][lid.y] = depth_mip3;
    }

    workgroupBarrier();

    if all(lid.xy % vec2<u32>(8u) == vec2<u32>(0u)) {
        let depth0 = previous_mip_depth[lid.x + 0u][lid.y + 0u];
        let depth1 = previous_mip_depth[lid.x + 4u][lid.y + 0u];
        let depth2 = previous_mip_depth[lid.x + 0u][lid.y + 4u];
        let depth3 = previous_mip_depth[lid.x + 4u][lid.y + 4u];
        let depth_mip4 = depth_mip_filter(depth0, depth1, depth2, depth3);
        store_mip4(base_mip1 / 8, layer, depth_mip4);
    }
}
