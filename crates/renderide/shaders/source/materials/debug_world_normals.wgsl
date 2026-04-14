//! Debug raster: world-space normals (RGB).
//!
//! Build emits two targets from this file via [`MULTIVIEW`](https://docs.rs/naga_oil) shader defs:
//! - `debug_world_normals_default.wgsl` — `MULTIVIEW` off (single-view desktop)
//! - `debug_world_normals_multiview.wgsl` — `MULTIVIEW` on (stereo `@builtin(view_index)`)
//!
//! [`PerDrawUniforms`] lives in [`renderide::per_draw`].

#import renderide::globals as rg
#import renderide::per_draw as pd

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_n: vec3<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = normalize((d.model * vec4<f32>(normal.xyz, 0.0)).xyz);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_n = world_n;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = in.world_n * 0.5 + 0.5;
    var lit: u32 = 0u;
    if (rg::frame.light_count > 0u) {
        lit = rg::lights[0].light_type;
    }
    let c = vec3<f32>(n) + rg::frame.camera_world_pos.xyz * 0.0001 + vec3<f32>(f32(lit) * 1e-10);
    let cluster_touch =
        f32(rg::cluster_light_counts[0u] & 255u) * 1e-10 + f32(rg::cluster_light_indices[0u] & 255u) * 1e-10;
    return vec4<f32>(c + vec3<f32>(cluster_touch), 1.0);
}
