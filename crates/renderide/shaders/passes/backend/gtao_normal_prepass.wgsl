//! World-mesh normal prepass for GTAO.
//!
//! The pass renders only position + normal streams against the active forward depth tier with
//! depth compare `Equal`. MSAA views use the multisampled forward depth and resolve normals to the
//! single-sample GTAO texture. RGB stores a normalized view-space normal in GTAO's positive-depth
//! convention, and alpha marks pixels with a valid mesh normal so the GTAO shader can fall back to
//! depth-derived normals for sky, outlines, or geometry whose material pass changes depth.

#import renderide::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::per_draw as pd
#import renderide::view_basis as vb

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) view_n: vec3<f32>,
}

fn world_normal_for_stream(draw: pd::PerDrawUniforms, n: vec4<f32>) -> vec3<f32> {
    let transformed = rmath::safe_normalize(
        draw.normal_matrix * n.xyz,
        vec3<f32>(0.0, 1.0, 0.0),
    );
    let stream_world = rmath::safe_normalize(n.xyz, vec3<f32>(0.0, 1.0, 0.0));
    return select(transformed, stream_world, pd::position_stream_is_world_space(draw));
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
) -> VertexOutput {
    let draw = pd::get_draw(instance_index);
#ifdef MULTIVIEW
    let view = view_idx;
#else
    let view = 0u;
#endif
    let world_p = mv::world_position(draw, pos);
    let vp = mv::select_view_proj(draw, view);
    let world_n = world_normal_for_stream(draw, n);
    let view_n = vb::world_to_view_normal(world_n, vp);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.view_n = vec3<f32>(view_n.x, view_n.y, -view_n.z);
    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let oriented = select(-in.view_n, in.view_n, is_front);
    let n = rmath::safe_normalize(oriented, vec3<f32>(0.0, 0.0, -1.0));
    return vec4<f32>(n, 1.0);
}
