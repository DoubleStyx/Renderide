//! Shared scene-depth intersection helpers for the PBSIntersect material family.

#define_import_path renderide::pbs::families::intersect

#import renderide::core::math as rmath
#import renderide::frame::scene_depth_sample as sds

const INTERSECTION_DEPTH_GRACE: f32 = 1e-4;

fn intersect_linear_factor(start_value: f32, end_value: f32, value: f32) -> f32 {
    let num = value - start_value;
    let denom = end_value - start_value;
    if (abs(denom) < INTERSECTION_DEPTH_GRACE) {
        return select(0.0, 1.0, num >= INTERSECTION_DEPTH_GRACE);
    }
    return rmath::saturate(num / denom);
}

fn intersection_lerp(
    frag_pos: vec4<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    begin_start: f32,
    begin_end: f32,
    end_start: f32,
    end_end: f32,
) -> f32 {
    let diff = sds::scene_linear_depth(frag_pos, view_layer) - sds::fragment_linear_depth(world_pos, view_layer);
    if (diff < min(begin_end, end_start) + INTERSECTION_DEPTH_GRACE) {
        return intersect_linear_factor(begin_start, begin_end, diff);
    }
    return 1.0 - intersect_linear_factor(end_start, end_end, diff);
}
