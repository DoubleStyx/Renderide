//! Shared mesh transform helpers for material roots.

#define_import_path renderide::mesh::transform

#import renderide::draw::types as dt
#import renderide::core::math as rmath

fn select_view_proj(draw: dt::PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
    return dt::select_view_proj(draw, view_idx);
}

fn world_position(draw: dt::PerDrawUniforms, pos: vec4<f32>) -> vec4<f32> {
    return draw.model * vec4<f32>(pos.xyz, 1.0);
}

fn world_normal(draw: dt::PerDrawUniforms, n: vec4<f32>) -> vec3<f32> {
    return rmath::safe_normalize(draw.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));
}

fn model_vector(draw: dt::PerDrawUniforms, v: vec3<f32>) -> vec3<f32> {
    return (draw.model * vec4<f32>(v, 0.0)).xyz;
}

fn model_handedness(draw: dt::PerDrawUniforms) -> f32 {
    if (dt::position_stream_is_world_space(draw)) {
        return 1.0;
    }
    let det = dot(draw.model[0].xyz, cross(draw.model[1].xyz, draw.model[2].xyz));
    return select(1.0, -1.0, det < 0.0);
}

fn tangent_w_sign(tangent_w: f32) -> f32 {
    return select(1.0, -1.0, tangent_w < 0.0);
}

fn world_tangent(draw: dt::PerDrawUniforms, t: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(normalize(model_vector(draw, t.xyz)), tangent_w_sign(t.w) * model_handedness(draw));
}

fn packed_view_layer(instance_index: u32, view_idx: u32) -> u32 {
    return (instance_index << 1u) | (view_idx & 1u);
}

fn ndc_xy(clip: vec4<f32>) -> vec2<f32> {
    return clip.xy / max(abs(clip.w), 1e-6);
}
