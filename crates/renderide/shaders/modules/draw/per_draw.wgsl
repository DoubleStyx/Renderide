//! Per-draw instance data (`@group(2)`) shared by mesh materials -- storage buffer indexed by
//! `@builtin(instance_index)`.
//! Import with `#import renderide::draw::per_draw as pd` from `shaders/materials/*.wgsl` and use
//! `pd::get_draw(instance_index)` in `vs_main`. Do not redeclare `@group(2)` in material roots.
//!
//! CPU packing must match [`crate::mesh_deform::PaddedPerDrawUniforms`].

#define_import_path renderide::draw::per_draw

#import renderide::draw::types as dt

@group(2) @binding(0) var<storage, read> instances: array<dt::PerDrawUniforms>;

fn get_draw(instance_idx: u32) -> dt::PerDrawUniforms {
    return instances[instance_idx];
}

fn local_reflection_probe_indices(draw: dt::PerDrawUniforms) -> vec4<u32> {
    return dt::local_reflection_probe_indices(draw);
}

fn fallback_reflection_probe_index(draw: dt::PerDrawUniforms) -> u32 {
    return dt::fallback_reflection_probe_index(draw);
}

fn reflection_probe_importance_mask(draw: dt::PerDrawUniforms) -> u32 {
    return dt::reflection_probe_importance_mask(draw);
}

fn has_reflection_probe_selection(draw: dt::PerDrawUniforms) -> bool {
    return dt::has_reflection_probe_selection(draw);
}

fn particle_kind(draw: dt::PerDrawUniforms) -> u32 {
    return dt::particle_kind(draw);
}

fn particle_alignment(draw: dt::PerDrawUniforms) -> u32 {
    return dt::particle_alignment(draw);
}

fn particle_min_screen_size(draw: dt::PerDrawUniforms) -> f32 {
    return dt::particle_min_screen_size(draw);
}

fn particle_max_screen_size(draw: dt::PerDrawUniforms) -> f32 {
    return dt::particle_max_screen_size(draw);
}

fn particle_color(draw: dt::PerDrawUniforms) -> vec4<f32> {
    return dt::particle_color(draw);
}
