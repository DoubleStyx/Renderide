//! Per-draw instance data (`@group(2)`) shared by mesh materials -- storage buffer indexed by
//! `@builtin(instance_index)`.
//! Import with `#import renderide::draw::per_draw as pd` from `shaders/materials/*.wgsl` and use
//! `pd::get_draw(instance_index)` in `vs_main`. Do not redeclare `@group(2)` in material roots.
//!
//! CPU packing must match [`crate::backend::mesh_deform::PaddedPerDrawUniforms`].

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
