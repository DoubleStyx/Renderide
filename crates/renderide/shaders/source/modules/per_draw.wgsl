//! Per-draw instance data (`@group(2)`) shared by mesh materials — storage buffer indexed by
//! `@builtin(instance_index)`.
//! Import with `#import renderide::per_draw as pd` from `source/materials/*.wgsl` and use
//! `pd::get_draw(instance_index)` in `vs_main`. Do not redeclare `@group(2)` in material roots.
//!
//! CPU packing must match [`crate::backend::mesh_deform::PaddedPerDrawUniforms`].

#define_import_path renderide::per_draw

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

@group(2) @binding(0) var<storage, read> instances: array<PerDrawUniforms>;

fn get_draw(instance_idx: u32) -> PerDrawUniforms {
    return instances[instance_idx];
}
