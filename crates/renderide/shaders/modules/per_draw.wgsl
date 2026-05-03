//! Per-draw instance data (`@group(2)`) shared by mesh materials -- storage buffer indexed by
//! `@builtin(instance_index)`.
//! Import with `#import renderide::per_draw as pd` from `shaders/materials/*.wgsl` and use
//! `pd::get_draw(instance_index)` in `vs_main`. Do not redeclare `@group(2)` in material roots.
//!
//! CPU packing must match [`crate::backend::mesh_deform::PaddedPerDrawUniforms`].

#define_import_path renderide::per_draw

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    /// Inverse transpose of the upper 3x3 of `model` (correct normals under non-uniform scale).
    normal_matrix: mat3x3<f32>,
    /// Metadata plus reserved padding. `x` is non-zero when `@location(0)` positions are already world-space.
    _pad: vec4<f32>,
}

/// `_pad.x` marker for world-space position streams.
const POSITION_STREAM_WORLD_SPACE_FLAG: f32 = 1.0;

@group(2) @binding(0) var<storage, read> instances: array<PerDrawUniforms>;

fn get_draw(instance_idx: u32) -> PerDrawUniforms {
    return instances[instance_idx];
}

/// `true` when the bound position stream has already been transformed into world space.
fn position_stream_is_world_space(draw: PerDrawUniforms) -> bool {
    return draw._pad.x > 0.5 * POSITION_STREAM_WORLD_SPACE_FLAG;
}
