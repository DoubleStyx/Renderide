//! Per-draw dynamic uniform slot (`@group(2)`) shared by mesh materials.
//! Import with `#import renderide::per_draw` from `source/materials/*.wgsl`.
//!
//! CPU packing must match [`crate::gpu::PaddedPerDrawUniforms`].

#define_import_path renderide::per_draw

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

//! Bind `@group(2) @binding(0) var<uniform> draw: PerDrawUniforms;` in each material root.
