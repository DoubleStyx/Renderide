//! Helpers for material passes that must retain otherwise-unused bind-group entries.
//!
//! Some depth-only passes intentionally share the forward pass material bind group even when the
//! color path is the only pass that consumes every texture. These helpers keep the sampled texture
//! and sampler visible to reflection while contributing zero to the fragment output.

#define_import_path renderide::material::layout_retention

#import renderide::core::texture_sampling as ts

fn sample_2d_zero(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, lod_bias: f32) -> f32 {
    return ts::sample_tex_2d(tex, samp, uv, lod_bias).x * 0.0;
}
