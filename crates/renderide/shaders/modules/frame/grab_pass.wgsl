//! Helpers for sampling the scene-color snapshot used by grab-pass materials.
//!
//! `scene_color` / `scene_color_array` is a renderer-produced top-down texture copied from the
//! resolved color attachment. Screen UVs derived from `@builtin(position)` therefore sample it
//! directly without the Unity-authored texture V flip used by material textures.

#define_import_path renderide::frame::grab_pass

#import renderide::frame::globals as rg

fn frag_screen_uv(frag_pos: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(
        frag_pos.x / f32(rg::frame.viewport_width),
        frag_pos.y / f32(rg::frame.viewport_height),
    );
}

fn sample_scene_color(uv: vec2<f32>, view_layer: u32) -> vec4<f32> {
#ifdef MULTIVIEW
    return textureSampleLevel(
        rg::scene_color_array,
        rg::scene_color_sampler,
        uv,
        i32(rg::view_index_from_layer(view_layer)),
        0.0,
    );
#else
    return textureSampleLevel(rg::scene_color, rg::scene_color_sampler, uv, 0.0);
#endif
}
