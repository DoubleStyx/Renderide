//! Shared rect clipping predicates used by UI and screen-space filter materials.

#define_import_path renderide::ui::rect_clip

#import renderide::core::math as rmath

fn outside_rect_clip(p: vec2<f32>, rect: vec4<f32>) -> bool {
    return rmath::outside_rect(p, rect);
}

fn should_clip_rect_kw(p: vec2<f32>, rect: vec4<f32>, kw_enabled: bool) -> bool {
    return kw_enabled && rmath::rect_has_area(rect) && outside_rect_clip(p, rect);
}
