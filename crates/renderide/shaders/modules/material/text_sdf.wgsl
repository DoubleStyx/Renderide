//! Shared helpers for MSDF/SDF/raster text font-atlas shading.
//!
//! Import with `#import renderide::material::text_sdf as tsdf`.

#define_import_path renderide::material::text_sdf

struct DistanceFieldStyle {
    tint_color: vec4<f32>,
    outline_color: vec4<f32>,
    background_color: vec4<f32>,
    range: vec4<f32>,
    face_dilate: f32,
    face_softness: f32,
    outline_size: f32,
}

struct DistanceFieldInput {
    sig_dist: f32,
    uv: vec2<f32>,
    extra_data: vec4<f32>,
    vertex_color: vec4<f32>,
}

/// Median of three scalars (MSDF channel combine).
fn median3(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

/// Derives the text mode int (`0`=MSDF, `1`=RASTER, `2`=SDF) from decoded variant bits.
///
/// Defaults to MSDF when neither RASTER nor SDF is set, matching the material keyword default
/// after Froox's variant resolution.
fn text_mode_from_keywords(raster: bool, sdf: bool) -> i32 {
    if (raster) {
        return 1;
    }
    if (sdf) {
        return 2;
    }
    return 0;
}

fn distance_field_style(
    tint_color: vec4<f32>,
    outline_color: vec4<f32>,
    background_color: vec4<f32>,
    range: vec4<f32>,
    face_dilate: f32,
    face_softness: f32,
    outline_size: f32,
) -> DistanceFieldStyle {
    return DistanceFieldStyle(
        tint_color,
        outline_color,
        background_color,
        range,
        face_dilate,
        face_softness,
        outline_size,
    );
}

fn shade_distance_field(
    style: DistanceFieldStyle,
    input: DistanceFieldInput,
    outline_enabled: bool,
) -> vec4<f32> {
    var sig_dist = input.sig_dist + style.face_dilate + input.extra_data.x;

    let fw = vec2<f32>(fwidth(input.uv.x), fwidth(input.uv.y));
    let anti_aliasing = dot(style.range.xy, vec2<f32>(0.5) / max(fw, vec2<f32>(1e-6)));
    let aa = max(anti_aliasing, 1.0);

    var glyph_lerp = mix(sig_dist * aa, sig_dist, style.face_softness);
    glyph_lerp = clamp(glyph_lerp + 0.5, 0.0, 1.0);

    if (max(glyph_lerp, style.background_color.a) < 0.001) {
        discard;
    }

    var fill_color = style.tint_color * input.vertex_color;
    let outline_w = style.outline_size + input.extra_data.y;
    if (outline_enabled && style.outline_size > 1e-6) {
        let outline_dist = sig_dist - outline_w;
        var outline_lerp = mix(outline_dist * aa, outline_dist, style.face_softness);
        outline_lerp = clamp(outline_lerp + 0.5, 0.0, 1.0);
        fill_color = mix(
            style.outline_color * vec4<f32>(1.0, 1.0, 1.0, input.vertex_color.a),
            fill_color,
            outline_lerp,
        );
    }

    return mix(style.background_color * input.vertex_color, fill_color, glyph_lerp);
}

fn shade_text_sample(
    atlas_color: vec4<f32>,
    style: DistanceFieldStyle,
    input: DistanceFieldInput,
    raster_tint: vec4<f32>,
    mode: i32,
    outline_enabled: bool,
) -> vec4<f32> {
    if (mode == 1) {
        let c = atlas_color * raster_tint;
        if (atlas_color.a * raster_tint.a < 0.001) {
            discard;
        }
        return c;
    }

    if (mode == 2) {
        let sdf_input = DistanceFieldInput(
            atlas_color.a - 0.5,
            input.uv,
            input.extra_data,
            input.vertex_color,
        );
        return shade_distance_field(style, sdf_input, outline_enabled);
    }

    let sdf_input = DistanceFieldInput(
        median3(atlas_color.r, atlas_color.g, atlas_color.b) - 0.5,
        input.uv,
        input.extra_data,
        input.vertex_color,
    );
    return shade_distance_field(style, sdf_input, outline_enabled);
}
