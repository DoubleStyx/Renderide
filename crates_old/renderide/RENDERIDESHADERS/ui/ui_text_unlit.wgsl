// WGSL equivalent for third_party/Resonite.UnityShaders/Assets/Shaders/UI/UI_TextUnlit.shader
// Property-name parity target:
// _FontAtlas _TintColor _OverlayTint _OutlineColor _BackgroundColor _Range _FaceDilate _FaceSoftness _OutlineSize _Rect

#import renderide_uniform_ring
#import renderide_ui_common

struct NativeUiOverlayUnproject {
    inv_scene_proj: mat4x4f,
    inv_ui_proj: mat4x4f,
}

struct UiTextUnlitMaterialUniform {
    _TintColor: vec4f,
    _OverlayTint: vec4f,
    _OutlineColor: vec4f,
    _BackgroundColor: vec4f,
    _Range: vec4f,
    _FaceDilate: f32,
    _FaceSoftness: f32,
    _OutlineSize: f32,
    _PadScalar: f32,
    _Rect: vec4f,
    _Flags: u32,
    _PadFlags: u32,
    _PadTail: vec2u,
}

const MODE_RASTER: u32 = 0u;
const MODE_SDF: u32 = 1u;
const MODE_MSDF: u32 = 2u;
const MODE_AUTO: u32 = 3u;
const FLAG_OUTLINE: u32 = 1u << 8u;
const FLAG_RECTCLIP: u32 = 1u << 9u;
const FLAG_OVERLAY: u32 = 1u << 10u;

@group(0) @binding(0) var<uniform> uniforms: array<renderide_uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var scene_depth: texture_depth_2d;
@group(1) @binding(1) var<uniform> overlay_unproject: NativeUiOverlayUnproject;
@group(2) @binding(0) var<uniform> material: UiTextUnlitMaterialUniform;
@group(2) @binding(1) var _FontAtlas: texture_2d<f32>;
@group(2) @binding(2) var _FontAtlas_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) color: vec4f,
    @location(3) extra_data: vec4f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) local_xy: vec2f,
    @location(3) extra_xy: vec2f,
}

fn median3(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

fn fwidth_uv(uv: vec2f) -> vec2f {
    return vec2f(abs(dpdx(uv.x)) + abs(dpdy(uv.x)), abs(dpdx(uv.y)) + abs(dpdy(uv.y)));
}

fn auto_text_mode(atlas: vec4f) -> u32 {
    let rgb_spread = max(abs(atlas.r - atlas.g), max(abs(atlas.g - atlas.b), abs(atlas.r - atlas.b)));
    let rgb_mean = (atlas.r + atlas.g + atlas.b) * 0.3333333333;
    let alpha_delta = abs(atlas.a - rgb_mean);
    if (rgb_spread > 0.02) {
        return MODE_MSDF;
    }
    if (alpha_delta > 0.02) {
        return MODE_SDF;
    }
    return MODE_RASTER;
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.local_xy = in.position.xy;
    out.extra_xy = in.extra_data.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    if ((material._Flags & FLAG_RECTCLIP) != 0u && !renderide_ui_common::rect_contains(in.local_xy, material._Rect)) {
        discard;
    }

    let atlas = textureSample(_FontAtlas, _FontAtlas_sampler, in.uv);
    var mode = material._Flags & 3u;
    if (mode == MODE_AUTO) {
        mode = auto_text_mode(atlas);
    }
    var c: vec4f;

    if (mode == MODE_RASTER) {
        c = atlas * in.color;
        if (c.a <= 0.001) {
            discard;
        }
    } else {
        var sig_dist = atlas.a - 0.5;
        if (mode == MODE_MSDF) {
            sig_dist = median3(atlas.r, atlas.g, atlas.b) - 0.5;
        }
        sig_dist = sig_dist + material._FaceDilate + in.extra_xy.x;

        let fw = fwidth_uv(in.uv);
        let anti_aliasing = max(dot(material._Range.xy, vec2f(0.5 / max(fw.x, 1e-8), 0.5 / max(fw.y, 1e-8))), 1.0);
        var glyph_lerp = mix(sig_dist * anti_aliasing, sig_dist, material._FaceSoftness);
        glyph_lerp = clamp(glyph_lerp + 0.5, 0.0, 1.0);

        if (max(glyph_lerp, material._BackgroundColor.a) <= 0.001) {
            discard;
        }

        var fill_color = material._TintColor * in.color;
        if ((material._Flags & FLAG_OUTLINE) != 0u) {
            let outline_dist = sig_dist - (material._OutlineSize + in.extra_xy.y);
            var outline_lerp = mix(outline_dist * anti_aliasing, outline_dist, material._FaceSoftness);
            outline_lerp = clamp(outline_lerp + 0.5, 0.0, 1.0);
            let outline_src = vec4f(material._OutlineColor.rgb, material._OutlineColor.a * in.color.a);
            fill_color = mix(outline_src, fill_color, outline_lerp);
        }
        c = mix(material._BackgroundColor * in.color, fill_color, glyph_lerp);
    }

    if ((material._Flags & FLAG_OVERLAY) != 0u) {
        let dims = textureDimensions(scene_depth);
        let px = vec2i(i32(in.clip_position.x), i32(in.clip_position.y));
        let sx = clamp(px.x, 0, i32(dims.x) - 1);
        let sy = clamp(px.y, 0, i32(dims.y) - 1);
        let scene_d = textureLoad(scene_depth, vec2i(sx, sy), 0);
        let ndc_xy = renderide_ui_common::overlay_ndc_xy(in.clip_position.xy, dims);
        let scene_vz = renderide_ui_common::overlay_view_z_from_depth(scene_d, ndc_xy, overlay_unproject.inv_scene_proj);
        let part_vz = renderide_ui_common::overlay_view_z_from_depth(in.clip_position.z, ndc_xy, overlay_unproject.inv_ui_proj);
        if (-part_vz > -scene_vz) {
            c = c * material._OverlayTint;
        }
    }

    return c;
}
