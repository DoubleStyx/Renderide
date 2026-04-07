// WGSL equivalent for third_party/Resonite.UnityShaders/Assets/Shaders/UI/UI_Unlit.shader
// Property-name parity target:
// _MainTex _Tint _OverlayTint _Cutoff _Rect _SrcBlend _DstBlend _ZWrite _Cull _MaskTex

#import renderide_uniform_ring
#import renderide_ui_common

struct NativeUiOverlayUnproject {
    inv_scene_proj: mat4x4f,
    inv_ui_proj: mat4x4f,
}

struct UiUnlitMaterialUniform {
    _Tint: vec4f,
    _OverlayTint: vec4f,
    _MainTex_ST: vec4f,
    _MaskTex_ST: vec4f,
    _Rect: vec4f,
    _Cutoff: f32,
    _Flags: u32,
    _Pad0: vec2u,
}

const FLAG_ALPHACLIP: u32 = 1u;
const FLAG_RECTCLIP: u32 = 2u;
const FLAG_OVERLAY: u32 = 4u;
const FLAG_TEXTURE_NORMALMAP: u32 = 8u;
const FLAG_TEXTURE_LERPCOLOR: u32 = 16u;
const FLAG_MASK_TEXTURE_MUL: u32 = 32u;
const FLAG_MASK_TEXTURE_CLIP: u32 = 64u;
const FLAG_HAS_MAIN_TEX: u32 = 128u;

@group(0) @binding(0) var<uniform> uniforms: array<renderide_uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var scene_depth: texture_depth_2d;
@group(1) @binding(1) var<uniform> overlay_unproject: NativeUiOverlayUnproject;
@group(2) @binding(0) var<uniform> material: UiUnlitMaterialUniform;
@group(2) @binding(1) var _MainTex: texture_2d<f32>;
@group(2) @binding(2) var _MainTex_sampler: sampler;
@group(2) @binding(3) var _MaskTex: texture_2d<f32>;
@group(2) @binding(4) var _MaskTex_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) color: vec4f,
    @location(3) lerp_color: vec4f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) lerp_color: vec4f,
    @location(3) local_xy: vec2f,
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = renderide_ui_common::apply_st(in.uv, material._MainTex_ST);
    out.color = in.color * material._Tint;
    out.lerp_color = in.lerp_color * material._Tint;
    out.local_xy = in.position.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    if ((material._Flags & FLAG_RECTCLIP) != 0u && !renderide_ui_common::rect_contains(in.local_xy, material._Rect)) {
        discard;
    }

    var color = vec4f(1.0, 1.0, 1.0, 1.0);
    if ((material._Flags & FLAG_HAS_MAIN_TEX) != 0u) {
        color = textureSample(_MainTex, _MainTex_sampler, in.uv);
    }
    if ((material._Flags & FLAG_TEXTURE_NORMALMAP) != 0u && (material._Flags & FLAG_HAS_MAIN_TEX) != 0u) {
        let n = color.xyz * 2.0 - 1.0;
        color = vec4f(n * 0.5 + 0.5, 1.0);
    }

    if ((material._Flags & FLAG_TEXTURE_LERPCOLOR) != 0u) {
        let l = (color.r + color.g + color.b) * 0.3333333333;
        let mixed = mix(in.color, in.lerp_color, l);
        color = vec4f(mixed.rgb, mixed.a * color.a);
    } else {
        color = color * in.color;
    }

    if ((material._Flags & FLAG_MASK_TEXTURE_MUL) != 0u || (material._Flags & FLAG_MASK_TEXTURE_CLIP) != 0u) {
        let mask_uv = renderide_ui_common::apply_st(in.uv, material._MaskTex_ST);
        let mask = textureSample(_MaskTex, _MaskTex_sampler, mask_uv);
        let mul = (mask.r + mask.g + mask.b) * 0.3333333 * mask.a;
        if ((material._Flags & FLAG_MASK_TEXTURE_MUL) != 0u) {
            color.a = color.a * mul;
        }
        if ((material._Flags & FLAG_MASK_TEXTURE_CLIP) != 0u && mul - material._Cutoff <= 0.0) {
            discard;
        }
    }

    if ((material._Flags & FLAG_ALPHACLIP) != 0u && (material._Flags & FLAG_MASK_TEXTURE_CLIP) == 0u) {
        if (color.a - material._Cutoff <= 0.0) {
            discard;
        }
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
            color = color * material._OverlayTint;
        }
    }

    return color;
}
