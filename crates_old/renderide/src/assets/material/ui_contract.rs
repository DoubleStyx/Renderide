//! Host contract for native WGSL UI materials (`UI_Unlit`, `UI_TextUnlit`).
//!
//! For Resonite world `Shader "Unlit"` (mesh [`VertexPosNormalUv`](crate::gpu::mesh::VertexPosNormalUv)),
//! see [`super::world_unlit_material_contract`].
//!
//! Shader asset IDs are assigned by the host at runtime; [`crate::assets::AssetRegistry::handle_shader_upload`]
//! resolves [`crate::assets::ShaderAsset::program`] and names so [`resolve_native_ui_shader_family`]
//! can map `set_shader` batches to [`NativeUiShaderFamily`].
//!
//! Logical Unity names (`Shader "UI/Unlit"`), plain stems in [`crate::shared::ShaderUpload::file`]
//! (e.g. `UI_Unlit` from a host mod), ShaderLab/WGSL payloads, and path hints are resolved in
//! [`super::shader_logical_name`] — see also [`crate::shared::shader_upload_extras`].
//!
//! ## Material property IDs
//!
//! Batches use numeric `property_id` values. When a field in [`UiUnlitPropertyIds`] or
//! [`UiTextUnlitPropertyIds`] is `-1`, that channel is skipped and the GPU uniform uses a default.
//!
//! **Automatic mapping:** On [`MaterialPropertyIdRequest`](crate::shared::MaterialPropertyIdRequest),
//! [`crate::assets::material_property_host`] interns each property name to an integer and replies with
//! [`MaterialPropertyIdResult`](crate::shared::MaterialPropertyIdResult). Those integers are **not**
//! Unity `Shader.PropertyToID` values; the host must use the renderer’s returned ids in batches (see
//! module docs on [`material_property_host`](crate::assets::material_property_host)). Matching
//! FrooxEngine shader property names also update [`crate::config::RenderConfig`] when
//! [`RenderConfig::use_native_ui_wgsl`](crate::config::RenderConfig::use_native_ui_wgsl) is true.
//!
//! **Manual mapping:** INI sections `[native_ui_unlit_properties]` and
//! `[native_ui_text_unlit_properties]` (see [`crate::config::RenderConfig`]) set the same fields.
//!
//! ### `UI_UnlitMaterial` (FrooxEngine) → INI key under `[native_ui_unlit_properties]`
//!
//! - `_MainTex` → `main_tex` (`set_texture` packed id → GPU bind)
//! - `_MaskTex` → `mask_tex`
//! - `_MainTex_ST` → `main_tex_st`, `_MaskTex_ST` → `mask_tex_st`
//! - `_Tint` → `tint`, `_OverlayTint` → `overlay_tint`, `_Cutoff` → `cutoff`, `_Rect` → `rect`
//! - Keyword floats (when the host assigns property ids): `alphaclip`, `rectclip`, `overlay`,
//!   `texture_normalmap`, `texture_lerpcolor`, `mask_texture_mul`, `mask_texture_clip`
//! - `_SrcBlend` / `_DstBlend` → `src_blend` / `dst_blend` (Unity `BlendMode` enum values as floats;
//!   mapped to [`crate::assets::native_ui_blend::NativeUiSurfaceBlend`] alpha / premultiplied / additive;
//!   unmapped pairs fall back to [`crate::config::RenderConfig::native_ui_default_surface_blend`] and log)
//!
//! **FrooxEngine shader keywords** (from `UpdateKeywords`, not only underscored property names) are
//! mirrored into the same INI fields when the host requests ids for those names:
//! `ALPHACLIP`, `RECTCLIP`, `OVERLAY`, `TEXTURE_NORMALMAP`, `TEXTURE_LERPCOLOR`,
//! `_MASK_TEXTURE_MUL`, `_MASK_TEXTURE_CLIP`, and for text `RASTER`, `SDF`, `MSDF`, `OUTLINE`.
//!
//! ### `TextUnlitMaterial` / `UI_TextUnlitMaterial` → `[native_ui_text_unlit_properties]`
//!
//! - `_FontAtlas` → `font_atlas`
//! - `_TintColor` → `tint_color`, `_OutlineColor` → `outline_color`, `_BackgroundColor` → `background_color`
//! - `_Range` → `range`, `_FaceDilate` → `face_dilate`, `_FaceSoftness` → `face_softness`,
//!   `_OutlineSize` → `outline_size`
//! - `_OverlayTint` → `overlay_tint`, `_Rect` → `rect`
//! - `src_blend` / `dst_blend` for `_SrcBlend` / `_DstBlend`
//! - Mode / keyword floats when present: `raster`, `sdf`, `msdf`, `outline`, `rectclip`, `overlay`
//!
//! ### Vertex stream (text)
//!
//! Resonite’s `UI_TextUnlit` shader packs glyph extras in the **NORMAL** slot (`extraData`); the mesh
//! path copies that into [`VertexUiCanvas`](crate::gpu::mesh::VertexUiCanvas) `aux` for WGSL.
//!
//! ## Render/state properties not mapped into native UI WGSL
//!
//! `UI_Unlit.shader` also exposes `_ZWrite`, `_ZTest`, `_Cull`, `_ColorMask`, stencil comparison/ref/mask,
//! and depth offset (`_Offset*`). Those drive Unity render state, not the data-driven uniform path: native
//! pipelines use fixed depth/stencil/rasterizer states from WGSL pipeline construction.
//!
//! ### Future parity (fixed-function)
//!
//! To approach Unity per-material behavior, new work would add either:
//! - additional [`crate::gpu::PipelineVariant`] branches (e.g. depth write on/off, cull mode) with matching
//!   [`wgpu::RenderPipelineDescriptor`] entries, or
//! - dynamic depth bias / stencil reference where the API exposes state compatible with the pass.
//!
//! Until then, overlay and world-space native UI use the states encoded in
//! [`crate::gpu::pipeline::ui_unlit_native::UiUnlitNativePipeline`].
//!
//! ## IPC parity: batch opcodes vs [`crate::assets::material_properties::MaterialPropertyStore`]
//!
//! [`crate::assets::material_update_batch`] applies FrooxEngine `MaterialsUpdateBatch` records. Unless
//! [`crate::config::RenderConfig::material_batch_persist_extended_payloads`] is enabled, `set_float4x4`
//! and float / float4 array payloads are **not** stored (matrices are still consumed from the wire cursor;
//! arrays are advanced only). Use [`crate::config::RenderConfig::material_batch_wire_metrics`] to count
//! those opcodes in the debug HUD. Generic PBR host-factor binding (`_Color`, `_Metallic`, `_Glossiness`)
//! and optional `_MainTex` → [`crate::gpu::PipelineVariant::PbrHostAlbedo`] are documented on
//! [`crate::assets::material_properties`] and [`crate::config::RenderConfig`]; full Standard stacks
//! (`_BumpMap`, `_EmissionMap`, …) are not wired into the generic PBR WGSL path yet.

use crate::assets::util::compact_alnum_lower;

/// Default `_OverlayTint` when the host omits the property (RGBA; alpha scales overlay contribution).
pub const DEFAULT_OVERLAY_TINT: [f32; 4] = [1.0, 1.0, 1.0, 0.5];

/// Identifies which native UI WGSL program to use for a host shader asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeUiShaderFamily {
    /// Resonite `UI/Unlit` ([`third_party/Resonite.UnityShaders/.../UI_Unlit.shader`](../../../../third_party/Resonite.UnityShaders/Assets/Shaders/UI/UI_Unlit.shader)).
    UiUnlit,
    /// Resonite `UI/Text/Unlit`.
    UiTextUnlit,
}

/// Infers [`NativeUiShaderFamily`] from the host shader upload string (`ShaderUpload.file`: path or label).
///
/// Matches path fragments such as `UI/Unlit`, `UI_Unlit`, `UI/Text/Unlit` (text is checked before unlit).
pub fn native_ui_family_from_shader_path_hint(hint: &str) -> Option<NativeUiShaderFamily> {
    let h = hint.to_ascii_lowercase();
    if h.contains("ui/text") && (h.contains("unlit") || h.contains("textunlit")) {
        return Some(NativeUiShaderFamily::UiTextUnlit);
    }
    if h.contains("ui/unlit") || h.contains("ui_unlit") || h.contains("uiunlit") {
        return Some(NativeUiShaderFamily::UiUnlit);
    }
    None
}

/// Maps a logical shader name or stem (first whitespace-delimited token) to [`NativeUiShaderFamily`]
/// when it matches Resonite `UI/Unlit` or `UI/Text/Unlit` (including file stems such as `UI_Unlit` / `UI_TextUnlit`).
pub fn native_ui_family_from_shader_label(label: &str) -> Option<NativeUiShaderFamily> {
    let token = label.split_whitespace().next()?;
    if token.is_empty() {
        return None;
    }
    let key = compact_alnum_lower(token);
    let k_unlit = compact_alnum_lower(super::shader_logical_name::CANONICAL_UNITY_UI_UNLIT);
    let k_text = compact_alnum_lower(super::shader_logical_name::CANONICAL_UNITY_UI_TEXT_UNLIT);
    match key.as_str() {
        k if k == k_unlit.as_str() => Some(NativeUiShaderFamily::UiUnlit),
        k if k == k_text.as_str() => Some(NativeUiShaderFamily::UiTextUnlit),
        _ => None,
    }
}

/// Maps Unity ShaderLab `Shader "…"` strings (and common aliases) to [`NativeUiShaderFamily`].
///
/// Uses [`native_ui_family_from_shader_label`] first, then [`native_ui_family_from_shader_path_hint`]
/// for legacy bundle paths and other substring matches.
pub fn native_ui_family_from_unity_shader_name(name: &str) -> Option<NativeUiShaderFamily> {
    native_ui_family_from_shader_label(name)
        .or_else(|| native_ui_family_from_shader_path_hint(name))
}

/// Resolves native UI shader family from stored logical shader identity.
pub fn resolve_native_ui_shader_family(
    shader_asset_id: i32,
    registry: &super::AssetRegistry,
) -> Option<NativeUiShaderFamily> {
    if let Some(s) = registry.get_shader(shader_asset_id) {
        match s.program {
            super::EssentialShaderProgram::UiUnlit => return Some(NativeUiShaderFamily::UiUnlit),
            super::EssentialShaderProgram::UiTextUnlit => {
                return Some(NativeUiShaderFamily::UiTextUnlit);
            }
            _ => {}
        }
        if let Some(f) = s
            .unity_shader_name
            .as_deref()
            .and_then(native_ui_family_from_unity_shader_name)
        {
            return Some(f);
        }
        if let Some(f) = s
            .wgsl_source
            .as_deref()
            .and_then(native_ui_family_from_unity_shader_name)
        {
            return Some(f);
        }
    }
    None
}

/// Property id map for `UI_Unlit` material batches. `-1` = omit (use GPU default).
#[derive(Clone, Debug)]
pub struct UiUnlitPropertyIds {
    /// `_Tint` (float4 linear color).
    pub tint: i32,
    /// `_OverlayTint` (float4).
    pub overlay_tint: i32,
    /// `_Cutoff` (float).
    pub cutoff: i32,
    /// `_Rect` min/max extents (float4), used with rect clip flag.
    pub rect: i32,
    /// `_MainTex_ST` scale.xy offset.zw (float4) or two floats — we expect float4 from host.
    pub main_tex_st: i32,
    /// `_MaskTex_ST` (float4).
    pub mask_tex_st: i32,
    /// `_MainTex` texture (`set_texture` packed id).
    pub main_tex: i32,
    /// `_MaskTex` texture.
    pub mask_tex: i32,
    /// Keyword-style flags sent as floats (0/1) when the host uses dedicated property ids.
    pub alphaclip: i32,
    pub rectclip: i32,
    pub overlay: i32,
    pub texture_normalmap: i32,
    pub texture_lerpcolor: i32,
    pub mask_texture_mul: i32,
    pub mask_texture_clip: i32,
    /// `_SrcBlend` (float; Unity blend mode enum value).
    pub src_blend: i32,
    /// `_DstBlend` (float).
    pub dst_blend: i32,
    /// `_ZWrite` (float render-state property).
    pub zwrite: i32,
    /// `_Cull` (float render-state property).
    pub cull: i32,
    /// `_ZTest` (float render-state property).
    pub ztest: i32,
    /// `_ColorMask` (float render-state property).
    pub color_mask: i32,
    /// `_StencilComp` (float render-state property).
    pub stencil_comp: i32,
    /// `_Stencil` (float render-state property).
    pub stencil_ref: i32,
    /// `_StencilOp` (float render-state property).
    pub stencil_op: i32,
    /// `_StencilWriteMask` (float render-state property).
    pub stencil_write_mask: i32,
    /// `_StencilReadMask` (float render-state property).
    pub stencil_read_mask: i32,
    /// `_OffsetFactor` (float render-state property).
    pub offset_factor: i32,
    /// `_OffsetUnits` (float render-state property).
    pub offset_units: i32,
}

impl Default for UiUnlitPropertyIds {
    fn default() -> Self {
        Self {
            tint: -1,
            overlay_tint: -1,
            cutoff: -1,
            rect: -1,
            main_tex_st: -1,
            mask_tex_st: -1,
            main_tex: -1,
            mask_tex: -1,
            alphaclip: -1,
            rectclip: -1,
            overlay: -1,
            texture_normalmap: -1,
            texture_lerpcolor: -1,
            mask_texture_mul: -1,
            mask_texture_clip: -1,
            src_blend: -1,
            dst_blend: -1,
            zwrite: -1,
            cull: -1,
            ztest: -1,
            color_mask: -1,
            stencil_comp: -1,
            stencil_ref: -1,
            stencil_op: -1,
            stencil_write_mask: -1,
            stencil_read_mask: -1,
            offset_factor: -1,
            offset_units: -1,
        }
    }
}

/// Property id map for `UI_TextUnlit`.
#[derive(Clone, Debug)]
pub struct UiTextUnlitPropertyIds {
    pub tint_color: i32,
    pub overlay_tint: i32,
    pub outline_color: i32,
    pub background_color: i32,
    pub range: i32,
    pub face_dilate: i32,
    pub face_softness: i32,
    pub outline_size: i32,
    pub rect: i32,
    pub font_atlas: i32,
    pub raster: i32,
    pub sdf: i32,
    pub msdf: i32,
    pub outline: i32,
    pub rectclip: i32,
    pub overlay: i32,
    /// `_SrcBlend` (float).
    pub src_blend: i32,
    /// `_DstBlend` (float).
    pub dst_blend: i32,
}

impl Default for UiTextUnlitPropertyIds {
    fn default() -> Self {
        Self {
            tint_color: -1,
            overlay_tint: -1,
            outline_color: -1,
            background_color: -1,
            range: -1,
            face_dilate: -1,
            face_softness: -1,
            outline_size: -1,
            rect: -1,
            font_atlas: -1,
            raster: -1,
            sdf: -1,
            msdf: -1,
            outline: -1,
            rectclip: -1,
            overlay: -1,
            src_blend: -1,
            dst_blend: -1,
        }
    }
}

/// GPU-packed flags for `UI_Unlit` (single u32 in uniform block).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UiUnlitFlags {
    pub alphaclip: bool,
    pub rectclip: bool,
    pub overlay: bool,
    pub has_main_tex: bool,
    pub texture_normalmap: bool,
    pub texture_lerpcolor: bool,
    pub mask_texture_mul: bool,
    pub mask_texture_clip: bool,
}

impl UiUnlitFlags {
    /// Packs flags into a little-endian bitfield for WGSL `u32`.
    pub fn to_bits(self) -> u32 {
        let mut b = 0u32;
        if self.alphaclip {
            b |= 1;
        }
        if self.rectclip {
            b |= 2;
        }
        if self.overlay {
            b |= 4;
        }
        if self.has_main_tex {
            b |= 128;
        }
        if self.texture_normalmap {
            b |= 8;
        }
        if self.texture_lerpcolor {
            b |= 16;
        }
        if self.mask_texture_mul {
            b |= 32;
        }
        if self.mask_texture_clip {
            b |= 64;
        }
        b
    }
}

/// CPU-side uniform data for `UI_Unlit` before upload (matches WGSL `UiUnlitMaterialUniform`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiUnlitMaterialUniform {
    pub tint: [f32; 4],
    pub overlay_tint: [f32; 4],
    pub main_tex_st: [f32; 4],
    pub mask_tex_st: [f32; 4],
    pub rect: [f32; 4],
    pub cutoff: f32,
    pub flags: u32,
    pub pad_tail: [u32; 2],
}

/// CPU-side uniform for `UI_TextUnlit`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiTextUnlitMaterialUniform {
    pub tint_color: [f32; 4],
    pub overlay_tint: [f32; 4],
    pub outline_color: [f32; 4],
    pub background_color: [f32; 4],
    pub range_xy: [f32; 4],
    pub face_dilate: f32,
    pub face_softness: f32,
    pub outline_size: f32,
    pub pad_scalar: f32,
    pub rect: [f32; 4],
    /// Lower bits: mode 0=raster,1=sdf,2=msdf; bit 8 outline; bit 9 rectclip; bit 10 overlay.
    pub flags: u32,
    pub pad_flags: u32,
    pub pad_tail: [u32; 2],
}

/// Reads a float4 property via merged material / property-block lookup, or `default` when unset.
fn float4(
    store: &super::MaterialPropertyStore,
    lookup: super::MaterialPropertyLookupIds,
    pid: i32,
    default: [f32; 4],
) -> [f32; 4] {
    if pid < 0 {
        return default;
    }
    match store.get_merged(lookup, pid) {
        Some(super::MaterialPropertyValue::Float4(v)) => *v,
        _ => default,
    }
}

/// Reads either a float4 or a scalar-expanded float4. Useful for host properties that may be sent
/// as a single float in one implementation and a vec4 in another.
fn float4_or_scalar(
    store: &super::MaterialPropertyStore,
    lookup: super::MaterialPropertyLookupIds,
    pid: i32,
    scalar_to_vec4: impl FnOnce(f32) -> [f32; 4],
    default: [f32; 4],
) -> [f32; 4] {
    if pid < 0 {
        return default;
    }
    match store.get_merged(lookup, pid) {
        Some(super::MaterialPropertyValue::Float4(v)) => *v,
        Some(super::MaterialPropertyValue::Float(v)) => scalar_to_vec4(*v),
        _ => default,
    }
}

/// Reads a float property via merged lookup, or `default` when unset.
fn float1(
    store: &super::MaterialPropertyStore,
    lookup: super::MaterialPropertyLookupIds,
    pid: i32,
    default: f32,
) -> f32 {
    if pid < 0 {
        return default;
    }
    match store.get_merged(lookup, pid) {
        Some(super::MaterialPropertyValue::Float(v)) => *v,
        _ => default,
    }
}

/// True when merged lookup has a float keyword property ≥ 0.5.
fn flag_f(
    store: &super::MaterialPropertyStore,
    lookup: super::MaterialPropertyLookupIds,
    pid: i32,
) -> bool {
    if pid < 0 {
        return false;
    }
    matches!(
        store.get_merged(lookup, pid),
        Some(super::MaterialPropertyValue::Float(f)) if *f >= 0.5
    )
}

/// Builds GPU uniform and texture handles for `UI_Unlit` from merged material + mesh property-block lookup.
pub fn ui_unlit_material_uniform(
    store: &super::MaterialPropertyStore,
    lookup: super::MaterialPropertyLookupIds,
    ids: &UiUnlitPropertyIds,
) -> (UiUnlitMaterialUniform, i32, i32) {
    let tint = float4(store, lookup, ids.tint, [1.0, 1.0, 1.0, 1.0]);
    let overlay_tint = float4(store, lookup, ids.overlay_tint, DEFAULT_OVERLAY_TINT);
    let cutoff = float1(store, lookup, ids.cutoff, 0.98);
    let main_tex_st = float4(store, lookup, ids.main_tex_st, [1.0, 1.0, 0.0, 0.0]);
    let mask_tex_st = float4(store, lookup, ids.mask_tex_st, [1.0, 1.0, 0.0, 0.0]);
    let rect = float4(store, lookup, ids.rect, [0.0, 0.0, 1.0, 1.0]);
    let main_tex = texture_handle(store, lookup, ids.main_tex);
    let mask_tex = texture_handle(store, lookup, ids.mask_tex);
    let flags = UiUnlitFlags {
        alphaclip: flag_f(store, lookup, ids.alphaclip),
        rectclip: flag_f(store, lookup, ids.rectclip),
        overlay: flag_f(store, lookup, ids.overlay),
        has_main_tex: main_tex != 0,
        texture_normalmap: main_tex != 0 && flag_f(store, lookup, ids.texture_normalmap),
        texture_lerpcolor: main_tex != 0 && flag_f(store, lookup, ids.texture_lerpcolor),
        mask_texture_mul: mask_tex != 0 && flag_f(store, lookup, ids.mask_texture_mul),
        mask_texture_clip: mask_tex != 0 && flag_f(store, lookup, ids.mask_texture_clip),
    };
    let u = UiUnlitMaterialUniform {
        tint,
        overlay_tint,
        main_tex_st,
        mask_tex_st,
        rect,
        cutoff,
        flags: flags.to_bits(),
        pad_tail: [0; 2],
    };
    (u, main_tex, mask_tex)
}

/// Packed host texture id from merged lookup, or `0` when unset.
fn texture_handle(
    store: &super::MaterialPropertyStore,
    lookup: super::MaterialPropertyLookupIds,
    pid: i32,
) -> i32 {
    if pid < 0 {
        return 0;
    }
    match store.get_merged(lookup, pid) {
        Some(super::MaterialPropertyValue::Texture(h)) if *h > 0 => *h,
        _ => 0,
    }
}

/// Builds uniform and font atlas handle for `UI_TextUnlit` from merged property lookup.
pub fn ui_text_unlit_material_uniform(
    store: &super::MaterialPropertyStore,
    lookup: super::MaterialPropertyLookupIds,
    ids: &UiTextUnlitPropertyIds,
) -> (UiTextUnlitMaterialUniform, i32) {
    let tint_color = float4(store, lookup, ids.tint_color, [1.0, 1.0, 1.0, 1.0]);
    let overlay_tint = float4(store, lookup, ids.overlay_tint, DEFAULT_OVERLAY_TINT);
    let outline_color = float4(store, lookup, ids.outline_color, [1.0, 1.0, 1.0, 0.0]);
    let background_color = float4(store, lookup, ids.background_color, [0.0, 0.0, 0.0, 0.0]);
    let range_v = float4_or_scalar(
        store,
        lookup,
        ids.range,
        |v| [v, v, 0.0, 0.0],
        [0.001, 0.001, 0.0, 0.0],
    );
    let face_dilate = float1(store, lookup, ids.face_dilate, 0.0);
    let face_softness = float1(store, lookup, ids.face_softness, 0.0);
    let outline_size = float1(store, lookup, ids.outline_size, 0.0);
    let rect = float4(store, lookup, ids.rect, [0.0, 0.0, 1.0, 1.0]);
    let raster = flag_f(store, lookup, ids.raster);
    let sdf = flag_f(store, lookup, ids.sdf);
    let msdf = flag_f(store, lookup, ids.msdf);
    let mode: u32 = if raster {
        0
    } else if msdf {
        2
    } else if sdf {
        1
    } else {
        // When the host did not expose explicit text-mode keywords, defer to shader-side atlas
        // inspection so MSDF nametag atlases still render correctly.
        3
    };
    let mut flags = mode & 3;
    if flag_f(store, lookup, ids.outline) {
        flags |= 1 << 8;
    }
    if flag_f(store, lookup, ids.rectclip) {
        flags |= 1 << 9;
    }
    if flag_f(store, lookup, ids.overlay) {
        flags |= 1 << 10;
    }
    let font_atlas = texture_handle(store, lookup, ids.font_atlas);
    let u = UiTextUnlitMaterialUniform {
        tint_color,
        overlay_tint,
        outline_color,
        background_color,
        range_xy: [range_v[0], range_v[1], range_v[2], range_v[3]],
        face_dilate,
        face_softness,
        outline_size,
        pad_scalar: 0.0,
        rect,
        flags,
        pad_flags: 0,
        pad_tail: [0; 2],
    };
    (u, font_atlas)
}

#[cfg(test)]
mod tests {
    use crate::assets::material_properties::{
        MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
    };
    use crate::assets::shader_logical_name::{
        CANONICAL_UNITY_UI_TEXT_UNLIT, CANONICAL_UNITY_UI_UNLIT,
    };

    use crate::assets::AssetRegistry;
    use crate::assets::texture2d_asset_id_from_packed;
    use crate::config::RenderConfig;
    use crate::shared::ShaderUpload;

    use super::{
        NativeUiShaderFamily, UiUnlitFlags, UiUnlitPropertyIds, native_ui_family_from_shader_label,
        native_ui_family_from_unity_shader_name, resolve_native_ui_shader_family,
        ui_unlit_material_uniform,
    };

    #[test]
    fn ui_unlit_flags_bit_positions_match_wgsl() {
        let f = UiUnlitFlags {
            alphaclip: true,
            rectclip: true,
            overlay: true,
            has_main_tex: false,
            texture_normalmap: true,
            texture_lerpcolor: true,
            mask_texture_mul: true,
            mask_texture_clip: true,
        };
        assert_eq!(f.to_bits(), 1 | 2 | 4 | 8 | 16 | 32 | 64);
    }

    #[test]
    fn native_ui_family_maps_canonical_unity_names() {
        assert_eq!(
            native_ui_family_from_unity_shader_name(CANONICAL_UNITY_UI_UNLIT),
            Some(NativeUiShaderFamily::UiUnlit)
        );
        assert_eq!(
            native_ui_family_from_unity_shader_name(CANONICAL_UNITY_UI_TEXT_UNLIT),
            Some(NativeUiShaderFamily::UiTextUnlit)
        );
    }

    #[test]
    fn native_ui_family_compact_alnum_ui_unlit() {
        assert_eq!(
            native_ui_family_from_unity_shader_name("uiunlit"),
            Some(NativeUiShaderFamily::UiUnlit)
        );
    }

    #[test]
    fn native_ui_family_from_shader_label_matches_stems() {
        assert_eq!(
            native_ui_family_from_shader_label("UI_Unlit"),
            Some(NativeUiShaderFamily::UiUnlit)
        );
        assert_eq!(
            native_ui_family_from_shader_label("UI_TextUnlit"),
            Some(NativeUiShaderFamily::UiTextUnlit)
        );
        assert_eq!(
            native_ui_family_from_shader_label("UI_TextUnlit EXTRA"),
            Some(NativeUiShaderFamily::UiTextUnlit)
        );
        assert_eq!(native_ui_family_from_shader_label("Unknown"), None);
    }

    #[test]
    fn ui_unlit_uniform_reads_float_keyword_flags() {
        let mut store = MaterialPropertyStore::new();
        let mid = 10;
        store.set_material(mid, 100, MaterialPropertyValue::Float(1.0));
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: mid,
            mesh_property_block_slot0: None,
        };
        let ids = UiUnlitPropertyIds {
            alphaclip: 100,
            ..Default::default()
        };
        let (u, _, _) = ui_unlit_material_uniform(&store, lookup, &ids);
        assert_ne!(u.flags & 1, 0);
    }

    #[test]
    fn ui_unlit_inventory_material_only_lookup_reads_main_tex() {
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(10, 42);
        let mut rc = RenderConfig::default();
        rc.ui_unlit_property_ids.main_tex = 7;
        store.set_material(10, 7, MaterialPropertyValue::Texture(99));
        let mut reg = AssetRegistry::new();
        reg.handle_shader_upload(ShaderUpload {
            asset_id: 42,
            file: Some("UI_Unlit".to_string()),
        });
        assert_eq!(
            resolve_native_ui_shader_family(42, &reg),
            Some(NativeUiShaderFamily::UiUnlit)
        );
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 10,
            mesh_property_block_slot0: None,
        };
        let (_, main_packed, _) =
            ui_unlit_material_uniform(&store, lookup, &rc.ui_unlit_property_ids);
        assert_eq!(main_packed, 99);
        assert_eq!(texture2d_asset_id_from_packed(99), Some(99));
    }
}
