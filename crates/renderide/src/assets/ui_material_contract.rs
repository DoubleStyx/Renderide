//! Host contract for native WGSL UI materials (`UI_Unlit`, `UI_TextUnlit`).
//!
//! Shader asset IDs are assigned by the host at runtime. Configure them under `[rendering]` in
//! `configuration.ini` (see [`crate::config::RenderConfig`]) or via environment variables so the
//! renderer maps `set_shader` batches to [`NativeUiShaderFamily`].
//!
//! Logical Unity names (`Shader "UI/Unlit"`) are resolved from ShaderLab/WGSL upload payloads, path hints,
//! or a host-provided hint — see [`super::shader_logical_name`] and [`crate::shared::shader_upload_extras`].
//!
//! ## Material property IDs
//!
//! Batches use numeric `property_id` values. When a field in [`UiUnlitPropertyIds`] or
//! [`UiTextUnlitPropertyIds`] is `-1`, that channel is skipped and the GPU uniform uses a default.
//!
//! **Automatic mapping:** On [`MaterialPropertyIdRequest`](crate::shared::MaterialPropertyIdRequest),
//! [`crate::assets::material_property_host`] interns each property name to an integer and replies with
//! [`MaterialPropertyIdResult`](crate::shared::MaterialPropertyIdResult); matching FrooxEngine shader
//! property names also update [`crate::config::RenderConfig`] when [`RenderConfig::use_native_ui_wgsl`](crate::config::RenderConfig::use_native_ui_wgsl) is true.
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
//! - `_SrcBlend` / `_DstBlend` → `src_blend` / `dst_blend` (Unity blend factors as floats; see
//!   [`crate::assets::native_ui_blend::NativeUiSurfaceBlend`])
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
//! pipelines use fixed depth/stencil/rasterizer states from WGSL pipeline construction. Extend the native
//! UI pipeline descriptors only if a Resonite workflow requires matching those toggles per material.

/// Identifies which native UI WGSL program to use for a host shader asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeUiShaderFamily {
    /// Resonite `UI/Unlit` ([`third_party/Resonite.UnityShaders/.../UI_Unlit.shader`](../../../../third_party/Resonite.UnityShaders/Assets/Shaders/UI/UI_Unlit.shader)).
    UiUnlit,
    /// Resonite `UI/Text/Unlit`.
    UiTextUnlit,
}

/// Resolves `shader_asset_id` to a native UI family using configured allowlist ids.
pub fn native_ui_family_for_shader(
    shader_asset_id: i32,
    ui_unlit_id: i32,
    ui_text_unlit_id: i32,
) -> Option<NativeUiShaderFamily> {
    if ui_unlit_id >= 0 && shader_asset_id == ui_unlit_id {
        return Some(NativeUiShaderFamily::UiUnlit);
    }
    if ui_text_unlit_id >= 0 && shader_asset_id == ui_text_unlit_id {
        return Some(NativeUiShaderFamily::UiTextUnlit);
    }
    None
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

fn compact_alnum_lower(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Maps Unity ShaderLab `Shader "…"` strings (and common aliases) to [`NativeUiShaderFamily`].
///
/// Accepts exact Resonite names [`super::shader_logical_name::CANONICAL_UNITY_UI_UNLIT`] and
/// [`super::shader_logical_name::CANONICAL_UNITY_UI_TEXT_UNLIT`], case-insensitive equality, a
/// compact alphanumeric form (e.g. `uiunlit`), and the same path-style heuristics as
/// [`native_ui_family_from_shader_path_hint`].
pub fn native_ui_family_from_unity_shader_name(name: &str) -> Option<NativeUiShaderFamily> {
    let t = name.trim();
    if t.is_empty() {
        return None;
    }
    let lower = t.to_ascii_lowercase();
    if lower == super::shader_logical_name::CANONICAL_UNITY_UI_UNLIT.to_ascii_lowercase()
        || compact_alnum_lower(t)
            == compact_alnum_lower(super::shader_logical_name::CANONICAL_UNITY_UI_UNLIT)
    {
        return Some(NativeUiShaderFamily::UiUnlit);
    }
    if lower == super::shader_logical_name::CANONICAL_UNITY_UI_TEXT_UNLIT.to_ascii_lowercase()
        || compact_alnum_lower(t)
            == compact_alnum_lower(super::shader_logical_name::CANONICAL_UNITY_UI_TEXT_UNLIT)
    {
        return Some(NativeUiShaderFamily::UiTextUnlit);
    }
    native_ui_family_from_shader_path_hint(t)
}

/// Resolves native UI shader family using configured allowlist ids, stored Unity shader name, then path hint.
pub fn resolve_native_ui_shader_family(
    shader_asset_id: i32,
    native_ui_unlit_shader_id: i32,
    native_ui_text_unlit_shader_id: i32,
    registry: &super::AssetRegistry,
) -> Option<NativeUiShaderFamily> {
    native_ui_family_for_shader(
        shader_asset_id,
        native_ui_unlit_shader_id,
        native_ui_text_unlit_shader_id,
    )
    .or_else(|| {
        registry
            .get_shader(shader_asset_id)
            .and_then(|s| s.unity_shader_name.as_deref())
            .and_then(native_ui_family_from_unity_shader_name)
    })
    .or_else(|| {
        registry
            .get_shader(shader_asset_id)
            .and_then(|s| s.wgsl_source.as_deref())
            .and_then(native_ui_family_from_shader_path_hint)
    })
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
    let overlay_tint = float4(store, lookup, ids.overlay_tint, [1.0, 1.0, 1.0, 0.73]);
    let cutoff = float1(store, lookup, ids.cutoff, 0.98);
    let main_tex_st = float4(store, lookup, ids.main_tex_st, [1.0, 1.0, 0.0, 0.0]);
    let mask_tex_st = float4(store, lookup, ids.mask_tex_st, [1.0, 1.0, 0.0, 0.0]);
    let rect = float4(store, lookup, ids.rect, [0.0, 0.0, 1.0, 1.0]);
    let flags = UiUnlitFlags {
        alphaclip: flag_f(store, lookup, ids.alphaclip),
        rectclip: flag_f(store, lookup, ids.rectclip),
        overlay: flag_f(store, lookup, ids.overlay),
        texture_normalmap: flag_f(store, lookup, ids.texture_normalmap),
        texture_lerpcolor: flag_f(store, lookup, ids.texture_lerpcolor),
        mask_texture_mul: flag_f(store, lookup, ids.mask_texture_mul),
        mask_texture_clip: flag_f(store, lookup, ids.mask_texture_clip),
    };
    let main_tex = texture_handle(store, lookup, ids.main_tex);
    let mask_tex = texture_handle(store, lookup, ids.mask_tex);
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
        Some(super::MaterialPropertyValue::Texture(h)) => *h,
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
    let overlay_tint = float4(store, lookup, ids.overlay_tint, [1.0, 1.0, 1.0, 0.73]);
    let outline_color = float4(store, lookup, ids.outline_color, [1.0, 1.0, 1.0, 0.0]);
    let background_color = float4(store, lookup, ids.background_color, [0.0, 0.0, 0.0, 0.0]);
    let range_v = float4(store, lookup, ids.range, [0.001, 0.001, 0.0, 0.0]);
    let face_dilate = float1(store, lookup, ids.face_dilate, 0.0);
    let face_softness = float1(store, lookup, ids.face_softness, 0.0);
    let outline_size = float1(store, lookup, ids.outline_size, 0.0);
    let rect = float4(store, lookup, ids.rect, [0.0, 0.0, 1.0, 1.0]);
    let mut mode: u32 = 0;
    if flag_f(store, lookup, ids.sdf) {
        mode = 1;
    }
    if flag_f(store, lookup, ids.msdf) {
        mode = 2;
    }
    if flag_f(store, lookup, ids.raster) {
        mode = 0;
    }
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

    use super::{
        NativeUiShaderFamily, UiUnlitFlags, UiUnlitPropertyIds,
        native_ui_family_from_unity_shader_name, ui_unlit_material_uniform,
    };

    #[test]
    fn ui_unlit_flags_bit_positions_match_wgsl() {
        let f = UiUnlitFlags {
            alphaclip: true,
            rectclip: true,
            overlay: true,
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
    fn ui_unlit_uniform_reads_float_keyword_flags() {
        let mut store = MaterialPropertyStore::new();
        let mid = 10;
        store.set(mid, 100, MaterialPropertyValue::Float(1.0));
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
}
