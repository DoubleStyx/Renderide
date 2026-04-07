//! Host contract for native WGSL world (non-UI) [`super::shader_logical_name::CANONICAL_UNITY_WORLD_UNLIT`]
//! (`Shader "Unlit"`, [`third_party/Resonite.UnityShaders/.../Unlit.shader`](../../../../third_party/Resonite.UnityShaders/Assets/Shaders/Common/Unlit.shader)).
//!
//! This is separate from [`super::ui_material_contract`] (`UI/Unlit`, canvas vertices): world unlit uses
//! [`crate::gpu::mesh::VertexPosNormalUv`] and [`crate::gpu::pipeline::WorldUnlitPipeline`].
//!
//! ## MVP vs later parity
//!
//! **MVP** (implemented in WGSL): `_Tex` / `_Tex_ST`, `_Color`, `_Cutoff`, `_MaskTex` / `_MaskTex_ST`,
//! keyword-driven paths matching Unity `#pragma` toggles for texture, color, alpha test, mask multiply,
//! mask clip, and texture-as-normal visualization.
//!
//! **Phase 2** (not yet in WGSL): vertex colors, `_MUL_RGB_BY_ALPHA`, `_MUL_ALPHA_INTENSITY`.
//!
//! **Phase 3**: `_OFFSET_TEXTURE`, `_POLARUV`, stereo `_RightEye_ST`.

use std::collections::HashMap;

use crate::assets::util::compact_alnum_lower;
use crate::config::RenderConfig;

use super::MaterialPropertyLookupIds;
use super::texture2d_asset_id_from_packed;

/// Identifies the Resonite world `Shader "Unlit"` for host shader asset routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorldUnlitShaderFamily {
    /// Resonite `Shader "Unlit"` ([`third_party/Resonite.UnityShaders/.../Unlit.shader`](../../../../third_party/Resonite.UnityShaders/Assets/Shaders/Common/Unlit.shader)).
    StandardUnlit,
}

/// Maps a logical shader name or stem (first whitespace-delimited token) to [`WorldUnlitShaderFamily`]
/// when it matches Resonite `Shader "Unlit"` / stem `Unlit`, and not `UI/Unlit`.
pub fn world_unlit_family_from_shader_label(label: &str) -> Option<WorldUnlitShaderFamily> {
    let token = label.split_whitespace().next()?;
    if token.is_empty() {
        return None;
    }
    let key = compact_alnum_lower(token);
    let k_world = compact_alnum_lower(super::shader_logical_name::CANONICAL_UNITY_WORLD_UNLIT);
    let k_ui = compact_alnum_lower(super::shader_logical_name::CANONICAL_UNITY_UI_UNLIT);
    let k_ui_text = compact_alnum_lower(super::shader_logical_name::CANONICAL_UNITY_UI_TEXT_UNLIT);
    if key == k_ui || key == k_ui_text {
        return None;
    }
    if key == k_world {
        return Some(WorldUnlitShaderFamily::StandardUnlit);
    }
    None
}

/// Infers [`WorldUnlitShaderFamily`] from bundle paths and file names (e.g. `Common/Unlit.shader`).
///
/// Does not match `UI/Unlit` / `UI_Unlit` paths (those belong to [`super::ui_material_contract`]).
pub fn world_unlit_family_from_shader_path_hint(hint: &str) -> Option<WorldUnlitShaderFamily> {
    let h = hint.to_ascii_lowercase();
    if h.contains("ui/unlit")
        || h.contains("ui_unlit")
        || h.contains("uiunlit")
        || h.contains("ui/text")
        || h.contains("ui_text")
    {
        return None;
    }
    if h.contains("common/unlit") || h.contains("common\\unlit") {
        return Some(WorldUnlitShaderFamily::StandardUnlit);
    }
    if h.contains("unlit.shader") && !h.contains("ui_") && !h.contains("/ui/") {
        return Some(WorldUnlitShaderFamily::StandardUnlit);
    }
    None
}

/// Maps Unity ShaderLab `Shader "…"` strings to [`WorldUnlitShaderFamily`] when they denote world Unlit.
pub fn world_unlit_family_from_unity_shader_name(name: &str) -> Option<WorldUnlitShaderFamily> {
    world_unlit_family_from_shader_label(name)
        .or_else(|| world_unlit_family_from_shader_path_hint(name))
}

/// Resolves world Unlit family from stored logical shader identity.
pub fn resolve_world_unlit_shader_family(
    shader_asset_id: i32,
    registry: &super::AssetRegistry,
) -> Option<WorldUnlitShaderFamily> {
    if let Some(s) = registry.get_shader(shader_asset_id) {
        if s.program == super::EssentialShaderProgram::WorldUnlit {
            return Some(WorldUnlitShaderFamily::StandardUnlit);
        }
        if let Some(f) = s
            .unity_shader_name
            .as_deref()
            .and_then(world_unlit_family_from_unity_shader_name)
        {
            return Some(f);
        }
        if let Some(f) = s
            .wgsl_source
            .as_deref()
            .and_then(world_unlit_family_from_unity_shader_name)
        {
            return Some(f);
        }
    }
    None
}

/// Material property indices for world `Shader "Unlit"` batches. `-1` = omit (use GPU default / skip path).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldUnlitPropertyIds {
    /// `_Tex`
    pub tex: i32,
    /// `_Tex_ST`
    pub tex_st: i32,
    /// `_Color` (linear float4)
    pub color: i32,
    /// `_Cutoff`
    pub cutoff: i32,
    /// `_MaskTex`
    pub mask_tex: i32,
    /// `_MaskTex_ST`
    pub mask_tex_st: i32,
    /// Keyword float `_TEXTURE` / `TEXTURE`
    pub texture_kw: i32,
    /// Keyword float `_COLOR` / `COLOR`
    pub color_kw: i32,
    /// Keyword float `_TEXTURE_NORMALMAP` / `TEXTURE_NORMALMAP`
    pub texture_normalmap_kw: i32,
    /// Keyword float `_ALPHATEST` / `ALPHATEST`
    pub alphatest_kw: i32,
    /// Keyword float `_MASK_TEXTURE_MUL` / `mask_texture_mul`
    pub mask_texture_mul: i32,
    /// Keyword float `_MASK_TEXTURE_CLIP` / `mask_texture_clip`
    pub mask_texture_clip: i32,
}

impl Default for WorldUnlitPropertyIds {
    fn default() -> Self {
        Self {
            tex: -1,
            tex_st: -1,
            color: -1,
            cutoff: -1,
            mask_tex: -1,
            mask_tex_st: -1,
            texture_kw: -1,
            color_kw: -1,
            texture_normalmap_kw: -1,
            alphatest_kw: -1,
            mask_texture_mul: -1,
            mask_texture_clip: -1,
        }
    }
}

/// GPU-packed flags for world Unlit (single `u32` in uniform block). Must match [`wgsl_modules/world_unlit.wgsl`].
#[derive(Clone, Copy, Debug)]
pub struct WorldUnlitFlags {
    pub texture: bool,
    pub color: bool,
    pub texture_normalmap: bool,
    pub alphatest: bool,
    pub mask_texture_mul: bool,
    pub mask_texture_clip: bool,
}

impl WorldUnlitFlags {
    // Bit positions MUST match the WGSL constants in `RENDERIDESHADERS/world/unlit.wgsl`:
    //   FLAG_TEXTURE            = 1u
    //   FLAG_COLOR              = 2u
    //   FLAG_ALPHATEST          = 4u   (not 8!)
    //   FLAG_VERTEXCOLORS       = 8u   (not exposed from Rust – vertex color not yet wired)
    //   FLAG_MUL_ALPHA_INTENSITY = 16u (not exposed)
    //   FLAG_OFFSET_TEXTURE     = 32u  (not exposed)
    //   FLAG_MASK_TEXTURE_MUL   = 64u  (not 16!)
    //   FLAG_MASK_TEXTURE_CLIP  = 128u (not 32!)
    //   FLAG_MUL_RGB_BY_ALPHA   = 256u (not exposed)
    //   FLAG_POLARUV            = 512u (not exposed)
    //   FLAG_TEXTURE_NORMALMAP  = 1024u (not 4!)
    const FLAG_TEXTURE: u32 = 1;
    const FLAG_COLOR: u32 = 2;
    const FLAG_ALPHATEST: u32 = 4;
    const FLAG_MASK_MUL: u32 = 64;
    const FLAG_MASK_CLIP: u32 = 128;
    const FLAG_TEXTURE_NORMALMAP: u32 = 1024;

    /// Packs flags into a `u32` for GPU upload.
    pub fn to_bits(self) -> u32 {
        let mut b = 0u32;
        if self.texture {
            b |= Self::FLAG_TEXTURE;
        }
        if self.color {
            b |= Self::FLAG_COLOR;
        }
        if self.texture_normalmap {
            b |= Self::FLAG_TEXTURE_NORMALMAP;
        }
        if self.alphatest {
            b |= Self::FLAG_ALPHATEST;
        }
        if self.mask_texture_mul {
            b |= Self::FLAG_MASK_MUL;
        }
        if self.mask_texture_clip {
            b |= Self::FLAG_MASK_CLIP;
        }
        b
    }
}

/// CPU-side uniform for world Unlit before upload (matches WGSL `WorldUnlitMaterialUniform`).
///
/// **Layout must exactly mirror** the WGSL struct in `RENDERIDESHADERS/world/unlit.wgsl`:
/// ```wgsl
/// struct WorldUnlitMaterialUniform {
///     _Color: vec4f,           // offset  0
///     _Tex_ST: vec4f,          // offset 16
///     _MaskTex_ST: vec4f,      // offset 32
///     _OffsetMagnitude: vec4f, // offset 48
///     _Cutoff: f32,            // offset 64
///     _PolarPow: f32,          // offset 68
///     _Flags: u32,             // offset 72
///     _Pad0: u32,              // offset 76
/// }  // total 80 bytes
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldUnlitMaterialUniform {
    /// `_Color` when used; defaults white.
    pub color: [f32; 4],
    /// `_Tex_ST`
    pub tex_st: [f32; 4],
    /// `_MaskTex_ST`
    pub mask_tex_st: [f32; 4],
    /// `_OffsetMagnitude` – xy applied as UV offset when `FLAG_OFFSET_TEXTURE` is set.
    /// Defaults `[0,0,0,0]` (no offset).
    pub offset_magnitude: [f32; 4],
    /// `_Cutoff`
    pub cutoff: f32,
    /// `_PolarPow` – polar-UV radius exponent. Default `1.0` (linear).
    pub polar_pow: f32,
    /// [`WorldUnlitFlags::to_bits`].
    pub flags: u32,
    pub pad_tail: u32,
}

fn float4(
    store: &super::MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
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

fn float1(
    store: &super::MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pid: i32,
    default: f32,
) -> f32 {
    if pid < 0 {
        return default;
    }
    match store.get_merged(lookup, pid) {
        Some(super::MaterialPropertyValue::Float(f)) => *f,
        _ => default,
    }
}

fn flag_f(
    store: &super::MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
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

fn texture_handle(
    store: &super::MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
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

/// Builds GPU uniform and packed texture handles for world `Shader "Unlit"` from merged material lookup.
pub fn world_unlit_material_uniform(
    store: &super::MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &WorldUnlitPropertyIds,
) -> (WorldUnlitMaterialUniform, i32, i32) {
    let color = float4(store, lookup, ids.color, [1.0, 1.0, 1.0, 1.0]);
    let tex_st = float4(store, lookup, ids.tex_st, [1.0, 1.0, 0.0, 0.0]);
    let mask_tex_st = float4(store, lookup, ids.mask_tex_st, [1.0, 1.0, 0.0, 0.0]);
    let cutoff = float1(store, lookup, ids.cutoff, 0.5);
    let tex = texture_handle(store, lookup, ids.tex);
    let mask_tex = texture_handle(store, lookup, ids.mask_tex);
    let flags = WorldUnlitFlags {
        // If no keyword property is configured (pid = -1), default to enabled when the resource
        // is present. This handles shaders like XSToon2.0 that have _MainTex / _Color but no
        // explicit _TEXTURE / _COLOR enable keywords.
        texture: tex != 0 && (ids.texture_kw < 0 || flag_f(store, lookup, ids.texture_kw)),
        color: ids.color >= 0 && (ids.color_kw < 0 || flag_f(store, lookup, ids.color_kw)),
        texture_normalmap: tex != 0 && flag_f(store, lookup, ids.texture_normalmap_kw),
        alphatest: flag_f(store, lookup, ids.alphatest_kw),
        mask_texture_mul: mask_tex != 0 && flag_f(store, lookup, ids.mask_texture_mul),
        mask_texture_clip: mask_tex != 0 && flag_f(store, lookup, ids.mask_texture_clip),
    };
    let u = WorldUnlitMaterialUniform {
        color,
        tex_st,
        mask_tex_st,
        offset_magnitude: [0.0, 0.0, 0.0, 0.0],
        cutoff,
        polar_pow: 1.0,
        flags: flags.to_bits(),
        pad_tail: 0,
    };
    (u, tex, mask_tex)
}

/// When [`RenderConfig::log_ui_unlit_material_inventory`] is true, logs one line per material asset
/// whose `set_shader` resolves to world [`Shader "Unlit"`](super::shader_logical_name::CANONICAL_UNITY_WORLD_UNLIT)
/// via [`resolve_world_unlit_shader_family`], excluding UI variants ([`super::shader_logical_name::CANONICAL_UNITY_UI_UNLIT`],
/// [`super::shader_logical_name::CANONICAL_UNITY_UI_TEXT_UNLIT`]).
///
/// Uses material-only [`MaterialPropertyLookupIds`] (no per-renderer property-block merge).
pub fn log_world_unlit_material_inventory_if_enabled(
    store: &super::MaterialPropertyStore,
    registry: &super::AssetRegistry,
    rc: &RenderConfig,
    texture2d_gpu: &HashMap<i32, (wgpu::Texture, wgpu::TextureView)>,
) {
    if !rc.log_ui_unlit_material_inventory {
        return;
    }
    for (material_id, shader_id) in store.iter_material_shader_bindings() {
        if resolve_world_unlit_shader_family(shader_id, registry).is_none() {
            continue;
        }
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: material_id,
            mesh_property_block_slot0: None,
        };
        let (_, tex_packed, mask_packed) =
            world_unlit_material_uniform(store, lookup, &rc.world_unlit_property_ids);
        let tex_id = texture2d_asset_id_from_packed(tex_packed);
        let mask_id = texture2d_asset_id_from_packed(mask_packed);
        let has_color_texture_property = tex_packed != 0 && tex_id.is_some();
        let gpu_texture_resident = tex_id.is_some_and(|id| texture2d_gpu.contains_key(&id));
        let mask_gpu_texture_resident = mask_id.is_some_and(|id| texture2d_gpu.contains_key(&id));
        logger::info!(
            "world_unlit_material_inventory: material_id={} shader_id={} tex_packed={} mask_tex_packed={} has_color_texture_property={} texture2d_asset_id={:?} gpu_texture_resident={} mask_texture2d_asset_id={:?} mask_gpu_texture_resident={}",
            material_id,
            shader_id,
            tex_packed,
            mask_packed,
            has_color_texture_property,
            tex_id,
            gpu_texture_resident,
            mask_id,
            mask_gpu_texture_resident,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::shader_logical_name::{
        CANONICAL_UNITY_UI_TEXT_UNLIT, CANONICAL_UNITY_UI_UNLIT, CANONICAL_UNITY_WORLD_UNLIT,
    };
    use crate::assets::{
        AssetRegistry, MaterialPropertyStore, MaterialPropertyValue, texture2d_asset_id_from_packed,
    };
    use crate::shared::ShaderUpload;

    #[test]
    fn world_unlit_label_matches_unlit_stem() {
        assert_eq!(
            world_unlit_family_from_shader_label("Unlit"),
            Some(WorldUnlitShaderFamily::StandardUnlit)
        );
        assert_eq!(
            world_unlit_family_from_shader_label("Unlit VARIANT"),
            Some(WorldUnlitShaderFamily::StandardUnlit)
        );
    }

    #[test]
    fn world_unlit_label_rejects_ui_unlit() {
        assert_eq!(world_unlit_family_from_shader_label("UI_Unlit"), None);
        assert_eq!(
            world_unlit_family_from_shader_label(CANONICAL_UNITY_UI_UNLIT),
            None
        );
        assert_eq!(
            world_unlit_family_from_shader_label(CANONICAL_UNITY_UI_TEXT_UNLIT),
            None
        );
    }

    #[test]
    fn world_unlit_from_unity_name() {
        assert_eq!(
            world_unlit_family_from_unity_shader_name(CANONICAL_UNITY_WORLD_UNLIT),
            Some(WorldUnlitShaderFamily::StandardUnlit)
        );
    }

    #[test]
    fn world_unlit_path_hint_common() {
        assert_eq!(
            world_unlit_family_from_shader_path_hint("Assets/Shaders/Common/Unlit.shader"),
            Some(WorldUnlitShaderFamily::StandardUnlit)
        );
    }

    #[test]
    fn world_unlit_flags_bit_positions_match_wgsl() {
        let f = WorldUnlitFlags {
            texture: true,
            color: true,
            texture_normalmap: true,
            alphatest: true,
            mask_texture_mul: true,
            mask_texture_clip: true,
        };
        // Matches WGSL constants: FLAG_TEXTURE=1, FLAG_COLOR=2, FLAG_ALPHATEST=4,
        // FLAG_MASK_TEXTURE_MUL=64, FLAG_MASK_TEXTURE_CLIP=128, FLAG_TEXTURE_NORMALMAP=1024
        assert_eq!(f.to_bits(), 1 | 2 | 4 | 64 | 128 | 1024);
    }

    #[test]
    fn resolve_world_unlit_uses_registry_shader_program() {
        let mut reg = AssetRegistry::new();
        reg.handle_shader_upload(ShaderUpload {
            asset_id: 42,
            file: Some("Unlit".to_string()),
        });
        assert_eq!(
            resolve_world_unlit_shader_family(42, &reg),
            Some(WorldUnlitShaderFamily::StandardUnlit)
        );
    }

    #[test]
    fn world_unlit_uniform_reads_texture_ids() {
        let mut store = MaterialPropertyStore::new();
        let mid = 5;
        store.set_shader_asset_for_material(mid, 99);
        store.set_material(mid, 10, MaterialPropertyValue::Texture(77));
        let ids = WorldUnlitPropertyIds {
            tex: 10,
            ..WorldUnlitPropertyIds::default()
        };
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: mid,
            mesh_property_block_slot0: None,
        };
        let (_, tex_p, _) = world_unlit_material_uniform(&store, lookup, &ids);
        assert_eq!(tex_p, 77);
        assert_eq!(texture2d_asset_id_from_packed(77), Some(77));
    }
}
