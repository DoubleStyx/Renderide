//! Host `MaterialPropertyIdRequest` / `MaterialPropertyIdResult` support.
//!
//! FrooxEngine’s render material manager sends [`MaterialPropertyIdRequest`](crate::shared::MaterialPropertyIdRequest)
//! with shader property **names** (e.g. `_MainTex`). The renderer must reply with one integer per name;
//! those integers are stored on host `MaterialProperty` values and used in `MaterialsUpdateBatch` as
//! `property_id`.
//!
//! ## Host contract (Renderite vs this renderer)
//!
//! Unity’s Renderite driver resolves names with `Shader.PropertyToID` and sends those integers back.
//! The standalone
//! renderer **does not** share Unity’s numeric id space: it interns each distinct name to a
//! process-stable integer via [`intern_host_material_property_id`]. The host **must** treat the
//! returned [`MaterialPropertyIdResult::property_ids`](crate::shared::MaterialPropertyIdResult)
//! as opaque handles and use them verbatim in subsequent `MaterialsUpdateBatch` records. Mixing in
//! Unity/ShaderLab `PropertyToID` values from another process will not match the store lookup keys
//! used for native UI WGSL (`UI_Unlit`, `UI_TextUnlit`) and PBR host bindings.
//!
//! When [`crate::config::RenderConfig::use_native_ui_wgsl`] is enabled, [`apply_froox_material_property_name_to_native_ui_config`]
//! maps known Froox/Unity property names (`_MainTex`, `_Tint`, …) onto [`crate::config::RenderConfig::ui_unlit_property_ids`]
//! so batches can drive [`crate::assets::ui_unlit_material_uniform`] without manual INI mapping.
//!
//! When [`crate::config::RenderConfig::use_host_unlit_pilot`] is enabled, [`apply_froox_material_property_name_to_world_unlit_config`]
//! maps `_Tex`, `_Color`, … onto [`crate::config::RenderConfig::world_unlit_property_ids`] for world `Shader "Unlit"`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::config::RenderConfig;

static NEXT_PROPERTY_ID: AtomicI32 = AtomicI32::new(1);
static NAME_TO_ID: LazyLock<Mutex<HashMap<String, i32>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Returns a stable integer id for `name`, allocating a new id the first time this name appears.
///
/// Empty names map to `0` and are not inserted (matches “no property” semantics).
pub fn intern_host_material_property_id(name: &str) -> i32 {
    if name.is_empty() {
        return 0;
    }
    let mut map = NAME_TO_ID
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(&id) = map.get(name) {
        return id;
    }
    let id = NEXT_PROPERTY_ID.fetch_add(1, Ordering::Relaxed);
    map.insert(name.to_string(), id);
    id
}

/// Copies known FrooxEngine / Unity property names into [`RenderConfig`] native UI id maps when
/// [`RenderConfig::use_native_ui_wgsl`] is enabled.
///
/// See the FrooxEngine-to-field table in [`crate::assets::ui_material_contract`] module documentation.
pub fn apply_froox_material_property_name_to_native_ui_config(
    config: &mut RenderConfig,
    name: &str,
    id: i32,
) {
    if !config.use_native_ui_wgsl || name.is_empty() {
        return;
    }
    let u = &mut config.ui_unlit_property_ids;
    let t = &mut config.ui_text_unlit_property_ids;
    match name {
        "_MainTex" => u.main_tex = id,
        "_MaskTex" => u.mask_tex = id,
        "_MainTex_ST" => u.main_tex_st = id,
        "_MaskTex_ST" => u.mask_tex_st = id,
        "_Tint" => u.tint = id,
        "_OverlayTint" => {
            u.overlay_tint = id;
            t.overlay_tint = id;
        }
        "_Cutoff" => u.cutoff = id,
        "_Rect" => {
            u.rect = id;
            t.rect = id;
        }
        "_FontAtlas" => t.font_atlas = id,
        "_TintColor" => t.tint_color = id,
        "_OutlineColor" => t.outline_color = id,
        "_BackgroundColor" => t.background_color = id,
        "_Range" => t.range = id,
        "_FaceDilate" => t.face_dilate = id,
        "_FaceSoftness" => t.face_softness = id,
        "_OutlineSize" => t.outline_size = id,
        "_SrcBlend" => {
            u.src_blend = id;
            t.src_blend = id;
        }
        "_DstBlend" => {
            u.dst_blend = id;
            t.dst_blend = id;
        }
        // FrooxEngine / Unity shader keywords (see `ui_material_contract` module docs).
        "ALPHACLIP" | "alphaclip" => u.alphaclip = id,
        "RECTCLIP" | "rectclip" => {
            u.rectclip = id;
            t.rectclip = id;
        }
        "OVERLAY" | "overlay" => {
            u.overlay = id;
            t.overlay = id;
        }
        "TEXTURE_NORMALMAP" | "texture_normalmap" => u.texture_normalmap = id,
        "TEXTURE_LERPCOLOR" | "texture_lerpcolor" => u.texture_lerpcolor = id,
        "_MASK_TEXTURE_MUL" | "mask_texture_mul" => u.mask_texture_mul = id,
        "_MASK_TEXTURE_CLIP" | "mask_texture_clip" => u.mask_texture_clip = id,
        "RASTER" | "raster" => t.raster = id,
        "SDF" | "sdf" => t.sdf = id,
        "MSDF" | "msdf" => t.msdf = id,
        "OUTLINE" | "outline" => t.outline = id,
        _ => {}
    }
}

/// Maps FrooxEngine / Unity property names for Resonite world [`crate::assets::CANONICAL_UNITY_WORLD_UNLIT`]
/// into [`crate::config::RenderConfig::world_unlit_property_ids`] when [`RenderConfig::use_host_unlit_pilot`] is on.
pub fn apply_froox_material_property_name_to_world_unlit_config(
    config: &mut RenderConfig,
    name: &str,
    id: i32,
) {
    if !config.use_host_unlit_pilot || name.is_empty() {
        return;
    }
    let w = &mut config.world_unlit_property_ids;
    match name {
        "_Tex" => w.tex = id,
        "_Tex_ST" => w.tex_st = id,
        "_Color" => w.color = id,
        "_Cutoff" => w.cutoff = id,
        "_MaskTex" => w.mask_tex = id,
        "_MaskTex_ST" => w.mask_tex_st = id,
        "_TEXTURE" | "TEXTURE" | "texture_kw" => w.texture_kw = id,
        "_COLOR" | "COLOR" | "color_kw" => w.color_kw = id,
        "_TEXTURE_NORMALMAP" | "TEXTURE_NORMALMAP" | "texture_normalmap_kw" => {
            w.texture_normalmap_kw = id;
        }
        "_ALPHATEST" | "ALPHATEST" | "alphatest_kw" => w.alphatest_kw = id,
        "_MASK_TEXTURE_MUL" | "mask_texture_mul" => w.mask_texture_mul = id,
        "_MASK_TEXTURE_CLIP" | "mask_texture_clip" => w.mask_texture_clip = id,
        _ => {}
    }
}

/// Records Unity Standard / FrooxEngine property names used by generic PBR for host-driven factors.
///
/// Independent of [`RenderConfig::use_native_ui_wgsl`]; PBR reads these ids when
/// [`RenderConfig::pbr_bind_host_material_properties`](crate::config::RenderConfig::pbr_bind_host_material_properties) is on.
pub fn apply_froox_material_property_name_to_pbr_host_config(
    config: &mut RenderConfig,
    name: &str,
    id: i32,
) {
    if name.is_empty() {
        return;
    }
    match name {
        "_Color" => config.pbr_host_color_property_id = id,
        "_Metallic" => config.pbr_host_metallic_property_id = id,
        "_Glossiness" | "Glossiness" => config.pbr_host_smoothness_property_id = id,
        "_MainTex" => config.pbr_host_main_tex_property_id = id,
        _ => {}
    }
}

#[cfg(test)]
pub(crate) fn reset_material_property_intern_table_for_tests() {
    NEXT_PROPERTY_ID.store(1, Ordering::Relaxed);
    let mut map = NAME_TO_ID
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    map.clear();
}

/// Serializes tests that mutate the process-global property name table (parallel `cargo test` safe).
#[cfg(test)]
static MATERIAL_PROPERTY_HOST_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use std::sync::MutexGuard;

    use super::{
        MATERIAL_PROPERTY_HOST_TEST_MUTEX, apply_froox_material_property_name_to_native_ui_config,
        apply_froox_material_property_name_to_pbr_host_config,
        apply_froox_material_property_name_to_world_unlit_config, intern_host_material_property_id,
        reset_material_property_intern_table_for_tests,
    };
    use crate::config::RenderConfig;

    fn test_lock() -> MutexGuard<'static, ()> {
        MATERIAL_PROPERTY_HOST_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn intern_returns_same_id_for_same_name() {
        let _lock = test_lock();
        reset_material_property_intern_table_for_tests();
        let a = intern_host_material_property_id("_MainTex");
        let b = intern_host_material_property_id("_MainTex");
        assert_eq!(a, b);
        assert_ne!(a, intern_host_material_property_id("_Tint"));
    }

    #[test]
    fn apply_maps_froox_names_to_world_unlit_when_host_unlit_pilot_on() {
        let _lock = test_lock();
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig {
            use_host_unlit_pilot: true,
            ..Default::default()
        };
        let id = intern_host_material_property_id("_Tex");
        apply_froox_material_property_name_to_world_unlit_config(&mut c, "_Tex", id);
        assert_eq!(c.world_unlit_property_ids.tex, id);
    }

    #[test]
    fn apply_skips_world_unlit_when_host_unlit_pilot_off() {
        let _lock = test_lock();
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig {
            use_host_unlit_pilot: false,
            ..Default::default()
        };
        let id = intern_host_material_property_id("_Tex");
        apply_froox_material_property_name_to_world_unlit_config(&mut c, "_Tex", id);
        assert_eq!(c.world_unlit_property_ids.tex, -1);
    }

    #[test]
    fn apply_maps_froox_names_to_ui_unlit_and_text_ids() {
        let _lock = test_lock();
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig {
            use_native_ui_wgsl: true,
            ..Default::default()
        };
        let m = intern_host_material_property_id("_MainTex");
        let f = intern_host_material_property_id("_FontAtlas");
        apply_froox_material_property_name_to_native_ui_config(&mut c, "_MainTex", m);
        apply_froox_material_property_name_to_native_ui_config(&mut c, "_FontAtlas", f);
        assert_eq!(c.ui_unlit_property_ids.main_tex, m);
        assert_eq!(c.ui_text_unlit_property_ids.font_atlas, f);
    }

    #[test]
    fn apply_skips_when_native_ui_wgsl_disabled() {
        let _lock = test_lock();
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig {
            use_native_ui_wgsl: false,
            ..Default::default()
        };
        let id = intern_host_material_property_id("_MainTex");
        apply_froox_material_property_name_to_native_ui_config(&mut c, "_MainTex", id);
        assert_eq!(c.ui_unlit_property_ids.main_tex, -1);
    }

    #[test]
    fn apply_maps_shader_keyword_alphaclip() {
        let _lock = test_lock();
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig {
            use_native_ui_wgsl: true,
            ..Default::default()
        };
        let id = intern_host_material_property_id("ALPHACLIP");
        apply_froox_material_property_name_to_native_ui_config(&mut c, "ALPHACLIP", id);
        assert_eq!(c.ui_unlit_property_ids.alphaclip, id);
    }

    #[test]
    fn apply_pbr_host_maps_color_metallic_glossiness() {
        let _lock = test_lock();
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig::default();
        let color_id = intern_host_material_property_id("_Color");
        let metal_id = intern_host_material_property_id("_Metallic");
        let gloss_id = intern_host_material_property_id("_Glossiness");
        apply_froox_material_property_name_to_pbr_host_config(&mut c, "_Color", color_id);
        apply_froox_material_property_name_to_pbr_host_config(&mut c, "_Metallic", metal_id);
        apply_froox_material_property_name_to_pbr_host_config(&mut c, "_Glossiness", gloss_id);
        assert_eq!(c.pbr_host_color_property_id, color_id);
        assert_eq!(c.pbr_host_metallic_property_id, metal_id);
        assert_eq!(c.pbr_host_smoothness_property_id, gloss_id);
    }

    #[test]
    fn apply_pbr_host_maps_main_tex() {
        let _lock = test_lock();
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig::default();
        let main_tex_id = intern_host_material_property_id("_MainTex");
        apply_froox_material_property_name_to_pbr_host_config(&mut c, "_MainTex", main_tex_id);
        assert_eq!(c.pbr_host_main_tex_property_id, main_tex_id);
    }

    /// Simulates `MaterialPropertyIdRequest` → intern → `MaterialsUpdateBatch` property_id lookup keys.
    #[test]
    fn material_property_ids_are_stable_for_batch_lookup_after_intern() {
        let _lock = test_lock();
        reset_material_property_intern_table_for_tests();
        let main_tex = intern_host_material_property_id("_MainTex");
        let tint = intern_host_material_property_id("_Tint");
        assert_ne!(main_tex, tint);
        let main_tex_again = intern_host_material_property_id("_MainTex");
        assert_eq!(main_tex, main_tex_again);
        let mut c = RenderConfig {
            use_native_ui_wgsl: true,
            ..Default::default()
        };
        apply_froox_material_property_name_to_native_ui_config(&mut c, "_MainTex", main_tex);
        apply_froox_material_property_name_to_native_ui_config(&mut c, "_Tint", tint);
        assert_eq!(c.ui_unlit_property_ids.main_tex, main_tex);
        assert_eq!(c.ui_unlit_property_ids.tint, tint);
    }
}
