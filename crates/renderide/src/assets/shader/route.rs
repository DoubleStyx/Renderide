//! Coarse classification of host shaders into [`MaterialFamilyId`] for [`MaterialRegistry`](crate::materials::MaterialRegistry).
//!
//! Extraction of Unity logical names lives in [`super::logical_name`] and [`super::unity_asset`].
//! Names listed in the embedded shader manifest resolve to [`MANIFEST_RASTER_FAMILY_ID`](crate::materials::MANIFEST_RASTER_FAMILY_ID);
//! other kinds use [`DEBUG_WORLD_NORMALS_FAMILY_ID`](crate::materials::DEBUG_WORLD_NORMALS_FAMILY_ID) until implemented.

pub use crate::assets::util::normalize_unity_shader_lookup_key;
use crate::materials::DEBUG_WORLD_NORMALS_FAMILY_ID;
use crate::materials::{manifest_stem_for_unity_name, MaterialFamilyId, MANIFEST_RASTER_FAMILY_ID};
use crate::shared::ShaderUpload;

use super::logical_name::{
    self, CANONICAL_UNITY_UI_TEXT_UNLIT, CANONICAL_UNITY_UI_UNLIT, CANONICAL_UNITY_WORLD_UNLIT,
};
use crate::assets::util::compact_alnum_lower;

/// Resolved upload: optional Unity-style logical name plus the material family for pipeline selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedShaderUpload {
    /// `Shader "…"` string, container stem, or label when resolution succeeded.
    pub unity_shader_name: Option<String>,
    /// Family passed to [`crate::materials::MaterialRegistry::map_shader_route`].
    pub family: MaterialFamilyId,
}

/// Coarse bucket after name/path heuristics (mirrors old `NativeShaderRoute` / UI+world contracts).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoarseShaderKind {
    /// Canvas `UI/Unlit` and related paths.
    Ui,
    /// World mesh `Shader "Unlit"` (non-UI).
    WorldUnlit,
    /// PBS metallic / specular family.
    PbsMetallic,
    /// No confident match; use default routing fallback.
    Unknown,
}

/// Full resolution pipeline for a host [`ShaderUpload`].
pub fn resolve_shader_upload(data: &ShaderUpload) -> ResolvedShaderUpload {
    let unity_shader_name = logical_name::resolve_logical_shader_name_from_upload(data);
    let kind = classify_shader(unity_shader_name.as_deref(), data.file.as_deref());
    let family = material_family_for_kind(kind, unity_shader_name.as_deref());
    ResolvedShaderUpload {
        unity_shader_name,
        family,
    }
}

/// Exposes classification for tests and diagnostics without touching the filesystem.
pub fn classify_shader(unity_name: Option<&str>, path_hint: Option<&str>) -> CoarseShaderKind {
    if let Some(h) = path_hint {
        if native_ui_family_from_shader_path_hint(h).is_some() {
            return CoarseShaderKind::Ui;
        }
        if world_unlit_family_from_shader_path_hint(h).is_some() {
            return CoarseShaderKind::WorldUnlit;
        }
        if pbs_metallic_family_from_shader_path_hint(h) {
            return CoarseShaderKind::PbsMetallic;
        }
    }
    if let Some(n) = unity_name {
        if native_ui_family_from_shader_label(n).is_some() {
            return CoarseShaderKind::Ui;
        }
        if world_unlit_family_from_shader_label(n).is_some() {
            return CoarseShaderKind::WorldUnlit;
        }
        if pbs_metallic_family_from_unity_shader_name(n) {
            return CoarseShaderKind::PbsMetallic;
        }
    }
    CoarseShaderKind::Unknown
}

/// Maps Unity logical name + coarse kind to a registered [`MaterialFamilyId`] (see `resolve_shader_upload`).
fn material_family_for_kind(kind: CoarseShaderKind, unity_name: Option<&str>) -> MaterialFamilyId {
    if let Some(name) = unity_name {
        if manifest_stem_for_unity_name(name).is_some() {
            return MANIFEST_RASTER_FAMILY_ID;
        }
        let key = normalize_unity_shader_lookup_key(name);
        if key.contains("ui_unlit") || key.contains("uiunlit") {
            return DEBUG_WORLD_NORMALS_FAMILY_ID;
        }
        if key.contains("ui_text") || key.contains("uitext") || key.contains("ui_textunlit") {
            return DEBUG_WORLD_NORMALS_FAMILY_ID;
        }
        if key.contains("pbsmetallic")
            || key.contains("pbs_specular")
            || key.contains("pbsspecular")
        {
            return DEBUG_WORLD_NORMALS_FAMILY_ID;
        }
    }
    match kind {
        CoarseShaderKind::Ui | CoarseShaderKind::WorldUnlit | CoarseShaderKind::PbsMetallic => {
            DEBUG_WORLD_NORMALS_FAMILY_ID
        }
        CoarseShaderKind::Unknown => DEBUG_WORLD_NORMALS_FAMILY_ID,
    }
}

/// Returns true when the first whitespace-delimited token of `name` matches the PBS metallic family.
pub fn pbs_metallic_family_from_unity_shader_name(name: &str) -> bool {
    let Some(token) = name.split_whitespace().next() else {
        return false;
    };
    compact_alnum_lower(token) == compact_alnum_lower("PBSMetallic")
}

/// Heuristic PBS metallic detection from a shader path or upload label string.
pub fn pbs_metallic_family_from_shader_path_hint(hint: &str) -> bool {
    let h = hint.to_ascii_lowercase();
    h.contains("pbsmetallic")
        || h.contains("pbs_specular")
        || h.contains("pbsspecular")
        || h.contains("pbs/specular")
}

/// Infers a UI family from path fragments such as `UI/Unlit`, `UI_Unlit`.
pub fn native_ui_family_from_shader_path_hint(hint: &str) -> Option<UiFamily> {
    let h = hint.to_ascii_lowercase();
    if h.contains("ui/text") && (h.contains("unlit") || h.contains("textunlit")) {
        return Some(UiFamily::TextUnlit);
    }
    if h.contains("ui/unlit") || h.contains("ui_unlit") || h.contains("uiunlit") {
        return Some(UiFamily::Unlit);
    }
    None
}

/// Maps a logical shader name or stem to UI when it matches Resonite `UI/Unlit` or `UI/Text/Unlit`.
pub fn native_ui_family_from_shader_label(label: &str) -> Option<UiFamily> {
    let token = label.split_whitespace().next()?;
    if token.is_empty() {
        return None;
    }
    let key = compact_alnum_lower(token);
    let k_unlit = compact_alnum_lower(CANONICAL_UNITY_UI_UNLIT);
    let k_text = compact_alnum_lower(CANONICAL_UNITY_UI_TEXT_UNLIT);
    if key == k_unlit {
        return Some(UiFamily::Unlit);
    }
    if key == k_text {
        return Some(UiFamily::TextUnlit);
    }
    None
}

/// UI shader bucket for routing (WGSL families TBD).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UiFamily {
    /// `UI/Unlit`.
    Unlit,
    /// `UI/Text/Unlit`.
    TextUnlit,
}

/// Infers world unlit from bundle paths and file names (e.g. `Common/Unlit.shader`).
pub fn world_unlit_family_from_shader_path_hint(hint: &str) -> Option<WorldUnlitFamily> {
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
        return Some(WorldUnlitFamily::StandardUnlit);
    }
    if h.contains("unlit.shader") && !h.contains("ui_") && !h.contains("/ui/") {
        return Some(WorldUnlitFamily::StandardUnlit);
    }
    None
}

/// Maps a logical shader name to world `Shader "Unlit"` when not UI.
pub fn world_unlit_family_from_shader_label(label: &str) -> Option<WorldUnlitFamily> {
    let token = label.split_whitespace().next()?;
    if token.is_empty() {
        return None;
    }
    let key = compact_alnum_lower(token);
    let k_world = compact_alnum_lower(CANONICAL_UNITY_WORLD_UNLIT);
    let k_ui = compact_alnum_lower(CANONICAL_UNITY_UI_UNLIT);
    let k_ui_text = compact_alnum_lower(CANONICAL_UNITY_UI_TEXT_UNLIT);
    if key == k_ui || key == k_ui_text {
        return None;
    }
    if key == k_world {
        return Some(WorldUnlitFamily::StandardUnlit);
    }
    None
}

/// World unlit bucket for routing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldUnlitFamily {
    /// Resonite `Shader "Unlit"`.
    StandardUnlit,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{DEBUG_WORLD_NORMALS_FAMILY_ID, MANIFEST_RASTER_FAMILY_ID};

    #[test]
    fn classify_ui_path_hint() {
        assert_eq!(
            classify_shader(None, Some("Assets/UI_Unlit.foo")),
            CoarseShaderKind::Ui
        );
    }

    #[test]
    fn classify_world_unlit_name() {
        assert_eq!(
            classify_shader(Some("Unlit"), None),
            CoarseShaderKind::WorldUnlit
        );
    }

    #[test]
    fn classify_ui_name_token() {
        assert_eq!(
            classify_shader(Some("UI/Unlit"), None),
            CoarseShaderKind::Ui
        );
    }

    #[test]
    fn classify_pbs_token() {
        assert_eq!(
            classify_shader(Some("PBSMetallic"), None),
            CoarseShaderKind::PbsMetallic
        );
    }

    #[test]
    fn resolve_shader_upload_sets_family_id() {
        let u = ShaderUpload {
            asset_id: 3,
            file: Some("Shader \"UI/Unlit\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.unity_shader_name.as_deref(), Some("UI/Unlit"));
        assert_eq!(r.family, DEBUG_WORLD_NORMALS_FAMILY_ID);
    }

    #[test]
    fn manifest_unity_name_maps_to_manifest_raster_family() {
        let u = ShaderUpload {
            asset_id: 4,
            file: Some("Shader \"Unlit\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.unity_shader_name.as_deref(), Some("Unlit"));
        assert_eq!(r.family, MANIFEST_RASTER_FAMILY_ID);
    }

    #[test]
    fn pbs_metallic_not_in_manifest_uses_debug_family() {
        let u = ShaderUpload {
            asset_id: 5,
            file: Some("Shader \"PBSMetallic\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.family, DEBUG_WORLD_NORMALS_FAMILY_ID);
    }
}
