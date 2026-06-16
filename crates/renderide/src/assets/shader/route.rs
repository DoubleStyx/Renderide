//! Resolves [`ShaderUpload`](crate::shared::ShaderUpload) to a [`RasterPipelineKind`] for [`MaterialRegistry`](crate::materials::MaterialRegistry).
//!
//! The stock host sends an on-disk shader AssetBundle path in [`ShaderUpload::file`]. Routing reads
//! that bundle, extracts the Shader object's `m_Container` asset filename, and maps that filename to
//! an embedded WGSL stem. The serialized Shader object's internal name is also parsed for a Froox
//! variant suffix, but that suffix is not used to choose the shader route.
//!
//! Names with an embedded `{asset_name}_default` WGSL target resolve to
//! [`RasterPipelineKind::EmbeddedStem`]; unresolved or non-embedded shaders use
//! [`RasterPipelineKind::Null`] (the black/grey checkerboard) as the **only** mesh fallback
//! (there is no separate solid-color pipeline).
//!
//! The integration harness can bypass AssetBundle parsing by setting [`ShaderUpload::file`] to
//! `RENDERIDE_TEST_STEM:<stem>` (see [`renderide_shared::test_hooks::RENDERIDE_TEST_STEM_PREFIX`]). The prefix
//! is never produced by the production host, so this path is inert outside the test harness.

use std::path::Path;
use std::sync::Arc;

use renderide_shared::test_hooks::RENDERIDE_TEST_STEM_PREFIX;

use crate::render_contract::RasterPipelineKind;

use crate::shared::ShaderUpload;

use super::unity_asset;

/// Lookup function mapping a Unity shader asset name to the default embedded material stem.
pub type ShaderRouteStemLookup = fn(&str) -> Option<String>;

/// Resolved upload: optional AssetBundle shader asset name plus the raster pipeline kind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedShaderUpload {
    /// Shader asset filename or stem from the AssetBundle `m_Container` entry.
    pub shader_asset_name: Option<String>,
    /// Froox shader variant bitmask parsed from the internal Shader name suffix, when available.
    pub shader_variant_bits: Option<u32>,
    /// Pipeline kind passed to [`crate::materials::MaterialRegistry::map_shader_route`].
    pub pipeline: RasterPipelineKind,
}

/// Pure shader route selected from an already-resolved shader asset name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderRoutePlan {
    /// Shader asset filename or stem from the AssetBundle `m_Container` entry.
    pub shader_asset_name: Option<String>,
    /// Froox shader variant bitmask parsed from the internal Shader name suffix, when available.
    pub shader_variant_bits: Option<u32>,
    /// Pipeline kind passed to [`crate::materials::MaterialRegistry::map_shader_route`].
    pub pipeline: RasterPipelineKind,
}

impl From<ShaderRoutePlan> for ResolvedShaderUpload {
    fn from(plan: ShaderRoutePlan) -> Self {
        Self {
            shader_asset_name: plan.shader_asset_name,
            shader_variant_bits: plan.shader_variant_bits,
            pipeline: plan.pipeline,
        }
    }
}

/// Selects the raster route for an optional shader asset name without filesystem access.
pub fn plan_shader_route(
    shader_asset_name: Option<String>,
    shader_variant_bits: Option<u32>,
    stem_lookup: ShaderRouteStemLookup,
) -> ShaderRoutePlan {
    let pipeline = match shader_asset_name.as_deref() {
        Some(name) => {
            if let Some(stem) = stem_lookup(name) {
                RasterPipelineKind::EmbeddedStem(Arc::from(stem))
            } else {
                RasterPipelineKind::Null
            }
        }
        None => RasterPipelineKind::Null,
    };
    ShaderRoutePlan {
        shader_asset_name,
        shader_variant_bits,
        pipeline,
    }
}

/// Full resolution pipeline for a host [`ShaderUpload`].
pub fn resolve_shader_upload(
    data: &ShaderUpload,
    stem_lookup: ShaderRouteStemLookup,
) -> ResolvedShaderUpload {
    if let Some(suffix) = data
        .file
        .as_deref()
        .and_then(|f| f.strip_prefix(RENDERIDE_TEST_STEM_PREFIX))
    {
        let (stem, shader_variant_bits) = normalize_test_stem_suffix(suffix);
        return plan_shader_route(Some(stem), shader_variant_bits, stem_lookup).into();
    }
    let resolved = data
        .file
        .as_deref()
        .and_then(|file| unity_asset::try_resolve_shader_asset_name_from_path(Path::new(file)));
    let shader_asset_name = resolved
        .as_ref()
        .map(|resolved| resolved.shader_asset_name.clone());
    let shader_variant_bits = resolved.and_then(|resolved| resolved.shader_variant_bits);
    plan_shader_route(shader_asset_name, shader_variant_bits, stem_lookup).into()
}

/// Normalizes a sentinel-prefix suffix the way the AssetBundle path resolves a `m_Container`
/// entry: drop a trailing `.shader` (case-insensitive) and lowercase. Lets the harness pass a
/// production-style name like `Unlit.shader` and have it match the embedded `unlit_default`
/// stem the same way the production host's AssetBundle entry would.
fn normalize_test_stem_suffix(suffix: &str) -> (String, Option<u32>) {
    let trimmed = suffix.trim();
    let without_ext = trimmed
        .strip_suffix(".shader")
        .or_else(|| trimmed.strip_suffix(".SHADER"))
        .or_else(|| trimmed.strip_suffix(".Shader"))
        .unwrap_or(trimmed);
    let (stem, shader_variant_bits) = split_variant_suffix(without_ext)
        .map_or((without_ext, None), |(stem, bits)| (stem, Some(bits)));
    (stem.to_ascii_lowercase(), shader_variant_bits)
}

fn split_variant_suffix(name: &str) -> Option<(&str, u32)> {
    let (stem, suffix) = name.rsplit_once('_')?;
    if stem.trim().is_empty() || suffix.len() != 8 || !suffix.chars().all(|c| c.is_ascii_hexdigit())
    {
        return None;
    }
    u32::from_str_radix(suffix, 16)
        .ok()
        .map(|bits| (stem, bits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::embedded_default_stem_for_shader_asset_name;

    fn missing_stem_lookup(_name: &str) -> Option<String> {
        None
    }

    #[test]
    fn missing_file_uses_null_pipeline() {
        let u = ShaderUpload {
            asset_id: 1,
            file: None,
        };
        let r = resolve_shader_upload(&u, missing_stem_lookup);
        assert_eq!(r.shader_asset_name, None);
        assert_eq!(r.shader_variant_bits, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn inline_shader_lab_text_is_not_a_routing_source() {
        let u = ShaderUpload {
            asset_id: 2,
            file: Some("Shader \"Unlit\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u, missing_stem_lookup);
        assert_eq!(r.shader_asset_name, None);
        assert_eq!(r.shader_variant_bits, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn non_assetbundle_file_uses_null_pipeline() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("unlit.shader");
        std::fs::write(&path, "Shader \"Unlit\" { }").expect("write test shader text");
        let u = ShaderUpload {
            asset_id: 3,
            file: Some(path.to_string_lossy().to_string()),
        };
        let r = resolve_shader_upload(&u, missing_stem_lookup);
        assert_eq!(r.shader_asset_name, None);
        assert_eq!(r.shader_variant_bits, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn route_plan_resolves_known_embedded_shader_name() {
        let r = plan_shader_route(
            Some("ui_textunlit".to_string()),
            None,
            embedded_default_stem_for_shader_asset_name,
        );

        assert_eq!(r.shader_asset_name.as_deref(), Some("ui_textunlit"));
        assert_eq!(r.shader_variant_bits, None);
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }

    #[test]
    fn route_plan_preserves_variant_bits_for_pbslerpspecular() {
        let r = plan_shader_route(
            Some("pbslerpspecular".to_string()),
            Some(0xB1),
            embedded_default_stem_for_shader_asset_name,
        );

        assert_eq!(r.shader_asset_name.as_deref(), Some("pbslerpspecular"));
        assert_eq!(r.shader_variant_bits, Some(0xB1));
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }

    #[test]
    fn route_plan_uses_null_for_unknown_name() {
        let r = plan_shader_route(
            Some("definitely_missing_shader".to_string()),
            None,
            missing_stem_lookup,
        );

        assert_eq!(
            r.shader_asset_name.as_deref(),
            Some("definitely_missing_shader")
        );
        assert_eq!(r.shader_variant_bits, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn stem_prefix_resolves_to_embedded_stem() {
        let u = ShaderUpload {
            asset_id: 7,
            file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}ui_textunlit")),
        };
        let r = resolve_shader_upload(&u, embedded_default_stem_for_shader_asset_name);
        assert_eq!(r.shader_asset_name.as_deref(), Some("ui_textunlit"));
        assert_eq!(r.shader_variant_bits, None);
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }

    #[test]
    fn stem_prefix_avoids_filesystem_lookup() {
        let nonexistent = "/this/path/should/never/exist/on/disk/anywhere/zzz";
        assert!(!Path::new(nonexistent).exists());
        let u = ShaderUpload {
            asset_id: 8,
            file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}{nonexistent}")),
        };
        let r = resolve_shader_upload(&u, missing_stem_lookup);
        assert_eq!(r.shader_asset_name.as_deref(), Some(nonexistent));
        assert_eq!(r.shader_variant_bits, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn stem_prefix_with_unknown_stem_falls_back_to_null() {
        let u = ShaderUpload {
            asset_id: 9,
            file: Some(format!(
                "{RENDERIDE_TEST_STEM_PREFIX}definitely_missing_shader"
            )),
        };
        let r = resolve_shader_upload(&u, embedded_default_stem_for_shader_asset_name);
        assert_eq!(
            r.shader_asset_name.as_deref(),
            Some("definitely_missing_shader")
        );
        assert_eq!(r.shader_variant_bits, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn stem_prefix_strips_dot_shader_suffix_and_lowercases() {
        let u = ShaderUpload {
            asset_id: 10,
            file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}Unlit.shader")),
        };
        let r = resolve_shader_upload(&u, embedded_default_stem_for_shader_asset_name);
        assert_eq!(r.shader_asset_name.as_deref(), Some("unlit"));
        assert_eq!(r.shader_variant_bits, None);
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }

    #[test]
    fn stem_prefix_accepts_uppercase_dot_shader_suffix() {
        let u = ShaderUpload {
            asset_id: 11,
            file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}TextureDebug.SHADER")),
        };
        let r = resolve_shader_upload(&u, embedded_default_stem_for_shader_asset_name);
        assert_eq!(r.shader_asset_name.as_deref(), Some("texturedebug"));
        assert_eq!(r.shader_variant_bits, None);
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }

    #[test]
    fn stem_prefix_strips_variant_suffix_for_route_and_preserves_bits() {
        let u = ShaderUpload {
            asset_id: 12,
            file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}Unlit_00002202.shader")),
        };
        let r = resolve_shader_upload(&u, embedded_default_stem_for_shader_asset_name);
        assert_eq!(r.shader_asset_name.as_deref(), Some("unlit"));
        assert_eq!(r.shader_variant_bits, Some(0x2202));
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }

    #[test]
    fn stem_prefix_accepts_unlit_texture_variant_bits() {
        let u = ShaderUpload {
            asset_id: 13,
            file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}Unlit_00000200.shader")),
        };
        let r = resolve_shader_upload(&u, embedded_default_stem_for_shader_asset_name);
        assert_eq!(r.shader_asset_name.as_deref(), Some("unlit"));
        assert_eq!(r.shader_variant_bits, Some(0x200));
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }
}
