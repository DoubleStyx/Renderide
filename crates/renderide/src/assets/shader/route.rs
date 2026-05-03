//! Resolves [`ShaderUpload`](crate::shared::ShaderUpload) to a [`RasterPipelineKind`] for [`MaterialRegistry`](crate::materials::MaterialRegistry).
//!
//! The stock host sends an on-disk shader AssetBundle path in [`ShaderUpload::file`]. Routing reads
//! that bundle, extracts the Shader object's `m_Container` asset filename, and maps that filename to
//! an embedded WGSL stem.
//!
//! Names with an embedded `{asset_name}_default` WGSL target (see [`crate::materials::embedded_shader_stem`]) resolve to
//! [`RasterPipelineKind::EmbeddedStem`]; unresolved or non-embedded shaders use
//! [`RasterPipelineKind::Null`] (the black/grey checkerboard) as the **only** mesh fallback
//! (there is no separate solid-color pipeline).
//!
//! The integration harness can bypass AssetBundle parsing by setting [`ShaderUpload::file`] to
//! `RENDERIDE_TEST_STEM:<stem>` (see [`renderide_shared::RENDERIDE_TEST_STEM_PREFIX`]). The prefix
//! is never produced by the production host, so this path is inert outside the test harness.

use std::path::Path;
use std::sync::Arc;

use renderide_shared::RENDERIDE_TEST_STEM_PREFIX;

use crate::materials::{RasterPipelineKind, embedded_default_stem_for_shader_asset_name};

use crate::shared::ShaderUpload;

use super::unity_asset;

/// Resolved upload: optional AssetBundle shader asset name plus the raster pipeline kind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedShaderUpload {
    /// Shader asset filename or stem from the AssetBundle `m_Container` entry.
    pub shader_asset_name: Option<String>,
    /// Pipeline kind passed to [`crate::materials::MaterialRegistry::map_shader_route`].
    pub pipeline: RasterPipelineKind,
}

/// Pure shader route selected from an already-resolved shader asset name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderRoutePlan {
    /// Shader asset filename or stem from the AssetBundle `m_Container` entry.
    pub shader_asset_name: Option<String>,
    /// Pipeline kind passed to [`crate::materials::MaterialRegistry::map_shader_route`].
    pub pipeline: RasterPipelineKind,
}

impl From<ShaderRoutePlan> for ResolvedShaderUpload {
    fn from(plan: ShaderRoutePlan) -> Self {
        Self {
            shader_asset_name: plan.shader_asset_name,
            pipeline: plan.pipeline,
        }
    }
}

/// Selects the raster route for an optional shader asset name without filesystem access.
pub fn plan_shader_route(shader_asset_name: Option<String>) -> ShaderRoutePlan {
    let pipeline = match shader_asset_name.as_deref() {
        Some(name) => {
            if let Some(stem) = embedded_default_stem_for_shader_asset_name(name) {
                RasterPipelineKind::EmbeddedStem(Arc::from(stem))
            } else {
                RasterPipelineKind::Null
            }
        }
        None => RasterPipelineKind::Null,
    };
    ShaderRoutePlan {
        shader_asset_name,
        pipeline,
    }
}

/// Full resolution pipeline for a host [`ShaderUpload`].
pub fn resolve_shader_upload(data: &ShaderUpload) -> ResolvedShaderUpload {
    if let Some(suffix) = data
        .file
        .as_deref()
        .and_then(|f| f.strip_prefix(RENDERIDE_TEST_STEM_PREFIX))
    {
        let stem = normalize_test_stem_suffix(suffix);
        return plan_shader_route(Some(stem)).into();
    }
    let shader_asset_name = data
        .file
        .as_deref()
        .and_then(|file| unity_asset::try_resolve_shader_asset_name_from_path(Path::new(file)));
    plan_shader_route(shader_asset_name).into()
}

/// Normalizes a sentinel-prefix suffix the way the AssetBundle path resolves a `m_Container`
/// entry: drop a trailing `.shader` (case-insensitive) and lowercase. Lets the harness pass a
/// production-style name like `Unlit.shader` and have it match the embedded `unlit_default`
/// stem the same way the production host's AssetBundle entry would.
fn normalize_test_stem_suffix(suffix: &str) -> String {
    let trimmed = suffix.trim();
    let without_ext = trimmed
        .strip_suffix(".shader")
        .or_else(|| trimmed.strip_suffix(".SHADER"))
        .or_else(|| trimmed.strip_suffix(".Shader"))
        .unwrap_or(trimmed);
    without_ext.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::RasterPipelineKind;

    #[test]
    fn missing_file_uses_null_pipeline() {
        let u = ShaderUpload {
            asset_id: 1,
            file: None,
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn inline_shader_lab_text_is_not_a_routing_source() {
        let u = ShaderUpload {
            asset_id: 2,
            file: Some("Shader \"Unlit\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name, None);
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
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn route_plan_resolves_known_embedded_shader_name() {
        let r = plan_shader_route(Some("ui_textunlit".to_string()));

        assert_eq!(r.shader_asset_name.as_deref(), Some("ui_textunlit"));
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }

    #[test]
    fn route_plan_uses_null_for_unknown_name() {
        let r = plan_shader_route(Some("definitely_missing_shader".to_string()));

        assert_eq!(
            r.shader_asset_name.as_deref(),
            Some("definitely_missing_shader")
        );
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn stem_prefix_resolves_to_embedded_stem() {
        let u = ShaderUpload {
            asset_id: 7,
            file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}ui_textunlit")),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name.as_deref(), Some("ui_textunlit"));
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
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name.as_deref(), Some(nonexistent));
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
        let r = resolve_shader_upload(&u);
        assert_eq!(
            r.shader_asset_name.as_deref(),
            Some("definitely_missing_shader")
        );
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn stem_prefix_strips_dot_shader_suffix_and_lowercases() {
        let u = ShaderUpload {
            asset_id: 10,
            file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}Unlit.shader")),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name.as_deref(), Some("unlit"));
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }

    #[test]
    fn stem_prefix_accepts_uppercase_dot_shader_suffix() {
        let u = ShaderUpload {
            asset_id: 11,
            file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}TextureDebug.SHADER")),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name.as_deref(), Some("texturedebug"));
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }
}
