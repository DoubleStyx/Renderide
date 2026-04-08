//! Resolves [`ShaderUpload`](crate::shared::ShaderUpload) to a [`RasterPipelineKind`] for [`MaterialRegistry`](crate::materials::MaterialRegistry).
//!
//! Extraction of Unity logical names lives in [`super::logical_name`] and [`super::unity_asset`].
//! [`resolve_shader_upload`] uses
//! [`super::logical_name::resolve_shader_routing_name_from_upload`] so filesystem paths prefer raw
//! AssetBundle / container stems before ShaderLab first-token canonicalization.
//!
//! Names with an embedded `{logical}_default` WGSL target (see [`crate::materials::embedded_shader_stem`]) resolve to
//! [`RasterPipelineKind::EmbeddedStem`]; unknown or non-embedded shaders use
//! [`RasterPipelineKind::DebugWorldNormals`] as the **only** mesh fallback
//! (there is no separate solid-color pipeline).

use std::sync::Arc;

use crate::materials::{embedded_default_stem_for_unity_name, RasterPipelineKind};

use crate::shared::ShaderUpload;

use super::logical_name;

/// Resolved upload: optional Unity-style logical name plus the raster pipeline kind for pipeline selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedShaderUpload {
    /// `Shader "…"` string, container stem, or label when resolution succeeded.
    pub unity_shader_name: Option<String>,
    /// Pipeline kind passed to [`crate::materials::MaterialRegistry::map_shader_route`].
    pub pipeline: RasterPipelineKind,
}

/// Full resolution pipeline for a host [`ShaderUpload`].
pub fn resolve_shader_upload(data: &ShaderUpload) -> ResolvedShaderUpload {
    let unity_shader_name = logical_name::resolve_shader_routing_name_from_upload(data, None);
    let pipeline = match unity_shader_name.as_deref() {
        Some(name) => {
            if let Some(stem) = embedded_default_stem_for_unity_name(name) {
                RasterPipelineKind::EmbeddedStem(Arc::from(stem))
            } else {
                RasterPipelineKind::DebugWorldNormals
            }
        }
        None => RasterPipelineKind::DebugWorldNormals,
    };
    ResolvedShaderUpload {
        unity_shader_name,
        pipeline,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::RasterPipelineKind;

    #[test]
    fn shader_lab_unlit_resolves_embedded_pipeline() {
        let u = ShaderUpload {
            asset_id: 1,
            file: Some("Shader \"Unlit\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert!(matches!(r.pipeline, RasterPipelineKind::EmbeddedStem(_)));
    }

    #[test]
    fn unknown_shader_uses_debug_pipeline() {
        let u = ShaderUpload {
            asset_id: 2,
            file: Some("Shader \"Custom/NoSuchEmbeddedShader\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.pipeline, RasterPipelineKind::DebugWorldNormals);
    }
}
