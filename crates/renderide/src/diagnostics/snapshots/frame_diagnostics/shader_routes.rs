//! Host-shader -> renderer-pipeline routing fragment of [`super::FrameDiagnosticsSnapshot`].
//!
//! Implemented routes precede fallback routes; within each group, rows are id-ascending so the
//! HUD displays a stable, scannable list across captures.

use crate::diagnostics::BackendDiagSnapshot;
use crate::materials::RasterPipelineKind;

/// One row in the **Shader routes** tab: identifies the host shader, its backing pipeline, and
/// whether the renderer has a real embedded shader for it or falls back to `null`.
#[derive(Clone, Debug)]
pub struct ShaderRouteRow {
    /// Host-assigned shader asset id.
    pub shader_asset_id: i32,
    /// Shader asset filename extracted from the uploaded AssetBundle `m_Container` entry.
    pub shader_asset_name: Option<String>,
    /// Human-readable pipeline label (composed stem, or `null`).
    pub pipeline_label: String,
    /// True when the route resolved to a real embedded shader; false when it fell back to the null fallback.
    pub implemented: bool,
}

/// Sorted shader route rows for the **Shader routes** tab.
#[derive(Clone, Debug, Default)]
pub struct ShaderRoutesFragment {
    /// Implemented routes first, then fallbacks; id-ascending within each group.
    pub rows: Vec<ShaderRouteRow>,
}

impl ShaderRoutesFragment {
    /// Builds the fragment from the backend snapshot's raw routing rows.
    pub fn capture(backend: &BackendDiagSnapshot) -> Self {
        let mut rows: Vec<ShaderRouteRow> = backend
            .shader_routes
            .iter()
            .map(|row| {
                let implemented = !matches!(row.pipeline, RasterPipelineKind::Null);
                let pipeline_label = match &row.pipeline {
                    RasterPipelineKind::EmbeddedStem(stem) => stem.to_string(),
                    RasterPipelineKind::Null => "null".to_string(),
                };
                ShaderRouteRow {
                    shader_asset_id: row.shader_asset_id,
                    shader_asset_name: row.shader_asset_name.clone(),
                    pipeline_label,
                    implemented,
                }
            })
            .collect();
        rows.sort_by(|a, b| {
            b.implemented
                .cmp(&a.implemented)
                .then(a.shader_asset_id.cmp(&b.shader_asset_id))
        });
        Self { rows }
    }
}

#[cfg(test)]
mod tests {
    use super::{ShaderRouteRow, ShaderRoutesFragment};

    fn row(id: i32, implemented: bool) -> ShaderRouteRow {
        ShaderRouteRow {
            shader_asset_id: id,
            shader_asset_name: None,
            pipeline_label: if implemented {
                "stem".to_string()
            } else {
                "null".to_string()
            },
            implemented,
        }
    }

    #[test]
    fn sort_groups_implemented_first_then_id_ascending() {
        let mut f = ShaderRoutesFragment {
            rows: vec![row(7, false), row(3, true), row(2, false), row(5, true)],
        };
        f.rows.sort_by(|a, b| {
            b.implemented
                .cmp(&a.implemented)
                .then(a.shader_asset_id.cmp(&b.shader_asset_id))
        });
        let order: Vec<(i32, bool)> = f
            .rows
            .iter()
            .map(|r| (r.shader_asset_id, r.implemented))
            .collect();
        assert_eq!(
            order,
            vec![(3, true), (5, true), (2, false), (7, false)],
            "implemented routes precede fallback routes; within each group id-ascending"
        );
    }
}
