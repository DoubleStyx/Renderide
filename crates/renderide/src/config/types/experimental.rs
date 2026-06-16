//! Experimental renderer settings. Persisted as `[experimental]`.

use serde::{Deserialize, Serialize};

use crate::render_contract::MAX_LOCAL_REFLECTION_PROBES;

/// Feature flags for renderer behavior that is still experimental.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentalSettings {
    /// Maximum number of local reflection probes that can contribute to reflections on a single mesh.
    pub max_local_reflection_probes: usize,
    /// Whether reflection probes may contribute SH2 indirect diffuse lighting.
    pub reflection_probe_sh2_enabled: bool,
    /// Whether runtime shader package WGSL edits invalidate and reload material pipelines in development builds.
    pub material_shader_hot_reload_enabled: bool,
}

impl Default for ExperimentalSettings {
    fn default() -> Self {
        Self {
            max_local_reflection_probes: MAX_LOCAL_REFLECTION_PROBES,
            reflection_probe_sh2_enabled: false,
            material_shader_hot_reload_enabled: false,
        }
    }
}

impl ExperimentalSettings {
    /// Returns the local reflection-probe count clamped to the fixed per-draw packing capacity.
    #[must_use]
    pub fn effective_max_local_reflection_probes(self) -> usize {
        self.max_local_reflection_probes
            .min(MAX_LOCAL_REFLECTION_PROBES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_local_reflection_probe_limit_matches_packed_capacity() {
        let settings = ExperimentalSettings::default();

        assert_eq!(
            settings.max_local_reflection_probes,
            MAX_LOCAL_REFLECTION_PROBES
        );
        assert_eq!(
            settings.effective_max_local_reflection_probes(),
            MAX_LOCAL_REFLECTION_PROBES
        );
    }

    #[test]
    fn effective_local_reflection_probe_limit_clamps_to_packed_capacity() {
        let settings = ExperimentalSettings {
            max_local_reflection_probes: MAX_LOCAL_REFLECTION_PROBES + 10,
            ..Default::default()
        };

        assert_eq!(
            settings.effective_max_local_reflection_probes(),
            MAX_LOCAL_REFLECTION_PROBES
        );
    }
}
