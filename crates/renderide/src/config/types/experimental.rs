//! Experimental renderer settings. Persisted as `[experimental]`.

use serde::{Deserialize, Serialize};

/// Feature flags for renderer behavior that is still experimental.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentalSettings {
    /// Whether reflection probes may contribute SH2 indirect diffuse lighting.
    pub reflection_probe_sh2_enabled: bool,
}
