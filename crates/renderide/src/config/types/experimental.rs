//! Experimental renderer settings. Persisted as `[experimental]`.

use serde::{Deserialize, Serialize};

/// Feature flags for renderer behavior that is still experimental.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentalSettings {
    /// Whether reflection probes may contribute SH2 indirect diffuse lighting.
    pub reflection_probe_sh2_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::ExperimentalSettings;

    #[test]
    fn default_keeps_reflection_probe_sh2_disabled() {
        let settings = ExperimentalSettings::default();

        assert!(!settings.reflection_probe_sh2_enabled);
    }

    #[test]
    fn missing_field_defaults_to_disabled() {
        let settings: ExperimentalSettings = toml::from_str("").expect("deserialize");

        assert!(!settings.reflection_probe_sh2_enabled);
    }
}
