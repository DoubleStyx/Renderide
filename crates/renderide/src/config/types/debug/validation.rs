//! Render-graph validation policy persisted under `[debug]`.

use serde::{Deserialize, Serialize};

/// Runtime policy for render-graph declaration and execution validation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderGraphValidationMode {
    /// Validation is disabled except for invariants already required by graph construction.
    Off,
    /// Validation diagnostics are logged and exposed but do not stop execution.
    #[default]
    Warn,
    /// Validation diagnostics become build or execute errors.
    Strict,
}

impl RenderGraphValidationMode {
    /// Every validation mode in renderer-config display order.
    pub const ALL: &'static [Self] = &[Self::Off, Self::Warn, Self::Strict];

    /// Human-readable label for the renderer config HUD.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Warn => "Warn",
            Self::Strict => "Strict",
        }
    }

    /// Returns whether this mode should collect diagnostics.
    pub const fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Returns whether diagnostics should be treated as errors.
    pub const fn is_strict(self) -> bool {
        matches!(self, Self::Strict)
    }
}
