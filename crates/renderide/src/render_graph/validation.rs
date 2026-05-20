//! Render-graph validation modes and diagnostic reports.

use serde::{Deserialize, Serialize};

use super::ids::PassId;
use super::pass::BlackboardSlotKey;

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

/// Validation diagnostics collected while compiling a graph.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphValidationReport {
    /// Validation mode used for this report.
    pub mode: RenderGraphValidationMode,
    /// Diagnostics found during graph build.
    pub diagnostics: Vec<GraphValidationDiagnostic>,
}

impl GraphValidationReport {
    /// Creates an empty report for `mode`.
    pub fn new(mode: RenderGraphValidationMode) -> Self {
        Self {
            mode,
            diagnostics: Vec::new(),
        }
    }

    /// Adds one diagnostic to the report when validation is enabled.
    pub fn push(&mut self, diagnostic: GraphValidationDiagnostic) {
        if self.mode.enabled() {
            self.diagnostics.push(diagnostic);
        }
    }

    /// Returns whether the report has no diagnostics.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Number of diagnostics in the report.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Logs report diagnostics according to the configured validation mode.
    pub fn log(&self) {
        if self.mode == RenderGraphValidationMode::Off || self.diagnostics.is_empty() {
            return;
        }
        for diagnostic in &self.diagnostics {
            match self.mode {
                RenderGraphValidationMode::Warn => {
                    logger::warn!("render graph validation: {diagnostic}");
                }
                RenderGraphValidationMode::Strict => {
                    logger::error!("render graph validation: {diagnostic}");
                }
                RenderGraphValidationMode::Off => {}
            }
        }
    }
}

impl std::fmt::Display for GraphValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} render graph validation diagnostic(s)",
            self.diagnostics.len()
        )
    }
}

/// One graph declaration validation issue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphValidationDiagnostic {
    /// A pass declared a required blackboard read without any declared seed or producer.
    MissingBlackboardProducer {
        /// Pass declaring the read.
        pass: PassId,
        /// Pass name.
        pass_name: String,
        /// Blackboard slot that was required.
        slot: BlackboardSlotKey,
    },
}

impl std::fmt::Display for GraphValidationDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBlackboardProducer {
                pass,
                pass_name,
                slot,
            } => write!(
                f,
                "pass {pass:?} `{pass_name}` reads required blackboard slot `{}` without a declared producer or seed",
                slot.type_name
            ),
        }
    }
}
