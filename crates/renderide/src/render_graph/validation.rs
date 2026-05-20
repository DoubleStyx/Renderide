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
    /// The graph declares more than one external seed for the same blackboard slot.
    DuplicateBlackboardSeed {
        /// Blackboard slot with duplicate seed declarations.
        slot: BlackboardSlotKey,
        /// First seed producer label.
        first_producer: &'static str,
        /// Duplicate seed producer label.
        duplicate_producer: &'static str,
    },
    /// One pass declared the same blackboard access more than once.
    DuplicateBlackboardAccess {
        /// Pass declaring the duplicate access.
        pass: PassId,
        /// Pass name.
        pass_name: String,
        /// Blackboard slot being accessed.
        slot: BlackboardSlotKey,
        /// Access kind label.
        kind: &'static str,
    },
    /// More than one pass appears to be a primary writer for the same slot.
    AmbiguousBlackboardWriters {
        /// Blackboard slot with multiple pure writers.
        slot: BlackboardSlotKey,
        /// First writer pass id.
        first_pass: PassId,
        /// First writer pass name.
        first_pass_name: String,
        /// Second writer pass id.
        second_pass: PassId,
        /// Second writer pass name.
        second_pass_name: String,
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
            Self::DuplicateBlackboardSeed {
                slot,
                first_producer,
                duplicate_producer,
            } => write!(
                f,
                "blackboard slot `{}` is seeded more than once (`{first_producer}` and `{duplicate_producer}`)",
                slot.type_name
            ),
            Self::DuplicateBlackboardAccess {
                pass,
                pass_name,
                slot,
                kind,
            } => write!(
                f,
                "pass {pass:?} `{pass_name}` declares duplicate {kind} access for blackboard slot `{}`",
                slot.type_name
            ),
            Self::AmbiguousBlackboardWriters {
                slot,
                first_pass,
                first_pass_name,
                second_pass,
                second_pass_name,
            } => write!(
                f,
                "blackboard slot `{}` has multiple primary writers: {first_pass:?} `{first_pass_name}` and {second_pass:?} `{second_pass_name}`",
                slot.type_name
            ),
        }
    }
}
