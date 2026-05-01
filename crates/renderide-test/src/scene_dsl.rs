//! Modular scene-driven integration test surface.
//!
//! Each integration test case is described by an [`cases::IntegrationCase`] (name, golden
//! path, render resolution, comparison [`tolerance::Tolerance`]) and dispatches to one of the
//! built-in scene templates that drive the harness end-to-end. New cases are added by
//! authoring a builder in [`cases`] and wiring its template into
//! [`runner::run_integration_case`].
//!
//! This is the modular spine for the integration suite: golden comparison, output layout, and
//! report emission all flow through this module. The intent is to grow the suite by adding
//! cases and templates rather than copy-pasting end-to-end orchestration.

pub mod cases;
pub mod output;
pub mod runner;
pub mod tolerance;

pub use cases::{CaseTemplate, IntegrationCase};
pub use output::CaseOutputLayout;
pub use runner::run_integration_case;
pub use tolerance::{Combine, CriterionResult, Tolerance, ToleranceEvaluation};
