//! Public API surface of the renderide-test harness.
//!
//! Exposes the harness modules used by the [`crate::cli::run`] entry point and by integration
//! tests under `tests/`. Cargo integration tests link against this library, not the binary, so
//! every type exercised from `tests/*.rs` is reached through this crate root.

#![warn(missing_docs)]

pub mod cli;
pub mod error;
pub mod golden;
pub mod host;
pub mod logging;
pub mod scene;
pub mod scene_dsl;

pub use error::HarnessError;
