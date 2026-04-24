//! Concrete post-processing render passes registered on the
//! [`crate::render_graph::post_processing::PostProcessChain`].
//!
//! The chain currently ships with two effects, executed in this order:
//! 1. [`GtaoPass`] — Ground-Truth Ambient Occlusion (pre-tonemap HDR modulation).
//! 2. [`AcesTonemapPass`] — Stephen Hill ACES Fitted tonemap.
//!
//! Future effects (bloom, color grading, etc.) live alongside them as sibling sub-modules and
//! implement [`crate::render_graph::post_processing::PostProcessEffect`].

mod aces_tonemap;
mod gtao;

pub use aces_tonemap::{AcesTonemapEffect, AcesTonemapGraphResources, AcesTonemapPass};
pub use gtao::{GtaoEffect, GtaoGraphResources, GtaoPass};
