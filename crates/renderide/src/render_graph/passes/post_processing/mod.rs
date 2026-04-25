//! Concrete post-processing render passes registered on the
//! [`crate::render_graph::post_processing::PostProcessChain`].
//!
//! The chain currently ships with three effects, executed in this order:
//! 1. [`GtaoPass`] — Ground-Truth Ambient Occlusion (pre-tonemap HDR modulation).
//! 2. [`BloomEffect`] — dual-filter physically-based bloom (pre-tonemap HDR scatter).
//! 3. [`AcesTonemapPass`] — Stephen Hill ACES Fitted tonemap.
//!
//! Future effects (color grading, etc.) live alongside them as sibling sub-modules and implement
//! [`crate::render_graph::post_processing::PostProcessEffect`].

mod aces_tonemap;
mod bloom;
mod gtao;

pub use aces_tonemap::{AcesTonemapEffect, AcesTonemapGraphResources, AcesTonemapPass};
pub use bloom::BloomEffect;
pub use gtao::{GtaoEffect, GtaoGraphResources, GtaoPass};
