//! Concrete screen-space render passes registered around the main forward graph.
//!
//! The regular post-processing chain currently ships with five effects, executed in this order:
//! 1. [`AutoExposureEffect`] -- histogram-based exposure adaptation (pre-bloom HDR scale).
//! 2. [`BloomEffect`] -- dual-filter physically-based bloom (post-exposure, pre-tonemap HDR scatter).
//! 3. [`MotionBlurEffect`] -- screen-space HDR motion blur (post-bloom, pre-tonemap).
//! 4. [`AcesTonemapEffect`] -- Stephen Hill ACES Fitted tonemap when selected.
//! 5. [`AgxTonemapEffect`] -- analytic AgX tonemap when selected.
//!
//! [`GtaoEffect`] is a screen-space effect too, but it is registered by the main graph before
//! transparent rendering so Unity-style opaque-only AO does not darken late transparent pixels.
//!
//! Future effects (color grading, etc.) live alongside them as sibling sub-modules and implement
//! [`crate::render_graph::post_process_chain::PostProcessEffect`].

mod aces_tonemap;
mod agx_tonemap;
mod auto_exposure;
mod bloom;
mod fullscreen_tonemap;
mod gtao;
mod motion_blur;
pub(crate) mod settings_slots;

pub use aces_tonemap::AcesTonemapEffect;
pub use agx_tonemap::AgxTonemapEffect;
pub use auto_exposure::AutoExposureEffect;
pub(crate) use auto_exposure::AutoExposureStateCache;
pub use bloom::BloomEffect;
pub(crate) use gtao::{GtaoEffect, GtaoGraphResources, GtaoPassRange, gpu_supports_gtao};
pub use motion_blur::MotionBlurEffect;
pub(crate) use motion_blur::MotionBlurStateCache;

/// Returns whether a graph view should execute the post-processing chain.
pub(crate) fn view_post_processing_enabled(
    view: &crate::graph_inputs::GraphPassFrameView<'_>,
) -> bool {
    view.post_processing.is_enabled()
}
