//! Concrete post-processing render passes registered on the
//! [`crate::render_graph::post_process_chain::PostProcessChain`].
//!
//! The chain currently ships with five effects, executed in this order:
//! 1. [`GtaoEffect`] -- Ground-Truth Ambient Occlusion (pre-tonemap HDR modulation, with a
//!    depth-aware bilateral denoise stage between AO production and apply).
//! 2. [`AutoExposureEffect`] -- histogram-based exposure adaptation (pre-bloom HDR scale).
//! 3. [`BloomEffect`] -- dual-filter physically-based bloom (post-exposure, pre-tonemap HDR scatter).
//! 4. [`MotionBlurEffect`] -- screen-space HDR motion blur (post-bloom, pre-tonemap).
//! 5. [`AcesTonemapEffect`] -- Stephen Hill ACES Fitted tonemap when selected.
//! 6. [`AgxTonemapEffect`] -- analytic AgX tonemap when selected.
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

pub use aces_tonemap::AcesTonemapEffect;
pub use agx_tonemap::AgxTonemapEffect;
pub use auto_exposure::AutoExposureEffect;
pub(crate) use auto_exposure::AutoExposureStateCache;
pub use bloom::BloomEffect;
pub use gtao::GtaoEffect;
pub(crate) use gtao::gpu_supports_gtao;
pub use motion_blur::MotionBlurEffect;
pub(crate) use motion_blur::MotionBlurStateCache;

/// Returns whether a graph view should execute the post-processing chain.
pub(crate) fn view_post_processing_enabled(
    view: &crate::render_graph::frame_params::GraphPassFrameView<'_>,
) -> bool {
    view.post_processing.is_enabled()
}
